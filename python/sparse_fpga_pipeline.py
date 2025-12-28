"""
End-to-end demo: train a tiny CNN (real or synthetic data), prune to induce
sparsity, export compressed weights for FPGA, simulate dense vs zero-skipping
sparse inference, add a naive INT8 post-training quantization sweep, and render
comparison plots + a simple block diagram.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_synthetic_dataset(
    n_train: int = 512, n_test: int = 128, size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate a simple 2-class dataset with bright quadrants so a tiny CNN can
    learn quickly without downloads.
    """

    def build_split(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = torch.zeros(n_samples, 1, size, size, dtype=torch.float32)
        ys = torch.zeros(n_samples, dtype=torch.long)
        for i in range(n_samples):
            label = np.random.randint(0, 2)
            ys[i] = label
            cx, cy = (4, 4) if label == 0 else (size - 4, size - 4)
            xs[i, 0, cx - 3 : cx + 3, cy - 3 : cy + 3] = torch.rand(6, 6) * 0.4 + 0.8
            xs[i] += torch.randn_like(xs[i]) * 0.05
        return xs, ys

    x_train, y_train = build_split(n_train)
    x_test, y_test = build_split(n_test)

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    return train_loader, test_loader


def limit_dataset(ds: Dataset, n: int) -> Dataset:
    if n is None or n <= 0 or n >= len(ds):
        return ds
    return Subset(ds, list(range(n)))


def load_real_dataset(
    name: str, data_dir: str, train_limit: int = None, test_limit: int = None
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Load MNIST / FashionMNIST / CIFAR-10 with optional subset limits.
    Returns loaders, num_classes, and input channels.
    """
    name = name.lower()
    if name == "mnist":
        dataset_cls = torchvision.datasets.MNIST
        num_classes = 10
        in_ch = 1
        tx = T.Compose([T.ToTensor()])
    elif name in ("fashion-mnist", "fashion_mnist", "fashion"):
        dataset_cls = torchvision.datasets.FashionMNIST
        num_classes = 10
        in_ch = 1
        tx = T.Compose([T.ToTensor()])
    elif name in ("cifar10", "cifar-10", "cifar"):
        dataset_cls = torchvision.datasets.CIFAR10
        num_classes = 10
        in_ch = 3
        tx = T.Compose([T.ToTensor()])
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_ds = limit_dataset(dataset_cls(root=data_dir, train=True, download=True, transform=tx), train_limit)
    test_ds = limit_dataset(dataset_cls(root=data_dir, train=False, download=True, transform=tx), test_limit)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader, num_classes, in_ch


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, avg_pool_size: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((avg_pool_size, avg_pool_size))
        self.fc = nn.Linear(16 * avg_pool_size * avg_pool_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# Training / eval / pruning

def train_model(
    model: nn.Module, loader: DataLoader, device: torch.device, epochs: int = 5
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def apply_pruning(model: nn.Module, amount: float = 0.7) -> None:
    to_prune = [
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.fc, "weight"),
    ]
    prune.global_unstructured(
        to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    for module, name in to_prune:
        prune.remove(module, name)


def apply_nm_pruning(model: nn.Module, n: int = 2, m: int = 4) -> None:
    """
    Structured N:M pruning per output channel: keep top-n magnitude entries per group of m.
    This preserves a regular pattern that is more scheduler-friendly than unstructured masks.
    """

    def prune_tensor(w: torch.Tensor) -> torch.Tensor:
        flat = w.view(w.shape[0], -1)
        groups = (flat.shape[1] + m - 1) // m
        padded = torch.zeros(flat.shape[0], groups * m, device=flat.device)
        padded[:, : flat.shape[1]] = flat
        padded = padded.view(flat.shape[0], groups, m)
        magnitudes = padded.abs()
        topk = torch.topk(magnitudes, k=n, dim=2, largest=True, sorted=False)
        mask = torch.zeros_like(padded, dtype=flat.dtype)
        # scatter on last dim
        mask.scatter_(2, topk.indices, torch.ones_like(topk.values))
        pruned = padded * mask
        pruned = pruned.view(flat.shape[0], -1)[:, : flat.shape[1]]
        return pruned.view_as(w)

    with torch.no_grad():
        model.conv1.weight.copy_(prune_tensor(model.conv1.weight))
        model.conv2.weight.copy_(prune_tensor(model.conv2.weight))
        model.fc.weight.copy_(prune_tensor(model.fc.weight))


# ----------------------------
# Quantization helpers (naive PTQ)
# ----------------------------


def collect_activation_scales(
    model: nn.Module, loader: DataLoader, device: torch.device, num_batches: int = 5
) -> Dict[str, float]:
    """
    Run a few batches to capture activation ranges for int8 scaling.
    """
    model.eval()
    act_ranges: Dict[str, float] = {"input": 1.0}

    def register_max(name: str):
        def hook(_mod, _inp, out):
            act_ranges[name] = max(act_ranges.get(name, 0.0), out.detach().abs().max().item())
        return hook

    h1 = model.conv1.register_forward_hook(register_max("conv1"))
    h2 = model.conv2.register_forward_hook(register_max("conv2"))
    pooled_max = {"val": 0.0}

    with torch.no_grad():
        for idx, (xb, _yb) in enumerate(loader):
            xb = xb.to(device)
            act_ranges["input"] = max(act_ranges.get("input", 0.0), xb.abs().max().item())
            out1 = F.relu(model.conv1(xb))
            out2 = F.relu(model.conv2(out1))
            pooled = model.avg_pool(out2)
            pooled_max["val"] = max(pooled_max["val"], pooled.abs().max().item())
            if idx + 1 >= num_batches:
                break

    act_ranges["fc_in"] = max(pooled_max["val"], 1e-6)
    h1.remove()
    h2.remove()
    return {k: max(v, 1e-6) for k, v in act_ranges.items()}


def quantize_tensor(t: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.clamp(torch.round(t / scale), -127, 127)
    return q


def quantized_eval(
    model: TinyCNN,
    loader: DataLoader,
    device: torch.device,
    act_ranges: Dict[str, float],
) -> float:
    """
    Simple per-tensor symmetric int8 PTQ simulation using calibration ranges.
    """
    model.eval()
    # Weight scales
    w1_scale = max(model.conv1.weight.detach().abs().max().item(), 1e-6) / 127.0
    w2_scale = max(model.conv2.weight.detach().abs().max().item(), 1e-6) / 127.0
    wf_scale = max(model.fc.weight.detach().abs().max().item(), 1e-6) / 127.0
    conv1_w = quantize_tensor(model.conv1.weight.detach(), w1_scale).to(device)
    conv2_w = quantize_tensor(model.conv2.weight.detach(), w2_scale).to(device)
    fc_w = quantize_tensor(model.fc.weight.detach(), wf_scale).to(device)

    def dequant_scale(max_abs: float) -> float:
        return max_abs / 127.0 if max_abs > 0 else 1.0

    in_scale = dequant_scale(act_ranges.get("input", 1.0))
    a1_scale = dequant_scale(act_ranges.get("conv1", 1.0))
    a2_scale = dequant_scale(act_ranges.get("conv2", 1.0))
    af_scale = dequant_scale(act_ranges.get("fc_in", 1.0))

    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            a = quantize_tensor(xb, in_scale).float() * in_scale
            a = F.conv2d(a, conv1_w.float() * w1_scale, bias=model.conv1.bias, padding=1)
            a = F.relu(a)
            a = quantize_tensor(a, a1_scale).float() * a1_scale

            a = F.conv2d(a, conv2_w.float() * w2_scale, bias=model.conv2.bias, padding=1)
            a = F.relu(a)
            a = quantize_tensor(a, a2_scale).float() * a2_scale

            a = model.avg_pool(a)
            a = quantize_tensor(a, af_scale).float() * af_scale
            a = torch.flatten(a, 1)
            logits = F.linear(a, fc_w.float() * wf_scale, bias=model.fc.bias)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def quantize_model_weights_int8(model: TinyCNN) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Quantize weights per tensor using symmetric int8 scales.
    Returns quantized tensors (int8) and per-layer scales (float).
    """
    q_weights: Dict[str, torch.Tensor] = {}
    scales: Dict[str, float] = {}
    for name, w in [
        ("conv1", model.conv1.weight.detach()),
        ("conv2", model.conv2.weight.detach()),
        ("fc", model.fc.weight.detach()),
    ]:
        max_abs = max(w.abs().max().item(), 1e-6)
        scale = max_abs / 127.0
        q_weights[name] = quantize_tensor(w, scale).to(torch.int8).cpu()
        scales[name] = scale
    return q_weights, scales


def export_int8_hex(
    layers: List["LayerMetadata"],
    idx_mm: np.memmap,
    q_weights: Dict[str, torch.Tensor],
    out_dir: str,
) -> Tuple[str, str]:
    """
    Export int8 sparse weights into BRAM-friendly hex files.
    Two formats:
      - single-entry words (32-bit): {idx[11:0], weight[7:0]} for simple readers
      - packed pair words (40-bit): two entries per line for dual-lane MACs
    Returns paths for both files.
    """
    single_path = os.path.join(out_dir, "weights_int8_single.hex")
    packed_path = os.path.join(out_dir, "weights_int8_packed.hex")
    single_lines: List[str] = []
    packed_lines: List[str] = []
    pair_buf: List[Tuple[int, int]] = []  # (idx, weight)

    for layer in layers:
        qw = flatten_conv_weights(q_weights[layer.name]).view(layer.out_channels, -1)
        base = layer.base_offset
        for oc, seg in enumerate(layer.channel_segments):
            if seg.length == 0:
                continue
            idxs = np.array(idx_mm[base + seg.offset : base + seg.offset + seg.length], copy=False)
            for idx in idxs:
                w_val = int(qw[oc, int(idx)].item())
                entry = ((int(idx) & 0xFFF) << 8) | (w_val & 0xFF)
                single_lines.append(f"{entry:08x}")
                pair_buf.append((int(idx), w_val))
                if len(pair_buf) == 2:
                    lo_idx, lo_w = pair_buf[0]
                    hi_idx, hi_w = pair_buf[1]
                    word = (((hi_idx & 0xFFF) << 8) | (hi_w & 0xFF)) << 20
                    word |= (((lo_idx & 0xFFF) << 8) | (lo_w & 0xFF))
                    packed_lines.append(f"{word:010x}")  # 40 bits -> 10 hex chars
                    pair_buf.clear()

    if len(pair_buf) == 1:
        lo_idx, lo_w = pair_buf[0]
        word = (((0 & 0xFFF) << 8) | (0 & 0xFF)) << 20
        word |= (((lo_idx & 0xFFF) << 8) | (lo_w & 0xFF))
        packed_lines.append(f"{word:010x}")

    with open(single_path, "w", encoding="utf-8") as f:
        f.write("\n".join(single_lines))
    with open(packed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(packed_lines))
    return single_path, packed_path


# ----------------------------
# Compression and metadata
# ----------------------------


@dataclass
class ChannelSegment:
    offset: int  # offset relative to the layer base
    length: int


@dataclass
class LayerMetadata:
    name: str
    kind: str  # "conv" or "linear"
    out_channels: int
    in_channels: int
    kernel_size: Tuple[int, int]  # (h, w) or (1, 1) for linear
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    out_hw: Tuple[int, int]
    channel_segments: List[ChannelSegment]
    dense_weights: int
    base_offset: int  # starting index into the global idx/val arrays
    bias: torch.Tensor


def flatten_conv_weights(weight: torch.Tensor) -> torch.Tensor:
    # reshape to (out_channels, in_channels * k_h * k_w)
    return weight.view(weight.shape[0], -1)


def compress_model_weights(
    model: TinyCNN, input_hw: Tuple[int, int], out_dir: str
) -> Tuple[List[LayerMetadata], np.memmap, np.memmap]:
    """
    Flattens each output channel, stores non-zero (idx, value) pairs into
    memory-mapped binary files for FPGA. Returns metadata and mmap handles.
    """
    os.makedirs(out_dir, exist_ok=True)
    layers: List[LayerMetadata] = []
    all_indices: List[np.ndarray] = []
    all_values: List[np.ndarray] = []
    global_offset = 0

    def handle_layer(
        name: str,
        layer: nn.Module,
        kernel_size,
        stride,
        padding,
        out_hw,
        kind: str,
    ):
        nonlocal global_offset
        w = flatten_conv_weights(layer.weight.detach().cpu())
        channel_segments: List[ChannelSegment] = []
        layer_base = global_offset
        local_cursor = 0
        bias = layer.bias.detach().cpu() if layer.bias is not None else torch.zeros(w.shape[0])
        for oc in range(w.shape[0]):
            nz_mask = w[oc] != 0
            nz_idx = nz_mask.nonzero(as_tuple=False).view(-1).numpy().astype(np.int32)
            nz_val = w[oc, nz_mask].numpy().astype(np.float32)
            all_indices.append(nz_idx)
            all_values.append(nz_val)
            channel_segments.append(ChannelSegment(offset=local_cursor, length=len(nz_idx)))
            local_cursor += len(nz_idx)
            global_offset += len(nz_idx)
        layers.append(
            LayerMetadata(
                name=name,
                kind=kind,
                out_channels=w.shape[0],
                in_channels=w.shape[1] // (kernel_size[0] * kernel_size[1]),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                out_hw=out_hw,
                channel_segments=channel_segments,
                dense_weights=w.numel(),
                base_offset=layer_base,
                bias=bias,
            )
        )

    # conv1
    out1_h = (input_hw[0] + 2 * 1 - 3) // 1 + 1
    out1_w = (input_hw[1] + 2 * 1 - 3) // 1 + 1
    handle_layer(
        "conv1", model.conv1, (3, 3), (1, 1), (1, 1), (out1_h, out1_w), kind="conv"
    )
    # conv2
    out2_h = (out1_h + 2 * 1 - 3) // 1 + 1
    out2_w = (out1_w + 2 * 1 - 3) // 1 + 1
    handle_layer(
        "conv2", model.conv2, (3, 3), (1, 1), (1, 1), (out2_h, out2_w), kind="conv"
    )
    # fc (treat as conv with 1x1)
    handle_layer(
        "fc",
        model.fc,
        (1, 1),
        (1, 1),
        (0, 0),
        (1, 1),
        kind="linear",
    )

    total_nz = sum(len(v) for v in all_indices)
    idx_path = os.path.join(out_dir, "weights_indices_run.bin")
    val_path = os.path.join(out_dir, "weights_values_run.bin")
    idx_mm = np.memmap(idx_path, dtype=np.int32, mode="w+", shape=(total_nz,))
    val_mm = np.memmap(val_path, dtype=np.float32, mode="w+", shape=(total_nz,))
    cursor = 0
    for idxs, vals in zip(all_indices, all_values):
        idx_mm[cursor : cursor + len(idxs)] = idxs
        val_mm[cursor : cursor + len(vals)] = vals
        cursor += len(idxs)
    idx_mm.flush()
    val_mm.flush()

    metadata = {
        "format": "idx_val_memmap_v1",
        "index_file": idx_path,
        "value_file": val_path,
        "index_dtype": "int32",
        "value_dtype": "float32",
        "layers": [
            {
                "name": l.name,
                "kind": l.kind,
                "out_channels": l.out_channels,
                "in_channels": l.in_channels,
                "kernel": list(l.kernel_size),
                "stride": list(l.stride),
                "padding": list(l.padding),
                "out_hw": list(l.out_hw),
                "base_offset": l.base_offset,
                "bias": l.bias.tolist(),
                "channel_segments": [
                    {"offset": seg.offset, "length": seg.length}
                    for seg in l.channel_segments
                ],
                "dense_weights": l.dense_weights,
            }
            for l in layers
        ],
        "total_nonzero": int(total_nz),
    }
    with open(os.path.join(out_dir, "weights_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return layers, idx_mm, val_mm


# ----------------------------
# MAC counting + latency model
# ----------------------------


def count_dense_macs(model: TinyCNN, input_hw: Tuple[int, int]) -> Dict[str, int]:
    h, w = input_hw
    macs = {}
    # conv1
    out1_h = (h + 2 * 1 - 3) // 1 + 1
    out1_w = (w + 2 * 1 - 3) // 1 + 1
    macs["conv1"] = out1_h * out1_w * model.conv1.out_channels * model.conv1.in_channels * 3 * 3
    # conv2
    out2_h = (out1_h + 2 * 1 - 3) // 1 + 1
    out2_w = (out1_w + 2 * 1 - 3) // 1 + 1
    macs["conv2"] = out2_h * out2_w * model.conv2.out_channels * model.conv2.in_channels * 3 * 3
    # avgpool reduces to 4x4
    macs["fc"] = model.fc.in_features * model.fc.out_features
    macs["total"] = macs["conv1"] + macs["conv2"] + macs["fc"]
    return macs


def count_sparse_macs(layers: List[LayerMetadata]) -> Dict[str, int]:
    macs = {}
    total = 0
    for layer in layers:
        layer_macs = 0
        if layer.kind == "conv":
            out_h, out_w = layer.out_hw
            for seg in layer.channel_segments:
                layer_macs += seg.length * out_h * out_w
        else:
            for seg in layer.channel_segments:
                layer_macs += seg.length
        macs[layer.name] = layer_macs
        total += layer_macs
    macs["total"] = total
    return macs


def estimate_latency_us(mac_count: int, macs_per_cycle: int = 4, freq_mhz: int = 200) -> float:
    cycles = math.ceil(mac_count / macs_per_cycle) + 4  # +4 for small pipeline fill/flush
    return cycles / (freq_mhz * 1e6) * 1e6  # microseconds


# ----------------------------
# Sparse inference simulation
# ----------------------------


def sparse_forward(
    x: torch.Tensor,
    metadata: List[LayerMetadata],
    idx_mm: np.memmap,
    val_mm: np.memmap,
    avg_pool_size: int = 4,
) -> torch.Tensor:
    """
    Executes inference using only non-zero weights from the compressed store.
    This is slow (Python loops) but matches zero-skipping behavior.
    """
    device = x.device
    out = x
    for layer in metadata:
        base = layer.base_offset
        if layer.kind == "conv":
            out = sparse_conv2d(out, layer, idx_mm, val_mm, base, device)
            out = F.relu(out)
        else:
            out = sparse_linear(out, layer, idx_mm, val_mm, base, device)
        if layer.name == "conv2":
            out = F.adaptive_avg_pool2d(out, (avg_pool_size, avg_pool_size))
    return out


def sparse_conv2d(
    x: torch.Tensor,
    layer: LayerMetadata,
    idx_mm: np.memmap,
    val_mm: np.memmap,
    base_offset: int,
    device: torch.device,
) -> torch.Tensor:
    batch, _, h, w = x.shape
    k_h, k_w = layer.kernel_size
    pad_h, pad_w = layer.padding
    stride_h, stride_w = layer.stride
    out_h, out_w = layer.out_hw
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
    out = torch.zeros(batch, layer.out_channels, out_h, out_w, device=device)
    bias = layer.bias.to(device)
    for oc, seg in enumerate(layer.channel_segments):
        if seg.length == 0:
            continue
        idxs = np.array(idx_mm[base_offset + seg.offset : base_offset + seg.offset + seg.length], copy=False)
        vals = np.array(val_mm[base_offset + seg.offset : base_offset + seg.offset + seg.length], copy=False)
        idxs_t = torch.from_numpy(idxs.astype(np.int64)).to(device)
        vals_t = torch.from_numpy(vals.astype(np.float32)).to(device)
        for idx_val, weight_val in zip(idxs_t, vals_t):
            ic = int(idx_val.item() // (k_h * k_w))
            kk = int(idx_val.item() % (k_h * k_w))
            kh = kk // k_w
            kw = kk % k_w
            patch = x_pad[:, ic, kh : kh + out_h * stride_h : stride_h, kw : kw + out_w * stride_w : stride_w]
            out[:, oc] += weight_val * patch
        out[:, oc] += bias[oc]
    return out


def sparse_linear(
    x: torch.Tensor,
    layer: LayerMetadata,
    idx_mm: np.memmap,
    val_mm: np.memmap,
    base_offset: int,
    device: torch.device,
) -> torch.Tensor:
    x_flat = torch.flatten(x, 1)
    batch = x_flat.shape[0]
    out = torch.zeros(batch, layer.out_channels, device=device)
    bias = layer.bias.to(device)
    for oc, seg in enumerate(layer.channel_segments):
        if seg.length == 0:
            continue
        idxs = np.array(idx_mm[base_offset + seg.offset : base_offset + seg.offset + seg.length], copy=False)
        vals = np.array(val_mm[base_offset + seg.offset : base_offset + seg.offset + seg.length], copy=False)
        idxs_t = torch.from_numpy(idxs.astype(np.int64)).to(device)
        vals_t = torch.from_numpy(vals.astype(np.float32)).to(device)
        out[:, oc] = (x_flat[:, idxs_t] * vals_t).sum(dim=1) + bias[oc]
    return out


# ----------------------------
# Plotting helpers
# ----------------------------


def plot_mac_chart(dense: Dict[str, int], sparse: Dict[str, int], out_path: str) -> None:
    labels = ["conv1", "conv2", "fc", "total"]
    dense_vals = [dense[k] for k in labels]
    sparse_vals = [sparse[k] for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, dense_vals, width, label="Dense")
    plt.bar(x + width / 2, sparse_vals, width, label="Sparse (zero-skip)")
    plt.xticks(x, labels)
    plt.ylabel("MAC operations")
    plt.title("Total MACs per inference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_latency_chart(
    dense_mac: int, sparse_mac: int, out_path: str, macs_per_cycle: int, freq_mhz: int
) -> None:
    dense_lat = estimate_latency_us(dense_mac, macs_per_cycle, freq_mhz)
    sparse_lat = estimate_latency_us(sparse_mac, macs_per_cycle, freq_mhz)
    plt.figure(figsize=(6, 4))
    plt.bar(["Dense", "Sparse"], [dense_lat, sparse_lat], color=["tab:blue", "tab:green"])
    plt.ylabel("Latency (microseconds)")
    plt.title(f"Estimated latency @ {freq_mhz} MHz, {macs_per_cycle} MAC/cycle")
    for i, v in enumerate([dense_lat, sparse_lat]):
        plt.text(i, v * 1.02, f"{v:.3f} us", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_block_diagram(out_path: str) -> None:
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        ("Input Buffer", (0.05, 0.25), 0.18, 0.5),
        ("Compressed Weight\nStore (idx,val)", (0.28, 0.25), 0.2, 0.5),
        ("Index Decoder", (0.52, 0.25), 0.14, 0.5),
        ("Zero-Skip MAC Array", (0.7, 0.25), 0.2, 0.5),
        ("Accumulator /\nActivation", (0.9, 0.25), 0.18, 0.5),
    ]
    for label, (x, y), w, h in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, color="#d0e1ff", ec="#1f4b99")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    arrows = [
        (0.23, 0.5, 0.28, 0.5),
        (0.48, 0.5, 0.52, 0.5),
        (0.66, 0.5, 0.7, 0.5),
        (0.9, 0.5, 1.06, 0.5),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.29, 0.82, "Non-zero weights only", fontsize=9, ha="left")
    plt.xlim(0, 1.15)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main orchestration
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse-aware FPGA pipeline")
    parser.add_argument("--dataset", type=str, default="mnist", help="mnist|fashion-mnist|cifar10|synthetic")
    parser.add_argument("--train-limit", type=int, default=20000, help="limit training samples (for speed); -1 for all")
    parser.add_argument("--test-limit", type=int, default=5000, help="limit test samples; -1 for all")
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--prune", type=float, default=0.7, help="global sparsity target (0-1)")
    parser.add_argument("--nm-structured", type=str, default="", help="optional N:M pruning, e.g., 2:4; overrides unstructured if set")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset.lower() == "synthetic":
        input_hw = (16, 16)
        train_loader, test_loader = make_synthetic_dataset(size=input_hw[0])
        num_classes = 2
        in_ch = 1
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        train_loader, test_loader, num_classes, in_ch = load_real_dataset(
            args.dataset, data_dir, None if args.train_limit < 0 else args.train_limit, None if args.test_limit < 0 else args.test_limit
        )
        # infer input HW from a batch
        sample_x, _ = next(iter(train_loader))
        input_hw = (sample_x.shape[2], sample_x.shape[3])

    # 1) data + train dense baseline
    model = TinyCNN(in_channels=in_ch, num_classes=num_classes).to(device)
    train_model(model, train_loader, device, epochs=args.epochs)
    dense_acc = evaluate(model, test_loader, device)

    dense_macs = count_dense_macs(model, input_hw)

    # 2) prune and evaluate sparse model
    nm_tuple = None
    if args.nm_structured:
        try:
            n_str, m_str = args.nm_structured.split(":")
            nm_tuple = (int(n_str), int(m_str))
        except Exception:
            nm_tuple = None
    if nm_tuple:
        apply_nm_pruning(model, n=nm_tuple[0], m=nm_tuple[1])
    else:
        apply_pruning(model, amount=args.prune)
    sparse_acc = evaluate(model, test_loader, device)
    layers, idx_mm, val_mm = compress_model_weights(model, input_hw, out_dir)
    sparse_macs = count_sparse_macs(layers)

    # 3) Sparse forward check on a small batch
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)
    dense_out = model(sample_x)
    sparse_out = sparse_forward(sample_x, layers, idx_mm, val_mm)
    diff = (dense_out - sparse_out).abs().max().item()

    # 4) Quantization (naive PTQ)
    act_ranges = collect_activation_scales(model, train_loader, device, num_batches=5)
    quant_acc = quantized_eval(model, test_loader, device, act_ranges)
    q_weights, weight_scales = quantize_model_weights_int8(model)
    int8_single_hex, int8_packed_hex = export_int8_hex(layers, idx_mm, q_weights, out_dir)
    act_scales = {
        "input": act_ranges.get("input", 1.0) / 127.0,
        "conv1": act_ranges.get("conv1", 1.0) / 127.0,
        "conv2": act_ranges.get("conv2", 1.0) / 127.0,
        "fc_in": act_ranges.get("fc_in", 1.0) / 127.0,
    }
    with open(os.path.join(out_dir, "int8_scales.json"), "w", encoding="utf-8") as f:
        json.dump({"weight_scales": weight_scales, "activation_scales": act_scales}, f, indent=2)

    # 5) Charts
    plot_mac_chart(dense_macs, sparse_macs, os.path.join(out_dir, "mac_ops.png"))
    macs_per_cycle = 4
    freq_mhz = 200
    plot_latency_chart(
        dense_macs["total"],
        sparse_macs["total"],
        os.path.join(out_dir, "latency.png"),
        macs_per_cycle,
        freq_mhz,
    )
    plot_block_diagram(os.path.join(out_dir, "block_diagram.png"))

    summary = {
        "dataset": args.dataset,
        "train_limit": args.train_limit,
        "test_limit": args.test_limit,
        "dense_accuracy": dense_acc,
        "sparse_accuracy": sparse_acc,
        "quantized_accuracy": quant_acc,
        "dense_macs": dense_macs,
        "sparse_macs": sparse_macs,
        "max_output_diff": diff,
        "macs_per_cycle": macs_per_cycle,
        "freq_mhz": freq_mhz,
        "compressed_nonzeros": int(idx_mm.shape[0]),
        "prune_amount": args.prune,
        "nm_structured": args.nm_structured or "",
        "int8_weight_hex_single": os.path.abspath(int8_single_hex),
        "int8_weight_hex_packed2": os.path.abspath(int8_packed_hex),
        "int8_scales_json": os.path.abspath(os.path.join(out_dir, "int8_scales.json")),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Pipeline summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
