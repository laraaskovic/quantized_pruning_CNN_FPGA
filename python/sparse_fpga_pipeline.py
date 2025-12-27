"""
End-to-end demo: train a tiny CNN, prune to induce sparsity, export compressed
weights for FPGA, simulate dense vs zero-skipping sparse inference, and render
comparison plots + a simple block diagram.
"""

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
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Dataset + model definitions
# ----------------------------


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


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 4 * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ----------------------------
# Training / eval / pruning
# ----------------------------


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
    idx_path = os.path.join(out_dir, "weights_indices.bin")
    val_path = os.path.join(out_dir, "weights_values.bin")
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
            out = F.avg_pool2d(out, 4)
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
    set_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_hw = (16, 16)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1) data + train dense baseline
    train_loader, test_loader = make_synthetic_dataset(size=input_hw[0])
    model = TinyCNN().to(device)
    train_model(model, train_loader, device, epochs=8)
    dense_acc = evaluate(model, test_loader, device)

    dense_macs = count_dense_macs(model, input_hw)

    # 2) prune and evaluate sparse model
    apply_pruning(model, amount=0.7)
    sparse_acc = evaluate(model, test_loader, device)
    layers, idx_mm, val_mm = compress_model_weights(model, input_hw, out_dir)
    sparse_macs = count_sparse_macs(layers)

    # 3) Sparse forward check on a small batch
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)
    dense_out = model(sample_x)
    sparse_out = sparse_forward(sample_x, layers, idx_mm, val_mm)
    diff = (dense_out - sparse_out).abs().max().item()

    # 4) Charts
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
        "dense_accuracy": dense_acc,
        "sparse_accuracy": sparse_acc,
        "dense_macs": dense_macs,
        "sparse_macs": sparse_macs,
        "max_output_diff": diff,
        "macs_per_cycle": macs_per_cycle,
        "freq_mhz": freq_mhz,
        "compressed_nonzeros": int(idx_mm.shape[0]),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Pipeline summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
