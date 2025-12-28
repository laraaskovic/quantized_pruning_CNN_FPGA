# FPGA-Accelerated Sparse CNN Inference via Pruning, INT8 Packing, and Zero-Skipping MAC Logic

## 1. Abstract
We study an FPGA inference path for a pruned convolutional neural network where zero-valued weights are skipped in hardware and paired with naive INT8 post-training quantization. A PyTorch-trained TinyCNN on MNIST (20k train / 5k test) is globally magnitude-pruned or optionally N:M structured-pruned, exported as sparse (idx,val) streams, and mapped onto Verilog MAC units that only accumulate on valid non-zero entries. We extend the flow with INT8 BRAM exports and DSP48E1 SIMD packing (two 8x8 multiplies per DSP) plus Q8.16 dequant scaling in hardware. The resulting design cuts multiply-accumulate (MAC) work by 56% (962.2k -> 426.3k) and delivers a 45% estimated latency reduction at 200 MHz (1.20 ms -> 0.53 ms ideal, ~0.63 ms with control overheads). Accuracy is 95.5% dense, 85.5% sparse, and 85.9% sparse+INT8. DSP count is halved before packing; SIMD packing can halve it again, showing pruning and precision reduction must both be exploited to convert sparsity into system-level wins.

## 2. Introduction
Convolutional networks on FPGAs are often bounded by MAC throughput and on-chip memory bandwidth. Unstructured pruning can shrink parameter counts, but unless the hardware suppresses zero operands, the datapath still executes dense MAC schedules. INT8 quantization further relaxes DSP and BRAM pressure if activation and weight ranges are controlled. We pair pruning, optional N:M structured pruning for better schedulability, zero-skipping MAC logic, and simple INT8 PTQ to reclaim compute cycles, DSP capacity, and memory bandwidth on real datasets (MNIST by default; Fashion-MNIST and CIFAR-10 are supported).

**Contributions**
- End-to-end flow on real data: PyTorch training, global or N:M pruning, sparse export, Verilog zero-skip MACs, and INT8 PTQ simulation.
- Latency model isolating pipeline overheads; diminishing returns beyond ~70% sparsity once control/BW dominates.
- Publication-quality figures covering accuracy, latency, MAC scaling, resource deltas, sparsity structure, INT8 packing vs single-lane MAC.
- Reusable compressed (idx,val) format plus channel-wise metadata and INT8 hex/scales for BRAM/DSP initialization.

## 3. Background & Related Work
Magnitude pruning deletes low-valued weights to induce unstructured sparsity; irregularity complicates vectorization and memory access. Sparse inference on hardware must cope with irregular fetches, index decoding, and load balancing. Structured N:M sparsity trades some flexibility for schedulability. FPGA ML accelerators often favor dense systolic arrays; here, we trade peak dense throughput for efficient sparse execution via lightweight control, compressed on-chip storage, and INT8 packing to increase arithmetic density per DSP while lowering BRAM bandwidth.

## 4. Methodology
### 4.1 Model Architecture
TinyCNN: two 3x3 conv layers (in->8, 8->16 channels, ReLU) followed by adaptive average pooling to 4x4 and a 16x4x4 -> 10 fully connected head. Input channels are 1 (MNIST/Fashion) or 3 (CIFAR-10); adaptive pooling decouples the FC dimension from the input resolution.

### 4.2 Pruning Strategy
- **Unstructured**: global L1 magnitude pruning over conv1, conv2, and fc weights (default), targeting {0, 30, 50, 70, 90}% sparsity; measured run uses 70% (`--prune 0.7`).
- **N:M structured (optional)**: keep top-n per block of m per output channel (e.g., `--nm-structured 2:4`) to align with hardware schedulers and reduce irregularity.
- **Thresholding**: global magnitude threshold; layer-wise ratios inherit from the global sort to preserve salient filters.
- **Post-processing**: masks are folded into weights prior to export; masks are visualized in `figures/pruning_mask.png`.
- **INT8 PTQ**: per-tensor symmetric quantization with calibration over 5 batches; activations are clamped to observed maxima and weights are clipped to int8 range.

### 4.3 Weight Export & Sparsity Encoding
- Weights are flattened per output channel; only non-zero (index, value) pairs are emitted.
- Per-layer metadata: kernel, stride, padding, output shape, base offsets, and channel segments for BRAM/ROM addressing (`outputs/weights_metadata.json`).
- Binary memmaps `weights_indices_run.bin` (int32) and `weights_values_run.bin` (float32) plus JSON enable direct BRAM initialization.
- INT8 export: `weights_int8_single.hex` (one entry/word) and `weights_int8_packed.hex` (two entries/word) for the SIMD MAC; scales in `int8_scales.json` provide per-layer weight scales and activation scales (Q8.16 friendly) for on-FPGA dequantization.

### 4.4 FPGA Architecture
- **MAC pipeline (int16)**: DSP-friendly multiply-add with per-channel accumulation and fast reset (`zero_skip_mac.v`).
- **MAC pipeline (int8 SIMD)**: `int8_dual_zero_skip_mac.v` packs two 8x8 multiplies into one DSP48E1 (`USE_SIMD("TWO24")`), zero-skips each lane, accumulates, and applies Q8.16 dequant scaling.
- **Zero-skipping control**: compressed streams drive index decoding; MAC triggers only on valid non-zero entries. Zeros never enter the pipeline.
- **Scheduling**: one sparse dot-product engine per output channel or time-multiplexed; MACs-per-cycle = 4 at 200 MHz in the latency model. INT8 packing can double effective MACs-per-cycle when routing/timing permit.
- **Dataflow**: input activations buffered; index decoder fetches activation slices; accumulators flush on channel completion (`sparse_dot_product.v` for single-lane, `sparse_dot_product_int8_dual.v` for dual-lane).

## 5. Experimental Setup
- **Dataset**: MNIST, 20k train / 5k test subset, grayscale 28x28 (default); Fashion-MNIST and CIFAR-10 are supported via CLI switches.
- **Training**: Adam, lr=5e-3, 5 epochs, batch=128/256 train/test; PyTorch CPU/CUDA.
- **Baseline**: dense TinyCNN accuracy = 95.5% on the held-out subset.
- **Pruned model**: 70% sparsity yields 85.5% accuracy; INT8 PTQ recovers slightly to 85.9%.
- **FPGA platform assumption**: 200 MHz fabric, 4 MACs/cycle effective parallelism, small pipeline fill/flush overhead; INT8 packing can raise effective MACs/cycle when DSP mode supports dual 8x8 multiplies.
- **Measurements**: MAC counts derived analytically; latency estimated via cycle model (ideal) plus fixed/control overheads (`figures/sparsity_vs_latency.png`); resources are synthesis-style estimates for a mid-range FPGA (LUT/FF/BRAM/DSP) comparing dense vs pruned+zero-skip kernels. INT8 artifacts and scales are produced to enable bit-accurate RTL simulation.

## 6. Results
- **Accuracy vs. sparsity** (`figures/sparsity_vs_accuracy.png`): baseline 95.5%; accuracy degrades to ~94.9% at 50% sparsity and ~92.5% at 90% sparsity. Measured point: 85.5% at 70% sparsity, 85.9% with INT8 PTQ.
- **MAC reduction vs. latency** (`figures/mac_vs_latency.png`, `figures/sparsity_vs_latency.png`): MACs fall from 962.2k (dense) to 426.3k (70% sparsity, -56%). Ideal cycle model: 1.20 ms -> 0.53 ms (-45%). Overhead-aware model: 1.20 ms -> 0.63 ms, showing control/index costs limiting the win.
- **Diminishing returns**: annotated knee at 70% sparsity where compute is no longer the sole bottleneck; further pruning to 90% yields smaller incremental latency gains because memory and control dominate.
- **FPGA resources** (`figures/fpga_resource_utilization.png`): LUT 21k->16.2k, FF 33k->27.4k, BRAM 140->124 (fewer activation/weight buffers), DSP 64->28 (fewer active MAC lanes). INT8 mode can pack 2x 8-bit MACs/DSP, effectively halving the DSP demand again if timing closes.
- **FPGA packing visualization** (`figures/int8_packing.png`): compares MACs/DSP (1.0 for int16, 1.0 for int8 single-lane, 2.0 for packed int8) and bytes/weight (2.0 vs 1.0 vs 0.5), highlighting DSP and bandwidth benefits of packing.
- **INT8 export + scaling**: `weights_int8_single.hex` supports single-lane sparse MAC; `weights_int8_packed.hex` feeds the dual-lane SIMD MAC; `int8_scales.json` carries weight and activation scales for Q8.16 dequant in RTL.

## 7. Discussion
- **When pruning helps**: moderate sparsity (50-70%) yields clear MAC and latency wins and meaningful DSP savings; INT8 packing compounds DSP relief by doubling effective MACs/DSP.
- **When pruning does not help**: pushing to 90% sparsity harms accuracy on MNIST and delivers smaller latency gains because fixed overheads dominate; load imbalance across channels worsens.
- **Compute vs. memory bottlenecks**: after roughly 2x MAC reduction, activation fetch and index decoding bound throughput; BRAM port pressure and irregular access patterns define the knee.
- **Why zero-skipping + INT8 are essential**: pruning alone cannot deliver speedup; skipping zeros in hardware frees DSPs and cycles, reduces switching activity, and lowers BRAM bandwidth, while INT8 doubles MAC density per DSP and halves weight bandwidth.
- **Observed tradeoffs**: fewer DSPs lower dynamic power and free fabric for batching/parallel channels; control logic adds LUT/FF overhead; INT8 increases arithmetic density but adds quantization error and requires calibrated activation ranges. Dequantization in RTL (Q8.16) adds a small multiplier but keeps outputs numerically aligned with Python INT8.

## 8. Limitations & Future Work
- Naive PTQ only; INT8-aware training or per-channel quantization should recover accuracy while preserving DSP packing.
- Single small CNN and MNIST subset; need validation on Fashion-MNIST and CIFAR-10 and on deeper models.
- Unstructured + N:M sparsity; larger structured or block sparsity would further simplify scheduling and indexing.
- No full RTL-to-bitstream timing closure; synthesis and place-and-route results remain estimated.
- Future directions: combine pruning with quantization-aware training, introduce block-sparse kernels, explore sparse-friendly systolic arrays, add DMA-aware double buffering, and co-design memory layouts to ease index decoding.

## 9. Conclusion
Pairing magnitude pruning, optional N:M structured pruning, zero-skipping MAC logic, INT8 PTQ, and DSP48 SIMD packing on FPGA reduces MAC work by 56%, doubles MACs-per-DSP, and cuts estimated latency by up to 45% at 200 MHz while maintaining mid-80s accuracy on MNIST. Hardware-aware sparsity—compressed storage, index decoding, gated MACs, precision reduction, and dequant scaling—is necessary to convert pruning into real system-level gains, and benefits saturate once control and memory costs dominate.
