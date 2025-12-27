# Sparse-Aware FPGA Inference Demo

End-to-end flow that trains a tiny CNN in PyTorch, prunes weights, exports a compressed memory-mapped format for FPGA, simulates dense vs zero-skipping sparse inference, and provides synthesizable Verilog building blocks.

## How to run
- Ensure Python with PyTorch and matplotlib is available (Torch 2.5+ already present in this environment).
- From the repo root run:  
  `python python/sparse_fpga_pipeline.py`
- Outputs are written to `outputs/`:
  - `weights_indices.bin` / `weights_values.bin` + `weights_metadata.json`: compressed (idx,val) memmap suitable for BRAM initialization.
  - `mac_ops.png`, `latency.png`, `block_diagram.png`: visuals comparing dense vs sparse MAC and latency plus the zero-skip pipeline diagram.
  - `summary.json`: metrics (accuracy, MAC counts, latency estimates, sparsity) and max diff between dense and sparse simulations.

## What the pipeline does
- Builds a synthetic 2-class image dataset (16×16) to avoid downloads.
- Trains `TinyCNN` (2×3×3 conv + FC) for a few epochs.
- Applies global unstructured L1 pruning to reach ~70% sparsity.
- Compresses non-zero weights per output channel into memory-mapped binary files with accompanying JSON metadata describing offsets/lengths for FPGA loaders.
- Simulates inference two ways: dense PyTorch path and a Python zero-skipping path that only touches non-zero weights from the compressed store to validate numerical equivalence.
- Counts MACs for dense vs sparse and estimates latency assuming a 200 MHz fabric with 4 MACs/cycle.

## RTL overview (`rtl/`)
- `sparse_weight_store.v`: ROM/BRAM for packed (index,value) pairs; parameterizable depth/width and optional `$readmemh` init.
- `zero_skip_mac.v`: DSP-friendly MAC that only accumulates when valid; resettable per output channel.
- `sparse_dot_product.v`: Streams compressed weights, fetches activations by index, and accumulates via `zero_skip_mac`; asserts `done` on the last non-zero. Use one instance per output channel or time-multiplex.

## Mapping metadata to RTL
- JSON fields `base_offset` and `channel_segments[offset,length]` describe where each channel’s non-zero entries live inside the packed binary streams. Convert to `.hex` for `$readmemh` or preload into on-chip BRAM.
- `index` selects the activation element; `value` is the weight. The MAC only fires on provided entries—zeros never enter the pipeline.

## Next steps (optional)
- Sweep pruning ratios to find accuracy/speed sweet spots.
- Replace synthetic data with a real dataset and retrain.
- Add Verilog testbenches that read the exported `.bin` files (converted to hex) and compare against the Python sparse simulation.
