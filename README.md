# Fused GPU Kernels in Triton: Performance and Energy Benchmarking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JawadurRahman/triton-fused-kernels/blob/main/notebooks/run_on_colab.ipynb)

This repo has fused Triton kernels for two memory-bound transformer operations.
Each kernel is checked against PyTorch and benchmarked against PyTorch eager
and `torch.compile` across input sizes and launch configurations. Energy per
call can optionally be measured through NVML.

> **Status:** the code and tests are written but have **not been run on a GPU
> yet**, so there are no results. The tables below stay empty until a real run fills them. Every
> number in this README comes from a CSV in [`results/`](results/), and the
> source file is cited next to it.

## Motivation

Inference on modern GPUs is often limited by memory bandwidth, not FLOPs,
and this is especially true for normalization and softmax. Those ops do very
little arithmetic per byte, so their runtime and much of their energy come
from DRAM traffic. Fusing them removes intermediate tensors that would
otherwise be written to DRAM and read back. This project measures how much
that helps in practice on a small GPU. It is part of my MCS thesis work on
hardware-aware transformer design and energy-efficient inference.

## Kernels

| Kernel | File | What it fuses | Minimum DRAM traffic |
|---|---|---|---|
| Softmax (row-wise) | [`kernels/softmax.py`](kernels/softmax.py) | max, subtract, exp, sum, divide | read x, write y: 2·M·N elements |
| Residual add + LayerNorm | [`kernels/layernorm.py`](kernels/layernorm.py) | `h = x + residual; y = LayerNorm(h)` (pre-LN block boundary) | read x, residual; write y, h: 4·M·N elements (eager PyTorch: 5·M·N) |

- **Softmax** is adapted from the official Triton
  [fused softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
  Changes: fp32 math for fp16 inputs, and `BLOCK_SIZE` and `num_warps` exposed for the sweep.
- **Add + LayerNorm** is my own kernel. Each program handles one row: it loads
  `x` and `residual`, writes `h` (the next residual), computes the mean and
  variance from registers with an exact two-pass method, and writes `y`. The
  sum is rounded to the storage dtype before the statistics are computed, so
  it computes exactly what unfused PyTorch computes.
- Both kernels are forward only (inference), and each row must fit in one
  block. That covers transformer hidden sizes (768–8192) and attention rows up
  to at least 16K keys.

## Repository layout

```
kernels/      Triton kernels + unfused PyTorch reference
tests/        Correctness tests against PyTorch (GPU, or CPU via TRITON_INTERPRET=1)
benchmarks/   bench.py (timing sweep), energy.py (NVML energy), plot.py (figures)
results/      Raw CSV logs from benchmark runs (committed: they are the evidence)
plots/        Figures generated from results/
notebooks/    run_on_colab.ipynb: runs everything on a Colab GPU
```

## Setup

**Colab (easiest):** open the notebook with the badge above, choose
Runtime → Change runtime type → T4 GPU, and run all cells. Colab already
ships `torch` and `triton`. The notebook installs only `nvidia-ml-py`
and anything else that's missing.

**Linux machine with an NVIDIA GPU:**

```bash
git clone https://github.com/JawadurRahman/triton-fused-kernels.git
cd triton-fused-kernels
pip install -r requirements.txt
```

**CPU only (correctness only):** the tests fall back to Triton's interpreter
when no GPU is present. That checks the kernel logic, masking and dtype
handling, but not timing, `num_warps`, or GPU numerics.

```bash
TRITON_INTERPRET=1 python -m pytest -q
```

## How to reproduce

From the repo root:

```bash
python -m pytest -q                 # correctness (run this first)
python -m benchmarks.bench          # timing sweep -> results/bench_<kernel>_<gpu>_<time>.csv
python -m benchmarks.energy         # optional     -> results/energy_<kernel>_<gpu>_<time>.csv
python -m benchmarks.plot           # figures from the newest CSVs -> plots/
```

Each CSV row records the GPU, compute capability, torch, triton and CUDA
versions, and a UTC timestamp. The notebook also saves `nvidia-smi -q`, `pip freeze`
and the git commit to `results/env_<time>.txt`.

## Methodology

**Timing.** [`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
uses CUDA events with ~25 ms of warmup and ~100 ms of timed repetitions. L2
is flushed before each repetition, so inputs come from DRAM. The CSV
reports the median with the 20th and 80th percentiles. Before a Triton
configuration is timed, its output is checked against the reference, so a
fast but wrong config can never appear in the results.

**Sweep.** M = 4096 rows. Softmax N ∈ {256, 512, 1000, 1024, 2048, 3000, 4096, 8192, 16384}.
Add + LayerNorm N ∈ {768, 1024, 2048, 2560, 4096, 5120, 8192}. Triton configs:
`num_warps` ∈ {1, 2, 4, 8, 16} × `BLOCK_SIZE` ∈ {1×, 2×} the smallest power of two ≥ N.
Both fp16 and fp32. Baselines: PyTorch eager and `torch.compile` (default mode).

**Effective bandwidth** = minimum required bytes (table above) / median
time. Every provider uses the same byte count, so the numbers compare
runtime on one scale. It is not the traffic each implementation actually
generates.

**Energy.** `nvmlDeviceGetTotalEnergyConsumption` is read before and after
each implementation runs back-to-back for ≥ 5 s. Idle power is measured at
the start and end of the run and the mean is subtracted, giving net
joules per call. Total joules per call, SM clock and temperature are logged
too. If the counter is not exposed, the script skips cleanly.

## Results

> TODO: fill from a logged run. Every cell must cite a CSV in `results/`.

**GPU:** — · **torch:** — · **triton:** — · **date:** —

### Runtime (median, µs; lower is better)

| Kernel | dtype | M × N | Triton default | Triton best (config) | PyTorch eager | torch.compile | Speedup vs eager | Speedup vs compile | Source |
|---|---|---|---|---|---|---|---|---|---|
| softmax | fp16 | 4096 × — | — | — | — | — | — | — | — |
| softmax | fp32 | 4096 × — | — | — | — | — | — | — | — |
| add + LayerNorm | fp16 | 4096 × — | — | — | — | — | — | — | — |
| add + LayerNorm | fp32 | 4096 × — | — | — | — | — | — | — | — |

### Energy (net µJ per call, idle subtracted; lower is better)

| Kernel | dtype | M × N | Triton | PyTorch eager | torch.compile | Idle power used (W) | Source |
|---|---|---|---|---|---|---|---|
| softmax | fp16 | 4096 × — | — | — | — | — | — |
| add + LayerNorm | fp16 | 4096 × — | — | — | — | — | — |

### Figures

`python -m benchmarks.plot` writes, per kernel, `plots/<kernel>_bandwidth.png`,
`plots/<kernel>_config_sweep.png` and (if measured) `plots/<kernel>_energy.png`.
They will be embedded here once they exist.

## Findings

> TODO: write after real runs, using only logged data. Questions to answer:
>
> - How close does each kernel get to the GPU's DRAM bandwidth, and at what N does it saturate?
> - Does the fused add + LayerNorm beat eager by roughly the 5→4 traffic ratio, and how does it compare with `torch.compile`, which can fuse the same ops?
> - Which `num_warps` / `BLOCK_SIZE` choices matter, and why (occupancy, register pressure, masked lanes for non-power-of-two N)? Back this up with Nsight Compute or `torch.profiler` output.
> - Does energy per call track runtime, or does average power differ between implementations?

## Limitations

- **Noise on shared Colab GPUs.** Colab VMs share physical hosts, and the
  T4 is passively cooled with a 70 W power cap. Its clocks drop under
  heat or when the cap is reached, and how often this happens varies between
  sessions and even within a run. Results can differ between sessions on the
  "same" GPU. Mitigations: medians with p20/p80, L2 flushing, SM clock and
  temperature logged in the energy CSV, and the advice to repeat runs before
  trusting small differences (say, under ~5%).
- **Energy readings on virtualized GPUs** may be unavailable, coarse
  (the counter can update every tens of milliseconds), or include
  activity outside this process. Idle subtraction assumes idle power stays
  constant during the run; the start/end idle values are logged so you can check that.
- **Energy loop and caches.** Unlike the timing benchmark, the energy loop
  does not flush L2. The smallest inputs (e.g. softmax fp16 at N = 256,
  2 MB in + 2 MB out) can stay partly L2-resident on a T4 (4 MB L2).
- **One GPU class.** The T4 (Turing, sm_75) has no bf16, and its bandwidth
  and cache sizes differ from A100 or H100 GPUs. Conclusions may not transfer.
- **Scope.** Forward only, single-block rows, and `torch.compile` in default
  mode only (not `max-autotune`). Effective bandwidth is a normalized
  runtime, not measured DRAM traffic.
- **CPU interpreter tests** check logic and masking only, not GPU code
  generation or GPU numerics.

## Credits

- Softmax kernel adapted from the Triton tutorial
  [Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
  (Triton authors, MIT license).
- Benchmarking uses `triton.testing.do_bench`; energy uses NVIDIA's NVML via `nvidia-ml-py`.
