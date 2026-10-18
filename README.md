# Fused GPU Kernels in Triton: Performance and Energy Benchmarking

Fused Triton kernels for transformer building blocks (softmax, and LayerNorm
fused with a residual add), checked for correctness against PyTorch and
benchmarked against PyTorch eager and `torch.compile` across input sizes and
launch configurations, with optional energy measurement through NVML.

> **Status:** work in progress. There are no results yet. Every number that
> ends up in this README will come from a logged run in `results/`.

## Layout

```
kernels/      Triton kernels
tests/        Correctness tests against PyTorch reference implementations
benchmarks/   Timing and energy benchmark scripts (write to results/*.csv)
results/      Raw CSV logs from benchmark runs
plots/        Figures generated from results/
notebooks/    Colab notebook to reproduce everything on a GPU
```
