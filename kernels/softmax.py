"""Fused row-wise softmax in Triton.

Based on the official Triton tutorial "Fused Softmax"
(https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html),
specifically its simpler one-program-per-row version. Changes from the tutorial:
  * math is done in fp32 even for fp16 inputs (the output keeps the input dtype),
  * BLOCK_SIZE and num_warps can be overridden so the benchmark can sweep them,
  * inputs with any number of leading dims are flattened to (rows, cols).

Why fuse: the eager PyTorch decomposition (max, subtract, exp, sum, divide) reads
and writes the tensor several times. Here each row is loaded from DRAM once,
reduced in registers, and written once, so the kernel is memory-bound at
roughly 2 * numel * element_size bytes of traffic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # One program instance handles one row.
    row = tl.program_id(0)
    in_row_ptr = in_ptr + row * in_row_stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # BLOCK_SIZE is a power of two >= n_cols, so the tail lanes are masked.
    # They are filled with -inf so they do not affect the max and exp(-inf) = 0
    # keeps them out of the sum.
    x = tl.load(in_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)

    # Subtract the row max for numerical stability (softmax is shift-invariant).
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    out_row_ptr = out_ptr + row * out_row_stride
    tl.store(out_row_ptr + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def default_num_warps(block_size: int) -> int:
    # Heuristic from the Triton tutorial: more warps for wider rows so each
    # thread handles a reasonable number of elements.
    if block_size >= 4096:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def softmax(x: torch.Tensor, block_size: int | None = None, num_warps: int | None = None) -> torch.Tensor:
    """Softmax over the last dimension of `x`.

    block_size: must be a power of two >= x.shape[-1]; defaults to the smallest one.
    num_warps:  defaults to `default_num_warps(block_size)`.
    """
    if x.shape[-1] == 0 or x.numel() == 0:
        return torch.empty_like(x)

    n_cols = x.shape[-1]
    # The kernel assumes unit stride along the row.
    x2d = x.reshape(-1, n_cols)
    if x2d.stride(-1) != 1:
        x2d = x2d.contiguous()
    n_rows = x2d.shape[0]

    min_block = triton.next_power_of_2(n_cols)
    if block_size is None:
        block_size = min_block
    if block_size < n_cols or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two >= {n_cols}, got {block_size}")
    if num_warps is None:
        num_warps = default_num_warps(block_size)

    out = torch.empty_like(x2d)
    _softmax_kernel[(n_rows,)](
        out,
        x2d,
        x2d.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)
