"""Fused residual add + LayerNorm (forward only) in Triton.

In a pre-LN transformer block, every sublayer ends with

    h = x + residual          # update the residual stream
    y = LayerNorm(h)          # input to the next sublayer

Eager PyTorch runs these as two kernels: the add writes `h` to DRAM, and
LayerNorm reads it back. This kernel does both in one pass per row: it loads
`x` and `residual` once, writes `h` (still needed as the next residual) and
`y`, and keeps the row in registers in between. That removes one full read of
`h`, so ideal traffic drops from 5 to 4 tensor-sized transfers (plus the small
weight and bias vectors).

This is forward only, aimed at inference. Each program normalizes one full
row, so the row must fit in one block (fine for transformer hidden sizes
such as 768 to 8192).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _add_layernorm_kernel(
    y_ptr,
    h_ptr,
    x_ptr,
    res_ptr,
    w_ptr,
    b_ptr,
    row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offs = row * row_stride + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Round the sum to the storage dtype before computing statistics. PyTorch's
    # `x + residual` produces an fp16 tensor in fp16, and LayerNorm then sees that
    # rounded value. Doing the same here means the kernel computes the same
    # function as the unfused reference, not a slightly more precise one.
    h = (x + r).to(h_ptr.dtype.element_ty)
    tl.store(h_ptr + offs, h, mask=mask)
    h = h.to(tl.float32)

    # Two-pass mean/variance over values already in registers: exact and
    # cheap, since the row is not re-read from memory. Masked lanes are 0,
    # so they do not change the sum. They are zeroed again before the
    # variance because (0 - mean) would otherwise count.
    mean = tl.sum(h, axis=0) / n_cols
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols  # biased, like nn.LayerNorm
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def default_num_warps(block_size: int) -> int:
    # Same heuristic as the softmax kernel; the benchmark sweeps around it.
    if block_size >= 4096:
        return 16
    if block_size >= 2048:
        return 8
    return 4


def add_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    block_size: int | None = None,
    num_warps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (y, h) where h = x + residual and y = layer_norm(h) over the last dim.

    block_size: must be a power of two >= hidden size; defaults to the smallest one.
    num_warps:  defaults to `default_num_warps(block_size)`.
    """
    if x.shape != residual.shape:
        raise ValueError(f"shape mismatch: x {tuple(x.shape)} vs residual {tuple(residual.shape)}")
    if x.dtype != residual.dtype:
        raise ValueError(f"dtype mismatch: x {x.dtype} vs residual {residual.dtype}")
    n_cols = x.shape[-1]
    if weight.shape != (n_cols,) or bias.shape != (n_cols,):
        raise ValueError(f"weight and bias must have shape ({n_cols},)")

    # The kernel uses one row stride for x, residual, h and y, so make all of
    # them contiguous 2D (rows, n_cols). reshape is free for contiguous inputs.
    x2d = x.reshape(-1, n_cols).contiguous()
    r2d = residual.reshape(-1, n_cols).contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    n_rows = x2d.shape[0]

    y = torch.empty_like(x2d)
    h = torch.empty_like(x2d)
    if n_rows == 0 or n_cols == 0:
        return y.reshape(x.shape), h.reshape(x.shape)

    min_block = triton.next_power_of_2(n_cols)
    if block_size is None:
        block_size = min_block
    if block_size < n_cols or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a power of two >= {n_cols}, got {block_size}")
    if num_warps is None:
        num_warps = default_num_warps(block_size)

    _add_layernorm_kernel[(n_rows,)](
        y,
        h,
        x2d,
        r2d,
        weight,
        bias,
        x2d.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y.reshape(x.shape), h.reshape(x.shape)


def torch_add_layernorm(x, residual, weight, bias, eps=1e-5):
    """Unfused PyTorch reference: the same computation as two separate ops."""
    h = x + residual
    y = F.layer_norm(h, (h.shape[-1],), weight, bias, eps)
    return y, h
