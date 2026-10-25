"""Correctness of kernels.softmax against PyTorch.

Reference: torch.softmax computed in fp32 on the same inputs, then cast to
the test dtype. The kernel also does its math in fp32, so in fp16 both sides
should differ by at most about one fp16 rounding step (relative 2^-11 ~ 4.9e-4).
"""

import pytest
import torch

from kernels import softmax

# fp32: only reduction order differs. fp16: allow ~4 ulp relative, and a tiny
# atol for outputs near zero, where relative error alone would be too strict.
TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-6),
    torch.float16: dict(rtol=2e-3, atol=1e-5),
}

# Column counts chosen to hit edge cases of the power-of-two block:
# 1 (degenerate), small primes, one below / at / above a power of two,
# and transformer-like sizes (attention rows over seq lengths, vocab-ish).
N_COLS = [1, 3, 17, 127, 128, 129, 781, 1000, 1023, 1024, 1025, 2048, 4097]
N_ROWS = [1, 7, 33]


def reference(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("n_cols", N_COLS)
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_matches_torch(device, dtype, n_rows, n_cols):
    torch.manual_seed(0)
    x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    torch.testing.assert_close(softmax(x), reference(x), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rows_sum_to_one(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(16, 1000, device=device, dtype=dtype)
    sums = softmax(x).float().sum(-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=1e-2 if dtype == torch.float16 else 1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_large_values_are_stable(device, dtype):
    # Without max subtraction exp() overflows (fp16 overflows above ~11).
    torch.manual_seed(0)
    x = (torch.randn(8, 513, device=device) * 1000).to(dtype)
    y = softmax(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, reference(x), **TOL[dtype])


def test_attention_shaped_input(device):
    # (batch, heads, query, key), as in attention scores, with an awkward key length.
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 197, device=device, dtype=torch.float16)
    y = softmax(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y, reference(x), **TOL[torch.float16])


def test_masked_scores(device):
    # Causal-style masking: -inf entries must give exactly 0 probability.
    torch.manual_seed(0)
    x = torch.randn(64, 64, device=device)
    x = x.masked_fill(torch.triu(torch.ones_like(x, dtype=torch.bool), diagonal=1), float("-inf"))
    y = softmax(x)
    assert (y[torch.isinf(x)] == 0).all()
    torch.testing.assert_close(y, reference(x), **TOL[torch.float32])


def test_non_contiguous_input(device):
    torch.manual_seed(0)
    x = torch.randn(300, 40, device=device).t()  # shape (40, 300), column stride != 1
    assert not x.is_contiguous()
    torch.testing.assert_close(softmax(x), reference(x), **TOL[torch.float32])


@pytest.mark.parametrize("block_size", [1024, 2048, 4096])
@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_launch_config_does_not_change_result(device, block_size, num_warps):
    # The benchmark sweeps these, so every setting must be correct.
    torch.manual_seed(0)
    x = torch.randn(9, 1000, device=device)
    y = softmax(x, block_size=block_size, num_warps=num_warps)
    torch.testing.assert_close(y, reference(x), **TOL[torch.float32])


@pytest.mark.parametrize("block_size", [512, 1000])
def test_invalid_block_size_raises(device, block_size):
    # 512 < n_cols, and 1000 is not a power of two.
    x = torch.randn(2, 1000, device=device)
    with pytest.raises(ValueError):
        softmax(x, block_size=block_size)
