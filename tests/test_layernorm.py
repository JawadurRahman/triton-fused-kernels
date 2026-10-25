"""Correctness of kernels.add_layernorm against PyTorch.

Reference (`reference` below): h = x + residual rounded to the test dtype,
then F.layer_norm in fp32 cast back to the test dtype. That is what the
kernel computes, so the outputs should agree to about one rounding step.
`test_matches_unfused_torch_ops` also checks against the plain PyTorch ops
the benchmark compares against.
"""

import pytest
import torch
import torch.nn.functional as F

from kernels import add_layernorm, torch_add_layernorm

# LayerNorm outputs are O(1) to O(few), so fp16 rounding at |y| ~ 4 is about
# 2e-3 absolute; atol covers outputs near zero where rtol is too strict.
TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=2e-3, atol=2e-3),
}

# Awkward sizes around powers of two, plus real hidden sizes:
# 768 (BERT-base / GPT-2), 4096 (LLaMA-7B), 5120 (LLaMA-13B), 8192 (LLaMA-70B).
N_COLS = [1, 3, 127, 128, 129, 768, 781, 1000, 1023, 1025, 4096, 5120, 8192]
N_ROWS = [1, 7, 33]


def make_inputs(n_rows, n_cols, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    x, res = rand(n_rows, n_cols), rand(n_rows, n_cols)
    # Non-trivial affine parameters so a bug in weight/bias handling is visible.
    w, b = rand(n_cols), rand(n_cols)
    return x, res, w, b


def reference(x, res, w, b, eps=1e-5):
    h = (x.float() + res.float()).to(x.dtype)
    y = F.layer_norm(h.float(), (h.shape[-1],), w.float(), b.float(), eps).to(x.dtype)
    return y, h


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("n_cols", N_COLS)
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_matches_reference(device, dtype, n_rows, n_cols):
    x, res, w, b = make_inputs(n_rows, n_cols, dtype, device)
    y, h = add_layernorm(x, res, w, b)
    y_ref, h_ref = reference(x, res, w, b)
    # The residual sum is one rounded add, so it must match exactly.
    assert torch.equal(h, h_ref)
    torch.testing.assert_close(y, y_ref, **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matches_unfused_torch_ops(device, dtype):
    if dtype == torch.float16 and device == "cpu":
        # The benchmark baseline is fp16 PyTorch on GPU; CPU fp16 kernels are
        # a different implementation and their support varies by torch version.
        pytest.skip("fp16 baseline comparison is only meaningful on CUDA")
    x, res, w, b = make_inputs(33, 1000, dtype, device)
    y, h = add_layernorm(x, res, w, b)
    y_ref, h_ref = torch_add_layernorm(x, res, w, b)
    torch.testing.assert_close(h, h_ref, rtol=0, atol=0)
    torch.testing.assert_close(y, y_ref, **TOL[dtype])


def test_transformer_shaped_input(device):
    # (batch, seq, hidden) with an awkward sequence length.
    x, res, w, b = make_inputs(2 * 77, 768, torch.float16, device)
    x, res = x.view(2, 77, 768), res.view(2, 77, 768)
    y, h = add_layernorm(x, res, w, b)
    y_ref, h_ref = reference(x, res, w, b)
    assert y.shape == h.shape == x.shape
    assert torch.equal(h, h_ref)
    torch.testing.assert_close(y, y_ref, **TOL[torch.float16])


def test_large_mean_input(device):
    # Rows with a large offset relative to their spread. A one-pass
    # E[x^2] - E[x]^2 variance loses precision here; two-pass should not.
    x, res, w, b = make_inputs(8, 1024, torch.float32, device)
    x = x + 1000.0
    y, _ = add_layernorm(x, res, w, b)
    y_ref, _ = reference(x, res, w, b)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-2])
def test_eps(device, eps):
    x, res, w, b = make_inputs(4, 781, torch.float32, device)
    y, _ = add_layernorm(x, res, w, b, eps=eps)
    y_ref, _ = reference(x, res, w, b, eps=eps)
    torch.testing.assert_close(y, y_ref, **TOL[torch.float32])


def test_non_contiguous_input(device):
    x, res, _, _ = make_inputs(768, 40, torch.float32, device)
    _, _, w, b = make_inputs(40, 768, torch.float32, device, seed=1)
    x, res = x.t(), res.t()  # shape (40, 768), column stride != 1
    assert not x.is_contiguous()
    y, h = add_layernorm(x, res, w, b)
    y_ref, h_ref = reference(x, res, w, b)
    assert torch.equal(h, h_ref)
    torch.testing.assert_close(y, y_ref, **TOL[torch.float32])


@pytest.mark.parametrize("block_size", [1024, 2048, 4096])
@pytest.mark.parametrize("num_warps", [1, 4, 8])
def test_launch_config_does_not_change_result(device, block_size, num_warps):
    # The benchmark sweeps these, so every setting must be correct.
    x, res, w, b = make_inputs(9, 1000, torch.float32, device)
    y, _ = add_layernorm(x, res, w, b, block_size=block_size, num_warps=num_warps)
    y_ref, _ = reference(x, res, w, b)
    torch.testing.assert_close(y, y_ref, **TOL[torch.float32])


def test_invalid_arguments_raise(device):
    x, res, w, b = make_inputs(2, 1000, torch.float32, device)
    with pytest.raises(ValueError):
        add_layernorm(x, res, w, b, block_size=512)  # smaller than the row
    with pytest.raises(ValueError):
        add_layernorm(x, res, w, b, block_size=1000)  # not a power of two
    with pytest.raises(ValueError):
        add_layernorm(x, res[:, :999], w, b)  # shape mismatch
    with pytest.raises(ValueError):
        add_layernorm(x, res.half(), w, b)  # dtype mismatch
    with pytest.raises(ValueError):
        add_layernorm(x, res, w[:999], b)  # wrong weight size
