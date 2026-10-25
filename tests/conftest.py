"""Shared test setup.

With a CUDA GPU, tests run there. Without one, they fall back to Triton's
interpreter (TRITON_INTERPRET=1), which runs kernels on CPU tensors with
NumPy. The interpreter checks the kernel logic, masking and dtype handling,
but not GPU-specific behaviour (num_warps, real fp16 hardware math, timing).
"""

import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()

# Must happen before `triton` is imported (it is read when kernels are
# decorated), which is why this lives in conftest and not in the tests.
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture(scope="session")
def device():
    if INTERPRET:
        return "cpu"
    if HAS_CUDA:
        return "cuda"
    pytest.skip("no CUDA device and TRITON_INTERPRET is not 1")


def pytest_report_header(config):
    mode = "Triton interpreter (CPU)" if INTERPRET else f"CUDA ({torch.cuda.get_device_name()})"
    return f"kernel backend: {mode}; torch {torch.__version__}"
