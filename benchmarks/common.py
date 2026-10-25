"""Helpers shared by the benchmark scripts: environment metadata, timing, CSV output."""

import csv
import datetime
import os
import re

import torch
import triton

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# triton.testing.do_bench budgets are in milliseconds of GPU time, not
# iteration counts: it runs the function for ~WARMUP_MS first, then enough
# repetitions to fill ~REP_MS, timing each with CUDA events. It also clears
# the L2 cache before every repetition, so each call reads its inputs from
# DRAM, as an activation coming from a previous layer usually would.
WARMUP_MS = 25
REP_MS = 100


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device found. Benchmarks need a GPU (see notebooks/run_on_colab.ipynb).")


def env_info() -> dict:
    """Metadata stored in every CSV row, so each number traces back to its setup."""
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def time_ms(fn) -> tuple[float, float, float]:
    """Returns (median, p20, p80) runtime of fn() in milliseconds."""
    median, p20, p80 = triton.testing.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, quantiles=[0.5, 0.2, 0.8])
    return median, p20, p80


def results_path(prefix: str) -> str:
    """results/<prefix>_<gpu>_<UTC timestamp>.csv, so runs never overwrite each other."""
    gpu = re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name(0).lower()).strip("-")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"{prefix}_{gpu}_{stamp}.csv")


def write_csv(rows: list[dict], path: str):
    # Union of keys in first-seen order, since error rows may lack some fields.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
