"""Optional energy benchmark using NVML's cumulative energy counter.

Method, per (kernel, dtype, N, provider):
  1. Measure idle power: read nvmlDeviceGetTotalEnergyConsumption (mJ since
     driver load) before and after sleeping for --idle-seconds with nothing
     queued. P_idle = delta energy / delta time. Done at the start and the
     end of the run; the mean of the two is used and both are logged.
  2. Run the kernel back-to-back for at least --duration seconds, reading
     the counter before and after (with the GPU synchronized at both ends).
  3. joules_per_call       = total energy / calls
     joules_per_call_net   = (total energy - P_idle * elapsed) / calls
  The "net" value is the energy above what the GPU draws when idle. It is
  the more comparable number between implementations, but idle power on a
  shared or virtualized GPU can drift, so both are logged.

Unlike the timing benchmark, the loop does not flush L2 between calls, so
inputs small enough to fit in L2 (T4: 4 MB; e.g. softmax fp16 at N=256 is
2 MB in + 2 MB out) may partly stay cached. Larger sizes do not fit.

If pynvml is not installed, or the GPU or driver does not expose the energy
counter (NVML_ERROR_NOT_SUPPORTED, common in VMs), the script prints why
and exits with status 0, so the Colab notebook continues.

Usage (from the repo root):
    python -m benchmarks.energy
    python -m benchmarks.energy --kernel add_layernorm --dtype fp16 --duration 10
"""

import argparse
import gc
import sys
import time

import torch

from benchmarks.bench import CASES, DTYPES, SWEEPS
from benchmarks.common import env_info, require_cuda, results_path, write_csv

try:
    import pynvml
except ImportError:
    pynvml = None


def skip(reason):
    print(f"Skipping energy measurement: {reason}")
    sys.exit(0)


def nvml_handle():
    """NVML handle for the GPU torch is using (cuda:0)."""
    # CUDA and NVML can number GPUs differently (CUDA_VISIBLE_DEVICES,
    # CUDA_DEVICE_ORDER), so match by UUID when torch exposes it.
    uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if uuid is not None:
        try:
            return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        except pynvml.NVMLError:
            pass
    print("warning: could not match the GPU by UUID; using NVML device 0")
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def energy_j(handle) -> float:
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0  # mJ -> J


def gpu_state(handle) -> dict:
    """Clock and temperature, to spot throttling or a GPU stuck in a low P-state."""
    state = {}
    try:
        state["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        state["temp_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError:
        pass
    return state


def idle_power_w(handle, seconds) -> float:
    torch.cuda.synchronize()
    time.sleep(1.0)  # let the last kernels drain and clocks start to settle
    e0, t0 = energy_j(handle), time.perf_counter()
    time.sleep(seconds)
    e1, t1 = energy_j(handle), time.perf_counter()
    return (e1 - e0) / (t1 - t0)


def calls_per_batch(fn, target_s=0.05) -> int:
    """Calls to enqueue between clock checks so a batch takes ~target_s.

    Synchronizing after every call would add CPU-GPU round trips and idle
    gaps, which cost energy but are not part of the kernel.
    """
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    per_call = (time.perf_counter() - t0) / 100
    return max(1, int(target_s / per_call))


def measure_energy(handle, fn, duration_s, p_idle_w) -> dict:
    batch = calls_per_batch(fn)
    calls = 0
    torch.cuda.synchronize()
    e0, t0 = energy_j(handle), time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        for _ in range(batch):
            fn()
        calls += batch
        torch.cuda.synchronize()
    e1, t1 = energy_j(handle), time.perf_counter()

    elapsed, total = t1 - t0, e1 - e0
    net = total - p_idle_w * elapsed
    return dict(
        calls=calls,
        elapsed_s=elapsed,
        energy_total_j=total,
        avg_power_w=total / elapsed,
        joules_per_call=total / calls,
        joules_per_call_net=net / calls,
        us_per_call_wall=elapsed / calls * 1e6,
        status="ok",
    )


def providers(case, compiled):
    """The implementations to measure, each a zero-argument callable."""
    args = case["args"]
    return {
        # None, None = Triton's default BLOCK_SIZE and num_warps.
        "triton": lambda: case["triton"](None, None),
        "torch": lambda: case["torch_fn"](*args),
        "torch_compile": lambda: compiled(*args),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", choices=[*CASES, "all"], default="all")
    parser.add_argument("--dtype", choices=[*DTYPES, "all"], default="all")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds of back-to-back calls per measurement")
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    args = parser.parse_args()

    require_cuda()
    if pynvml is None:
        skip("pynvml is not installed (pip install nvidia-ml-py)")
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        skip(f"NVML could not be initialized ({e})")
    handle = nvml_handle()
    try:
        energy_j(handle)
    except pynvml.NVMLError as e:
        skip(f"this GPU/driver does not report total energy consumption ({e})")

    torch.manual_seed(0)
    env = env_info()
    print(f"GPU: {env['gpu']}, torch {env['torch']}, triton {env['triton']}")

    p_idle_start = idle_power_w(handle, args.idle_seconds)
    print(f"idle power (start): {p_idle_start:.1f} W")

    kernels = list(CASES) if args.kernel == "all" else [args.kernel]
    dtypes = list(DTYPES) if args.dtype == "all" else [args.dtype]
    measured = []  # (kernel, row) pairs; idle power is filled in at the end

    for kernel in kernels:
        sizes = SWEEPS[kernel]["N"]
        # Smallest, middle and largest size, to keep the run to a few minutes.
        sizes = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
        M = SWEEPS[kernel]["M"]
        for dtype_name in dtypes:
            for N in sizes:
                case = CASES[kernel](M, N, DTYPES[dtype_name])
                case["check"](case["triton"](None, None))
                torch._dynamo.reset()
                compiled = torch.compile(case["torch_fn"])
                compiled(*case["args"])  # compile before measuring
                for provider, fn in providers(case, compiled).items():
                    row = dict(kernel=kernel, dtype=dtype_name, M=M, N=N, provider=provider,
                               bytes=case["bytes"], duration_target_s=args.duration)
                    row.update({f"{k}_start": v for k, v in gpu_state(handle).items()})
                    try:
                        row.update(measure_energy(handle, fn, args.duration, p_idle_start))
                    except Exception as e:
                        row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
                    row.update({f"{k}_end": v for k, v in gpu_state(handle).items()})
                    measured.append((kernel, row))
                    if row["status"] == "ok":
                        # Preliminary: the CSV recomputes net energy with the mean idle power.
                        print(f"{kernel} {dtype_name} N={N} {provider:>13}: "
                              f"{row['joules_per_call_net'] * 1e6:9.2f} uJ/call net (start idle), "
                              f"{row['avg_power_w']:.1f} W avg")
                    else:
                        print(f"{kernel} {dtype_name} N={N} {provider:>13}: {row['status']}")
                del case, compiled
                gc.collect()
                torch.cuda.empty_cache()

    p_idle_end = idle_power_w(handle, args.idle_seconds)
    p_idle = (p_idle_start + p_idle_end) / 2
    print(f"idle power (end): {p_idle_end:.1f} W; using mean {p_idle:.1f} W")

    # Recompute the net energy with the mean idle power now that both ends are known.
    for _, row in measured:
        row.update(idle_power_start_w=p_idle_start, idle_power_end_w=p_idle_end, idle_power_used_w=p_idle)
        if row["status"] == "ok":
            net = row["energy_total_j"] - p_idle * row["elapsed_s"]
            row["joules_per_call_net"] = net / row["calls"]
        row.update(env)

    for kernel in kernels:
        rows = [row for k, row in measured if k == kernel]
        path = results_path(f"energy_{kernel}")
        write_csv(rows, path)
        print(f"wrote {len(rows)} rows to {path}")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
