"""Runtime benchmark: Triton kernels vs PyTorch eager vs torch.compile.

For each kernel, dtype and row width N (with a fixed number of rows M) it
times:
  * triton        every (BLOCK_SIZE, num_warps) pair in the sweep,
  * torch         the native PyTorch op(s), eager mode,
  * torch_compile the same PyTorch code through torch.compile (Inductor),
and writes one CSV row per measurement to results/.

Usage (from the repo root):
    python -m benchmarks.bench                      # full sweep, fp16 and fp32
    python -m benchmarks.bench --kernel softmax --dtype fp16
    python -m benchmarks.bench --quick              # a few sizes, default configs only

Bandwidth is "effective": the minimum bytes the operation must move (read
inputs once, write outputs once) divided by runtime. The same byte count is
used for every provider, so it compares runtimes on one scale. It is not
the traffic each implementation actually generates; eager PyTorch moves
more bytes than this for add+LayerNorm, which is the point of fusing.
"""

import argparse
import gc

import torch
import torch.nn.functional as F
import triton

from benchmarks.common import env_info, require_cuda, results_path, time_ms, write_csv
from kernels import add_layernorm, softmax, torch_add_layernorm
from kernels.layernorm import default_num_warps as ln_default_warps
from kernels.softmax import default_num_warps as sm_default_warps

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}

# M = number of rows. Softmax: rows of an attention score matrix, e.g.
# batch * heads * queries, with N = key length. LayerNorm: tokens
# (batch * seq), with N = hidden size of common models.
SWEEPS = {
    "softmax": dict(
        M=4096,
        N=[256, 512, 1000, 1024, 2048, 3000, 4096, 8192, 16384],
    ),
    "add_layernorm": dict(
        M=4096,
        N=[768, 1024, 2048, 2560, 4096, 5120, 8192],
    ),
}
NUM_WARPS = [1, 2, 4, 8, 16]
# BLOCK_SIZE as a multiple of the smallest valid block (next power of two >= N).
# Larger blocks only add masked lanes, which shows what over-sizing costs.
BLOCK_MULTIPLIERS = [1, 2]

# Same tolerances as the test suite. The benchmark checks each Triton config
# before timing it, so a fast but wrong config is never reported.
TOL = {torch.float16: dict(rtol=2e-3, atol=2e-3), torch.float32: dict(rtol=1e-5, atol=1e-5)}


def softmax_case(M, N, dtype):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)

    def torch_fn(t):
        return torch.softmax(t, dim=-1)

    return dict(
        triton=lambda bs, nw: softmax(x, block_size=bs, num_warps=nw),
        torch_fn=torch_fn,
        args=(x,),
        check=lambda out: torch.testing.assert_close(out, ref, **TOL[dtype]),
        bytes=2 * M * N * x.element_size(),  # read x, write y
        default_warps=sm_default_warps,
    )


def add_layernorm_case(M, N, dtype):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    b = torch.randn(N, device="cuda", dtype=dtype)
    h_ref = (x.float() + res.float()).to(dtype)
    y_ref = F.layer_norm(h_ref.float(), (N,), w.float(), b.float()).to(dtype)

    def check(out):
        y, h = out
        torch.testing.assert_close(h, h_ref, rtol=0, atol=0)
        torch.testing.assert_close(y, y_ref, **TOL[dtype])

    es = x.element_size()
    return dict(
        triton=lambda bs, nw: add_layernorm(x, res, w, b, block_size=bs, num_warps=nw),
        torch_fn=torch_add_layernorm,
        args=(x, res, w, b),
        check=check,
        bytes=4 * M * N * es + 2 * N * es,  # read x, res, w, b; write y, h
        default_warps=ln_default_warps,
    )


CASES = {"softmax": softmax_case, "add_layernorm": add_layernorm_case}


def measure(row, fn, n_bytes):
    """Times fn and fills in row; failures are recorded instead of crashing the sweep."""
    try:
        median, p20, p80 = time_ms(fn)
        row.update(
            ms_median=median,
            ms_p20=p20,
            ms_p80=p80,
            gbps_effective=n_bytes / (median * 1e-3) / 1e9,
            status="ok",
        )
    except Exception as e:  # e.g. out of registers / shared memory, OOM
        row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
        torch.cuda.empty_cache()
    return row


def run_case(kernel, dtype_name, M, N, quick, env):
    dtype = DTYPES[dtype_name]
    case = CASES[kernel](M, N, dtype)
    base = dict(kernel=kernel, dtype=dtype_name, M=M, N=N, bytes=case["bytes"])
    rows = []

    min_block = triton.next_power_of_2(N)
    default_nw = case["default_warps"](min_block)
    blocks = [min_block] if quick else [min_block * m for m in BLOCK_MULTIPLIERS]
    warps = [default_nw] if quick else NUM_WARPS

    for bs in blocks:
        for nw in warps:
            row = dict(base, provider="triton", block_size=bs, num_warps=nw,
                       is_default=(bs == min_block and nw == default_nw))
            try:
                case["check"](case["triton"](bs, nw))  # also compiles the kernel
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
                rows.append(dict(row, **env))
                torch.cuda.empty_cache()
                continue
            rows.append(dict(measure(row, lambda: case["triton"](bs, nw), case["bytes"]), **env))

    args = case["args"]
    row = dict(base, provider="torch", is_default=True)
    rows.append(dict(measure(row, lambda: case["torch_fn"](*args), case["bytes"]), **env))

    # Reset Dynamo so each shape compiles from scratch and never hits the
    # recompile limit, which would silently fall back to eager.
    torch._dynamo.reset()
    compiled = torch.compile(case["torch_fn"])
    row = dict(base, provider="torch_compile", is_default=True)
    try:
        compiled(*args)  # compile outside the timed region
        measure(row, lambda: compiled(*args), case["bytes"])
    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    rows.append(dict(row, **env))

    del case, compiled
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def print_summary(rows):
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['provider']:>13} bs={r.get('block_size', '-')} nw={r.get('num_warps', '-')}: {r['status']}")
        elif r["provider"] != "triton" or r["is_default"]:
            print(f"  {r['provider']:>13}: {r['ms_median'] * 1e3:8.1f} us  {r['gbps_effective']:7.1f} GB/s")
    best = min((r for r in rows if r["provider"] == "triton" and r.get("status") == "ok"),
               key=lambda r: r["ms_median"], default=None)
    if best is not None:
        print(f"  {'best triton':>13}: {best['ms_median'] * 1e3:8.1f} us  "
              f"(BLOCK_SIZE={best['block_size']}, num_warps={best['num_warps']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", choices=[*CASES, "all"], default="all")
    parser.add_argument("--dtype", choices=[*DTYPES, "all"], default="all")
    parser.add_argument("--rows", type=int, default=None, help="override M (number of rows)")
    parser.add_argument("--quick", action="store_true", help="3 sizes, default Triton config only")
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    env = env_info()
    print(f"GPU: {env['gpu']} (sm_{env['compute_capability']}), torch {env['torch']}, triton {env['triton']}")

    kernels = list(CASES) if args.kernel == "all" else [args.kernel]
    dtypes = list(DTYPES) if args.dtype == "all" else [args.dtype]

    for kernel in kernels:
        sweep = SWEEPS[kernel]
        M = args.rows or sweep["M"]
        sizes = sweep["N"]
        if args.quick:
            sizes = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
        rows = []
        for dtype_name in dtypes:
            for N in sizes:
                print(f"{kernel} {dtype_name} M={M} N={N}")
                case_rows = run_case(kernel, dtype_name, M, N, args.quick, env)
                print_summary(case_rows)
                rows += case_rows
        path = results_path(f"bench_{kernel}")
        write_csv(rows, path)
        print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
