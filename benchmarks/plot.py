"""Make figures in plots/ from the CSVs in results/.

By default it uses the newest bench_*/energy_* CSV for each kernel; pass
--csv to pick specific files. Every figure's subtitle names the GPU,
library versions and source CSV, so a figure traces back to its run.

Usage (from the repo root):
    python -m benchmarks.plot
    python -m benchmarks.plot --csv results/bench_softmax_tesla-t4_20261101-120000.csv

Figures, per kernel:
  <kernel>_bandwidth.png     effective GB/s vs row width, all providers (fp16 | fp32)
  <kernel>_config_sweep.png  Triton runtime per (BLOCK_SIZE, num_warps), relative to the best
  <kernel>_energy.png        net energy per call (only if an energy CSV exists)
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")  # write files only; no display needed (Colab, headless servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
OUT_DIR = PLOTS  # overridden by --out-dir

# Fixed color per provider (never reassigned when a provider is missing),
# taken in order from a colorblind-checked categorical palette. Markers and
# line styles differ too, so color is never the only way to tell them apart.
STYLE = {
    "triton (default)": dict(color="#2a78d6", marker="o", linestyle="-"),
    "triton (best config)": dict(color="#eb6834", marker="s", linestyle="--"),
    "torch": dict(color="#1baf7a", marker="^", linestyle="-"),
    "torch_compile": dict(color="#eda100", marker="D", linestyle="-"),
}
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,  # grid behind bars and lines
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "font.size": 10,
    "legend.frameon": False,
})


def latest(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS, pattern)))  # timestamped names sort by time
    return files[-1] if files else None


def provenance(df, path):
    r = df.iloc[0]
    return f"{r['gpu']} · torch {r['torch']} · triton {r['triton']} · {os.path.basename(path)}"


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def summary_lines(ok):
    """One series per provider: default Triton config, best Triton config, torch, torch_compile."""
    tri = ok[ok.provider == "triton"]
    series = {
        "triton (default)": tri[tri.is_default.astype(str) == "True"],
        "triton (best config)": tri.loc[tri.groupby(["dtype", "N"]).ms_median.idxmin()],
        "torch": ok[ok.provider == "torch"],
        "torch_compile": ok[ok.provider == "torch_compile"],
    }
    return {name: s.sort_values("N") for name, s in series.items() if len(s)}


def plot_bandwidth(df, path, kernel):
    ok = df[df.status == "ok"]
    dtypes = [d for d in ["fp16", "fp32"] if d in set(ok.dtype)]
    if not dtypes:
        return
    fig, axes = plt.subplots(1, len(dtypes), figsize=(5.5 * len(dtypes), 4), squeeze=False, sharey=True)
    for ax, dtype in zip(axes[0], dtypes):
        for name, s in summary_lines(ok[ok.dtype == dtype]).items():
            ax.plot(s.N, s.gbps_effective, label=name, **STYLE[name])
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(ok.N.unique()))
        ax.set_xticklabels(sorted(ok.N.unique()), rotation=45)
        ax.set_ylim(bottom=0)
        ax.set_title(dtype, loc="left")
        ax.set_xlabel(f"row width N (M = {int(ok.M.iloc[0])} rows)")
    axes[0][0].set_ylabel("effective bandwidth (GB/s, higher is better)")
    # Legend above the panels so it never covers data points.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(1, 1.02))
    fig.suptitle(f"{kernel}: effective memory bandwidth", x=0.01, y=1.06, ha="left", color=TEXT, fontsize=12)
    fig.text(0.01, -0.12, provenance(df, path) + " · median of do_bench, L2 flushed per call",
             color=TEXT_MUTED, fontsize=8)
    save(fig, f"{kernel}_bandwidth.png")


def plot_config_sweep(df, path, kernel):
    tri = df[df.provider == "triton"].copy()
    dtypes = [d for d in ["fp16", "fp32"] if d in set(tri.dtype)]
    if not dtypes or tri.num_warps.nunique() < 2:
        return  # a --quick run has only the default config; nothing to compare
    tri["block_x"] = (tri.block_size / tri.N.map(lambda n: 1 << (int(n) - 1).bit_length())).round().astype(int)
    tri["config"] = tri.apply(lambda r: f"BLOCK {r.block_x}x, {int(r.num_warps)} warps", axis=1)
    configs = (tri[["block_x", "num_warps", "config"]].drop_duplicates()
               .sort_values(["block_x", "num_warps"]).config.tolist())
    sizes = sorted(tri.N.unique())

    fig, axes = plt.subplots(1, len(dtypes), figsize=(0.75 * len(sizes) * len(dtypes) + 2.5, 0.45 * len(configs) + 1.8),
                             squeeze=False)
    for ax, dtype in zip(axes[0], dtypes):
        d = tri[tri.dtype == dtype]
        ms = d.pivot_table(index="config", columns="N", values="ms_median").reindex(index=configs, columns=sizes)
        ms[d.pivot_table(index="config", columns="N", values="status", aggfunc="first")
           .reindex(index=configs, columns=sizes) != "ok"] = np.nan
        rel = ms / ms.min(axis=0)  # 1.00 = fastest config for that N
        im = ax.imshow(np.log2(rel.values.astype(float)), cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=2)
        for i in range(len(configs)):
            for j in range(len(sizes)):
                v = rel.values[i, j]
                text, color = ("err", TEXT_MUTED) if np.isnan(v) else (f"{v:.2f}", TEXT if v < 2 else SURFACE)
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
        ax.set_xticks(range(len(sizes)), sizes, rotation=45)
        ax.set_yticks(range(len(configs)), configs if ax is axes[0][0] else [""] * len(configs))
        ax.grid(False)
        ax.set_title(dtype, loc="left")
        ax.set_xlabel("row width N")
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    cbar.set_ticks([0, 1, 2], labels=["1x (best)", "2x", ">=4x"])
    cbar.set_label("runtime relative to best config", color=TEXT_MUTED)
    fig.suptitle(f"{kernel}: Triton launch config sweep (lower is better; 'err' = failed to run)",
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.text(0.01, -0.08, provenance(df, path), color=TEXT_MUTED, fontsize=8)
    save(fig, f"{kernel}_config_sweep.png")


def plot_energy(df, path, kernel):
    ok = df[df.status == "ok"]
    dtypes = [d for d in ["fp16", "fp32"] if d in set(ok.dtype)]
    if not dtypes:
        return
    providers = [("triton", "triton (default)"), ("torch", "torch"), ("torch_compile", "torch_compile")]
    fig, axes = plt.subplots(1, len(dtypes), figsize=(5.5 * len(dtypes), 4), squeeze=False)
    for ax, dtype in zip(axes[0], dtypes):
        d = ok[ok.dtype == dtype]
        sizes = sorted(d.N.unique())
        width = 0.8 / len(providers)
        for k, (prov, name) in enumerate(providers):
            s = d[d.provider == prov].set_index("N").reindex(sizes)
            x = np.arange(len(sizes)) + (k - 1) * width
            ax.bar(x, s.joules_per_call_net * 1e6, width * 0.92, label=name, color=STYLE[name]["color"],
                   edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set_title(dtype, loc="left")
        ax.set_xlabel("row width N")
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    axes[0][0].set_ylabel("net energy per call (µJ, lower is better)")
    axes[0][-1].legend()
    idle = df.idle_power_used_w.iloc[0]
    fig.suptitle(f"{kernel}: energy per call above idle", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.text(0.01, -0.08, provenance(df, path) + f" · idle power subtracted: {idle:.1f} W",
             color=TEXT_MUTED, fontsize=8)
    save(fig, f"{kernel}_energy.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="*", help="specific CSV files (default: newest per kernel)")
    parser.add_argument("--out-dir", default=PLOTS, help="where to write PNGs (default: plots/)")
    args = parser.parse_args()
    global OUT_DIR
    OUT_DIR = args.out_dir

    paths = args.csv or [p for p in (latest(f"{kind}_{k}_*.csv")
                                      for kind in ["bench", "energy"]
                                      for k in ["softmax", "add_layernorm"]) if p]
    if not paths:
        raise SystemExit(f"No CSVs found in {RESULTS}. Run the benchmarks first.")

    for path in paths:
        df = pd.read_csv(path)
        kernel = df.kernel.iloc[0]
        if "joules_per_call_net" in df.columns:
            plot_energy(df, path, kernel)
        else:
            plot_bandwidth(df, path, kernel)
            plot_config_sweep(df, path, kernel)


if __name__ == "__main__":
    main()
