"""
Generate individual convergence plots and boxplots from existing CSV files.
Reads from results/ directory and saves PNGs into results/convergence/ and results/boxplots/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
CONV_CSV = os.path.join(RESULTS_DIR, "classical_30D_convergence_all_algorithms.csv")
RAW_CSV  = os.path.join(RESULTS_DIR, "classical_30D_raw_runs_all_algorithms.csv")

# Output folders
CONV_OUT = os.path.join(RESULTS_DIR, "convergence")
BOX_OUT  = os.path.join(RESULTS_DIR, "boxplots")
os.makedirs(CONV_OUT, exist_ok=True)
os.makedirs(BOX_OUT, exist_ok=True)

# ── Colors / style ──────────────────────────────────────────────────
COLORS = {"SSO": "#1f77b4", "ISSO": "#d62728"}
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

# ═══════════════════════════════════════════════════════════════════
# 1.  Convergence plots  (from median convergence CSV)
# ═══════════════════════════════════════════════════════════════════
print("Reading convergence CSV …")
conv_df = pd.read_csv(CONV_CSV)

benchmarks = conv_df["Benchmark"].unique()
# Each benchmark has 2 rows (SSO, ISSO) with Iter_0 … Iter_N
iter_cols = [c for c in conv_df.columns if c.startswith("Iter_")]
iters = np.arange(len(iter_cols))

for bm in benchmarks:
    sub = conv_df[conv_df["Benchmark"] == bm]
    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect all values to decide scale
    all_vals = []
    for _, row in sub.iterrows():
        algo = row["Algorithm"]
        vals = row[iter_cols].values.astype(float)
        all_vals.append(vals)
        ax.plot(iters, vals, label=algo,
                color=COLORS.get(algo, "gray"), linewidth=1.5)

    all_vals = np.concatenate(all_vals)
    if np.all(all_vals > 0):
        ax.set_yscale("log")
        ylabel = "Fitness (log scale)"
    elif np.any(all_vals < 0):
        # symlog works for data spanning negative → positive or all-negative
        ax.set_yscale("symlog", linthresh=1e-2)
        ylabel = "Fitness (symlog scale)"
    else:
        ylabel = "Fitness"

    ax.set_title(f"{bm} — Convergence (Median)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe = bm.replace(" ", "_").replace("/", "_")
    path = os.path.join(CONV_OUT, f"{safe}_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [conv] {path}")

# ═══════════════════════════════════════════════════════════════════
# 2.  Box plots  (from raw runs CSV)
# ═══════════════════════════════════════════════════════════════════
print("\nReading raw-runs CSV …")
raw_df = pd.read_csv(RAW_CSV, usecols=["Benchmark", "Algorithm", "BestFitness"])

for bm in raw_df["Benchmark"].unique():
    sub = raw_df[raw_df["Benchmark"] == bm]
    algos  = sub["Algorithm"].unique()
    data   = [sub[sub["Algorithm"] == a]["BestFitness"].values for a in algos]
    colors = [COLORS.get(a, "gray") for a in algos]

    fig, ax = plt.subplots(figsize=(6, 5))

    bp = ax.boxplot(data, tick_labels=algos, patch_artist=True, widths=0.5,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="gold",
                                   markeredgecolor="black", markersize=6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    # strip (jitter) overlay
    for i, (d, c) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(d))
        ax.scatter(np.full_like(d, i) + jitter, d,
                   color=c, alpha=0.5, s=18, zorder=3, edgecolors="k",
                   linewidths=0.3)

    ax.set_title(f"{bm} — Best Fitness Distribution")
    ax.set_ylabel("Best Fitness")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    safe = bm.replace(" ", "_").replace("/", "_")
    path = os.path.join(BOX_OUT, f"{safe}_boxplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [box]  {path}")

print("\nDone — all plots saved.")
