#!/usr/bin/env python3
"""
Ablation Study — F14-F23 ONLY
Completes the ablation for fixed-dimension functions, then merges with
existing F1-F13 results and regenerates all plots/summaries.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isso_benchmark.algorithms.sso import SSO
from isso_benchmark.algorithms.isso import ISSO
from isso_benchmark.algorithms.ablation import ISSO_AdaptiveOnly, ISSO_LevyOnly
from isso_benchmark.benchmarks.classical import get_classical_functions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── config ───
DIM       = 30
N_RUNS    = 30
MAX_ITER  = 500
POP       = 30
SEED      = 42
OUT_DIR   = "results_ablation"

os.makedirs(OUT_DIR, exist_ok=True)

# ─── helpers ───
def run_single(algo_class, algo_kwargs, func, dim, lb, ub, N, max_iter, seed):
    algo = algo_class(func=func, dim=dim, lb=lb, ub=ub,
                      N=N, max_iter=max_iter, seed=seed, **algo_kwargs)
    best_fit, best_pos, conv = algo.run()
    return best_fit, conv

def run_algo_on_func(algo_class, algo_kwargs, bench_func, n_runs, N, max_iter, base_seed):
    fitness_list, conv_curves = [], []
    for run in range(n_runs):
        seed = base_seed + run
        best_fit, conv = run_single(
            algo_class, algo_kwargs,
            bench_func.func, bench_func.dim,
            bench_func.lb, bench_func.ub,
            N, max_iter, seed
        )
        fitness_list.append(best_fit)
        conv_curves.append(conv)
    return fitness_list, conv_curves

# ─── variants ───
variants = {
    "SSO":        (SSO,               {}),
    "SSO+AdaptP": (ISSO_AdaptiveOnly, {"P_max": 0.8, "P_min": 0.2}),
    "SSO+Levy":   (ISSO_LevyOnly,     {"alpha": 0.01, "lam": 1.5}),
    "ISSO(full)": (ISSO,              {"P_max": 0.8, "P_min": 0.2,
                                        "alpha": 0.01, "lam": 1.5}),
}
variant_names = list(variants.keys())

# ─── get F14-F23 only ───
all_funcs = get_classical_functions(dim=DIM)
funcs_14_23 = [f for f in all_funcs if int(f.name.split("-")[0][1:]) >= 14]

print("=" * 65)
print("  ABLATION STUDY — F14-F23 (completing remaining functions)")
print("=" * 65)
print(f"  Dim={DIM}  Runs={N_RUNS}  Iter={MAX_ITER}  Pop={POP}")
print(f"  Functions: {len(funcs_14_23)}")
for f in funcs_14_23:
    print(f"    {f.name}  dim={f.dim}")
print("=" * 65)

# ─── run ───
results = {}
for fi, bench_func in enumerate(funcs_14_23):
    fname = bench_func.name
    results[fname] = {}
    print(f"\n[{fi+1}/{len(funcs_14_23)}] {fname}  (dim={bench_func.dim})")

    for vname in variant_names:
        algo_class, algo_kwargs = variants[vname]
        t0 = time.perf_counter()
        fit_list, conv_list = run_algo_on_func(
            algo_class, algo_kwargs, bench_func,
            N_RUNS, POP, MAX_ITER, SEED
        )
        elapsed = time.perf_counter() - t0
        results[fname][vname] = {
            "fitness": fit_list,
            "convergence": conv_list,
        }
        mean_f = np.mean(fit_list)
        print(f"    {vname:<14s}  mean={mean_f:.4e}  ({elapsed:.1f}s)")

# ─── build new rows ───
new_rows = []
for fname in results:
    row = [fname]
    means = {}
    for v in variant_names:
        arr = np.array(results[fname][v]["fitness"])
        m = float(np.mean(arr))
        means[v] = m
        row += [f"{m:.6e}", f"{np.std(arr):.6e}", f"{np.min(arr):.6e}",
                f"{np.max(arr):.6e}", f"{np.median(arr):.6e}"]
    best_v = min(means, key=means.get)
    row.append(best_v)
    new_rows.append(row)

# ─── merge with existing CSV ───
csv_path = os.path.join(OUT_DIR, f"ablation_{DIM}D_results.csv")
with open(csv_path, "r") as f:
    existing_lines = f.readlines()

# Keep header + existing F1-F13 rows
with open(csv_path, "w") as f:
    for line in existing_lines:
        f.write(line)
    for row in new_rows:
        f.write(",".join(str(c) for c in row) + "\n")

print(f"\n[CSV] Appended {len(new_rows)} rows to {csv_path}")

# ─── rebuild summary ───
# Re-read the full CSV to count wins
import csv
with open(csv_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    all_rows = list(reader)

win_count = {v: 0 for v in variant_names}
for row in all_rows:
    best_v = row[-1]
    if best_v in win_count:
        win_count[best_v] += 1

summary_path = os.path.join(OUT_DIR, f"ablation_{DIM}D_summary.csv")
with open(summary_path, "w") as f:
    f.write("Variant,Wins\n")
    for v in variant_names:
        f.write(f"{v},{win_count[v]}\n")
print(f"[CSV] {summary_path}")

# ─── convergence plots for F14-F23 ───
conv_dir = os.path.join(OUT_DIR, "convergence")
os.makedirs(conv_dir, exist_ok=True)

colors = {"SSO": "#2196F3", "SSO+AdaptP": "#FF9800",
          "SSO+Levy": "#4CAF50", "ISSO(full)": "#F44336"}
styles = {"SSO": "--", "SSO+AdaptP": "-.", "SSO+Levy": ":", "ISSO(full)": "-"}

for fname in results:
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variant_names:
        curves = np.array(results[fname][v]["convergence"])
        median_curve = np.median(curves, axis=0)
        iters = np.arange(1, len(median_curve) + 1)
        ax.semilogy(iters, np.clip(np.abs(median_curve), 1e-300, None),
                    color=colors[v], linestyle=styles[v], linewidth=2.0, label=v)
    ax.set_title(f"Ablation: {fname} (D={DIM})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness (log)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(os.path.join(conv_dir, f"{safe}_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close()

print(f"[Plots] {len(results)} convergence plots → {conv_dir}/")

# ─── regenerate combined grid (all 23) ───
# For the grid we need F1-F13 convergence data too, but we only have F14-F23 in memory.
# We'll just make a grid for F14-F23 + update the full grid later with the main script.

n = len(funcs_14_23)
cols = 4
rows_grid = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 4, rows_grid * 3))
axes = np.array(axes).flatten()

for idx, fname in enumerate(results):
    ax = axes[idx]
    for v in variant_names:
        curves = np.array(results[fname][v]["convergence"])
        median_curve = np.median(curves, axis=0)
        iters = np.arange(1, len(median_curve) + 1)
        ax.semilogy(iters, np.clip(np.abs(median_curve), 1e-300, None),
                    color=colors[v], linestyle=styles[v], linewidth=1.5, label=v)
    ax.set_title(fname.split("-")[1] if "-" in fname else fname, fontsize=8, fontweight="bold")
    ax.set_xlabel("Iter", fontsize=7)
    ax.set_ylabel("Fitness", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=5)

for idx in range(n, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f"Ablation Study — Convergence F14-F23 (D={DIM}, median of {N_RUNS} runs)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
grid_path = os.path.join(OUT_DIR, f"ablation_{DIM}D_convergence_F14_F23.png")
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Plot] {grid_path}")

# ─── boxplots for F14-F23 ───
fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 4, rows_grid * 3.5))
axes = np.array(axes).flatten()

for idx, fname in enumerate(results):
    ax = axes[idx]
    data = [results[fname][v]["fitness"] for v in variant_names]
    bp = ax.boxplot(data, labels=[v.replace("(full)", "") for v in variant_names],
                    patch_artist=True, medianprops=dict(color="black", linewidth=2))
    box_colors = [colors[v] for v in variant_names]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_title(fname.split("-")[1] if "-" in fname else fname, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3, axis="y")

for idx in range(n, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f"Ablation Study — Fitness Distribution F14-F23 (D={DIM}, {N_RUNS} runs)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
box_path = os.path.join(OUT_DIR, f"ablation_{DIM}D_boxplots_F14_F23.png")
plt.savefig(box_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Plot] {box_path}")

# ─── print summary ───
print("\n" + "=" * 65)
print("  ABLATION SUMMARY (ALL 23 FUNCTIONS)")
print("=" * 65)
print(f"{'Variant':<16}  Wins")
print("-" * 30)
for v in variant_names:
    print(f"{v:<16}  {win_count[v]}")
print("=" * 65)

# Print per-function table for F14-F23
print(f"\n{'Function':<22}", end="")
for v in variant_names:
    print(f"  {v:>13}", end="")
print("   Best")
print("-" * 85)
for fname in results:
    print(f"{fname:<22}", end="")
    means = {}
    for v in variant_names:
        m = np.mean(results[fname][v]["fitness"])
        means[v] = m
        print(f"  {m:>13.4e}", end="")
    best_v = min(means, key=means.get)
    print(f"   {best_v}")
print("-" * 85)
print(f"\nAll results saved to: {OUT_DIR}/")
