#!/usr/bin/env python3
"""
Ablation Study for ISSO
=========================
Runs 4 algorithm variants on classical benchmark functions (F1-F13, dim=30)
to isolate the contribution of each ISSO component:

  1. SSO              — baseline (no enhancements)
  2. SSO + Adaptive P — adaptive P(t) only, no Lévy
  3. SSO + Lévy       — Lévy flight only, fixed P=0.5
  4. ISSO (full)      — adaptive P(t) + Lévy flight

Output: results_ablation/ folder with CSV tables, convergence plots, boxplots.

Usage:
  python run_ablation.py                   # Full (30 runs, 500 iter)
  python run_ablation.py --quick           # Quick test (5 runs, 100 iter)
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isso_benchmark.algorithms.sso import SSO
from isso_benchmark.algorithms.isso import ISSO
from isso_benchmark.algorithms.ablation import ISSO_AdaptiveOnly, ISSO_LevyOnly
from isso_benchmark.benchmarks.classical import get_classical_functions

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────── helpers ───────────────────

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


# ─────────────────── main ───────────────────

def main():
    parser = argparse.ArgumentParser(description="ISSO Ablation Study")
    parser.add_argument("--runs",   type=int, default=30)
    parser.add_argument("--pop",    type=int, default=30)
    parser.add_argument("--iter",   type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--dim",    type=int, default=30)
    parser.add_argument("--output", type=str, default="results_ablation")
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    n_runs   = 5   if args.quick else args.runs
    max_iter = 100 if args.quick else args.iter
    dim      = args.dim

    os.makedirs(args.output, exist_ok=True)

    print("=" * 65)
    print("  ISSO ABLATION STUDY")
    print("=" * 65)
    print(f"  Dim={dim}  Runs={n_runs}  Iter={max_iter}  Pop={args.pop}")
    print("=" * 65)

    # All 23 classical functions (F1-F13 scalable + F14-F23 fixed-dim)
    all_funcs = get_classical_functions(dim=dim)
    funcs = all_funcs
    print(f"Functions: {len(funcs)} (F1-F23, scalable dim={dim}, fixed-dim as defined)")
    print()

    # 4 ablation variants
    variants = {
        "SSO":          (SSO,               {}),
        "SSO+AdaptP":   (ISSO_AdaptiveOnly, {"P_max": 0.8, "P_min": 0.2}),
        "SSO+Levy":     (ISSO_LevyOnly,     {"alpha": 0.01, "lam": 1.5}),
        "ISSO(full)":   (ISSO,              {"P_max": 0.8, "P_min": 0.2,
                                              "alpha": 0.01, "lam": 1.5}),
    }
    variant_names = list(variants.keys())

    # ──── Run all ────
    # results[func_name][variant_name] = {"fitness": [...], "convergence": [[...]]}
    results = {}

    for fi, bench_func in enumerate(funcs):
        fname = bench_func.name
        results[fname] = {}
        print(f"[{fi+1}/{len(funcs)}] {fname}")

        for vname in variant_names:
            algo_class, algo_kwargs = variants[vname]
            t0 = time.perf_counter()
            fit_list, conv_list = run_algo_on_func(
                algo_class, algo_kwargs, bench_func,
                n_runs, args.pop, max_iter, args.seed
            )
            elapsed = time.perf_counter() - t0
            results[fname][vname] = {
                "fitness": fit_list,
                "convergence": conv_list,
            }
            mean_f = np.mean(fit_list)
            print(f"    {vname:<14s}  mean={mean_f:.4e}  ({elapsed:.1f}s)")

    # ──── Build CSV table ────
    print("\nBuilding CSV...")
    header = ["Function"]
    for v in variant_names:
        header += [f"{v}_Mean", f"{v}_Std", f"{v}_Best", f"{v}_Worst", f"{v}_Median"]
    header.append("Best_Variant")

    csv_rows = [header]
    for fname in results:
        row = [fname]
        means = {}
        for v in variant_names:
            arr = np.array(results[fname][v]["fitness"])
            m = float(np.mean(arr))
            means[v] = m
            row += [f"{m:.6e}",
                    f"{np.std(arr):.6e}",
                    f"{np.min(arr):.6e}",
                    f"{np.max(arr):.6e}",
                    f"{np.median(arr):.6e}"]
        # Determine best variant (lowest mean)
        best_v = min(means, key=means.get)
        row.append(best_v)
        csv_rows.append(row)

    # Summary row
    win_count = {v: 0 for v in variant_names}
    for r in csv_rows[1:]:
        win_count[r[-1]] += 1

    csv_path = os.path.join(args.output, f"ablation_{dim}D_results.csv")
    with open(csv_path, "w") as f:
        for r in csv_rows:
            f.write(",".join(str(c) for c in r) + "\n")
    print(f"[CSV] {csv_path}")

    # ──── Summary CSV ────
    summary_path = os.path.join(args.output, f"ablation_{dim}D_summary.csv")
    with open(summary_path, "w") as f:
        f.write("Variant,Wins\n")
        for v in variant_names:
            f.write(f"{v},{win_count[v]}\n")
    print(f"[CSV] {summary_path}")

    # ──── Convergence plots (individual per function) ────
    conv_dir = os.path.join(args.output, "convergence")
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
                        color=colors[v], linestyle=styles[v],
                        linewidth=2.0, label=v)
        ax.set_title(f"Ablation: {fname} (D={dim})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Best Fitness (log)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(os.path.join(conv_dir, f"{safe}_ablation.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[Plots] {len(results)} convergence plots → {conv_dir}/")

    # ──── Combined convergence grid ────
    n = len(funcs)
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
                        color=colors[v], linestyle=styles[v],
                        linewidth=1.5, label=v)
        ax.set_title(fname.split("-")[1] if "-" in fname else fname, fontsize=8, fontweight="bold")
        ax.set_xlabel("Iter", fontsize=7)
        ax.set_ylabel("Fitness", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=5)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Ablation Study — Convergence (D={dim}, median of {n_runs} runs)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    grid_path = os.path.join(args.output, f"ablation_{dim}D_convergence_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {grid_path}")

    # ──── Boxplot grid ────
    fig, axes = plt.subplots(rows_grid, cols, figsize=(cols * 4, rows_grid * 3.5))
    axes = np.array(axes).flatten()

    for idx, fname in enumerate(results):
        ax = axes[idx]
        data = [results[fname][v]["fitness"] for v in variant_names]
        bp = ax.boxplot(data, labels=[v.replace("(full)", "") for v in variant_names],
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        box_colors = [colors[v] for v in variant_names]
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(fname.split("-")[1] if "-" in fname else fname, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Ablation Study — Fitness Distribution (D={dim}, {n_runs} runs)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    box_path = os.path.join(args.output, f"ablation_{dim}D_boxplots_grid.png")
    plt.savefig(box_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {box_path}")

    # ──── Print summary ────
    print("\n" + "=" * 65)
    print("  ABLATION SUMMARY")
    print("=" * 65)
    print(f"{'Function':<22}", end="")
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
    print(f"{'Wins':<22}", end="")
    for v in variant_names:
        print(f"  {win_count[v]:>13d}", end="")
    print()
    print("=" * 65)
    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
