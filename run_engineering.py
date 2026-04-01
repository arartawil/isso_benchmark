#!/usr/bin/env python3
"""
Run ISSO vs SSO on Engineering Design Problems
================================================
Usage:
  python run_engineering.py                   # Full run (30 runs, 500 iter)
  python run_engineering.py --quick           # Quick test (5 runs, 200 iter)
  python run_engineering.py --runs 30 --iter 1000

Output goes to  results/  folder.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isso_benchmark.benchmarks.engineering import (
    get_engineering_functions,
    welded_beam_penalised,
    _welded_beam_raw,
    _welded_beam_constraints,
)
from isso_benchmark.algorithms import SSO, ISSO
from isso_benchmark.comparison import Runner, build_stats_table, print_summary, Exporter


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SSO vs ISSO on Engineering Problems")
    parser.add_argument("--runs",   type=int, default=30,  help="Independent runs (default: 30)")
    parser.add_argument("--pop",    type=int, default=30,  help="Population size (default: 30)")
    parser.add_argument("--iter",   type=int, default=500, help="Max iterations (default: 500)")
    parser.add_argument("--seed",   type=int, default=42,  help="Base seed (default: 42)")
    parser.add_argument("--output", type=str, default="results", help="Output dir")
    parser.add_argument("--quick",  action="store_true",   help="Quick test: 5 runs, 200 iter")
    args = parser.parse_args()

    n_runs   = 5   if args.quick else args.runs
    max_iter = 200 if args.quick else args.iter

    print("=" * 60)
    print("  Engineering Optimization: SSO vs ISSO")
    print("=" * 60)
    print(f"  Runs    : {n_runs}")
    print(f"  PopSize : {args.pop}")
    print(f"  MaxIter : {max_iter}")
    print(f"  Output  : {args.output}/")
    print("=" * 60)

    eng_funcs = get_engineering_functions()
    print(f"Engineering functions loaded: {len(eng_funcs)}")
    print()

    # --- Run comparison ---
    runner = Runner(
        n_runs=n_runs,
        N=args.pop,
        max_iter=max_iter,
        base_seed=args.seed,
        verbose=True,
    )

    results = runner.run_comparison(eng_funcs)

    # --- Statistics ---
    print()
    df = build_stats_table(results)
    print_summary(df)

    # --- Export ---
    print()
    exporter = Exporter(output_dir=args.output)
    exporter.export_all(df, results, prefix="engineering")

    # --- Print detailed engineering results ---
    print("\n" + "=" * 70)
    print("  DETAILED ENGINEERING RESULTS")
    print("=" * 70)
    known_opt = 1.724852

    for fname, data in results.items():
        for algo in ["SSO", "ISSO"]:
            fits = np.array(data[algo]["fitness"])
            best_idx = np.argmin(fits)
            best_fit = fits[best_idx]

            print(f"\n  [{algo}] {fname}")
            print(f"    Best  = {best_fit:.6f}")
            print(f"    Mean  = {np.mean(fits):.6f}")
            print(f"    Std   = {np.std(fits):.6f}")
            print(f"    Worst = {np.max(fits):.6f}")
            print(f"    Known optimum = {known_opt:.6f}")
            print(f"    Relative error = {abs(best_fit - known_opt) / known_opt * 100:.4f}%")

    print("\n" + "=" * 70)
    print("Done! Check the results/ folder for CSV, LaTeX table, and plots.")


if __name__ == "__main__":
    main()
