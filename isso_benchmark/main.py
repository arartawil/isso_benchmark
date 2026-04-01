#!/usr/bin/env python3
"""
ISSO Benchmark - Main Entry Point
===================================
Usage examples:

  # Quick test (5 classical functions, 5 runs)
  python main.py --mode classical --runs 5 --quick

  # Full classical comparison (23 functions, 30 runs)
  python main.py --mode classical --runs 30

  # CEC2017 at specific dimensions
  python main.py --mode cec2017 --dims 10 30 --runs 30

  # Full comparison (classical + CEC2017)
  python main.py --mode both --runs 30

  # Custom output directory
  python main.py --mode classical --runs 10 --output my_results
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isso_benchmark.benchmarks import get_classical_functions, get_cec2017_functions, get_engineering_functions
from isso_benchmark.comparison import Runner, build_stats_table, print_summary, Exporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SSO vs ISSO on benchmark functions"
    )
    parser.add_argument("--mode", choices=["classical", "cec2017", "both", "engineering", "all"],
                        default="classical",
                        help="Which benchmark set to run (default: classical)")
    parser.add_argument("--dims", nargs="+", type=int,
                        default=[10, 30, 50, 100],
                        help="Dimensions for CEC2017 (default: 10 30 50 100)")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of independent runs per function (default: 30)")
    parser.add_argument("--pop", type=int, default=30,
                        help="Population size (default: 30)")
    parser.add_argument("--iter", type=int, default=500,
                        help="Max iterations (default: 500)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only F1-F5, 5 runs, 100 iter (for testing)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix for output filenames")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  ISSO Benchmark: SSO vs ISSO Comparison")
    print("=" * 60)
    print(f"  Mode    : {args.mode}")
    print(f"  Runs    : {args.runs}")
    print(f"  PopSize : {args.pop}")
    print(f"  MaxIter : {args.iter}")
    print(f"  Output  : {args.output}/")
    if args.mode in ["cec2017", "both"]:
        print(f"  Dims    : {args.dims}")
    print("=" * 60)

    # --- Collect benchmark functions ---
    functions = []

    if args.mode in ["classical", "both", "all"]:
        classical = get_classical_functions(dim=30)
        if args.quick:
            classical = classical[:5]
        functions.extend(classical)
        print(f"Classical functions loaded: {len(classical)}")

    if args.mode in ["cec2017", "both", "all"]:
        cec_dims = [10, 30] if args.quick else args.dims
        cec_funcs = get_cec2017_functions(dims=cec_dims)
        if args.quick:
            cec_funcs = cec_funcs[:6]
        functions.extend(cec_funcs)
        print(f"CEC2017 functions loaded: {len(cec_funcs)}")

    if args.mode in ["engineering", "all"]:
        eng_funcs = get_engineering_functions()
        functions.extend(eng_funcs)
        print(f"Engineering functions loaded: {len(eng_funcs)}")

    print(f"Total functions: {len(functions)}")
    print()

    # Quick mode adjustments
    n_runs = 5 if args.quick else args.runs
    max_iter = 100 if args.quick else args.iter

    if args.quick:
        print("[QUICK MODE] Using 5 runs, 100 iterations")

    # --- Run comparison ---
    runner = Runner(
        n_runs=n_runs,
        N=args.pop,
        max_iter=max_iter,
        base_seed=args.seed,
        verbose=True
    )

    print()
    results = runner.run_comparison(functions)

    # --- Compute statistics ---
    print()
    df = build_stats_table(results)
    print_summary(df)

    # --- Export results ---
    print()
    exporter = Exporter(output_dir=args.output)
    exporter.export_all(df, results, prefix=args.prefix or args.mode)

    print()
    print("Done! Check the results/ folder for:")
    print("  - comparison_results.csv   (full data)")
    print("  - comparison_table.tex     (paste into your paper)")
    print("  - convergence_curves.png   (convergence plots)")
    print("  - boxplots.png             (fitness distributions)")


if __name__ == "__main__":
    main()
