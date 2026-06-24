#!/usr/bin/env python3
"""
Run SSO vs ISSO on classical benchmarks at D=50 and D=100.
Saves separate CSV / LaTeX / plots per dimension into result_50_100/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isso_benchmark.benchmarks import get_classical_functions
from isso_benchmark.comparison import Runner, build_stats_table, print_summary, Exporter

DIMS      = [50, 100]
N_RUNS    = 30
POP       = 30
MAX_ITER  = 500
BASE_SEED = 42
OUTPUT    = "result_50_100"


def run_one_dimension(dim):
    prefix = f"classical_{dim}D"
    print("\n" + "=" * 70)
    print(f"  RUNNING ALL FUNCTIONS AT D = {dim}")
    print("=" * 70)

    funcs = get_classical_functions(dim=dim)
    print(f"  Functions loaded : {len(funcs)}")
    print(f"  Runs={N_RUNS}  Pop={POP}  Iter={MAX_ITER}  Seed={BASE_SEED}")
    print("=" * 70 + "\n")

    runner = Runner(
        n_runs=N_RUNS,
        N=POP,
        max_iter=MAX_ITER,
        base_seed=BASE_SEED,
        verbose=True,
    )

    results = runner.run_comparison(funcs)

    # --- Statistics ---
    df = build_stats_table(results)
    print()
    print_summary(df)

    # --- Export everything with dimension-specific prefix ---
    exporter = Exporter(output_dir=OUTPUT)
    exporter.export_all(df, results, prefix=prefix)

    print(f"\n  [DONE] D={dim} — files saved with prefix '{prefix}_' in {OUTPUT}/\n")


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    for dim in DIMS:
        run_one_dimension(dim)

    print("\n" + "=" * 70)
    print(f"  ALL DONE — check  {OUTPUT}/  for:")
    for dim in DIMS:
        p = f"classical_{dim}D"
        print(f"    {p}_comparison_results.csv")
        print(f"    {p}_comparison_table.tex")
        print(f"    {p}_convergence_all_algorithms.csv")
        print(f"    {p}_raw_runs_all_algorithms.csv")
        print(f"    {p}_results_all_algorithms.csv")
        print(f"    {p}_convergence_curves.png")
        print(f"    {p}_boxplots.png")
        print()
    print("=" * 70)


if __name__ == "__main__":
    main()
