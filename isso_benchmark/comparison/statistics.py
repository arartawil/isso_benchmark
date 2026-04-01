"""
Statistics: Compute comparison metrics between SSO and ISSO
============================================================
- Mean, Std, Best, Worst
- Wilcoxon rank-sum test
- Significance markers: + (ISSO better), - (ISSO worse), = (no difference)
"""
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd


def compute_stats(fitness_list):
    """Compute descriptive statistics for a list of fitness values."""
    arr = np.array(fitness_list)
    return {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "best":  float(np.min(arr)),
        "worst": float(np.max(arr)),
        "median":float(np.median(arr)),
    }


def wilcoxon_test(sso_fitness, isso_fitness, alpha=0.05):
    """
    Perform Wilcoxon signed-rank test between SSO and ISSO results.

    Returns
    -------
    p_value : float
    marker  : str  ('+' ISSO better, '-' ISSO worse, '=' no difference)
    """
    arr_sso  = np.array(sso_fitness)
    arr_isso = np.array(isso_fitness)

    # Check if arrays are identical (wilcoxon requires differences)
    if np.allclose(arr_sso, arr_isso):
        return 1.0, "="

    try:
        stat, p_value = wilcoxon(arr_sso, arr_isso)
    except ValueError:
        return 1.0, "="

    if p_value < alpha:
        if np.mean(arr_isso) < np.mean(arr_sso):
            marker = "+"   # ISSO significantly better
        else:
            marker = "-"   # ISSO significantly worse
    else:
        marker = "="   # No significant difference

    return float(p_value), marker


def build_stats_table(results, alpha=0.05):
    """
    Build a full statistics comparison table.

    Parameters
    ----------
    results : dict  (output from Runner.run_comparison)
    alpha   : float significance level

    Returns
    -------
    pd.DataFrame with columns:
        Function | SSO_Mean | SSO_Std | ISSO_Mean | ISSO_Std |
        SSO_Best | ISSO_Best | p_value | Sig | Category | Dim
    """
    rows = []
    for fname, data in results.items():
        sso_fit  = data["SSO"]["fitness"]
        isso_fit = data["ISSO"]["fitness"]
        bench    = data["SSO"]["func"]

        sso_s  = compute_stats(sso_fit)
        isso_s = compute_stats(isso_fit)
        p_val, marker = wilcoxon_test(sso_fit, isso_fit, alpha)

        rows.append({
            "Function":  fname,
            "Dim":       bench.dim,
            "Category":  getattr(bench, "category", ""),
            "SSO_Mean":  sso_s["mean"],
            "SSO_Std":   sso_s["std"],
            "SSO_Best":  sso_s["best"],
            "SSO_Worst": sso_s["worst"],
            "ISSO_Mean": isso_s["mean"],
            "ISSO_Std":  isso_s["std"],
            "ISSO_Best": isso_s["best"],
            "ISSO_Worst":isso_s["worst"],
            "p_value":   p_val,
            "Sig":       marker,
        })

    df = pd.DataFrame(rows)
    return df


def print_summary(df):
    """Print a readable summary of the comparison."""
    total = len(df)
    better = (df["Sig"] == "+").sum()
    worse  = (df["Sig"] == "-").sum()
    equal  = (df["Sig"] == "=").sum()

    print("=" * 70)
    print("COMPARISON SUMMARY: SSO vs ISSO")
    print("=" * 70)
    print(f"Total functions tested : {total}")
    print(f"ISSO significantly better (+) : {better} ({100*better/total:.1f}%)")
    print(f"ISSO significantly worse  (-) : {worse}  ({100*worse/total:.1f}%)")
    print(f"No significant difference (=) : {equal}  ({100*equal/total:.1f}%)")
    print("=" * 70)

    print(f"{'Function':<30} {'SSO Mean':>12} {'ISSO Mean':>12} {'p-value':>10} {'Sig':>4}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['Function']:<30} {row['SSO_Mean']:>12.4e} {row['ISSO_Mean']:>12.4e} "
              f"{row['p_value']:>10.4f} {row['Sig']:>4}")
    print("=" * 70)
