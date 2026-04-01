# ISSO Benchmark Package

Compare **Original SSO** vs **Improved SSO (ISSO)** with Lévy Flight + Adaptive P(t).

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Quick test (5 functions, 5 runs)
python main.py --mode classical --quick

# Full classical benchmark (23 functions, 30 runs)
python main.py --mode classical --runs 30

# CEC2017 at specific dimensions
python main.py --mode cec2017 --dims 10 30 50 100 --runs 30

# Both benchmarks
python main.py --mode both --runs 30
```

## Python API

```python
from isso_benchmark.algorithms import SSO, ISSO
from isso_benchmark.benchmarks import get_classical_functions, get_cec2017_functions
from isso_benchmark.comparison import Runner, build_stats_table, Exporter

# Load functions
functions = get_classical_functions(dim=30)

# Run comparison
runner = Runner(n_runs=30, N=30, max_iter=500)
results = runner.run_comparison(functions)

# Compute stats
df = build_stats_table(results)
print(df[["Function","SSO_Mean","ISSO_Mean","Sig"]])

# Export
exp = Exporter(output_dir="results")
exp.export_all(df, results)
```

## Package Structure

```
isso_benchmark/
├── algorithms/
│   ├── sso.py          # Original SSO
│   └── isso.py         # Improved SSO (Lévy + Adaptive P(t))
├── benchmarks/
│   ├── classical.py    # 23 classical functions (F1-F23)
│   └── cec2017.py      # CEC-BC-2017 functions (F1,F3-F13) × [10,30,50,100]D
├── comparison/
│   ├── runner.py       # Run both algorithms, collect results
│   ├── statistics.py   # Mean, Std, Wilcoxon test, significance markers
│   └── exporter.py     # CSV, LaTeX table, convergence plots, boxplots
└── main.py             # CLI entry point
```

## Output Files

| File | Description |
|------|-------------|
| `comparison_results.csv` | Full numerical results table |
| `comparison_table.tex` | LaTeX table — paste directly into your paper |
| `convergence_curves.png` | Convergence curves for all functions |
| `boxplots.png` | Fitness distribution box plots |

## ISSO Enhancements

### 1. Adaptive P(t)
Replaces the fixed 0.5 threshold with a nonlinear adaptive parameter:

```
P(t) = P_max - (P_max - P_min) * (t / T_max)^2
```
- `P_max = 0.8` (start: more exploitation)
- `P_min = 0.2` (end: more exploration)

### 2. Lévy Flight (Mantegna's Algorithm)
Adds heavy-tailed random jumps to the position update:

```
sigma_u = ((Γ(1+λ)·sin(π·λ/2)) / (Γ((1+λ)/2)·λ·2^((λ-1)/2)))^(1/λ)
L(λ) = u / |v|^(1/λ),   λ = 1.5
X_new += alpha · L(λ) · (X_i - X_best)
```

## Citation

If you use this package, please cite the original SSO paper:

> Nemati, M., Zandi, Y., & Agdas, A.S. (2024). Application of a novel metaheuristic 
> algorithm inspired by stadium spectators in global optimization problems. 
> Scientific Reports, 14, 3078. https://doi.org/10.1038/s41598-024-53602-2
