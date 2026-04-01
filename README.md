# ISSO: Improved Stadium Spectators Optimizer

An enhanced version of the **Stadium Spectators Optimizer (SSO)** with two key improvements:
1. **Adaptive P(t)** — nonlinear exploitation-to-exploration transition
2. **Lévy Flight (Mantegna's Algorithm)** — heavy-tailed random jumps for better global search

Benchmarked on **Classical (F1–F23)**, **CEC-2017**, and **CEC-2022** test suites with full statistical analysis using Wilcoxon signed-rank tests.

---

## Repository Structure

```
├── isso_benchmark/                         # Core package
│   ├── algorithms/
│   │   ├── sso.py                          # Original SSO (baseline)
│   │   ├── isso.py                         # Improved SSO (Adaptive P + Lévy)
│   │   └── ablation.py                     # Ablation variants (AdaptP-only, Lévy-only)
│   ├── benchmarks/
│   │   ├── classical.py                    # 23 classical functions (F1–F23)
│   │   ├── cec2017.py                      # CEC-2017 (12 functions, dims 10–100)
│   │   └── engineering.py                  # 3 engineering design problems
│   ├── comparison/
│   │   ├── runner.py                       # Experiment runner (SSO vs ISSO)
│   │   ├── statistics.py                   # Wilcoxon test & stats
│   │   └── exporter.py                     # CSV, LaTeX, plots export
│   └── main.py                             # CLI entry point
│
├── run_cec2022_heurilab.py                 # SSO vs ISSO on CEC-2022 via heurilab
├── run_ablation_cec2022_heurilab.py        # Ablation study on CEC-2022 via heurilab
├── run_ablation.py                         # Ablation study on Classical functions
├── run_50_100.py                           # Classical benchmarks at D=50, D=100
├── run_engineering.py                      # Engineering optimization problems
│
├── results/                                # Classical + CEC-2017 results
├── results_ablation/                       # Ablation study (Classical)
├── results_cec2022_heurilab/               # CEC-2022 results (SSO vs ISSO)
│   ├── CSV Data/                           # results.csv, convergence.csv, raw_runs.csv
│   ├── Convergence Curves/                 # 12 convergence plots
│   ├── Box Plots/                          # 12 box plots
│   └── Excel Files/                        # Wilcoxon.xlsx, Friedman.xlsx, Results.xlsx
├── results_ablation_cec2022_heurilab/      # CEC-2022 ablation (4 variants)
│   ├── CSV Data/
│   ├── Convergence Curves/
│   ├── Box Plots/
│   └── Excel Files/
└── result_50_100/                          # High-dimension results
```

---

## Algorithms

### Original SSO
The Stadium Spectators Optimizer uses a fixed probability threshold (P = 0.5) and a linear intensity parameter (LI) to update candidate solutions based on the global best and random individuals.

### ISSO (This Work)
Two enhancements over the original SSO:

**1. Adaptive P(t)** — replaces the fixed threshold with a nonlinear decay:
```
P(t) = P_max - (P_max - P_min) × (t / T_max)²
```
- Starts at P_max = 0.8 (strong exploitation)
- Decays to P_min = 0.2 (more exploration)

**2. Lévy Flight (Mantegna)** — adds heavy-tailed jumps for escaping local optima:
```
L(λ) = u / |v|^(1/λ),   u ~ N(0, σ_u²),   v ~ N(0,1)
X_new += α_t × L(λ) × (ub - lb)
```
- λ = 1.5, α = 0.01, with decaying intensity: α_t = α × (1 − LI)²

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Quick test
```bash
python isso_benchmark/main.py --mode classical --quick
```

### Full benchmarks (native runner)
```bash
# Classical functions (F1–F23), dim=30, 30 runs
python isso_benchmark/main.py --mode classical --runs 30

# CEC-2017 at multiple dimensions
python isso_benchmark/main.py --mode cec2017 --dims 10 30 50 100 --runs 30

# Engineering problems
python run_engineering.py
```

### CEC-2022 via heurilab
```bash
# SSO vs ISSO comparison (dim=10, pop=30, iter=500, 30 runs)
python run_cec2022_heurilab.py

# Ablation study: SSO, SSO+AdaptP, SSO+Lévy, ISSO
python run_ablation_cec2022_heurilab.py
```

### Ablation study (Classical)
```bash
python run_ablation.py --dim 30 --runs 30 --iter 500
```

---

## Results Summary

### CEC-2022 (D=10) — SSO vs ISSO

| Function | ISSO Mean | SSO Mean | p-value | Sig |
|----------|-----------|----------|---------|-----|
| F1  | 300.00   | 300.00   | 1.86e-09 | - |
| F2  | 406.66   | 412.15   | 4.04e-01 | = |
| F3  | 600.00   | 600.00   | 1.86e-09 | - |
| F4  | 843.23   | 844.84   | 9.19e-01 | = |
| F5  | 900.59   | 900.83   | 1.71e-01 | = |
| F6  | 22362.88 | 16999.27 | 3.84e-02 | - |
| F7  | 2038.75  | 2040.63  | 7.61e-01 | = |
| F8  | 2223.89  | 2257.02  | 5.59e-05 | + |
| F9  | 2550.65  | 2611.34  | 2.21e-02 | + |
| F10 | 1998.15  | 2551.80  | 1.34e-03 | + |
| F11 | 2688.46  | 2734.05  | 1.35e-01 | = |
| F12 | 2883.29  | 2903.71  | 2.34e-02 | + |

**W/T/L = 4/5/3** (+: ISSO better, =: tie, -: SSO better)

### Ablation Study — CEC-2022 (D=10)

| Variant | Wins (Best Mean) |
|---------|-----------------|
| ISSO (full) | 5 |
| SSO + Lévy | 3 |
| SSO + AdaptP | 3 |
| SSO (baseline) | 1 |

Both components contribute, and the full ISSO combining both wins most often.

---

## Output Formats

| File | Description |
|------|-------------|
| `results.csv` | Mean, Std, Best, Worst, Median, p-value, Significance |
| `convergence.csv` | Median convergence curves per algorithm |
| `raw_runs.csv` | Per-run data (30 runs × all functions) |
| `Wilcoxon.xlsx` | Pairwise Wilcoxon signed-rank tests |
| `Friedman.xlsx` | Friedman ranking test |
| `*.png` | Convergence curves and box plots |

---

## Citation

If you use this work, please cite the original SSO paper:

> Nemati, M., Zandi, Y., & Agdas, A.S. (2024). Application of a novel metaheuristic
> algorithm inspired by stadium spectators in global optimization problems.
> Scientific Reports, 14, 3078. https://doi.org/10.1038/s41598-024-53602-2

---

## License

MIT
