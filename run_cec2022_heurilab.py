"""
Run SSO vs ISSO on CEC-2022 benchmark functions using heurilab framework.
Settings: dim=10, pop_size=30, max_iter=500, 30 independent runs.
"""

import numpy as np
import opfunu
from math import gamma, pi, sin

from heurilab.algorithms.base import _Base
from heurilab.core.benchmarks import BenchmarkConfig, BenchmarkSuite
from heurilab.core.runner import run_experiment


# ── SSO wrapped for heurilab ──────────────────────────────────────────────

class SSO(_Base):
    """Original Stadium Spectators Optimizer."""

    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        X_best = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        for t in range(1, self.max_iter + 1):
            LI = t / self.max_iter
            for i in range(self.pop_size):
                candidates = [j for j in range(self.pop_size) if j != i]
                r1_idx, r2_idx = np.random.choice(candidates, size=2, replace=False)
                X_r1, X_r2 = X[r1_idx], X[r2_idx]
                HR = np.random.random()

                if np.random.random() < 0.5:
                    RND = X_best + HR * (X_r1 - X_r2)
                else:
                    RND = X_best - HR * np.random.random(self.dim)

                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                X_new = X[i] + r1 * (RND - X[i]) + LI * r2 * (X_r1 - X_r2)
                X_new = self._clip(X_new)

                f_new = self._eval(X_new)
                if f_new < fitness[i]:
                    X[i] = X_new
                    fitness[i] = f_new
                    if f_new < best_fit:
                        best_fit = f_new
                        X_best = X_new.copy()

            convergence.append(best_fit)

        return X_best, best_fit, convergence


# ── ISSO wrapped for heurilab ─────────────────────────────────────────────

class ISSO(_Base):
    """Improved Stadium Spectators Optimizer (Adaptive P + Lévy Flight)."""

    P_MAX = 0.8
    P_MIN = 0.2
    ALPHA = 0.01
    LAM = 1.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lam = self.LAM
        num = gamma(1 + lam) * sin(pi * lam / 2)
        den = gamma((1 + lam) / 2) * lam * (2 ** ((lam - 1) / 2))
        self._sigma_u = (num / den) ** (1 / lam)

    def _levy_step(self):
        u = np.random.normal(0, self._sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / self.LAM))

    def _adaptive_P(self, t):
        return self.P_MAX - (self.P_MAX - self.P_MIN) * (t / self.max_iter) ** 2

    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        X_best = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        range_scale = self.ub - self.lb

        for t in range(1, self.max_iter + 1):
            LI = t / self.max_iter
            P_t = self._adaptive_P(t)
            alpha_t = self.ALPHA * (1 - LI) ** 2

            for i in range(self.pop_size):
                candidates = [j for j in range(self.pop_size) if j != i]
                r1_idx, r2_idx = np.random.choice(candidates, size=2, replace=False)
                X_r1, X_r2 = X[r1_idx], X[r2_idx]
                HR = np.random.random()

                if np.random.random() < P_t:
                    RND = X_best + HR * (X_r1 - X_r2)
                else:
                    RND = X_best - HR * np.random.random(self.dim)

                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                L = self._levy_step()

                X_new = (X[i] + r1 * (RND - X[i])
                         + LI * r2 * (X_r1 - X_r2)
                         + alpha_t * L * range_scale)
                X_new = self._clip(X_new)

                f_new = self._eval(X_new)
                if f_new < fitness[i]:
                    X[i] = X_new
                    fitness[i] = f_new
                    if f_new < best_fit:
                        best_fit = f_new
                        X_best = X_new.copy()

            convergence.append(best_fit)

        return X_best, best_fit, convergence


# ── Build CEC-2022 benchmark suite (12 functions, dim=10) ─────────────────

def get_cec2022_suite(dim=10):
    """Create heurilab BenchmarkSuite from opfunu CEC-2022 functions."""
    cec2022_classes = sorted(
        [f for f in opfunu.get_cecs(2022) if f.__name__.endswith("2022")],
        key=lambda c: int(c.__name__.replace("F", "").replace("2022", "")),
    )

    benchmarks = []
    for cls in cec2022_classes:
        obj = cls(ndim=dim)
        fid = cls.__name__.replace("F", "").replace("2022", "")
        benchmarks.append(
            BenchmarkConfig(
                name=f"CEC2022-F{fid}",
                obj_func=obj.evaluate,
                lb=float(obj.lb[0]),
                ub=float(obj.ub[0]),
                dim=dim,
            )
        )
    return BenchmarkSuite(category="CEC2022", benchmarks=benchmarks)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DIM = 10
    POP_SIZE = 30
    MAX_ITER = 500
    N_RUNS = 30

    algorithms = [
        ("ISSO", ISSO),
        ("SSO",  SSO),
    ]

    suites = [get_cec2022_suite(dim=DIM)]

    print(f"CEC-2022 | dim={DIM} | pop={POP_SIZE} | iter={MAX_ITER} | runs={N_RUNS}")
    print(f"Algorithms: {[a[0] for a in algorithms]}")
    print(f"Functions : {[b.name for b in suites[0]]}")
    print("-" * 60)

    run_experiment(
        algorithms=algorithms,
        benchmark_suites=suites,
        output_dir="results_cec2022_heurilab",
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        dim=DIM,
        n_runs=N_RUNS,
        run_engineering=False,
    )

    print("\nDone! Results saved to results_cec2022_heurilab/")
