"""
Run ISSO vs SSO on the CEC-2017 suite at D = 30 using the heurilab package.

Settings : pop_size = 30 | max_iter = 500 | n_runs = 30
Functions: 29 CEC-2017 functions (F2 excluded, as standard), launched in
           3 waves of 10 / 10 / 9 functions to bound parallelism.

Both algorithms run together per function, so each output folder contains a
direct ISSO-vs-SSO comparison (results.csv with p-values, plots, Excel).

Output   : experiments/CEC_2017_30_500/ISSO_vs_SSO_CEC2017_D30/<function>/
"""

import os
import sys
import multiprocessing
from math import gamma, pi, sin

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from heurilab import run_experiment
from heurilab.core.benchmarks import BenchmarkSuite
from heurilab.core.cec2017 import CEC2017_FUNCTIONS
from heurilab.algorithms.base import _Base


# ── Experiment settings ───────────────────────────────────────────────────
POP    = 30
ITERS  = 500
N_RUNS = 30
DIM    = 50

# 3 waves of function indices (29 functions -> 10 / 10 / 9)
WAVES = [list(range(0, 10)), list(range(10, 20)), list(range(20, 29))]


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


ALGOS = [("ISSO", ISSO), ("SSO", SSO)]


# ── One function = both algorithms, compared in one output folder ─────────
def run_single(index, dim=DIM):
    os.chdir(HERE)
    name, func, lb, ub, _default_dim = CEC2017_FUNCTIONS[index]
    safe_name = name.replace(":", "").replace("  ", " ").strip()

    suite = BenchmarkSuite(f"CEC2017_D{dim}")
    suite.add(safe_name, func, lb, ub, dim=dim)

    print(f"[Process] Starting {safe_name} D={dim}")
    run_experiment(
        algorithms=ALGOS,
        benchmark_suites=[suite],
        output_dir=f"experiments/CEC_2017_30_500/ISSO_vs_SSO_CEC2017_D{dim}/{safe_name}",
        pop_size=POP,
        max_iter=ITERS,
        dim=dim,
        n_runs=N_RUNS,
        run_engineering=False,
    )
    print(f"[Process] Finished {safe_name} D={dim}")


if __name__ == "__main__":
    print(f"CEC-2017 | ISSO vs SSO | dim={DIM} | pop={POP} | iter={ITERS} | runs={N_RUNS}")
    for w, wave in enumerate(WAVES, start=1):
        funcs = [CEC2017_FUNCTIONS[i][0] for i in wave]
        print(f"\n=== Wave {w}/{len(WAVES)} ({len(wave)} functions) ===")
        print("   " + ", ".join(funcs))
        processes = []
        for idx in wave:
            p = multiprocessing.Process(target=run_single, args=(idx, DIM))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        print(f"=== Wave {w} done ===")
    print(f"\nAll ISSO vs SSO CEC-2017 D={DIM} done! "
          f"Results in experiments/CEC_2017_30_500/ISSO_vs_SSO_CEC2017_D{DIM}/")
