"""
Ablation Study for ISSO on CEC-2017 (5 functions) using heurilab framework.
============================================================================
4 algorithm variants to isolate each ISSO component:

  1. SSO              — baseline (no enhancements)
  2. SSO + Adaptive P — adaptive P(t) only, no Lévy
  3. SSO + Lévy       — Lévy flight only, fixed P=0.5
  4. ISSO (full)      — adaptive P(t) + Lévy flight

Settings : pop_size = 30 | max_iter = 500 | n_runs = 30 | dim = 30
Functions: first 5 CEC-2017 functions (F1, F3, F4, F5, F6) — 2 unimodal +
           3 simple-multimodal, a representative ablation set.

Each function runs all 4 variants together in its own output folder, so every
folder is a direct 4-way ablation comparison (results.csv, plots, Excel).

Output   : results_ablation_cec2017_heurilab/<function>/
"""

import os
import sys
import multiprocessing
from math import gamma, pi, sin

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from heurilab.core.runner import run_experiment
from heurilab.core.benchmarks import BenchmarkSuite
from heurilab.core.cec2017 import CEC2017_FUNCTIONS
from heurilab.algorithms.base import _Base


# ── Experiment settings ───────────────────────────────────────────────────
POP         = 30
ITERS       = 500
N_RUNS      = 30
DIM         = 30
N_FUNCS     = 5          # first 5 CEC-2017 functions (F1, F3, F4, F5, F6)
OUTPUT_ROOT = "results_ablation_cec2017_heurilab"


# ── 1. SSO (baseline) ─────────────────────────────────────────────────────
class SSO(_Base):
    """Original Stadium Spectators Optimizer — no enhancements."""

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


# ── 2. SSO + Adaptive P(t) only ───────────────────────────────────────────
class SSO_AdaptP(_Base):
    """SSO + Adaptive P(t) — no Lévy flight."""

    P_MAX = 0.8
    P_MIN = 0.2

    def _adaptive_P(self, t):
        return self.P_MAX - (self.P_MAX - self.P_MIN) * (t / self.max_iter) ** 2

    def optimize(self):
        X = self._init_pop()
        fitness = np.array([self._eval(X[i]) for i in range(self.pop_size)])

        best_idx = np.argmin(fitness)
        X_best = X[best_idx].copy()
        best_fit = fitness[best_idx]
        convergence = [best_fit]

        for t in range(1, self.max_iter + 1):
            LI = t / self.max_iter
            P_t = self._adaptive_P(t)

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


# ── 3. SSO + Lévy flight only ─────────────────────────────────────────────
class SSO_Levy(_Base):
    """SSO + Lévy flight — fixed P=0.5 (no adaptive P)."""

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
            alpha_t = self.ALPHA * (1 - LI) ** 2

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


# ── 4. ISSO (full) — Adaptive P + Lévy ────────────────────────────────────
class ISSO(_Base):
    """Improved SSO — Adaptive P(t) + Lévy Flight (Mantegna)."""

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


# 4 ablation variants — ISSO first so it is treated as the proposed method.
ALGOS = [
    ("ISSO",       ISSO),
    ("SSO",        SSO),
    ("SSO+AdaptP", SSO_AdaptP),
    ("SSO+Levy",   SSO_Levy),
]


# ── One function = all 4 variants, compared in one output folder ──────────
def run_single(index, dim):
    os.chdir(HERE)
    name, func, lb, ub, _default_dim = CEC2017_FUNCTIONS[index]
    safe_name = name.replace(":", "").replace("  ", " ").strip()

    suite = BenchmarkSuite(f"CEC2017_D{dim}")
    suite.add(safe_name, func, lb, ub, dim=dim)

    print(f"[Process] Starting {safe_name} D={dim}", flush=True)
    run_experiment(
        algorithms=ALGOS,
        benchmark_suites=[suite],
        output_dir=f"{OUTPUT_ROOT}/{safe_name}",
        pop_size=POP,
        max_iter=ITERS,
        dim=dim,
        n_runs=N_RUNS,
        run_engineering=False,
    )
    print(f"[Process] Finished {safe_name} D={dim}", flush=True)


if __name__ == "__main__":
    funcs = [CEC2017_FUNCTIONS[i][0] for i in range(N_FUNCS)]
    print("=" * 65)
    print("  ISSO ABLATION STUDY — CEC-2017 (5 functions)")
    print("=" * 65)
    print(f"  Dim={DIM}  Pop={POP}  Iter={ITERS}  Runs={N_RUNS}")
    print(f"  Variants : {[a[0] for a in ALGOS]}")
    print(f"  Functions: {funcs}")
    print("=" * 65)

    processes = []
    for idx in range(N_FUNCS):
        p = multiprocessing.Process(target=run_single, args=(idx, DIM))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    print(f"\nAll ISSO ablation CEC-2017 D={DIM} done! "
          f"Results in {OUTPUT_ROOT}/")
