"""
ISSO Ablation Variants
=======================
Two intermediate variants for the ablation study:
  1. ISSO_AdaptiveOnly  — Adaptive P(t), NO Lévy flight
  2. ISSO_LevyOnly      — Fixed P=0.5, WITH Lévy flight

Combined with SSO (no enhancements) and ISSO (both), this gives
four configurations to isolate each component's contribution.
"""
import numpy as np
from math import gamma, pi, sin


class ISSO_AdaptiveOnly:
    """SSO + Adaptive P(t) only — no Lévy flight."""

    def __init__(self, func, dim, lb, ub, N=30, max_iter=500, seed=None,
                 P_max=0.8, P_min=0.2, **_kw):
        self.func = func
        self.dim = dim
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb, dtype=float)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub, dtype=float)
        self.N = N
        self.max_iter = max_iter
        self.seed = seed
        self.P_max = P_max
        self.P_min = P_min
        self.best_fitness = None
        self.best_position = None
        self.convergence_curve = []

    def _adaptive_P(self, t):
        return self.P_max - (self.P_max - self.P_min) * (t / self.max_iter) ** 2

    def _boundary_correction(self, x):
        x_new = x.copy()
        below = x_new < self.lb
        above = x_new > self.ub
        range_ = self.ub - self.lb
        range_[range_ == 0] = 1e-10
        x_new[below] = self.lb[below] + np.abs(x_new[below] - self.lb[below]) % range_[below]
        x_new[above] = self.ub[above] - np.abs(x_new[above] - self.ub[above]) % range_[above]
        return x_new

    def run(self):
        rng = np.random.default_rng(self.seed)
        pop = rng.uniform(0, 1, (self.N, self.dim)) * (self.ub - self.lb) + self.lb
        fitness = np.array([self.func(pop[i]) for i in range(self.N)])
        best_idx = np.argmin(fitness)
        X_best = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        self.convergence_curve = []

        for t in range(1, self.max_iter + 1):
            LI = t / self.max_iter
            P_t = self._adaptive_P(t)                     # <-- adaptive

            for i in range(self.N):
                candidates = [j for j in range(self.N) if j != i]
                r1_idx, r2_idx = rng.choice(candidates, size=2, replace=False)
                X_r1, X_r2 = pop[r1_idx], pop[r2_idx]
                HR = rng.random()

                if rng.random() < P_t:                    # <-- adaptive P
                    RND = X_best + HR * (X_r1 - X_r2)
                else:
                    RND = X_best - HR * rng.random(self.dim)

                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)

                # NO Lévy term
                X_new = pop[i] + r1 * (RND - pop[i]) + LI * r2 * (X_r1 - X_r2)
                X_new = self._boundary_correction(X_new)

                f_new = self.func(X_new)
                if f_new < fitness[i]:
                    pop[i] = X_new
                    fitness[i] = f_new
                    if f_new < best_fit:
                        best_fit = f_new
                        X_best = X_new.copy()

            self.convergence_curve.append(best_fit)

        self.best_fitness = best_fit
        self.best_position = X_best
        return self.best_fitness, self.best_position, self.convergence_curve


class ISSO_LevyOnly:
    """SSO + Lévy flight only — fixed P = 0.5 (same as original SSO)."""

    def __init__(self, func, dim, lb, ub, N=30, max_iter=500, seed=None,
                 alpha=0.01, lam=1.5, **_kw):
        self.func = func
        self.dim = dim
        self.lb = np.full(dim, lb) if np.isscalar(lb) else np.array(lb, dtype=float)
        self.ub = np.full(dim, ub) if np.isscalar(ub) else np.array(ub, dtype=float)
        self.N = N
        self.max_iter = max_iter
        self.seed = seed
        self.alpha = alpha
        self.lam = lam
        self.best_fitness = None
        self.best_position = None
        self.convergence_curve = []
        num = gamma(1 + lam) * sin(pi * lam / 2)
        den = gamma((1 + lam) / 2) * lam * (2 ** ((lam - 1) / 2))
        self.sigma_u = (num / den) ** (1 / lam)

    def _levy_step(self, rng):
        u = rng.normal(0, self.sigma_u, self.dim)
        v = rng.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / self.lam))

    def _boundary_correction(self, x):
        x_new = x.copy()
        below = x_new < self.lb
        above = x_new > self.ub
        range_ = self.ub - self.lb
        range_[range_ == 0] = 1e-10
        x_new[below] = self.lb[below] + np.abs(x_new[below] - self.lb[below]) % range_[below]
        x_new[above] = self.ub[above] - np.abs(x_new[above] - self.ub[above]) % range_[above]
        return x_new

    def run(self):
        rng = np.random.default_rng(self.seed)
        pop = rng.uniform(0, 1, (self.N, self.dim)) * (self.ub - self.lb) + self.lb
        fitness = np.array([self.func(pop[i]) for i in range(self.N)])
        best_idx = np.argmin(fitness)
        X_best = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        self.convergence_curve = []

        range_scale = self.ub - self.lb

        for t in range(1, self.max_iter + 1):
            LI = t / self.max_iter
            alpha_t = self.alpha * (1 - LI) ** 2

            for i in range(self.N):
                candidates = [j for j in range(self.N) if j != i]
                r1_idx, r2_idx = rng.choice(candidates, size=2, replace=False)
                X_r1, X_r2 = pop[r1_idx], pop[r2_idx]
                HR = rng.random()

                if rng.random() < 0.5:                    # <-- fixed P = 0.5
                    RND = X_best + HR * (X_r1 - X_r2)
                else:
                    RND = X_best - HR * rng.random(self.dim)

                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)

                L = self._levy_step(rng)                  # <-- Lévy flight

                X_new = (pop[i] + r1 * (RND - pop[i])
                         + LI * r2 * (X_r1 - X_r2)
                         + alpha_t * L * range_scale)     # <-- Lévy term
                X_new = self._boundary_correction(X_new)

                f_new = self.func(X_new)
                if f_new < fitness[i]:
                    pop[i] = X_new
                    fitness[i] = f_new
                    if f_new < best_fit:
                        best_fit = f_new
                        X_best = X_new.copy()

            self.convergence_curve.append(best_fit)

        self.best_fitness = best_fit
        self.best_position = X_best
        return self.best_fitness, self.best_position, self.convergence_curve
