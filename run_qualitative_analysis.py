"""
Qualitative Analysis of ISSO on CEC-2017 and CEC-2022.
======================================================
For each function a 6-panel strip is produced (the standard
HHO/WOA-style qualitative figure):

  1. 3D function landscape        (search space, first 2 dims)
  2. Search history               (all visited positions, x1 vs x2)
  3. Average fitness              (mean population fitness per iter)
  4. Trajectory of 1st dimension  (x1 of the first agent per iter)
  5. Convergence curve            (best-so-far fitness, log scale)
  6. Exploration vs Exploitation  (population diversity balance, %)

Runs are done at dim = 2 (the convention for qualitative analysis) so the
search history and the 3D landscape correspond to the same 2-D problem.

Settings : pop_size = 30 | max_iter = 500 | dim = 2 | one seeded run/function.
Output   : results_qualitative/CEC2017/<func>.png
           results_qualitative/CEC2022/<func>.png
"""

import os
import sys
from math import gamma, pi, sin

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from heurilab.core.cec2017 import CEC2017_FUNCTIONS
import opfunu


# ── Settings ──────────────────────────────────────────────────────────────
POP       = 30
ITERS     = 500
SEED      = 42
GRID      = 80          # landscape mesh resolution
OUT       = os.path.join(HERE, "results_qualitative")

# All functions of each suite, run at the dimension used in the main study.
# (At dim=2 most hybrids/compositions are invalid, so the search runs at full
#  dim and the 3D landscape is a 2-D slice: vary x1,x2, other dims fixed at 0.)
DIM_2017     = 30
DIM_2022     = 20
CEC2017_PICK = [name for (name, *_rest) in CEC2017_FUNCTIONS]   # all 29
CEC2022_PICK = list(range(1, 13))                              # F1-F12


# ── Instrumented ISSO (records everything the figure needs) ───────────────
def run_isso_instrumented(func, lb, ub, dim, pop, iters, seed):
    np.random.seed(seed)

    # Lévy (Mantegna) scale
    P_MAX, P_MIN, ALPHA, LAM = 0.8, 0.2, 0.01, 1.5
    num = gamma(1 + LAM) * sin(pi * LAM / 2)
    den = gamma((1 + LAM) / 2) * LAM * (2 ** ((LAM - 1) / 2))
    sigma_u = (num / den) ** (1 / LAM)
    range_scale = ub - lb

    def levy():
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        return u / (np.abs(v) ** (1 / LAM))

    X = lb + (ub - lb) * np.random.random((pop, dim))
    fitness = np.array([func(X[i]) for i in range(pop)])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    best_fit = fitness[best_idx]

    conv, avg_fit, traj, history, diversity = [], [], [], [], []

    for t in range(1, iters + 1):
        LI = t / iters
        P_t = P_MAX - (P_MAX - P_MIN) * (t / iters) ** 2
        alpha_t = ALPHA * (1 - LI) ** 2

        for i in range(pop):
            candidates = [j for j in range(pop) if j != i]
            r1_idx, r2_idx = np.random.choice(candidates, size=2, replace=False)
            X_r1, X_r2 = X[r1_idx], X[r2_idx]
            HR = np.random.random()

            if np.random.random() < P_t:
                RND = X_best + HR * (X_r1 - X_r2)
            else:
                RND = X_best - HR * np.random.random(dim)

            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            X_new = (X[i] + r1 * (RND - X[i])
                     + LI * r2 * (X_r1 - X_r2)
                     + alpha_t * levy() * range_scale)
            X_new = np.clip(X_new, lb, ub)

            f_new = func(X_new)
            if f_new < fitness[i]:
                X[i] = X_new
                fitness[i] = f_new
                if f_new < best_fit:
                    best_fit = f_new
                    X_best = X_new.copy()

        # ── per-iteration records ──
        conv.append(best_fit)
        avg_fit.append(float(np.mean(fitness)))
        traj.append(float(X[0, 0]))
        history.append(X[:, :2].copy())
        # Diversity (Morales-Castañeda 2020): median-based, dimension-averaged
        med = np.median(X, axis=0)
        diversity.append(float(np.mean(np.mean(np.abs(med - X), axis=0))))

    diversity = np.array(diversity)
    div_max = diversity.max() if diversity.max() > 0 else 1.0
    xpl = diversity / div_max * 100.0
    xpt = np.abs(diversity - div_max) / div_max * 100.0

    return {
        "conv": np.array(conv),
        "avg": np.array(avg_fit),
        "traj": np.array(traj),
        "history": np.vstack(history),   # (pop*iters, 2)
        "xpl": xpl,
        "xpt": xpt,
    }


# ── Build the 6-panel qualitative figure ──────────────────────────────────
def make_figure(name, func, lb, ub, rec, out_path, dim):
    fig = plt.figure(figsize=(22, 3.4))

    # 1) 3D landscape — 2-D slice (vary x1,x2; remaining dims fixed at 0)
    ax1 = fig.add_subplot(1, 6, 1, projection="3d")
    xs = np.linspace(lb, ub, GRID)
    ys = np.linspace(lb, ub, GRID)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.empty_like(XX)
    base = np.zeros(dim)
    for a in range(GRID):
        for b in range(GRID):
            base[0] = XX[a, b]
            base[1] = YY[a, b]
            ZZ[a, b] = func(base)
    ax1.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_title(name, fontsize=9)
    ax1.set_xlabel("x1", fontsize=7); ax1.set_ylabel("x2", fontsize=7)
    ax1.tick_params(labelsize=6)

    # 2) Search history
    ax2 = fig.add_subplot(1, 6, 2)
    h = rec["history"]
    ax2.scatter(h[:, 0], h[:, 1], s=3, c="#1f5fbf", alpha=0.35, edgecolors="none")
    ax2.set_title("Search history", fontsize=9)
    ax2.set_xlabel("x1", fontsize=7); ax2.set_ylabel("x2", fontsize=7)
    ax2.set_xlim(lb, ub); ax2.set_ylim(lb, ub)
    ax2.tick_params(labelsize=6)

    iters = np.arange(1, len(rec["conv"]) + 1)

    # 3) Average fitness
    ax3 = fig.add_subplot(1, 6, 3)
    ax3.plot(iters, rec["avg"], color="#1f5fbf", linewidth=1.2)
    ax3.set_title("Average fitness", fontsize=9)
    ax3.set_xlabel("Iteration", fontsize=7); ax3.set_ylabel("Fitness", fontsize=7)
    ax3.tick_params(labelsize=6)

    # 4) Trajectory of 1st dimension
    ax4 = fig.add_subplot(1, 6, 4)
    ax4.plot(iters, rec["traj"], color="#1f5fbf", linewidth=1.0)
    ax4.set_title("Trajectory of 1st dimension", fontsize=9)
    ax4.set_xlabel("Iteration", fontsize=7); ax4.set_ylabel("x1 value", fontsize=7)
    ax4.tick_params(labelsize=6)

    # 5) Convergence curve
    ax5 = fig.add_subplot(1, 6, 5)
    ax5.semilogy(iters, np.clip(np.abs(rec["conv"]), 1e-300, None),
                 color="#1f5fbf", linewidth=1.4)
    ax5.set_title("Convergence curve", fontsize=9)
    ax5.set_xlabel("Iteration", fontsize=7); ax5.set_ylabel("Best score", fontsize=7)
    ax5.tick_params(labelsize=6)

    # 6) Exploration vs Exploitation
    ax6 = fig.add_subplot(1, 6, 6)
    ax6.plot(iters, rec["xpl"], color="#d62728", linewidth=1.3, label="Exploration")
    ax6.plot(iters, rec["xpt"], color="#1f5fbf", linewidth=1.3, label="Exploitation")
    ax6.set_title(name, fontsize=9)
    ax6.set_xlabel("Iteration", fontsize=7); ax6.set_ylabel("Percentage (%)", fontsize=7)
    ax6.legend(fontsize=6, loc="center right")
    ax6.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────
def run_cec2017():
    out_dir = os.path.join(OUT, "CEC2017")
    os.makedirs(out_dir, exist_ok=True)
    by_name = {n: (n, f, lb, ub) for (n, f, lb, ub, _d) in CEC2017_FUNCTIONS}
    ok, skipped = 0, []
    for name in CEC2017_PICK:
        _, func, lb, ub = by_name[name]
        print(f"[CEC2017] {name} (D={DIM_2017}) ...", flush=True)
        try:
            rec = run_isso_instrumented(func, float(lb), float(ub),
                                        DIM_2017, POP, ITERS, SEED)
            make_figure(name, func, float(lb), float(ub), rec,
                        os.path.join(out_dir, f"{name}.png"), DIM_2017)
            ok += 1
        except Exception as e:
            print(f"    SKIP {name}: {type(e).__name__}: {str(e)[:60]}", flush=True)
            skipped.append(name)
    print(f"[CEC2017] done -> {out_dir}  ({ok} figures, {len(skipped)} skipped)")
    if skipped:
        print(f"           skipped: {skipped}")


def run_cec2022():
    out_dir = os.path.join(OUT, "CEC2022")
    os.makedirs(out_dir, exist_ok=True)
    classes = {int(c.__name__.replace("F", "").replace("2022", "")): c
               for c in opfunu.get_cecs(2022) if c.__name__.endswith("2022")}
    ok, skipped = 0, []
    for fid in CEC2022_PICK:
        name = f"CEC2022-F{fid}"
        print(f"[CEC2022] {name} (D={DIM_2022}) ...", flush=True)
        try:
            obj = classes[fid](ndim=DIM_2022)
            lb, ub = float(obj.lb[0]), float(obj.ub[0])
            rec = run_isso_instrumented(obj.evaluate, lb, ub,
                                        DIM_2022, POP, ITERS, SEED)
            make_figure(name, obj.evaluate, lb, ub, rec,
                        os.path.join(out_dir, f"{name}.png"), DIM_2022)
            ok += 1
        except Exception as e:
            print(f"    SKIP {name}: {type(e).__name__}: {str(e)[:60]}", flush=True)
            skipped.append(name)
    print(f"[CEC2022] done -> {out_dir}  ({ok} figures, {len(skipped)} skipped)")
    if skipped:
        print(f"           skipped: {skipped}")


if __name__ == "__main__":
    print("=" * 65)
    print(f"  ISSO QUALITATIVE ANALYSIS  (CEC2017 D={DIM_2017}, "
          f"CEC2022 D={DIM_2022}, pop={POP}, iter={ITERS})")
    print("=" * 65)
    run_cec2017()
    run_cec2022()
    print(f"\nAll qualitative figures saved under {OUT}/")
