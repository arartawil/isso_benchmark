"""
Runner: Executes SSO and ISSO on benchmark functions
=====================================================
Runs each algorithm 30 independent times per function.
Supports parallel execution via joblib.
"""
import time
import numpy as np
from typing import List, Dict, Optional
from ..algorithms import SSO, ISSO


def _run_single(algo_class, algo_kwargs, func, dim, lb, ub, N, max_iter, seed):
    """Run one trial of an algorithm on one function."""
    algo = algo_class(
        func=func, dim=dim, lb=lb, ub=ub,
        N=N, max_iter=max_iter, seed=seed, **algo_kwargs
    )
    best_fit, best_pos, conv_curve = algo.run()
    return best_fit, conv_curve


class Runner:
    """
    Runs SSO and ISSO on a list of benchmark functions.

    Parameters
    ----------
    n_runs   : int  - number of independent runs per function (default 30)
    N        : int  - population size (default 30)
    max_iter : int  - max iterations (default 500)
    n_jobs   : int  - parallel jobs (-1 = use all CPUs, 1 = serial)
    base_seed: int  - base random seed (each run uses base_seed + run_index)
    verbose  : bool - print progress
    """

    def __init__(self, n_runs=30, N=30, max_iter=500, n_jobs=1,
                 base_seed=42, verbose=True):
        self.n_runs = n_runs
        self.N = N
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.base_seed = base_seed
        self.verbose = verbose

    def run_function(self, bench_func, algo_class, algo_kwargs=None):
        """
        Run one algorithm on one benchmark function for n_runs trials.

        Returns
        -------
        fitness_list  : list of float  (best fitness per run)
        conv_curves   : list of lists  (convergence curve per run)
        """
        if algo_kwargs is None:
            algo_kwargs = {}

        fitness_list = []
        conv_curves = []
        time_list = []

        for run in range(self.n_runs):
            seed = self.base_seed + run
            t_start = time.perf_counter()
            best_fit, conv = _run_single(
                algo_class, algo_kwargs,
                bench_func.func, bench_func.dim,
                bench_func.lb, bench_func.ub,
                self.N, self.max_iter, seed
            )
            t_elapsed = time.perf_counter() - t_start
            fitness_list.append(best_fit)
            conv_curves.append(conv)
            time_list.append(t_elapsed)

        return fitness_list, conv_curves, time_list

    def run_comparison(self, functions, isso_kwargs=None):
        """
        Run both SSO and ISSO on all provided benchmark functions.

        Parameters
        ----------
        functions   : list of BenchmarkFunction or CEC2017Function
        isso_kwargs : dict - extra kwargs for ISSO (P_max, P_min, alpha, lam)

        Returns
        -------
        results : dict
            {func_name: {
                "SSO":  {"fitness": [...], "convergence": [...], "func": bench_func},
                "ISSO": {"fitness": [...], "convergence": [...], "func": bench_func}
            }}
        """
        if isso_kwargs is None:
            isso_kwargs = {}

        results = {}

        for i, bench_func in enumerate(functions):
            fname = bench_func.name
            if self.verbose:
                print(f"[{i+1}/{len(functions)}] Running: {fname} (dim={bench_func.dim})")

            # Run SSO
            sso_fitness, sso_conv, sso_times = self.run_function(bench_func, SSO)

            # Run ISSO
            isso_fitness, isso_conv, isso_times = self.run_function(bench_func, ISSO, isso_kwargs)

            results[fname] = {
                "SSO":  {"fitness": sso_fitness,  "convergence": sso_conv,  "times": sso_times,  "func": bench_func},
                "ISSO": {"fitness": isso_fitness, "convergence": isso_conv, "times": isso_times, "func": bench_func},
            }

            if self.verbose:
                sso_mean  = np.mean(sso_fitness)
                isso_mean = np.mean(isso_fitness)
                flag = "✓ ISSO better" if isso_mean < sso_mean else "✗ SSO better"
                print(f"   SSO mean={sso_mean:.4e}  ISSO mean={isso_mean:.4e}  {flag}")

        return results
