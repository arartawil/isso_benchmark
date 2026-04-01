"""
Exporter: Save results as CSV, LaTeX table, and convergence plots
=================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Exporter:
    """
    Export comparison results to various formats.

    Parameters
    ----------
    output_dir : str - directory to save files (created if not exists)
    """

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_csv(self, df, filename="comparison_results.csv"):
        """Save full results table as CSV."""
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"[CSV] Saved: {path}")
        return path

    def export_latex(self, df, filename="comparison_table.tex"):
        """
        Generate LaTeX table ready to paste in paper.
        Format: Function | SSO Mean±Std | ISSO Mean±Std | p-value | Sig
        """
        path = os.path.join(self.output_dir, filename)

        def fmt(mean, std):
            return f"${mean:.2e} \\pm {std:.2e}$"

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Comparison of SSO and ISSO on Benchmark Functions (30 runs)}")
        lines.append(r"\label{tab:comparison}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{llllll}")
        lines.append(r"\hline")
        lines.append(r"\textbf{Function} & \textbf{Dim} & \textbf{SSO (Mean$\pm$Std)} & "
                     r"\textbf{ISSO (Mean$\pm$Std)} & \textbf{p-value} & \textbf{Sig} \\")
        lines.append(r"\hline")

        prev_cat = None
        for _, row in df.iterrows():
            cat = row.get("Category", "")
            if cat and cat != prev_cat:
                lines.append(r"\multicolumn{6}{l}{\textit{" + cat + r"}} \\")
                prev_cat = cat

            sso_str  = fmt(row["SSO_Mean"],  row["SSO_Std"])
            isso_str = fmt(row["ISSO_Mean"], row["ISSO_Std"])
            p_str    = f"{row['p_value']:.4f}"
            sig      = row["Sig"]

            # Bold ISSO if better
            if sig == "+":
                isso_str = r"\textbf{" + isso_str + "}"
            elif sig == "-":
                sso_str = r"\textbf{" + sso_str + "}"

            line = (f"{row['Function']} & {int(row['Dim'])} & {sso_str} & "
                    f"{isso_str} & {p_str} & {sig} \\\\")
            lines.append(line)

        lines.append(r"\hline")
        lines.append(r"\multicolumn{6}{l}{\small +: ISSO significantly better; "
                     r"-: ISSO significantly worse; =: no significant difference ($\alpha=0.05$)} \\")
        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table}")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"[LaTeX] Saved: {path}")
        return path

    def export_convergence_plots(self, results, filename="convergence_curves.png",
                                  max_funcs=12):
        """
        Plot convergence curves for SSO vs ISSO.
        Shows median convergence curve across 30 runs.
        """
        func_names = list(results.keys())[:max_funcs]
        n = len(func_names)
        cols = 4
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
        axes = np.array(axes).flatten()

        colors = {"SSO": "#2196F3", "ISSO": "#F44336"}
        styles = {"SSO": "--",      "ISSO": "-"}

        for idx, fname in enumerate(func_names):
            ax = axes[idx]
            data = results[fname]

            for algo_name in ["SSO", "ISSO"]:
                curves = np.array(data[algo_name]["convergence"])
                median_curve = np.median(curves, axis=0)
                iters = np.arange(1, len(median_curve)+1)
                ax.semilogy(iters, np.clip(np.abs(median_curve), 1e-300, None),
                            color=colors[algo_name],
                            linestyle=styles[algo_name],
                            linewidth=1.8,
                            label=algo_name)

            ax.set_title(fname.split("-D")[0], fontsize=8, fontweight="bold")
            ax.set_xlabel("Iteration", fontsize=7)
            ax.set_ylabel("Best Fitness (log)", fontsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

        # Hide unused subplots
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Convergence Curves: SSO vs ISSO (Median of 30 Runs)",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved: {path}")
        return path

    def export_boxplots(self, results, filename="boxplots.png", max_funcs=12):
        """
        Box plots comparing SSO vs ISSO fitness distributions.
        """
        func_names = list(results.keys())[:max_funcs]
        n = len(func_names)
        cols = 4
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.5))
        axes = np.array(axes).flatten()

        for idx, fname in enumerate(func_names):
            ax = axes[idx]
            data = results[fname]

            sso_fit  = data["SSO"]["fitness"]
            isso_fit = data["ISSO"]["fitness"]

            bp = ax.boxplot([sso_fit, isso_fit],
                            labels=["SSO", "ISSO"],
                            patch_artist=True,
                            medianprops=dict(color="black", linewidth=2))

            bp["boxes"][0].set_facecolor("#90CAF9")
            bp["boxes"][1].set_facecolor("#EF9A9A")

            ax.set_title(fname.split("-D")[0], fontsize=8, fontweight="bold")
            ax.set_ylabel("Best Fitness", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, axis="y")

        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Fitness Distribution: SSO vs ISSO (30 Runs)",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Boxplot] Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # NEW: Three CSV exports matching the requested formats
    # ------------------------------------------------------------------

    def export_convergence_csv(self, results, filename="convergence_all_algorithms.csv"):
        """
        One row per (Benchmark × Algorithm) with the **median** convergence
        curve across all runs.

        Columns: Benchmark | Algorithm | Iter_0 | Iter_1 | … | Iter_N
        """
        rows = []
        for fname, data in results.items():
            for algo_name in data:
                if algo_name in ("SSO", "ISSO"):
                    curves = np.array(data[algo_name]["convergence"])
                    median_curve = np.median(curves, axis=0)
                    row = {"Benchmark": fname, "Algorithm": algo_name}
                    for idx, val in enumerate(median_curve):
                        row[f"Iter_{idx}"] = val
                    rows.append(row)

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"[CSV] Saved: {path}")
        return path

    def export_raw_runs_csv(self, results, filename="raw_runs_all_algorithms.csv"):
        """
        One row per individual run.

        Columns: Benchmark | Algorithm | Run | BestFitness | Time_s |
                 Conv_0 | Conv_1 | … | Conv_N
        """
        rows = []
        for fname, data in results.items():
            for algo_name in data:
                if algo_name in ("SSO", "ISSO"):
                    algo_data = data[algo_name]
                    fitness_list = algo_data["fitness"]
                    conv_curves = algo_data["convergence"]
                    time_list = algo_data.get("times", [None] * len(fitness_list))

                    for run_idx in range(len(fitness_list)):
                        row = {
                            "Benchmark": fname,
                            "Algorithm": algo_name,
                            "Run": run_idx,
                            "BestFitness": fitness_list[run_idx],
                            "Time_s": round(time_list[run_idx], 2) if time_list[run_idx] is not None else "",
                        }
                        for c_idx, val in enumerate(conv_curves[run_idx]):
                            row[f"Conv_{c_idx}"] = val
                        rows.append(row)

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"[CSV] Saved: {path}")
        return path

    def export_results_csv(self, results, filename="results_all_algorithms.csv"):
        """
        One row per benchmark with summary statistics for every algorithm.

        Columns: Benchmark | SSO_Best | SSO_Mean | SSO_Std | SSO_Worst | SSO_Median |
                 ISSO_Best | ISSO_Mean | ISSO_Std | ISSO_Worst | ISSO_Median
        """
        rows = []
        for fname, data in results.items():
            row = {"Benchmark": fname}
            for algo_name in data:
                if algo_name in ("SSO", "ISSO"):
                    arr = np.array(data[algo_name]["fitness"])
                    row[f"{algo_name}_Best"]   = float(np.min(arr))
                    row[f"{algo_name}_Mean"]   = float(np.mean(arr))
                    row[f"{algo_name}_Std"]    = float(np.std(arr))
                    row[f"{algo_name}_Worst"]  = float(np.max(arr))
                    row[f"{algo_name}_Median"] = float(np.median(arr))
            rows.append(row)

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"[CSV] Saved: {path}")
        return path

    def export_individual_convergence(self, results, prefix=""):
        """
        Generate one convergence plot per function, saved as individual PNG files.
        Each plot shows SSO vs ISSO median convergence curve (log scale).
        """
        subdir = os.path.join(self.output_dir, f"{prefix}convergence_plots" if prefix else "convergence_plots")
        os.makedirs(subdir, exist_ok=True)

        colors = {"SSO": "#2196F3", "ISSO": "#F44336"}
        styles = {"SSO": "--",      "ISSO": "-"}

        for fname, data in results.items():
            fig, ax = plt.subplots(figsize=(8, 5))

            for algo_name in ["SSO", "ISSO"]:
                curves = np.array(data[algo_name]["convergence"])
                median_curve = np.median(curves, axis=0)
                q25 = np.percentile(curves, 25, axis=0)
                q75 = np.percentile(curves, 75, axis=0)
                iters = np.arange(1, len(median_curve) + 1)

                ax.semilogy(iters, np.clip(np.abs(median_curve), 1e-300, None),
                            color=colors[algo_name],
                            linestyle=styles[algo_name],
                            linewidth=2.0,
                            label=f"{algo_name} (median)")
                ax.fill_between(iters,
                                np.clip(np.abs(q25), 1e-300, None),
                                np.clip(np.abs(q75), 1e-300, None),
                                color=colors[algo_name], alpha=0.15)

            ax.set_title(f"Convergence: {fname}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Iteration", fontsize=11)
            ax.set_ylabel("Best Fitness (log scale)", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            plt.tight_layout()

            safe_name = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
            path = os.path.join(subdir, f"{safe_name}_convergence.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"[Plots] {len(results)} individual convergence plots saved to: {subdir}/")

    def export_individual_boxplots(self, results, prefix=""):
        """
        Generate one boxplot per function, saved as individual PNG files.
        Each plot compares SSO vs ISSO fitness distributions across 30 runs.
        """
        subdir = os.path.join(self.output_dir, f"{prefix}boxplots" if prefix else "boxplots")
        os.makedirs(subdir, exist_ok=True)

        for fname, data in results.items():
            fig, ax = plt.subplots(figsize=(6, 5))

            sso_fit  = data["SSO"]["fitness"]
            isso_fit = data["ISSO"]["fitness"]

            bp = ax.boxplot([sso_fit, isso_fit],
                            labels=["SSO", "ISSO"],
                            patch_artist=True,
                            widths=0.5,
                            medianprops=dict(color="black", linewidth=2),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2))

            bp["boxes"][0].set_facecolor("#90CAF9")
            bp["boxes"][1].set_facecolor("#EF9A9A")

            # Add individual data points (strip plot)
            for j, (vals, xpos) in enumerate([(sso_fit, 1), (isso_fit, 2)]):
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
                ax.scatter(np.full(len(vals), xpos) + jitter, vals,
                           alpha=0.4, s=15, color=["#1565C0", "#C62828"][j], zorder=3)

            ax.set_title(f"Fitness Distribution: {fname}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Best Fitness", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()

            safe_name = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
            path = os.path.join(subdir, f"{safe_name}_boxplot.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"[Plots] {len(results)} individual boxplots saved to: {subdir}/")

    def export_all(self, df, results, prefix=""):
        """Export CSV, LaTeX, convergence plots, boxplots, and the 3 algorithm CSVs."""
        p = prefix + "_" if prefix else ""
        self.export_csv(df,             f"{p}comparison_results.csv")
        self.export_latex(df,           f"{p}comparison_table.tex")
        self.export_convergence_plots(results, f"{p}convergence_curves.png")
        self.export_boxplots(results,   f"{p}boxplots.png")
        self.export_individual_convergence(results, prefix=p)
        self.export_individual_boxplots(results, prefix=p)
        self.export_convergence_csv(results,  f"{p}convergence_all_algorithms.csv")
        self.export_raw_runs_csv(results,     f"{p}raw_runs_all_algorithms.csv")
        self.export_results_csv(results,      f"{p}results_all_algorithms.csv")
        print(f"\nAll files saved to: {self.output_dir}/")
