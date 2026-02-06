# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Comparison tools for benchmark results.

Generates plots and reports comparing two benchmark runs.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .runner import BenchmarkResult


class BenchmarkComparator:
    """Compare two benchmark runs and generate visualizations."""

    def __init__(
        self,
        baseline: BenchmarkResult,
        comparison: BenchmarkResult,
        baseline_name: str = "baseline",
        comparison_name: str = "comparison",
    ):
        self.baseline = baseline
        self.comparison = comparison
        self.baseline_name = baseline_name
        self.comparison_name = comparison_name

        # Build lookup by trajectory name
        self.baseline_by_name = {s.trajectory_name: s for s in baseline.simulations}
        self.comparison_by_name = {s.trajectory_name: s for s in comparison.simulations}

        # Find common trajectories
        self.common_names = set(self.baseline_by_name.keys()) & set(
            self.comparison_by_name.keys()
        )

    def print_summary(self) -> None:
        """Print a text summary of the comparison."""
        print("=" * 70)
        print("BENCHMARK COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\nBaseline:   {self.baseline_name}")
        print(f"  Timestamp: {self.baseline.timestamp}")
        print(f"  Description: {self.baseline.description or 'N/A'}")

        print(f"\nComparison: {self.comparison_name}")
        print(f"  Timestamp: {self.comparison.timestamp}")
        print(f"  Description: {self.comparison.description or 'N/A'}")

        print(f"\nCommon trajectories: {len(self.common_names)}")

        # Aggregate timing stats
        base_stats = self.baseline.summary_stats()
        comp_stats = self.comparison.summary_stats()

        print("\n" + "-" * 70)
        print("TIMING COMPARISON (all iterations)")
        print("-" * 70)
        print(f"{'Metric':<25} {'Baseline':>15} {'Comparison':>15} {'Change':>12}")
        print("-" * 70)

        metrics = [
            ("Total time (s)", "total_time_s", "{:.2f}"),
            ("Mean solve (ms)", "solve_time_mean_ms", "{:.2f}"),
            ("Median solve (ms)", "solve_time_median_ms", "{:.2f}"),
            ("Min solve (ms)", "solve_time_min_ms", "{:.2f}"),
            ("Max solve (ms)", "solve_time_max_ms", "{:.2f}"),
        ]

        for label, key, fmt in metrics:
            base_val = base_stats[key]
            comp_val = comp_stats[key]
            if base_val > 0:
                change_pct = ((comp_val - base_val) / base_val) * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"

            print(
                f"{label:<25} {fmt.format(base_val):>15} {fmt.format(comp_val):>15} {change_str:>12}"
            )

        print("-" * 70)

        # Per-trajectory comparison
        print("\n" + "-" * 70)
        print("PER-TRAJECTORY TIMING (mean solve time)")
        print("-" * 70)
        print(f"{'Trajectory':<35} {'Baseline':>10} {'Comparison':>10} {'Change':>10}")
        print("-" * 70)

        for name in sorted(self.common_names):
            base_sim = self.baseline_by_name[name]
            comp_sim = self.comparison_by_name[name]

            base_mean = base_sim.mean_solve_time_ms
            comp_mean = comp_sim.mean_solve_time_ms

            if base_mean > 0:
                change_pct = ((comp_mean - base_mean) / base_mean) * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"

            # Truncate long names
            display_name = name[:33] + ".." if len(name) > 35 else name
            print(
                f"{display_name:<35} {base_mean:>10.2f} {comp_mean:>10.2f} {change_str:>10}"
            )

        print("=" * 70)

    def plot_timing_comparison(self, output_path: Optional[Path] = None) -> None:
        """Generate timing comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Timing Comparison: {self.baseline_name} vs {self.comparison_name}",
            fontsize=14,
        )

        # 1. Histogram of solve times
        ax = axes[0, 0]
        base_times = [
            it.solve_time_ms
            for sim in self.baseline.simulations
            for it in sim.iterations
        ]
        comp_times = [
            it.solve_time_ms
            for sim in self.comparison.simulations
            for it in sim.iterations
        ]

        bins = np.linspace(0, max(max(base_times), max(comp_times)) * 1.1, 50)
        ax.hist(base_times, bins=bins, alpha=0.5, label=self.baseline_name)
        ax.hist(comp_times, bins=bins, alpha=0.5, label=self.comparison_name)
        ax.set_xlabel("Solve Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Solve Time Distribution")
        ax.legend()

        # 2. Mean solve time per trajectory (bar chart)
        ax = axes[0, 1]
        common_sorted = sorted(self.common_names)
        x = np.arange(len(common_sorted))
        width = 0.35

        base_means = [
            self.baseline_by_name[n].mean_solve_time_ms for n in common_sorted
        ]
        comp_means = [
            self.comparison_by_name[n].mean_solve_time_ms for n in common_sorted
        ]

        ax.bar(x - width / 2, base_means, width, label=self.baseline_name, alpha=0.7)
        ax.bar(x + width / 2, comp_means, width, label=self.comparison_name, alpha=0.7)
        ax.set_xlabel("Trajectory")
        ax.set_ylabel("Mean Solve Time (ms)")
        ax.set_title("Mean Solve Time by Trajectory")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n[:10] for n in common_sorted], rotation=45, ha="right", fontsize=6
        )
        ax.legend()

        # 3. Speedup/slowdown scatter
        ax = axes[1, 0]
        speedups = []
        for name in common_sorted:
            base_mean = self.baseline_by_name[name].mean_solve_time_ms
            comp_mean = self.comparison_by_name[name].mean_solve_time_ms
            if base_mean > 0:
                speedups.append(base_mean / comp_mean)
            else:
                speedups.append(1.0)

        colors = ["green" if s > 1 else "red" for s in speedups]
        ax.bar(x, speedups, color=colors, alpha=0.7)
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Trajectory")
        ax.set_ylabel("Speedup (>1 = faster)")
        ax.set_title(f"Speedup: {self.baseline_name} / {self.comparison_name}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [n[:10] for n in common_sorted], rotation=45, ha="right", fontsize=6
        )

        # 4. Cumulative time comparison
        ax = axes[1, 1]
        base_cumulative = np.cumsum(base_times)
        comp_cumulative = np.cumsum(comp_times)

        ax.plot(base_cumulative / 1000, label=self.baseline_name)
        ax.plot(comp_cumulative / 1000, label=self.comparison_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative Time (s)")
        ax.set_title("Cumulative Solve Time")
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved timing comparison to {output_path}")
        else:
            plt.show()

    def plot_trajectory_comparison(
        self,
        trajectory_name: str,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot position traces for a single trajectory comparison."""
        if trajectory_name not in self.common_names:
            available = ", ".join(sorted(self.common_names)[:5]) + "..."
            raise ValueError(
                f"Trajectory '{trajectory_name}' not found in both runs. "
                f"Available: {available}"
            )

        base_sim = self.baseline_by_name[trajectory_name]
        comp_sim = self.comparison_by_name[trajectory_name]

        # Create figure with 2x2 grid on top and a half-height row at bottom
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.25)
        fig.suptitle(f"Trajectory Comparison: {trajectory_name}", fontsize=14)

        # 1. X-Y plot (bird's eye view)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(
            base_sim.ref_x_positions,
            base_sim.ref_y_positions,
            "k--",
            label="Reference",
            linewidth=2,
        )
        ax.plot(
            base_sim.x_positions,
            base_sim.y_positions,
            "b-",
            label=self.baseline_name,
            alpha=0.7,
        )
        ax.plot(
            comp_sim.x_positions,
            comp_sim.y_positions,
            "r-",
            label=self.comparison_name,
            alpha=0.7,
        )
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("X-Y Trajectory")
        ax.legend()
        ax.axis("equal")
        ax.grid(True, alpha=0.3)

        # 2. X position vs time
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(
            base_sim.timestamps_s,
            base_sim.ref_x_positions,
            "k--",
            label="Reference",
            linewidth=2,
        )
        ax.plot(
            base_sim.timestamps_s,
            base_sim.x_positions,
            "b-",
            label=self.baseline_name,
            alpha=0.7,
        )
        ax.plot(
            comp_sim.timestamps_s,
            comp_sim.x_positions,
            "r-",
            label=self.comparison_name,
            alpha=0.7,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("X Position (m)")
        ax.set_title("X Position vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Y position vs time
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(
            base_sim.timestamps_s,
            base_sim.ref_y_positions,
            "k--",
            label="Reference",
            linewidth=2,
        )
        ax.plot(
            base_sim.timestamps_s,
            base_sim.y_positions,
            "b-",
            label=self.baseline_name,
            alpha=0.7,
        )
        ax.plot(
            comp_sim.timestamps_s,
            comp_sim.y_positions,
            "r-",
            label=self.comparison_name,
            alpha=0.7,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title("Y Position vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Vehicle speed vs time
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(
            base_sim.timestamps_s,
            base_sim.velocities,
            "b-",
            label=self.baseline_name,
            alpha=0.7,
        )
        ax.plot(
            comp_sim.timestamps_s,
            comp_sim.velocities,
            "r-",
            label=self.comparison_name,
            alpha=0.7,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vehicle Speed (m/s)")
        ax.set_title("Vehicle Speed vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Solve time comparison (bottom row, spans both columns)
        ax = fig.add_subplot(gs[2, :])
        base_solve_times = [it.solve_time_ms for it in base_sim.iterations]
        comp_solve_times = [it.solve_time_ms for it in comp_sim.iterations]
        ax.plot(
            base_sim.timestamps_s,
            base_solve_times,
            "b-",
            label=f"{self.baseline_name} (mean={np.mean(base_solve_times):.1f}ms)",
            alpha=0.7,
        )
        ax.plot(
            comp_sim.timestamps_s,
            comp_solve_times,
            "r-",
            label=f"{self.comparison_name} (mean={np.mean(comp_solve_times):.1f}ms)",
            alpha=0.7,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Solve Time (ms)")
        ax.set_title("MPC Solve Time vs Simulation Time")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved trajectory comparison to {output_path}")
        else:
            plt.show()

    def plot_all_trajectories(self, output_dir: Path) -> None:
        """Generate comparison plots for all common trajectories."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for name in sorted(self.common_names):
            safe_name = name.replace("/", "_").replace(" ", "_")
            output_path = output_dir / f"trajectory_{safe_name}.png"
            self.plot_trajectory_comparison(name, output_path)

    def generate_report(self, output_dir: Path) -> None:
        """Generate a complete comparison report with all plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save timing comparison plot
        self.plot_timing_comparison(output_dir / "timing_comparison.png")

        # Save trajectory plots
        traj_dir = output_dir / "trajectories"
        self.plot_all_trajectories(traj_dir)

        # Save text summary to file
        summary_path = output_dir / "summary.txt"
        import io
        import sys

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.print_summary()
        sys.stdout = old_stdout

        with open(summary_path, "w") as f:
            f.write(buffer.getvalue())

        print(f"\nReport generated in {output_dir}")
        print("  - timing_comparison.png")
        print("  - summary.txt")
        print(f"  - trajectories/ ({len(self.common_names)} plots)")
