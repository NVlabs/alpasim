# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
CLI entry point for controller benchmarks.

Usage (from src/controller directory):
    # Run full benchmark and save results
    uv run python -m benchmark run --output results/baseline.json --description "Baseline MPC"

    # Run quick benchmark (10 trajectories)
    uv run python -m benchmark run --quick --output results/quick_test.json

    # Compare two benchmark runs
    uv run python -m benchmark compare results/baseline.json results/optimized.json --output-dir comparison_report/

    # Compare and show plots interactively
    uv run python -m benchmark compare results/baseline.json results/optimized.json --interactive
"""

import argparse
import sys
from pathlib import Path

from alpasim_controller.mpc_controller import MPCImplementation


def cmd_run(args: argparse.Namespace) -> int:
    """Run the benchmark suite."""
    from .runner import BenchmarkRunner

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"benchmark_results_{timestamp}.json")

    # Run benchmark
    runner = BenchmarkRunner(mpc_implementation=args.mpc_implementation)
    result = runner.run(
        quick=args.quick,
        description=args.description or "",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    stats = result.summary_stats()
    print(f"Simulations:        {stats['n_simulations']}")
    print(f"Total iterations:   {stats['total_iterations']}")
    print(f"Total time:         {stats['total_time_s']:.2f} s")
    print(f"Mean solve time:    {stats['solve_time_mean_ms']:.2f} ms")
    print(f"Median solve time:  {stats['solve_time_median_ms']:.2f} ms")
    print(f"Min solve time:     {stats['solve_time_min_ms']:.2f} ms")
    print(f"Max solve time:     {stats['solve_time_max_ms']:.2f} ms")
    print("=" * 60)

    # Save results
    result.save(output_path)
    print(f"\nResults saved to: {output_path}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two benchmark runs."""
    from .compare import BenchmarkComparator
    from .runner import BenchmarkResult

    # Load results
    baseline_path = Path(args.baseline)
    comparison_path = Path(args.comparison)

    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        return 1
    if not comparison_path.exists():
        print(f"Error: Comparison file not found: {comparison_path}")
        return 1

    print(f"Loading baseline: {baseline_path}")
    baseline = BenchmarkResult.load(baseline_path)

    print(f"Loading comparison: {comparison_path}")
    comparison = BenchmarkResult.load(comparison_path)

    # Create comparator
    baseline_name = args.baseline_name or baseline_path.stem
    comparison_name = args.comparison_name or comparison_path.stem

    comparator = BenchmarkComparator(
        baseline=baseline,
        comparison=comparison,
        baseline_name=baseline_name,
        comparison_name=comparison_name,
    )

    # Print summary
    comparator.print_summary()

    # Generate plots
    if args.output_dir:
        output_dir = Path(args.output_dir)
        comparator.generate_report(output_dir)
    elif args.interactive:
        comparator.plot_timing_comparison()

        # Also plot a few example trajectories
        common = sorted(comparator.common_names)
        if common:
            print("\nShowing first trajectory comparison (close to see next)...")
            for name in common[:3]:
                comparator.plot_trajectory_comparison(name)

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available trajectories."""
    from .trajectories import TrajectoryGenerator

    generator = TrajectoryGenerator()

    if args.quick:
        trajectories = generator.generate_quick_set()
        print(f"Quick benchmark trajectories ({len(trajectories)}):")
    else:
        trajectories = generator.generate_all()
        print(f"Full benchmark trajectories ({len(trajectories)}):")

    print("-" * 60)
    for traj in trajectories:
        print(f"  {traj.name:<35} {traj.description}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Controller MPC Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the benchmark suite")
    run_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with 14 trajectories instead of 120",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path (default: benchmark_results_<timestamp>.json)",
    )
    run_parser.add_argument(
        "--description",
        "-d",
        type=str,
        help="Description of this benchmark run",
    )
    run_parser.add_argument(
        "--mpc-implementation",
        "-m",
        type=MPCImplementation,
        choices=list(MPCImplementation),
        default=MPCImplementation.LINEAR,
        help="MPC implementation: linear (default, faster) or nonlinear (CasADi)",
    )
    run_parser.set_defaults(func=cmd_run)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark runs")
    compare_parser.add_argument(
        "baseline",
        type=str,
        help="Path to baseline benchmark results JSON",
    )
    compare_parser.add_argument(
        "comparison",
        type=str,
        help="Path to comparison benchmark results JSON",
    )
    compare_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help="Output directory for comparison report",
    )
    compare_parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Show plots interactively instead of saving",
    )
    compare_parser.add_argument(
        "--baseline-name",
        type=str,
        help="Display name for baseline (default: filename)",
    )
    compare_parser.add_argument(
        "--comparison-name",
        type=str,
        help="Display name for comparison (default: filename)",
    )
    compare_parser.set_defaults(func=cmd_compare)

    # List command
    list_parser = subparsers.add_parser("list", help="List available trajectories")
    list_parser.add_argument(
        "--quick",
        action="store_true",
        help="Show quick benchmark trajectories only",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
