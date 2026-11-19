# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""CLI entry point for controller benchmarks.

Usage (from src/controller directory):
    # Run full benchmark and save results
    uv run python -m benchmark run --output results/baseline.json --description "Baseline MPC"

    # Run quick benchmark (10 trajectories)
    uv run python -m benchmark run --quick --output results/quick_test.json

    # Compare two benchmark runs
    uv run python -m benchmark compare results/baseline.json results/optimized.json \
        --output-dir comparison_report/

    # Compare and show plots interactively
    uv run python -m benchmark compare results/baseline.json results/optimized.json --interactive
"""

import argparse
import sys
from pathlib import Path

from alpasim_controller.mpc_controller import ControllerConfig
from alpasim_utils.yaml_utils import typed_parse_config


def cmd_run(args: argparse.Namespace) -> int:
    """Run the benchmark suite."""
    from .ego_motion_catalog import EgoMotionCatalog
    from .runner import BenchmarkRunner

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"benchmark_results_{timestamp}.json")

    if args.config is None:
        controller_config = ControllerConfig()
    else:
        controller_config = typed_parse_config(Path(args.config), ControllerConfig)

    trajectories = None
    if args.trajectory_catalog:
        catalog = EgoMotionCatalog.load(Path(args.trajectory_catalog))
        trajectories = catalog.to_reference_trajectories()
        print(
            f"Loaded {len(trajectories)} ego-motion trajectories "
            f"from {args.trajectory_catalog}"
        )

    # Run benchmark
    runner = BenchmarkRunner(controller_config=controller_config)
    result = runner.run(
        quick=args.quick,
        description=args.description or "",
        trajectories=trajectories,
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
    print(f"Tracking RMSE:      {stats['tracking_rmse_m']:.3f} m")
    print(f"Tracking max error: {stats['tracking_max_error_m']:.3f} m")
    print(f"Control aggression: {stats['control_aggressiveness_score']:.3f}")
    print(f"Control violations: {stats['total_control_violation_count']}")
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
    from .ego_motion_catalog import EgoMotionCatalog
    from .trajectories import TrajectoryGenerator

    if args.trajectory_catalog:
        catalog = EgoMotionCatalog.load(Path(args.trajectory_catalog))
        trajectories = catalog.to_reference_trajectories()
        print(f"Ego-motion catalog trajectories ({len(trajectories)}):")
        if catalog.selection_metadata:
            radius = catalog.selection_metadata.get("covering_radius")
            if radius is not None:
                print(f"K-disk covering radius: {radius:.3f}")
    elif args.quick:
        generator = TrajectoryGenerator()
        trajectories = generator.generate_quick_set()
        print(f"Quick benchmark trajectories ({len(trajectories)}):")
    else:
        generator = TrajectoryGenerator()
        trajectories = generator.generate_all()
        print(f"Full benchmark trajectories ({len(trajectories)}):")

    print("-" * 60)
    for traj in trajectories:
        print(f"  {traj.name:<35} {traj.description}")

    return 0


def cmd_select_egomotion(args: argparse.Namespace) -> int:
    """Select a k-disk ego-motion benchmark subset."""
    from .ego_motion_catalog import EgoMotionCatalog

    catalog = EgoMotionCatalog.load(Path(args.catalog))
    selected_catalog = catalog.with_selected_cover(args.k)
    selected_catalog.save(Path(args.output))

    metadata = selected_catalog.selection_metadata
    print(
        f"Selected {len(selected_catalog.selected_names)} / {len(selected_catalog.records)} "
        f"ego-motion records."
    )
    print(f"Covering radius:       {metadata['covering_radius']:.3f}")
    print(f"Mean cover distance:   {metadata['mean_covering_distance']:.3f}")
    print(f"Max-distance record:   {metadata['max_distance_record']}")
    print(f"Selected catalog:      {args.output}")
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
        "--config",
        type=str,
        default=None,
        help="Path to controller YAML config file. If omitted, uses ControllerConfig defaults.",
    )
    run_parser.add_argument(
        "--trajectory-catalog",
        type=str,
        default=None,
        help=(
            "Path to an ego-motion catalog JSON. "
            "Selected records override synthetic trajectories."
        ),
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
    list_parser.add_argument(
        "--trajectory-catalog",
        type=str,
        default=None,
        help="Path to an ego-motion catalog JSON to list instead of synthetic trajectories.",
    )
    list_parser.set_defaults(func=cmd_list)

    # Ego-motion selection command
    select_parser = subparsers.add_parser(
        "select-egomotion",
        help="Select a k-disk covering subset from an ego-motion catalog",
    )
    select_parser.add_argument(
        "catalog",
        type=str,
        help="Input ego-motion catalog JSON",
    )
    select_parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of ego-motion records to select",
    )
    select_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output selected ego-motion catalog JSON",
    )
    select_parser.set_defaults(func=cmd_select_egomotion)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
