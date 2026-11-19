# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math

import pytest
from alpasim_controller.mpc_controller import ControllerConfig
from benchmark.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    IterationResult,
    SimulationResult,
    _curvature_from_state,
)
from benchmark.trajectories import ReferenceTrajectory, TrajectoryPoint


def test_summary_stats_include_tracking_and_control_aggressiveness() -> None:
    result = BenchmarkResult(
        timestamp="now",
        description="control metrics",
        simulations=[
            SimulationResult(
                trajectory_name="turn",
                trajectory_description="",
                total_time_ms=20.0,
                iterations=[
                    IterationResult(
                        timestamp_us=0,
                        solve_time_ms=1.0,
                        x=0.0,
                        y=0.0,
                        yaw=0.0,
                        vx=5.0,
                        ref_x=0.0,
                        ref_y=0.0,
                        ref_yaw=0.0,
                        steering_cmd=0.1,
                        acceleration_cmd=0.0,
                        jerk_cmd=0.0,
                        curvature=0.01,
                        curvature_rate=0.0,
                    ),
                    IterationResult(
                        timestamp_us=100_000,
                        solve_time_ms=2.0,
                        x=1.0,
                        y=1.0,
                        yaw=0.1,
                        vx=5.0,
                        ref_x=1.0,
                        ref_y=0.0,
                        ref_yaw=0.0,
                        steering_cmd=0.2,
                        acceleration_cmd=3.0,
                        jerk_cmd=30.0,
                        curvature=0.02,
                        curvature_rate=0.1,
                    ),
                ],
            )
        ],
    )

    stats = result.summary_stats()

    assert stats["tracking_rmse_m"] > 0.0
    assert stats["steering_cmd_abs_mean_rad"] == pytest.approx(0.15)
    assert stats["acceleration_cmd_abs_max_mps2"] == pytest.approx(3.0)
    assert stats["acceleration_cmd_abs_mean_mps2"] == pytest.approx(1.5)
    assert stats["jerk_cmd_abs_mean_mps3"] == pytest.approx(15.0)
    assert stats["control_aggressiveness_score"] > 0.0
    assert stats["control_violation_counts"]["jerk_cmd_abs"] == 1
    assert stats["total_control_violation_count"] == 1


def test_benchmark_runner_rejects_empty_trajectories(tmp_path) -> None:
    runner = BenchmarkRunner(ControllerConfig(), log_dir=tmp_path)

    with pytest.raises(ValueError, match="No trajectories to benchmark"):
        runner.run(trajectories=[])


def test_benchmark_runner_rejects_off_grid_trajectories(tmp_path) -> None:
    runner = BenchmarkRunner(ControllerConfig(), log_dir=tmp_path)
    trajectory = ReferenceTrajectory(
        name="off_grid",
        description="",
        points=[TrajectoryPoint(0, 0.0, 0.0, 0.0, 0.0)],
        simulation_duration_us=150_000,
        trajectory_duration_us=150_000,
    )

    with pytest.raises(ValueError, match="multiples of 100000 us"):
        runner.run(trajectories=[trajectory])


def test_low_speed_curvature_clamps_steering_singularity() -> None:
    curvature = _curvature_from_state(
        vx=0.0,
        yaw_rate=0.0,
        steering_cmd=math.pi / 2,
        wheelbase=2.5,
    )

    assert math.isfinite(curvature)
    assert curvature > 100.0
