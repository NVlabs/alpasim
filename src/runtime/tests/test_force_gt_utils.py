# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for force-GT trajectory helpers."""

import numpy as np
import pytest
from alpasim_runtime.events.force_gt_utils import (
    FORCE_GT_REFERENCE_HORIZON_US,
    controller_reference_trajectory,
    force_gt_dynamic_trajectory,
)
from alpasim_utils import geometry


def test_force_gt_dynamic_trajectory_filters_gt_derivatives() -> None:
    """Filtered dynamics recover acceleration and curvature from GT poses."""
    timestamps_us = np.arange(0, 1_100_000, 100_000, dtype=np.uint64)
    timestamps_s = timestamps_us / 1e6
    acceleration = 2.0
    yaw_rate = 0.2
    speed = 10.0 + acceleration * timestamps_s
    yaw = yaw_rate * timestamps_s
    positions = np.stack(
        [
            10.0 * timestamps_s + 0.5 * acceleration * timestamps_s**2,
            np.zeros_like(timestamps_s),
            np.zeros_like(timestamps_s),
        ],
        axis=-1,
    )
    poses = [
        geometry.Pose(
            position.astype(np.float32),
            np.array(
                [0.0, 0.0, np.sin(heading / 2.0), np.cos(heading / 2.0)],
                dtype=np.float32,
            ),
        )
        for position, heading in zip(positions, yaw, strict=True)
    ]
    trajectory = geometry.Trajectory.from_poses(timestamps_us, poses)

    dynamic_trajectory = force_gt_dynamic_trajectory(trajectory)

    middle = len(timestamps_us) // 2
    assert dynamic_trajectory.dynamics[middle, 0] == pytest.approx(
        speed[middle] * np.cos(yaw[middle]), rel=0.02
    )
    assert dynamic_trajectory.dynamics[middle, 5] == pytest.approx(yaw_rate, rel=0.02)
    assert dynamic_trajectory.dynamics[middle, 6] == pytest.approx(
        acceleration * np.cos(yaw[middle]), rel=0.05
    )


def test_controller_reference_continues_past_the_end_of_a_short_history() -> None:
    speed_mps = 10.0
    yaw_rate = 0.2
    radius_m = speed_mps / yaw_rate

    def arc_pose(timestamp_us: int) -> geometry.Pose:
        yaw = yaw_rate * timestamp_us / 1e6
        return geometry.Pose(
            np.array(
                [radius_m * np.sin(yaw), radius_m * (1.0 - np.cos(yaw)), 0.0],
                dtype=np.float32,
            ),
            np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32),
        )

    history_timestamps_us = np.arange(0, 2_100_000, 100_000, dtype=np.uint64)
    history = geometry.Trajectory.from_poses(
        history_timestamps_us, [arc_pose(int(t)) for t in history_timestamps_us]
    )
    step_start_us = 1_500_000

    reference = controller_reference_trajectory(history, step_start_us, 100_000)

    assert reference.time_range_us.start == step_start_us
    assert int(reference.timestamps_us[-1]) >= (
        step_start_us + FORCE_GT_REFERENCE_HORIZON_US
    )
    expected = [arc_pose(int(t)) for t in reference.timestamps_us]
    np.testing.assert_allclose(
        reference.positions, [pose.vec3 for pose in expected], atol=1e-3
    )
    np.testing.assert_allclose(
        reference.yaws, [pose.yaw() for pose in expected], atol=1e-4
    )
