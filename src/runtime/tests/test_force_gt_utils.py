# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for force-GT trajectory helpers."""

import numpy as np
import pytest
from alpasim_runtime.events.force_gt_utils import force_gt_dynamic_trajectory
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
