# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Helpers for force-GT event behavior."""

from __future__ import annotations

import numpy as np
from alpasim_utils import geometry

FORCE_GT_REFERENCE_HORIZON_US = 5_000_000
_FORCE_GT_DYNAMICS_SMOOTHING_FACTOR = 0.9


def force_gt_dynamic_trajectory(
    trajectory: geometry.Trajectory,
) -> geometry.DynamicTrajectory:
    """Estimate smooth rig-frame dynamics from a force-GT pose trajectory."""
    velocities_local = geometry.trajectory_velocities_cubic(
        trajectory, _FORCE_GT_DYNAMICS_SMOOTHING_FACTOR
    )
    accelerations_local = geometry.trajectory_accelerations_cubic(
        trajectory, _FORCE_GT_DYNAMICS_SMOOTHING_FACTOR
    )
    yaw_rates = geometry.trajectory_yaw_rates_cubic(
        trajectory, _FORCE_GT_DYNAMICS_SMOOTHING_FACTOR
    )
    yaws = trajectory.yaws

    cos_yaw = np.cos(yaws)
    sin_yaw = np.sin(yaws)
    dynamics = np.zeros((len(trajectory), 12), dtype=np.float64)
    dynamics[:, 0] = cos_yaw * velocities_local[:, 0] + sin_yaw * velocities_local[:, 1]
    dynamics[:, 1] = (
        -sin_yaw * velocities_local[:, 0] + cos_yaw * velocities_local[:, 1]
    )
    dynamics[:, 2] = velocities_local[:, 2]
    dynamics[:, 5] = yaw_rates
    dynamics[:, 6] = (
        cos_yaw * accelerations_local[:, 0] + sin_yaw * accelerations_local[:, 1]
    )
    dynamics[:, 7] = (
        -sin_yaw * accelerations_local[:, 0] + cos_yaw * accelerations_local[:, 1]
    )
    dynamics[:, 8] = accelerations_local[:, 2]
    return geometry.DynamicTrajectory.from_trajectory_and_dynamics(trajectory, dynamics)


def controller_reference_trajectory(
    force_gt_trajectory: geometry.Trajectory, step_start_us: int
) -> geometry.Trajectory:
    """Build the controller reference used while the rollout is force-GT driven."""
    clip_end_us = min(
        step_start_us + FORCE_GT_REFERENCE_HORIZON_US + 1,
        force_gt_trajectory.time_range_us.stop,
    )
    return force_gt_trajectory.clip(step_start_us, clip_end_us)
