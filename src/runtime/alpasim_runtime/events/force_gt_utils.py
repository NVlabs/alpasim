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
    force_gt_trajectory: geometry.Trajectory,
    step_start_us: int,
    control_timestep_us: int,
) -> geometry.Trajectory:
    """Build the controller reference used while the rollout is force-GT driven.

    The reference covers ``FORCE_GT_REFERENCE_HORIZON_US`` from the step start.
    Past the end of the force-GT trajectory it repeats the trajectory's last
    control-step motion. A fork's force-GT trajectory is the parent history,
    which ends at the policy start; a reference cut off there would have the
    controller brake towards that pose before the policy takes over.
    """
    reference_end_us = step_start_us + FORCE_GT_REFERENCE_HORIZON_US
    trajectory_end_us = force_gt_trajectory.time_range_us.stop - 1
    reference = force_gt_trajectory.clip(
        step_start_us, min(reference_end_us, trajectory_end_us) + 1
    )
    if trajectory_end_us >= reference_end_us:
        return reference

    pose_step = force_gt_trajectory.interpolate_delta(
        trajectory_end_us - control_timestep_us, trajectory_end_us
    )
    extension_timestamps_us = np.arange(
        trajectory_end_us + control_timestep_us,
        reference_end_us + control_timestep_us,
        control_timestep_us,
        dtype=np.uint64,
    )
    extension_poses = []
    pose = force_gt_trajectory.last_pose
    for _ in extension_timestamps_us:
        pose = pose @ pose_step
        extension_poses.append(pose)
    return reference.append(
        geometry.Trajectory.from_poses(extension_timestamps_us, extension_poses)
    )
