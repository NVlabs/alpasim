# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from alpasim_runtime.events.state import RolloutState
from alpasim_utils.geometry import Trajectory


def test_force_gt_trajectory_defaults_to_recorded_gt(
    rollout_state: RolloutState,
) -> None:
    assert rollout_state.force_gt_trajectory is rollout_state.unbound.gt_ego_trajectory


def test_force_gt_trajectory_prefers_blended_override(
    rollout_state: RolloutState,
    simple_trajectory: Trajectory,
) -> None:
    blended_trajectory = simple_trajectory.clip(0, 300_001)

    rollout_state.force_gt_ego_trajectory = blended_trajectory

    assert rollout_state.force_gt_trajectory is blended_trajectory
