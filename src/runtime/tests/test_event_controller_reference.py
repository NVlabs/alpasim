# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Held and delayed references use the rig frame at their first timestamp."""

import numpy as np
import pytest
from alpasim_runtime.delay_buffer import DelayBuffer
from alpasim_runtime.events.controller import ControllerEvent
from alpasim_runtime.events.state import StepContext
from alpasim_utils.geometry import Pose, Trajectory


@pytest.mark.asyncio
@pytest.mark.parametrize("delay_us", [0, 100_000])
@pytest.mark.parametrize("first_timestamp_us", [0, 100_000])
async def test_delayed_plan_keeps_first_waypoint_frame(
    rollout_state,
    service_bundle,
    simple_trajectory,
    mock_controller,
    delay_us,
    first_timestamp_us,
):
    """Decode with the historical ego pose, independently of delivery time."""
    service_bundle.planner_delay_buffer = DelayBuffer(delay_us)
    rollout_state.unbound.force_gt_duration_us = 0
    event = ControllerEvent(100_000, 100_000, service_bundle)
    plan = simple_trajectory.clip(first_timestamp_us, 1_000_000)
    plans = [plan, plan.transform(simple_trajectory.last_pose)]
    seeded_ego = Trajectory.from_poses(
        np.array([0, 100_000, 200_000], dtype=np.uint64),
        [
            Pose(
                np.array([3.0 * i, 0.5 * i * i, 0]),
                np.array([0, 0, np.sin(0.1 * i), np.cos(0.1 * i)]),
            )
            for i in range(3)
        ],
    )
    for index, timestamp in enumerate([100_000, 200_000]):
        rollout_state.ego_trajectory = seeded_ego.clip(0, timestamp + 1)
        rollout_state.step_context = StepContext(
            step_start_us=timestamp,
            target_time_us=timestamp + 100_000,
            force_gt=False,
            driver_trajectory=plans[index],
        )
        await event._run_controller(rollout_state)
        request = mock_controller.run_controller_and_vehicle.call_args.kwargs
        selected_plan = plans[0 if delay_us else index]
        anchor = seeded_ego.interpolate_pose(int(selected_plan.timestamps_us[0]))
        observed_in_local = request["rig_reference_trajectory_in_rig"].transform(anchor)
        np.testing.assert_allclose(
            observed_in_local.positions,
            selected_plan.positions,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            observed_in_local.quaternions,
            selected_plan.quaternions,
            atol=1e-6,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_timestamp_us", [0, 200_000])
async def test_reference_anchor_must_be_in_ego_history(
    rollout_state,
    service_bundle,
    simple_trajectory,
    mock_controller,
    first_timestamp_us,
):
    rollout_state.unbound.force_gt_duration_us = 0
    rollout_state.ego_trajectory = simple_trajectory.interpolate(
        np.array([50_000, 100_000], dtype=np.uint64)
    )
    rollout_state.step_context = StepContext(
        step_start_us=100_000,
        target_time_us=200_000,
        force_gt=False,
        driver_trajectory=simple_trajectory.clip(first_timestamp_us, 1_000_000),
    )
    event = ControllerEvent(100_000, 100_000, service_bundle)
    with pytest.raises(ValueError, match="Interpolation timestamp .* outside range"):
        await event._run_controller(rollout_state)
    mock_controller.run_controller_and_vehicle.assert_not_called()
