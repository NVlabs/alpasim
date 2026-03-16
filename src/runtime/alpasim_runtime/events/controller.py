# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Controller event for the split policy pipeline.

Runs the controller and vehicle model, writing the unconstrained pose and
estimated state to StepContext.
"""

from __future__ import annotations

import numpy as np
from alpasim_runtime.events.base import EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_utils import geometry


class ControllerEvent(RecurringEvent):
    """Run controller + vehicle model, writing results to StepContext."""

    priority: int = EventPriority.CONTROLLER

    def __init__(
        self,
        timestamp_us: int,
        control_timestep_us: int,
        services: ServiceBundle,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = control_timestep_us
        self.services = services

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        ctx = state.step_context

        # --- Sanity checks ---
        assert ctx is not None, "StepContext missing — PolicyEvent did not run"
        assert (
            ctx.step_start_us == self.timestamp_us
        ), f"StepContext timestamp mismatch: {ctx.step_start_us} != {self.timestamp_us}"
        assert (
            ctx.driver_trajectory is not None
        ), "driver_trajectory not set by PolicyEvent"

        # --- Controller + vehicle model ---
        ctx.ego_true, ctx.ego_estimated = await self._run_controller(state)

    async def _run_controller(
        self, state: RolloutState
    ) -> tuple[geometry.DynamicTrajectory, geometry.DynamicTrajectory]:
        """Run the controller and vehicle model.

        Returns (ego_true, ego_estimated) as DynamicTrajectory pairs.
        """
        ctx = state.step_context
        assert ctx is not None
        planner_delay_buffer = self.services.planner_delay_buffer
        controller = self.services.controller

        step_start_us = ctx.step_start_us
        target_time_us = ctx.target_time_us
        force_gt = ctx.force_gt
        reference_trajectory_of_rig_in_local = ctx.driver_trajectory
        assert reference_trajectory_of_rig_in_local is not None

        pose_local_to_rig = state.ego_trajectory.last_pose

        if force_gt and (len(reference_trajectory_of_rig_in_local.timestamps_us) == 0):
            max_interp_time_us = min(
                step_start_us + int(5e6),
                state.unbound.gt_ego_trajectory.time_range_us.stop - 1,
            )
            reference_trajectory_of_rig_in_local = (
                state.unbound.gt_ego_trajectory.interpolate(
                    np.linspace(
                        step_start_us, max_interp_time_us, num=51, dtype=np.uint64
                    )
                )
            )

        if force_gt:
            dt = (target_time_us - step_start_us) / 1e6
            pose_local_to_rig_t0 = state.unbound.gt_ego_trajectory.interpolate_pose(
                step_start_us
            )
            pose_local_to_rig_t1 = state.unbound.gt_ego_trajectory.interpolate_pose(
                target_time_us
            )
            fallback_trajectory_local_to_rig = state.unbound.gt_ego_trajectory
        else:
            assert len(state.ego_trajectory.timestamps_us) >= 2, (
                "ego_trajectory must have at least 2 entries by the time the "
                "controller runs (seeded with [t0, t1] during initialization)"
            )
            dt = (
                state.ego_trajectory.timestamps_us[-1]
                - state.ego_trajectory.timestamps_us[-2]
            ) / 1e6
            pose_local_to_rig_t0 = state.ego_trajectory.get_pose(-2)
            pose_local_to_rig_t1 = state.ego_trajectory.last_pose

            fallback_trajectory_local_to_rig = reference_trajectory_of_rig_in_local

        planner_delay_buffer.add(
            reference_trajectory_of_rig_in_local.transform(pose_local_to_rig.inverse()),
            step_start_us,
        )
        rig_reference_trajectory = planner_delay_buffer.at(step_start_us)

        pose_rig0_to_rig1 = pose_local_to_rig_t0.inverse() @ pose_local_to_rig_t1
        rig_linear_velocity_in_rig = pose_rig0_to_rig1.vec3 / dt
        rig_angular_velocity_in_rig = 2.0 * pose_rig0_to_rig1.quat[0:3] / dt

        propagated_states = await controller.run_controller_and_vehicle(
            now_us=step_start_us,
            pose_local_to_rig=pose_local_to_rig,
            rig_linear_velocity_in_rig=rig_linear_velocity_in_rig,
            rig_angular_velocity_in_rig=rig_angular_velocity_in_rig,
            rig_reference_trajectory_in_rig=rig_reference_trajectory,
            future_us=target_time_us,
            force_gt=force_gt,
            fallback_trajectory_local_to_rig=fallback_trajectory_local_to_rig,
            pose_reporting_interval_us=state.unbound.pose_reporting_interval_us
            or state.unbound.control_timestep_us,
        )

        # Convert list[PropagatedPosesAtTime] → two DynamicTrajectory instances.
        timestamps = np.array(
            [s.timestamp_us for s in propagated_states], dtype=np.uint64
        )
        true_poses = [s.pose_local_to_rig for s in propagated_states]
        est_poses = [s.pose_local_to_rig_estimate for s in propagated_states]
        true_dynamics = geometry.dynamic_states_to_array(
            [s.dynamic_state for s in propagated_states]
        )
        est_dynamics = geometry.dynamic_states_to_array(
            [s.dynamic_state_estimated for s in propagated_states]
        )

        ego_true = geometry.DynamicTrajectory.from_trajectory_and_dynamics(
            geometry.Trajectory.from_poses(timestamps, true_poses),
            true_dynamics,
        )
        ego_estimated = geometry.DynamicTrajectory.from_trajectory_and_dynamics(
            geometry.Trajectory.from_poses(timestamps, est_poses),
            est_dynamics,
        )
        return ego_true, ego_estimated
