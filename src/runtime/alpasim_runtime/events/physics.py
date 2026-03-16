# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Physics event for the split policy pipeline.

A single class instantiated twice per pipeline — once for ego, once for
traffic. The PhysicsTarget enum selects which physics step to run and
determines the event's priority.
"""

import enum

import numpy as np
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.events.base import EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_utils import geometry
from alpasim_utils.geometry import Trajectory


class PhysicsTarget(enum.Enum):
    EGO = "ego"
    TRAFFIC = "traffic"


_PHYSICS_PRIORITY: dict[PhysicsTarget, int] = {
    PhysicsTarget.EGO: EventPriority.PHYSICS_EGO,
    PhysicsTarget.TRAFFIC: EventPriority.PHYSICS_TRAFFIC,
}


class PhysicsEvent(RecurringEvent):
    """Apply physics constraints to ego or traffic poses."""

    def __init__(
        self,
        timestamp_us: int,
        control_timestep_us: int,
        services: ServiceBundle,
        target: PhysicsTarget,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = control_timestep_us
        self.services = services
        self.target = target
        self.priority = _PHYSICS_PRIORITY[target]

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        ctx = state.step_context
        assert ctx is not None

        if self.target == PhysicsTarget.EGO:
            assert (
                ctx.step_start_us == self.timestamp_us
            ), f"EGO physics timestamp mismatch: {ctx.step_start_us} != {self.timestamp_us}"
            await self._run_ego(state, ctx)
        else:
            await self._run_traffic(state, ctx)

    async def _run_ego(self, state: RolloutState, ctx: StepContext) -> None:
        """Apply physics constraints to the ego poses.

        Writes the physics-corrected trajectory (poses only — dynamics
        unchanged) into ``ctx.corrected_ego_trajectory``.
        """
        assert ctx.ego_true is not None, "ego_true not set by ControllerEvent"

        traj = ctx.ego_true.trajectory()

        if ctx.force_gt or state.unbound.physics_update_mode == PhysicsUpdateMode.NONE:
            ctx.corrected_ego_trajectory = traj
            return

        physics = self.services.physics
        ds_to_aabb = state.unbound.transform_ego_coords_ds_to_aabb
        aabb_to_ds = ds_to_aabb.inverse()

        traj_aabb = traj.transform(ds_to_aabb, is_relative=True)

        corrected_aabb, _ = await physics.ground_intersection(
            scene_id=state.unbound.scene_id,
            delta_start_us=ctx.step_start_us,
            delta_end_us=ctx.target_time_us,
            ego_trajectory_aabb=traj_aabb,
            traffic_poses={},
            ego_aabb=state.unbound.ego_aabb,
        )

        ctx.corrected_ego_trajectory = corrected_aabb.transform(
            aabb_to_ds, is_relative=True
        )

    async def _run_traffic(self, state: RolloutState, ctx: StepContext) -> None:
        """Apply physics constraints to traffic object poses."""
        assert ctx.corrected_ego_trajectory is not None, "ego physics did not run"
        assert ctx.traffic_response is not None, "traffic simulation did not run"

        physics = self.services.physics
        trafficsim = self.services.trafficsim

        target_time_us = self.timestamp_us + self.interval_us
        force_gt = target_time_us in state.unbound.force_gt_period

        ego_ds_pose = ctx.corrected_ego_trajectory.interpolate_pose(target_time_us)

        traffic_poses_future: dict[str, geometry.Pose] = {}

        if force_gt:
            for key, traffic_obj in state.unbound.traffic_objs.items():
                if target_time_us not in traffic_obj.trajectory.time_range_us:
                    continue
                traffic_poses_future[key] = traffic_obj.trajectory.interpolate_pose(
                    target_time_us
                )
        else:
            for obj_update in ctx.traffic_response.object_trajectory_updates:
                object_trajectory = geometry.trajectory_from_grpc(obj_update.trajectory)

                if target_time_us not in object_trajectory.time_range_us:
                    continue

                traffic_poses_future[obj_update.object_id] = (
                    object_trajectory.interpolate_pose(target_time_us)
                )

        if state.unbound.physics_update_mode == PhysicsUpdateMode.ALL_ACTORS:
            ego_aabb_pose_future = (
                ego_ds_pose @ state.unbound.transform_ego_coords_ds_to_aabb
            )

            ego_traj_aabb = Trajectory.from_poses(
                np.array([target_time_us], dtype=np.uint64),
                [ego_aabb_pose_future],
            )

            _, traffic_poses_future = await physics.ground_intersection(
                scene_id=state.unbound.scene_id,
                delta_start_us=self.timestamp_us,
                delta_end_us=target_time_us,
                ego_trajectory_aabb=ego_traj_aabb,
                traffic_poses=traffic_poses_future,
                ego_aabb=state.unbound.ego_aabb,
                skip=trafficsim.skip or force_gt,
            )

        # Accumulate into per-object trajectories
        for obj_id, pose in traffic_poses_future.items():
            if obj_id in ctx.traffic_trajectories:
                ctx.traffic_trajectories[obj_id].update_absolute(target_time_us, pose)
            else:
                ctx.traffic_trajectories[obj_id] = geometry.Trajectory.from_poses(
                    np.array([target_time_us], dtype=np.uint64), [pose]
                )
