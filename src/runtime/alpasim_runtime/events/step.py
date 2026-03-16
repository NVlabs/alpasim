# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Close the current step and open the next.

Commits all computed state to RolloutState, logs actor poses, and clears
StepContext. This is the only event that mutates trajectory state.
"""

import time

import numpy as np
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.events.base import EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_runtime.telemetry.telemetry_context import try_get_context
from alpasim_utils import geometry
from numpy.typing import NDArray


class StepEvent(RecurringEvent):
    """Close the current step and open the next.

    Commits all computed state to RolloutState, logs actor poses, and clears
    StepContext. This is the only event that mutates trajectory state.
    """

    priority: int = EventPriority.STEP

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

        if ctx is not None and ctx.ego_true is not None:
            # --- Normal case: commit pipeline results ---
            assert (
                ctx.step_start_us == self.timestamp_us
            ), f"StepEvent timestamp mismatch: {ctx.step_start_us} != {self.timestamp_us}"
            assert ctx.ego_estimated is not None
            assert ctx.corrected_ego_trajectory is not None

            # Commit ego trajectories (bulk concat)
            corrected_ego = geometry.DynamicTrajectory.from_trajectory_and_dynamics(
                ctx.corrected_ego_trajectory,
                ctx.ego_true.dynamics,
            )
            state.ego_trajectory = state.ego_trajectory.concat(corrected_ego)
            state.ego_trajectory_estimate = state.ego_trajectory_estimate.concat(
                ctx.ego_estimated
            )

            # Commit accumulated traffic trajectories
            for obj_id, accumulated_traj in ctx.traffic_trajectories.items():
                if obj_id == "EGO":
                    continue
                for i in range(len(accumulated_traj)):
                    ts = int(accumulated_traj.timestamps_us[i])
                    pose = accumulated_traj.get_pose(i)
                    state.traffic_objs[obj_id].trajectory.update_absolute(ts, pose)

            # Log actor poses at each intermediate timestamp
            await log_actor_poses(
                state, ctx.ego_true.timestamps_us, self.services.broadcaster
            )

            # Record step duration telemetry
            step_duration = time.perf_counter() - state.step_wall_start
            telemetry_ctx = try_get_context()
            if telemetry_ctx is not None:
                telemetry_ctx.step_duration.observe(step_duration)
        else:
            # --- Initial case: log initial actor poses ---
            t0_us = state.unbound.control_timestamps_us[0]
            t1_us = state.unbound.control_timestamps_us[1]
            await log_actor_poses(
                state,
                np.array([t0_us, t1_us], dtype=np.uint64),
                self.services.broadcaster,
            )

        # --- Always: create fresh StepContext for next step ---
        state.step_context = StepContext()
        state.step_wall_start = time.perf_counter()


async def log_actor_poses(
    state: RolloutState,
    timestamps_us: NDArray[np.uint64],
    broadcaster: MessageBroadcaster,
) -> None:
    """Log actor poses (ego + traffic) to the ASL file.

    Builds the trajectory dict and ego coordinate transform once, then
    batch-interpolates each trajectory at all requested timestamps before
    assembling and broadcasting per-timestamp ``LogEntry`` messages.
    """
    trajectories = {
        obj_id: obj.trajectory for obj_id, obj in state.traffic_objs.items()
    }
    trajectories["EGO"] = state.ego_trajectory.trajectory().transform(
        state.unbound.transform_ego_coords_ds_to_aabb,
        is_relative=True,
    )

    # Pre-allocate per-timestamp actor pose lists
    poses_by_ts: dict[int, list[ActorPoses.ActorPose]] = {
        int(ts): [] for ts in timestamps_us
    }

    for obj_id, trajectory in trajectories.items():
        time_range = trajectory.time_range_us

        if obj_id == "EGO":
            if (
                int(timestamps_us[0]) not in time_range
                or int(timestamps_us[-1]) not in time_range
            ):
                raise AssertionError("Ego trajectory ended early.")
            valid_ts = timestamps_us
        else:
            valid_ts = timestamps_us[
                (timestamps_us >= time_range.start) & (timestamps_us < time_range.stop)
            ]

        if len(valid_ts) == 0:
            continue

        poses = trajectory.interpolate_poses_list(valid_ts)
        for ts, pose in zip(valid_ts, poses, strict=True):
            poses_by_ts[int(ts)].append(
                ActorPoses.ActorPose(
                    actor_id=obj_id,
                    actor_pose=geometry.pose_to_grpc(pose),
                )
            )

    for ts in timestamps_us:
        ts_int = int(ts)
        poses_message = LogEntry(
            actor_poses=ActorPoses(
                timestamp_us=ts_int,
                actor_poses=poses_by_ts[ts_int],
            )
        )
        await broadcaster.broadcast(poses_message)
