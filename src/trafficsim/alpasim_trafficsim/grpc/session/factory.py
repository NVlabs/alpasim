# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Construction of a fully-seeded ``SessionState`` from request inputs.

Keeps ``start_session`` thin: given the loaded base env data and the logged
trajectories, this builds the per-session env, computes the initial history
anchor, constructs the session, and seeds the maintained CATK history window.
"""

from alpasim_grpc.v0 import traffic_pb2
from alpasim_trafficsim.grpc.pipeline.env_builder import build_session_env_data
from alpasim_trafficsim.grpc.service_structures import SessionState, SimEnvData
from alpasim_utils.geometry import Trajectory, trajectory_from_grpc

from .history import build_resampled_env_data


def first_ego_pose_ts_us(logged_trajectories: dict[str, Trajectory]) -> int:
    ego_logged_trajectory = logged_trajectories.get("EGO")
    if ego_logged_trajectory is None:
        raise ValueError("logged_object_trajectories must include an EGO trajectory")
    if ego_logged_trajectory.is_empty():
        raise ValueError("logged EGO trajectory must contain at least one pose")
    return int(ego_logged_trajectory.timestamps_us[0])


def build_session_state(
    request: traffic_pb2.TrafficSessionRequest,
    *,
    base_env_data: SimEnvData,
    dt_us: int,
    minimum_history_length: int,
) -> SessionState:
    """Build a ``SessionState`` for ``start_session``."""
    logged_object_trajectories = list(request.logged_object_trajectories)
    logged_trajectories = {
        logged_object.object_id: trajectory_from_grpc(logged_object.trajectory)
        for logged_object in logged_object_trajectories
    }
    ego_ts_us = first_ego_pose_ts_us(logged_trajectories)

    env_data = build_session_env_data(
        base_env_data=base_env_data,
        logged_object_trajectories=logged_object_trajectories,
    )

    history_end_idx = minimum_history_length - 1
    initial_ts_us = ego_ts_us + history_end_idx * dt_us

    handover_time_us = int(request.handover_time_us)
    if handover_time_us <= 0:
        raise ValueError("handover_time_us must be positive")

    session_state = SessionState(
        session_uuid=request.session_uuid,
        scene_id=request.scene_id,
        current_ts_us=initial_ts_us,
        handover_time_us=handover_time_us,
        closed_loop_trajectories=logged_trajectories,
        env_data=env_data,
    )
    session_state.env_data = build_resampled_env_data(
        session_state,
        end_ts_us=initial_ts_us,
        history_steps=minimum_history_length,
        dt_us=dt_us,
    )

    return session_state
