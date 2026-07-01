# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math

import numpy as np
import torch
from alpasim_grpc.v0 import common_pb2, traffic_pb2
from alpasim_trafficsim.grpc.service_structures import SimEnvData
from alpasim_utils.geometry import Trajectory, pose_to_grpc_at_time
from loguru import logger

from .env_builder import sample_start_timestamp_us, step_idx_to_timestamp_us


def build_agent_pose_at_timestamp(
    env_data: SimEnvData,
    *,
    agent_idx: int,
    timestamp_us: int,
    dt_us: int,
) -> common_pb2.PoseAtTime | None:
    valid_step_indices = (
        torch.nonzero(
            env_data["agents"]["valid_mask"][agent_idx],
            as_tuple=False,
        )
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    if not valid_step_indices:
        return None

    sample_start_t_us = sample_start_timestamp_us(env_data)
    timestamps_us = np.asarray(
        [
            sample_start_t_us + (int(step_idx) * dt_us)
            for step_idx in valid_step_indices
        ],
        dtype=np.uint64,
    )
    positions = (
        env_data["agents"]["xyz"][agent_idx, valid_step_indices, :]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    headings = (
        env_data["agents"]["heading"][agent_idx, valid_step_indices]
        .detach()
        .cpu()
        .numpy()
    )
    quaternions = np.zeros((len(valid_step_indices), 4), dtype=np.float32)
    half_yaws = 0.5 * headings
    quaternions[:, 2] = np.sin(half_yaws)
    quaternions[:, 3] = np.cos(half_yaws)
    trajectory = Trajectory(timestamps_us, positions, quaternions)
    try:
        pose = trajectory.interpolate_pose(int(timestamp_us))
    except ValueError:
        return None
    return pose_to_grpc_at_time(pose, int(timestamp_us))


def build_agent_updates_from_env(
    env_data: SimEnvData,
    *,
    timestamp_us: int,
    forecast_end_timestamp_us: int | None = None,
    dt_us: int,
) -> list[traffic_pb2.ObjectTrajectoryUpdate]:
    updates: list[traffic_pb2.ObjectTrajectoryUpdate] = []
    num_agents = env_data["agents"]["xyz"].shape[0]
    agent_object_ids = env_data["env"].get("agent_object_ids")
    if agent_object_ids is None:
        agent_object_ids = [
            str(int(track_id))
            for track_id in env_data["agents"]["track_ids"].detach().cpu().tolist()
        ]
    agent_is_static = env_data["env"].get("agent_is_static")
    if agent_is_static is None:
        static_mask = [False] * num_agents
    else:
        static_mask = [bool(v) for v in agent_is_static]
        if len(static_mask) != num_agents:
            static_mask = [False] * num_agents

    sample_start_t_us = sample_start_timestamp_us(env_data)
    forecast_timestamps_us = [int(timestamp_us)]
    if (
        forecast_end_timestamp_us is not None
        and forecast_end_timestamp_us > timestamp_us
    ):
        first_step_idx = math.floor((int(timestamp_us) - sample_start_t_us) / dt_us) + 1
        last_step_idx = math.floor(
            (int(forecast_end_timestamp_us) - sample_start_t_us) / dt_us
        )
        forecast_timestamps_us.extend(
            sample_start_t_us + (step_idx * dt_us)
            for step_idx in range(max(first_step_idx, 0), last_step_idx + 1)
            if sample_start_t_us + (step_idx * dt_us) > int(timestamp_us)
        )

    for agent_idx in range(num_agents):
        timestamps_for_agent = (
            [int(timestamp_us)] if static_mask[agent_idx] else forecast_timestamps_us
        )
        poses_at_time = [
            pose_at_time
            for ts_us in timestamps_for_agent
            if (
                pose_at_time := build_agent_pose_at_timestamp(
                    env_data,
                    agent_idx=agent_idx,
                    timestamp_us=ts_us,
                    dt_us=dt_us,
                )
            )
            is not None
        ]
        if not poses_at_time:
            continue
        trajectory = common_pb2.Trajectory(poses=poses_at_time)
        updates.append(
            traffic_pb2.ObjectTrajectoryUpdate(
                object_id=str(agent_object_ids[agent_idx]),
                trajectory=trajectory,
            )
        )
    return updates


def build_simulation_response(
    *,
    session_uuid: str,
    env_data: SimEnvData,
    query_ts_us: int,
    future_step_indices: list[int],
    forecast_step_indices: list[int] | None = None,
    dt_us: int,
    minimum_history_length: int,
) -> tuple[traffic_pb2.TrafficReturn, int]:
    current_step_idx = int(env_data["env"]["curr_t"])
    current_ts_us = step_idx_to_timestamp_us(
        env_data,
        current_step_idx,
        dt_us=dt_us,
    )
    env_data["current_time_us"] = torch.tensor([[current_ts_us]], dtype=torch.long)
    forecast_end_ts_us = None
    if forecast_step_indices:
        forecast_end_ts_us = step_idx_to_timestamp_us(
            env_data,
            forecast_step_indices[-1],
            dt_us=dt_us,
        )
    response_updates = build_agent_updates_from_env(
        env_data,
        timestamp_us=query_ts_us,
        forecast_end_timestamp_us=forecast_end_ts_us,
        dt_us=dt_us,
    )
    logger.debug(
        "catk_closed_loop_response: session={} requested_ts_us={} "
        "current_ts_us={} request_gap_us={} response_updates={} "
        "future_steps={} forecast_steps={} history_steps={}",
        session_uuid,
        query_ts_us,
        current_ts_us,
        query_ts_us - current_ts_us,
        len(response_updates),
        len(future_step_indices),
        len(forecast_step_indices or []),
        minimum_history_length,
    )
    return (
        traffic_pb2.TrafficReturn(object_trajectory_updates=response_updates),
        current_ts_us,
    )
