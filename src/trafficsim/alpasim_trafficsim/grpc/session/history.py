# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Traffic session trajectory history and env resampling helpers."""

import numpy as np
import torch
from alpasim_trafficsim.grpc.pipeline.env_builder import (
    agent_is_static_by_object_id,
    agent_object_id_to_index,
    snapshot_dynamic_env_data,
    step_idx_to_timestamp_us,
)
from alpasim_trafficsim.grpc.service_structures import SessionState, SimEnvData
from alpasim_utils.geometry import Trajectory, trajectory_from_grpc

_EGO_OBJECT_ID = "EGO"


def _trajectory_from_env_samples(
    timestamps_us: list[int],
    xyz,
    heading,
) -> Trajectory:
    positions = xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    headings = heading.detach().cpu().numpy()
    quaternions = np.zeros((len(timestamps_us), 4), dtype=np.float32)
    half_yaws = 0.5 * headings
    quaternions[:, 2] = np.sin(half_yaws)
    quaternions[:, 3] = np.cos(half_yaws)
    return Trajectory(
        np.asarray(timestamps_us, dtype=np.uint64),
        positions,
        quaternions,
    )


def merge_trajectory(
    trajectories: dict[str, Trajectory],
    *,
    object_id: str,
    trajectory: Trajectory,
) -> None:
    object_id = str(object_id)
    if trajectory.is_empty():
        return
    target = trajectories.get(object_id)
    if target is None:
        trajectories[object_id] = trajectory.clone()
        return

    first_incoming_ts_us = int(trajectory.timestamps_us[0])
    retained_target = target.filter(target.timestamps_us < first_incoming_ts_us)
    trajectories[object_id] = retained_target.append(trajectory)


def merge_object_trajectory_updates(
    trajectories: dict[str, Trajectory],
    updates,
) -> None:
    """Merge object trajectory updates into an existing trajectory history.

    Args:
        trajectories: Input/output mapping updated in place.
        updates: Input trajectory updates to merge.
    """
    for update in updates:
        merge_trajectory(
            trajectories,
            object_id=str(update.object_id),
            trajectory=trajectory_from_grpc(update.trajectory),
        )


def merge_env_step_trajectories(
    trajectories: dict[str, Trajectory],
    env_data: SimEnvData,
    *,
    step_indices: list[int],
    dt_us: int,
    include_ego: bool,
) -> None:
    if not step_indices:
        return

    step_indices = [int(step_idx) for step_idx in step_indices]
    timestamps_by_step_idx = {
        step_idx: step_idx_to_timestamp_us(env_data, step_idx, dt_us=dt_us)
        for step_idx in step_indices
    }

    if include_ego:
        ego_steps = [
            step_idx
            for step_idx in step_indices
            if step_idx < env_data["ego"]["xyz"].shape[0]
        ]
        if ego_steps:
            merge_trajectory(
                trajectories,
                object_id=_EGO_OBJECT_ID,
                trajectory=_trajectory_from_env_samples(
                    [timestamps_by_step_idx[step_idx] for step_idx in ego_steps],
                    env_data["ego"]["xyz"][ego_steps, :],
                    env_data["ego"]["heading"][ego_steps],
                ),
            )

    num_agents = int(env_data["agents"]["xyz"].shape[0])
    agent_object_ids = env_data["env"].get("agent_object_ids")
    if agent_object_ids is None:
        agent_object_ids = [
            str(int(track_id))
            for track_id in env_data["agents"]["track_ids"].detach().cpu().tolist()
        ]

    for agent_idx in range(num_agents):
        agent_steps = [
            step_idx
            for step_idx in step_indices
            if step_idx < env_data["agents"]["valid_mask"].shape[1]
            and bool(env_data["agents"]["valid_mask"][agent_idx, step_idx].item())
        ]
        if not agent_steps:
            continue
        merge_trajectory(
            trajectories,
            object_id=str(agent_object_ids[agent_idx]),
            trajectory=_trajectory_from_env_samples(
                [timestamps_by_step_idx[step_idx] for step_idx in agent_steps],
                env_data["agents"]["xyz"][agent_idx, agent_steps, :],
                env_data["agents"]["heading"][agent_idx, agent_steps],
            ),
        )


def _resample_trajectory_history(
    trajectory: Trajectory,
    timestamps_us: np.ndarray,
    *,
    is_static: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one trajectory over a history window.

    Returns history step indices and matching position/heading arrays. Dynamic
    trajectories omit timestamps outside their recorded range. Static
    trajectories use their first pose for out-of-range timestamps while still
    interpolating any in-range samples.
    """
    empty_indices = np.empty((0,), dtype=np.int64)
    empty_positions = np.empty((0, 3), dtype=np.float32)
    empty_headings = np.empty((0,), dtype=np.float32)
    if trajectory.is_empty():
        return empty_indices, empty_positions, empty_headings

    trajectory_start_us, trajectory_end_us = trajectory.get_time_range_tuple()
    in_range = (timestamps_us >= trajectory_start_us) & (
        timestamps_us < trajectory_end_us
    )

    if is_static:
        first_pose = trajectory.first_pose
        positions = np.broadcast_to(
            np.asarray(first_pose.vec3, dtype=np.float32),
            (len(timestamps_us), 3),
        ).copy()
        headings = np.full(
            (len(timestamps_us),),
            first_pose.yaw(),
            dtype=np.float32,
        )
        step_indices = np.arange(len(timestamps_us), dtype=np.int64)
        if bool(in_range.any()):
            resampled = trajectory.interpolate(timestamps_us[in_range])
            positions[in_range] = resampled.positions
            headings[in_range] = resampled.yaws
        return step_indices, positions, headings

    step_indices = np.flatnonzero(in_range).astype(np.int64, copy=False)
    if step_indices.size == 0:
        return empty_indices, empty_positions, empty_headings

    resampled = trajectory.interpolate(timestamps_us[in_range])
    return (
        step_indices,
        np.asarray(resampled.positions, dtype=np.float32),
        np.asarray(resampled.yaws, dtype=np.float32),
    )


def build_resampled_env_data(
    session_state: SessionState,
    *,
    end_ts_us: int,
    history_steps: int,
    dt_us: int,
) -> SimEnvData:
    history_steps = max(int(history_steps), 1)
    end_ts_us = int(end_ts_us)
    start_ts_us = end_ts_us - ((history_steps - 1) * dt_us)
    timestamps_us = start_ts_us + (
        np.arange(history_steps, dtype=np.uint64) * np.uint64(dt_us)
    )

    resampled_env_data = snapshot_dynamic_env_data(session_state.env_data)
    source_ego_xyz = session_state.env_data["ego"]["xyz"]
    source_ego_heading = session_state.env_data["ego"]["heading"]
    source_agent_xyz = session_state.env_data["agents"]["xyz"]
    source_agent_heading = session_state.env_data["agents"]["heading"]
    source_agent_valid = session_state.env_data["agents"]["valid_mask"]
    num_agents = int(source_agent_xyz.shape[0])

    ego_xyz = np.zeros((history_steps, 3), dtype=np.float32)
    ego_heading = np.zeros((history_steps,), dtype=np.float32)
    agent_xyz = np.zeros((num_agents, history_steps, 3), dtype=np.float32)
    agent_heading = np.zeros((num_agents, history_steps), dtype=np.float32)
    agent_valid = np.zeros((num_agents, history_steps), dtype=np.bool_)

    static_by_object_id = agent_is_static_by_object_id(resampled_env_data)
    object_id_to_idx = agent_object_id_to_index(resampled_env_data)
    object_ids = [
        _EGO_OBJECT_ID,
        *[str(v) for v in resampled_env_data["env"].get("agent_object_ids", [])],
    ]
    for object_id in object_ids:
        object_id = str(object_id)
        trajectory = session_state.closed_loop_trajectories.get(object_id)
        if trajectory is None:
            continue
        is_static = (
            False if object_id == _EGO_OBJECT_ID else static_by_object_id[object_id]
        )
        step_indices, positions, headings = _resample_trajectory_history(
            trajectory,
            timestamps_us,
            is_static=is_static,
        )
        if step_indices.size == 0:
            continue

        if object_id == _EGO_OBJECT_ID:
            ego_xyz[step_indices] = positions
            ego_heading[step_indices] = headings
            continue

        agent_idx = object_id_to_idx[object_id]
        agent_xyz[agent_idx, step_indices] = positions
        agent_heading[agent_idx, step_indices] = headings
        agent_valid[agent_idx, step_indices] = True

    resampled_env_data["ego"]["xyz"] = torch.as_tensor(
        ego_xyz, dtype=source_ego_xyz.dtype, device=source_ego_xyz.device
    )
    resampled_env_data["ego"]["heading"] = torch.as_tensor(
        ego_heading, dtype=source_ego_heading.dtype, device=source_ego_heading.device
    )
    resampled_env_data["agents"]["xyz"] = torch.as_tensor(
        agent_xyz, dtype=source_agent_xyz.dtype, device=source_agent_xyz.device
    )
    resampled_env_data["agents"]["heading"] = torch.as_tensor(
        agent_heading,
        dtype=source_agent_heading.dtype,
        device=source_agent_heading.device,
    )
    resampled_env_data["agents"]["valid_mask"] = torch.as_tensor(
        agent_valid, dtype=source_agent_valid.dtype, device=source_agent_valid.device
    )
    resampled_env_data["env"]["curr_t"] = history_steps - 1
    resampled_env_data["env"]["sample_start_t_us"] = start_ts_us
    resampled_env_data["current_time_us"] = torch.tensor(
        [[end_ts_us]],
        dtype=torch.long,
    )

    return resampled_env_data
