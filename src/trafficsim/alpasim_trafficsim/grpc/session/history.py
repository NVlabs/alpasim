# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Traffic session trajectory history and env resampling helpers."""

import numpy as np
from alpasim_trafficsim.grpc.pipeline.env_builder import (
    agent_is_static_by_object_id,
    agent_object_id_to_index,
    first_trajectory_pose_at_timestamp,
    reset_env_dynamic_state,
    snapshot_dynamic_env_data,
    step_idx_to_timestamp_us,
    write_pose_to_env,
)
from alpasim_trafficsim.grpc.service_structures import SessionState, SimEnvData
from alpasim_utils.geometry import (
    Trajectory,
    pose_to_grpc_at_time,
    trajectory_from_grpc,
)


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
                object_id="EGO",
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

    resampled_env_data = snapshot_dynamic_env_data(session_state.env_data)
    reset_env_dynamic_state(
        resampled_env_data,
        total_steps=history_steps,
        curr_t=history_steps - 1,
        sample_start_t_us=start_ts_us,
        dt_us=dt_us,
    )

    static_by_object_id = agent_is_static_by_object_id(resampled_env_data)
    object_id_to_idx = agent_object_id_to_index(resampled_env_data)
    object_ids = [
        "EGO",
        *[str(v) for v in resampled_env_data["env"].get("agent_object_ids", [])],
    ]
    for object_id in object_ids:
        trajectory = session_state.closed_loop_trajectories.get(str(object_id))
        if trajectory is None:
            continue
        for step_idx in range(history_steps):
            timestamp_us = start_ts_us + (step_idx * dt_us)
            try:
                pose = pose_to_grpc_at_time(
                    trajectory.interpolate_pose(int(timestamp_us)),
                    int(timestamp_us),
                )
            except ValueError:
                pose = None
            if pose is None and static_by_object_id.get(str(object_id), False):
                pose = first_trajectory_pose_at_timestamp(
                    trajectory,
                    timestamp_us=timestamp_us,
                )
            if pose is None:
                continue
            write_pose_to_env(
                resampled_env_data,
                object_id=str(object_id),
                step_idx=step_idx,
                pose_at_time=pose,
                agent_object_id_to_idx=object_id_to_idx,
            )

    return resampled_env_data
