# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import copy

import numpy as np
import torch
from alpasim_grpc.v0 import common_pb2, traffic_pb2
from alpasim_trafficsim.grpc.service_structures import SimEnvData
from alpasim_utils.geometry import (
    Pose,
    Trajectory,
    pose_from_grpc,
    pose_to_grpc_at_time,
    quat_to_yaw,
    yaw_to_quat_components,
)
from loguru import logger


class InsufficientEgoTrajectoryError(ValueError):
    """Raised when ego conditioning does not cover required CATK samples."""


def sample_start_timestamp_us(env_data: SimEnvData) -> int:
    sample_start_t_us = env_data["env"].get("sample_start_t_us")
    if sample_start_t_us is not None:
        return int(sample_start_t_us)

    metadata = env_data.get("metadata", {})
    if "t0_us" in metadata:
        return int(metadata["t0_us"])

    raise KeyError("env_data is missing both env.sample_start_t_us and metadata.t0_us")


def step_idx_to_timestamp_us(env_data: SimEnvData, step_idx: int, *, dt_us: int) -> int:
    return sample_start_timestamp_us(env_data) + step_idx * dt_us


def snapshot_dynamic_env_data(env_data: SimEnvData) -> SimEnvData:
    """Return a shallow snapshot of the dynamic ``env_data`` sub-dicts.

    The ``env``/``ego``/``agents`` mappings are copied into new dicts so the
    caller can re-key them without mutating the source, while the tensors and the
    ``map``/``metadata`` containers are shared by reference (no deep copy).
    """
    return {
        "metadata": env_data.get("metadata", {}),
        "env": dict(env_data.get("env", {})),
        "ego": dict(env_data["ego"]),
        "agents": dict(env_data["agents"]),
        "map": env_data.get("map", {}),
    }


def ensure_time_axis_length(env_data: SimEnvData, required_step_idx: int) -> None:
    required_steps = required_step_idx + 1
    current_steps = env_data["ego"]["xyz"].shape[0]
    if required_steps <= current_steps:
        return

    pad_steps = required_steps - current_steps
    ego_device = env_data["ego"]["xyz"].device
    agent_device = env_data["agents"]["xyz"].device
    num_agents = env_data["agents"]["xyz"].shape[0]

    env_data["ego"]["xyz"] = torch.cat(
        [
            env_data["ego"]["xyz"],
            torch.zeros((pad_steps, 3), dtype=torch.float32, device=ego_device),
        ],
        dim=0,
    )
    env_data["ego"]["heading"] = torch.cat(
        [
            env_data["ego"]["heading"],
            torch.zeros((pad_steps,), dtype=torch.float32, device=ego_device),
        ],
        dim=0,
    )
    env_data["agents"]["xyz"] = torch.cat(
        [
            env_data["agents"]["xyz"],
            torch.zeros(
                (num_agents, pad_steps, 3), dtype=torch.float32, device=agent_device
            ),
        ],
        dim=1,
    )
    env_data["agents"]["heading"] = torch.cat(
        [
            env_data["agents"]["heading"],
            torch.zeros(
                (num_agents, pad_steps), dtype=torch.float32, device=agent_device
            ),
        ],
        dim=1,
    )
    env_data["agents"]["valid_mask"] = torch.cat(
        [
            env_data["agents"]["valid_mask"],
            torch.zeros((num_agents, pad_steps), dtype=torch.bool, device=agent_device),
        ],
        dim=1,
    )


def reset_env_dynamic_state(
    env_data: SimEnvData,
    *,
    total_steps: int,
    curr_t: int,
    sample_start_t_us: int,
    dt_us: int,
) -> None:
    total_steps = max(total_steps, 1)
    ego_device = env_data["ego"]["xyz"].device
    agent_device = env_data["agents"]["xyz"].device
    num_agents = env_data["agents"]["xyz"].shape[0]

    env_data["ego"]["xyz"] = torch.zeros(
        (total_steps, 3),
        dtype=torch.float32,
        device=ego_device,
    )
    env_data["ego"]["heading"] = torch.zeros(
        (total_steps,),
        dtype=torch.float32,
        device=ego_device,
    )
    env_data["agents"]["xyz"] = torch.zeros(
        (num_agents, total_steps, 3),
        dtype=torch.float32,
        device=agent_device,
    )
    env_data["agents"]["heading"] = torch.zeros(
        (num_agents, total_steps),
        dtype=torch.float32,
        device=agent_device,
    )
    env_data["agents"]["valid_mask"] = torch.zeros(
        (num_agents, total_steps),
        dtype=torch.bool,
        device=agent_device,
    )
    env_data["env"]["curr_t"] = max(curr_t, 0)
    env_data["env"]["sample_start_t_us"] = sample_start_t_us
    env_data["current_time_us"] = torch.tensor(
        [[sample_start_t_us + max(curr_t, 0) * dt_us]],
        dtype=torch.long,
    )


def parse_object_id(object_id: str, fallback_idx: int) -> int:
    try:
        return int(object_id)
    except (TypeError, ValueError):
        return fallback_idx


def agent_object_id_to_index(env_data: SimEnvData) -> dict[str, int]:
    agent_object_ids = env_data["env"].get("agent_object_ids")
    if agent_object_ids is not None:
        return {str(object_id): idx for idx, object_id in enumerate(agent_object_ids)}
    track_ids = env_data["agents"]["track_ids"]
    return {str(int(track_id)): idx for idx, track_id in enumerate(track_ids.tolist())}


def build_session_env_data(
    *,
    base_env_data: SimEnvData,
    logged_object_trajectories: list[traffic_pb2.ObjectTrajectory],
) -> SimEnvData:
    env_data = copy.deepcopy(base_env_data)
    agent_xyz_template = env_data["agents"]["xyz"]
    agent_heading_template = env_data["agents"]["heading"]
    agent_valid_template = env_data["agents"]["valid_mask"]
    agent_lwh_template = env_data["agents"]["lwh"]
    agent_steps = max(agent_xyz_template.shape[1], 1)

    loader_track_ids = torch.as_tensor(
        env_data["agents"]["track_ids"],
        dtype=torch.long,
        device=agent_xyz_template.device,
    ).detach()
    loader_class_ids = torch.as_tensor(
        env_data["agents"]["class_ids"],
        dtype=torch.long,
        device=agent_xyz_template.device,
    ).detach()
    class_id_by_track_id = {
        int(track_id): int(class_id)
        for track_id, class_id in zip(
            loader_track_ids.detach().cpu().tolist(),
            loader_class_ids.detach().cpu().tolist(),
            strict=False,
        )
    }
    lwh_by_track_id = {
        int(track_id): agent_lwh_template[idx].clone()
        for idx, track_id in enumerate(loader_track_ids.detach().cpu().tolist())
    }

    obstacle_class_name_to_id = env_data["metadata"].get("obstacle_class_name_2_id", {})
    default_class_id = int(
        obstacle_class_name_to_id.get("car", obstacle_class_name_to_id.get("others", 0))
    )

    ego_object = next(
        (
            logged_object
            for logged_object in logged_object_trajectories
            if str(logged_object.object_id).upper() == "EGO"
        ),
        None,
    )
    if ego_object is not None:
        env_data["ego"]["lwh"] = agent_lwh_template.new_tensor(
            [
                float(ego_object.aabb.size_x),
                float(ego_object.aabb.size_y),
                float(ego_object.aabb.size_z),
            ],
            dtype=torch.float32,
        )

    agent_objects = [
        logged_object
        for logged_object in logged_object_trajectories
        if str(logged_object.object_id).upper() != "EGO"
    ]
    num_agents = len(agent_objects)
    agent_xyz = torch.zeros(
        (num_agents, agent_steps, 3),
        dtype=agent_xyz_template.dtype,
        device=agent_xyz_template.device,
    )
    agent_heading = torch.zeros(
        (num_agents, agent_steps),
        dtype=agent_heading_template.dtype,
        device=agent_heading_template.device,
    )
    agent_valid_mask = torch.zeros(
        (num_agents, agent_steps),
        dtype=agent_valid_template.dtype,
        device=agent_valid_template.device,
    )
    agent_lwh = torch.zeros(
        (num_agents, 3),
        dtype=agent_lwh_template.dtype,
        device=agent_lwh_template.device,
    )
    agent_track_ids: list[int] = []
    agent_class_ids: list[int] = []
    agent_object_ids: list[str] = []
    agent_is_static: list[bool] = []

    for agent_idx, logged_object in enumerate(agent_objects):
        object_id = str(logged_object.object_id)
        track_id = parse_object_id(object_id, agent_idx + 1)
        agent_track_ids.append(track_id)
        agent_object_ids.append(object_id)
        agent_is_static.append(bool(logged_object.is_static))
        agent_class_ids.append(class_id_by_track_id.get(track_id, default_class_id))

        if logged_object.HasField("aabb"):
            agent_lwh[agent_idx] = agent_lwh_template.new_tensor(
                [
                    float(logged_object.aabb.size_x),
                    float(logged_object.aabb.size_y),
                    float(logged_object.aabb.size_z),
                ],
                dtype=agent_lwh_template.dtype,
            )
        elif track_id in lwh_by_track_id:
            agent_lwh[agent_idx] = lwh_by_track_id[track_id].to(
                device=agent_lwh_template.device,
                dtype=agent_lwh_template.dtype,
            )

    agent_track_ids_tensor = torch.tensor(
        agent_track_ids,
        dtype=torch.long,
        device=agent_xyz_template.device,
    )
    agent_class_ids_tensor = torch.tensor(
        agent_class_ids,
        dtype=torch.long,
        device=agent_xyz_template.device,
    )
    env_data["agents"] = {
        "xyz": agent_xyz,
        "heading": agent_heading,
        "valid_mask": agent_valid_mask,
        "lwh": agent_lwh,
        "track_ids": agent_track_ids_tensor,
        "class_ids": agent_class_ids_tensor,
        "num_obstacles": num_agents,
    }
    env_data.setdefault("env", {})
    env_data["env"]["agent_object_ids"] = agent_object_ids
    env_data["env"]["agent_is_static"] = agent_is_static
    return env_data


def write_pose_to_env(
    env_data: SimEnvData,
    *,
    object_id: str,
    step_idx: int,
    pose_at_time: common_pb2.PoseAtTime,
    agent_object_id_to_idx: dict[str, int] | None = None,
) -> None:
    ensure_time_axis_length(env_data, step_idx)
    if str(object_id).upper() == "EGO":
        env_data["ego"]["xyz"][step_idx, 0] = float(pose_at_time.pose.vec.x)
        env_data["ego"]["xyz"][step_idx, 1] = float(pose_at_time.pose.vec.y)
        env_data["ego"]["xyz"][step_idx, 2] = float(pose_at_time.pose.vec.z)
        env_data["ego"]["heading"][step_idx] = quat_to_yaw(pose_at_time.pose.quat)
        return

    object_id_to_idx = (
        agent_object_id_to_idx
        if agent_object_id_to_idx is not None
        else agent_object_id_to_index(env_data)
    )
    agent_idx = object_id_to_idx.get(str(object_id))

    if agent_idx is None:
        return

    env_data["agents"]["xyz"][agent_idx, step_idx, 0] = float(pose_at_time.pose.vec.x)
    env_data["agents"]["xyz"][agent_idx, step_idx, 1] = float(pose_at_time.pose.vec.y)
    env_data["agents"]["xyz"][agent_idx, step_idx, 2] = float(pose_at_time.pose.vec.z)
    env_data["agents"]["heading"][agent_idx, step_idx] = quat_to_yaw(
        pose_at_time.pose.quat
    )
    env_data["agents"]["valid_mask"][agent_idx, step_idx] = True


def agent_is_static_by_object_id(env_data: SimEnvData) -> dict[str, bool]:
    agent_object_ids = env_data["env"].get("agent_object_ids") or []
    agent_is_static = env_data["env"].get("agent_is_static") or []
    return {
        str(object_id): (
            bool(agent_is_static[idx]) if idx < len(agent_is_static) else False
        )
        for idx, object_id in enumerate(agent_object_ids)
    }


def static_agent_mask(env_data: SimEnvData, *, device: torch.device) -> torch.Tensor:
    num_agents = int(env_data["agents"]["xyz"].shape[0])
    raw_mask = env_data["env"].get("agent_is_static")
    if raw_mask is None:
        return torch.zeros((num_agents,), dtype=torch.bool, device=device)
    mask = torch.as_tensor(raw_mask, dtype=torch.bool, device=device).flatten()
    if int(mask.numel()) >= num_agents:
        return mask[:num_agents]
    padded = torch.zeros((num_agents,), dtype=torch.bool, device=device)
    padded[: int(mask.numel())] = mask
    return padded


def backfill_static_agent_history(
    env_data: SimEnvData,
    *,
    curr_t: int,
    history_window_steps: int,
    predict_static: bool,
) -> None:
    """Fill static-agent history with the current pose for model input."""
    agents = env_data["agents"]
    total_agents = int(agents["xyz"].shape[0])
    num_agents = min(int(agents["num_obstacles"]), total_agents)
    if num_agents <= 0 or curr_t < 0:
        return

    curr_t = min(curr_t, int(agents["xyz"].shape[1]) - 1)
    history_beg = max(curr_t - history_window_steps + 1, 0)
    history_slice = slice(history_beg, curr_t + 1)
    prev_valid = agents["valid_mask"][:num_agents, curr_t]
    frozen_static_mask = (
        static_agent_mask(env_data, device=prev_valid.device)[:num_agents]
        if not predict_static
        else torch.zeros((num_agents,), dtype=torch.bool, device=prev_valid.device)
    )
    backfill_mask = frozen_static_mask & prev_valid
    if not bool(backfill_mask.any().item()):
        return

    agent_indices = torch.where(backfill_mask)[0]
    history_len = curr_t - history_beg + 1
    agents["xyz"][agent_indices, history_slice, :] = (
        agents["xyz"][agent_indices, curr_t, :].unsqueeze(1).expand(-1, history_len, -1)
    )
    agents["heading"][agent_indices, history_slice] = (
        agents["heading"][agent_indices, curr_t].unsqueeze(1).expand(-1, history_len)
    )
    agents["valid_mask"][agent_indices, history_slice] = True


def first_trajectory_pose_at_timestamp(
    trajectory: Trajectory,
    *,
    timestamp_us: int,
) -> common_pb2.PoseAtTime | None:
    if trajectory.is_empty():
        return None
    return pose_to_grpc_at_time(trajectory.first_pose, int(timestamp_us))


def populate_ego_future_from_trajectory(
    env_data: SimEnvData,
    ego_trajectory: Trajectory,
    *,
    current_step_idx: int,
    requested_timestamp_us: int,
    future_step_indices: list[int],
    dt_us: int,
) -> None:
    if not future_step_indices:
        return

    anchor_timestamp_us = step_idx_to_timestamp_us(
        env_data,
        current_step_idx,
        dt_us=dt_us,
    )
    anchor_heading = float(env_data["ego"]["heading"][current_step_idx].item())
    anchor_quat_w, anchor_quat_x, anchor_quat_y, anchor_quat_z = yaw_to_quat_components(
        anchor_heading
    )
    anchor_pose = Pose.from_denormalized_quat(
        env_data["ego"]["xyz"][current_step_idx, :]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False),
        np.asarray(
            [anchor_quat_x, anchor_quat_y, anchor_quat_z, anchor_quat_w],
            dtype=np.float32,
        ),
    )
    try:
        requested_pose = pose_to_grpc_at_time(
            ego_trajectory.interpolate_pose(int(requested_timestamp_us)),
            int(requested_timestamp_us),
        )
    except ValueError:
        requested_pose = None
    future_start_ts_us = step_idx_to_timestamp_us(
        env_data,
        future_step_indices[0],
        dt_us=dt_us,
    )
    future_end_ts_us = step_idx_to_timestamp_us(
        env_data,
        future_step_indices[-1],
        dt_us=dt_us,
    )
    if requested_pose is None:
        logger.debug(
            "catk_ego_conditioning: mode=sample_stream anchor_ts_us={} "
            "requested_ts_us={} future_steps={} future_start_ts_us={} "
            "future_end_ts_us={}",
            anchor_timestamp_us,
            requested_timestamp_us,
            len(future_step_indices),
            future_start_ts_us,
            future_end_ts_us,
        )
        for step_idx in future_step_indices:
            timestamp_us = step_idx_to_timestamp_us(
                env_data,
                step_idx,
                dt_us=dt_us,
            )
            try:
                sampled_pose = pose_to_grpc_at_time(
                    ego_trajectory.interpolate_pose(int(timestamp_us)),
                    int(timestamp_us),
                )
            except ValueError as exc:
                raise InsufficientEgoTrajectoryError(
                    "EGO trajectory does not cover required simulation timestamp "
                    f"{timestamp_us}; request timestamp is {requested_timestamp_us} "
                    f"and future sample end is {future_end_ts_us}"
                ) from exc
            write_pose_to_env(
                env_data,
                object_id="EGO",
                step_idx=step_idx,
                pose_at_time=sampled_pose,
            )
        return

    interpolation_trajectory = Trajectory.from_poses(
        np.asarray(
            [anchor_timestamp_us, requested_pose.timestamp_us],
            dtype=np.uint64,
        ),
        [
            anchor_pose,
            pose_from_grpc(requested_pose.pose),
        ],
    )
    logger.debug(
        "catk_ego_conditioning: mode=interpolate_single_endpoint "
        "anchor_ts_us={} requested_ts_us={} future_steps={} "
        "future_start_ts_us={} future_end_ts_us={}",
        anchor_timestamp_us,
        requested_timestamp_us,
        len(future_step_indices),
        future_start_ts_us,
        future_end_ts_us,
    )
    for step_idx in future_step_indices:
        interpolation_timestamp_us = step_idx_to_timestamp_us(
            env_data,
            step_idx,
            dt_us=dt_us,
        )
        if interpolation_timestamp_us <= requested_timestamp_us:
            try:
                sampled_pose = pose_to_grpc_at_time(
                    interpolation_trajectory.interpolate_pose(
                        int(interpolation_timestamp_us)
                    ),
                    int(interpolation_timestamp_us),
                )
            except ValueError as exc:
                raise InsufficientEgoTrajectoryError(
                    "EGO trajectory does not cover required simulation "
                    f"timestamp {interpolation_timestamp_us}; request timestamp is "
                    f"{requested_timestamp_us} and future sample end is "
                    f"{future_end_ts_us}"
                ) from exc
        else:
            try:
                sampled_pose = pose_to_grpc_at_time(
                    ego_trajectory.interpolate_pose(int(interpolation_timestamp_us)),
                    int(interpolation_timestamp_us),
                )
            except ValueError:
                sampled_pose = common_pb2.PoseAtTime(
                    timestamp_us=int(interpolation_timestamp_us),
                    pose=requested_pose.pose,
                )

        write_pose_to_env(
            env_data,
            object_id="EGO",
            step_idx=step_idx,
            pose_at_time=sampled_pose,
        )
