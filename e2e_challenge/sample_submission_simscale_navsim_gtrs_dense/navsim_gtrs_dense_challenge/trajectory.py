# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Trajectory, Vec3

EXPECTED_MODEL_SHAPE = (40, 3)
MODEL_INTERVAL_S = 0.1
MODEL_HORIZON_S = 4.0


@dataclass(frozen=True)
class CachedPlan:
    created_time_us: int
    times_s: np.ndarray
    positions_xy: np.ndarray
    yaws: np.ndarray


def yaw_from_quat(quat: Quat) -> float:
    siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw: float) -> Quat:
    half = 0.5 * yaw
    return Quat(w=float(math.cos(half)), x=0.0, y=0.0, z=float(math.sin(half)))


def rig_offsets_to_local_positions(
    anchor_pose: PoseAtTime,
    offsets_xy: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(offsets_xy, dtype=np.float64).reshape(-1, 2)
    yaw = yaw_from_quat(anchor_pose.pose.quat)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    origin = np.array(
        [anchor_pose.pose.vec.x, anchor_pose.pose.vec.y], dtype=np.float64
    )
    return offsets @ rotation.T + origin


def _warped_source_times(times_s: np.ndarray, scale: float) -> np.ndarray:
    if not math.isfinite(scale) or not 1.0 <= scale <= 1.25:
        raise ValueError("trajectory_time_scale must be finite and in [1.0, 1.25]")
    if scale == 1.0:
        return times_s

    lead_s = scale - 1.0
    source_times = np.where(
        times_s <= 1.0,
        scale * times_s,
        np.where(
            times_s <= 3.0,
            times_s + lead_s,
            times_s + lead_s * (MODEL_HORIZON_S - times_s),
        ),
    )
    return np.clip(source_times, 0.0, MODEL_HORIZON_S)


def make_cached_plan(
    created_time_us: int,
    anchor_pose: PoseAtTime,
    relative_poses: np.ndarray,
    *,
    trajectory_time_scale: float = 1.0,
) -> CachedPlan:
    poses = np.asarray(relative_poses, dtype=np.float64)
    if poses.shape != EXPECTED_MODEL_SHAPE:
        raise ValueError(
            f"GTRS prediction must have shape {EXPECTED_MODEL_SHAPE}; got {poses.shape}"
        )
    if not np.isfinite(poses).all():
        raise ValueError("GTRS prediction contains non-finite values")

    times_s = np.arange(41, dtype=np.float64) * MODEL_INTERVAL_S
    source_times_s = _warped_source_times(times_s, trajectory_time_scale)
    if trajectory_time_scale != 1.0:
        source_offsets = np.vstack((np.zeros((1, 2), dtype=np.float64), poses[:, :2]))
        source_yaws = np.concatenate(([0.0], np.unwrap(poses[:, 2])))
        warped_poses = poses.copy()
        warped_poses[:, 0] = np.interp(
            source_times_s[1:], times_s, source_offsets[:, 0]
        )
        warped_poses[:, 1] = np.interp(
            source_times_s[1:], times_s, source_offsets[:, 1]
        )
        warped_poses[:, 2] = np.interp(source_times_s[1:], times_s, source_yaws)
        poses = warped_poses
    anchor_xy = np.array(
        [[anchor_pose.pose.vec.x, anchor_pose.pose.vec.y]], dtype=np.float64
    )
    positions_xy = np.vstack(
        (anchor_xy, rig_offsets_to_local_positions(anchor_pose, poses[:, :2]))
    )
    anchor_yaw = yaw_from_quat(anchor_pose.pose.quat)
    yaws = np.concatenate(([anchor_yaw], anchor_yaw + poses[:, 2]))
    return CachedPlan(created_time_us, times_s, positions_xy, yaws)


def build_trajectory_from_plan(
    plan: CachedPlan | None,
    current_pose: PoseAtTime | None,
    time_now_us: int,
    time_query_us: int,
    *,
    callback_frequency_hz: float = 10.0,
    fallback_speed_mps: float = 5.0,
) -> Trajectory:
    if current_pose is None:
        current_pose = PoseAtTime(
            timestamp_us=time_now_us,
            pose=Pose(
                vec=Vec3(x=0.0, y=0.0, z=0.0),
                quat=Quat(w=1.0),
            ),
        )

    required_horizon_s = max(4.0, (time_query_us - time_now_us) / 1_000_000.0)
    if not cached_plan_covers_query(
        plan,
        time_now_us,
        time_query_us,
        callback_frequency_hz=callback_frequency_hz,
    ):
        return build_straight_line_trajectory(
            current_pose,
            time_now_us,
            speed_mps=max(1.0, fallback_speed_mps),
            horizon_s=required_horizon_s,
            frequency_hz=callback_frequency_hz,
        )

    dt_s = 1.0 / callback_frequency_hz
    elapsed_s = max(0.0, (time_now_us - plan.created_time_us) / 1_000_000.0)
    plan_end_s = float(plan.times_s[-1])
    first_step = int(math.ceil(elapsed_s / dt_s - 1e-9))
    last_step = int(math.floor(plan_end_s / dt_s + 1e-9))
    sample_times = np.arange(first_step, last_step + 1, dtype=np.float64) * dt_s
    xs = np.interp(sample_times, plan.times_s, plan.positions_xy[:, 0])
    ys = np.interp(sample_times, plan.times_s, plan.positions_xy[:, 1])
    yaws = np.interp(sample_times, plan.times_s, np.unwrap(plan.yaws))

    current_z = float(current_pose.pose.vec.z)
    trajectory = Trajectory()
    for time_s, x, y, yaw in zip(sample_times, xs, ys, yaws, strict=True):
        trajectory.poses.append(
            PoseAtTime(
                timestamp_us=plan.created_time_us
                + int(round(float(time_s) * 1_000_000)),
                pose=Pose(
                    vec=Vec3(x=float(x), y=float(y), z=current_z),
                    quat=quat_from_yaw(float(yaw)),
                ),
            )
        )
    return trajectory


def cached_plan_covers_query(
    plan: CachedPlan | None,
    time_now_us: int,
    time_query_us: int,
    *,
    callback_frequency_hz: float = 10.0,
) -> bool:
    if plan is None or len(plan.times_s) < 2:
        return False
    plan_end_us = plan.created_time_us + int(round(float(plan.times_s[-1]) * 1_000_000))
    if plan_end_us < time_query_us:
        return False

    dt_s = 1.0 / callback_frequency_hz
    elapsed_s = max(0.0, (time_now_us - plan.created_time_us) / 1_000_000.0)
    first_step = int(math.ceil(elapsed_s / dt_s - 1e-9))
    last_step = int(math.floor(float(plan.times_s[-1]) / dt_s + 1e-9))
    return last_step - first_step >= 1


def build_straight_line_trajectory(
    start_pose: PoseAtTime,
    time_now_us: int,
    *,
    speed_mps: float,
    horizon_s: float,
    frequency_hz: float,
) -> Trajectory:
    yaw = yaw_from_quat(start_pose.pose.quat)
    dx = math.cos(yaw)
    dy = math.sin(yaw)
    dt_s = 1.0 / frequency_hz
    point_count = max(2, int(math.ceil(horizon_s * frequency_hz)) + 1)

    trajectory = Trajectory()
    for index in range(point_count):
        time_s = index * dt_s
        trajectory.poses.append(
            PoseAtTime(
                timestamp_us=time_now_us + int(round(time_s * 1_000_000)),
                pose=Pose(
                    vec=Vec3(
                        x=float(start_pose.pose.vec.x + dx * speed_mps * time_s),
                        y=float(start_pose.pose.vec.y + dy * speed_mps * time_s),
                        z=float(start_pose.pose.vec.z),
                    ),
                    quat=start_pose.pose.quat,
                ),
            )
        )
    return trajectory
