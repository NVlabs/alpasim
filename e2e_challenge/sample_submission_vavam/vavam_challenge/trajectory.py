# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Turn one VAVAM prediction into the trajectories the Drive RPC returns.

The model predicts an ego-relative path.  When inference runs we convert that
path into the rollout's local (inertial) frame once, anchored at the current
ego pose, and cache it.

Between inferences we hand the controller the *same* cached path, just shorter:
each Drive call samples the cached plan on a grid fixed to the plan's own clock
and emits only the samples at or after "now".  The points never move until the
next inference -- the trajectory only shrinks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Trajectory, Vec3


@dataclass(frozen=True)
class CachedPlan:
    """One VAVAM prediction expressed in the rollout's local frame.

    ``times_s`` are seconds relative to ``created_time_us``; ``positions_xy``
    and ``yaws`` are absolute local-frame poses along the predicted path.
    """

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
    """Project ego-relative (rig-frame) offsets into the rollout local frame."""
    offsets = np.asarray(offsets_xy, dtype=np.float64).reshape(-1, 2)
    yaw = yaw_from_quat(anchor_pose.pose.quat)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    origin = np.array(
        [anchor_pose.pose.vec.x, anchor_pose.pose.vec.y], dtype=np.float64
    )
    return offsets @ rot.T + origin


def make_cached_plan(
    *,
    created_time_us: int,
    anchor_pose: PoseAtTime,
    trajectory_xy: np.ndarray,
    headings: np.ndarray,
    source_frequency_hz: float = 2.0,
) -> CachedPlan | None:
    """Convert raw model output into a local-frame cached plan (or ``None``)."""
    offsets = np.asarray(trajectory_xy, dtype=np.float64).reshape(-1, 2)
    if offsets.shape[0] == 0:
        return None

    step_s = 1.0 / source_frequency_hz
    future_times = np.arange(1, offsets.shape[0] + 1, dtype=np.float64) * step_s
    times_s = np.concatenate(([0.0], future_times))

    anchor_xy = np.array(
        [[anchor_pose.pose.vec.x, anchor_pose.pose.vec.y]], dtype=np.float64
    )
    positions_xy = np.vstack(
        (anchor_xy, rig_offsets_to_local_positions(anchor_pose, offsets))
    )

    anchor_yaw = yaw_from_quat(anchor_pose.pose.quat)
    model_yaws = np.asarray(headings, dtype=np.float64).reshape(-1) + anchor_yaw
    if model_yaws.size != offsets.shape[0]:
        model_yaws = _headings_from_positions(positions_xy)[1:]
    yaws = np.concatenate(([anchor_yaw], model_yaws))

    return CachedPlan(created_time_us, times_s, positions_xy, yaws)


def build_trajectory_from_plan(
    plan: CachedPlan | None,
    current_pose: PoseAtTime | None,
    time_now_us: int,
    *,
    callback_frequency_hz: float = 10.0,
    fallback_speed_mps: float = 5.0,
    max_horizon_s: float = 5.0,
) -> Trajectory:
    if current_pose is None:
        current_pose = PoseAtTime(
            timestamp_us=time_now_us,
            pose=Pose(vec=Vec3(x=0.0, y=0.0, z=0.0), quat=Quat(w=1.0)),
        )

    dt_s = 1.0 / callback_frequency_hz
    if plan is None or len(plan.times_s) < 2:
        return build_straight_line_trajectory(
            current_pose,
            time_now_us,
            speed_mps=max(1.0, fallback_speed_mps),
            horizon_s=max_horizon_s,
            frequency_hz=callback_frequency_hz,
        )

    # How far along the cached plan we are.  The plan stays fixed in the local
    # frame; we only stop emitting the points the ego has already driven past.
    elapsed_s = max(0.0, (time_now_us - plan.created_time_us) / 1_000_000.0)
    horizon_end_s = min(float(plan.times_s[-1]), elapsed_s + max_horizon_s)

    # Sample the plan on a grid fixed to its own clock (multiples of dt from
    # created_time), so a given plan-time always maps to the same point.  We
    # keep only the samples at or after "now": the trajectory shrinks, but no
    # point moves until the next inference replaces the plan.
    first_step = int(math.ceil(elapsed_s / dt_s - 1e-9))
    last_step = int(math.floor(horizon_end_s / dt_s + 1e-9))
    if last_step - first_step < 1:
        return build_straight_line_trajectory(
            current_pose,
            time_now_us,
            speed_mps=max(1.0, fallback_speed_mps),
            horizon_s=max_horizon_s,
            frequency_hz=callback_frequency_hz,
        )

    sample_times = np.arange(first_step, last_step + 1, dtype=np.float64) * dt_s
    xs = np.interp(sample_times, plan.times_s, plan.positions_xy[:, 0])
    ys = np.interp(sample_times, plan.times_s, plan.positions_xy[:, 1])
    yaws = np.interp(sample_times, plan.times_s, np.unwrap(plan.yaws))

    cur_z = float(current_pose.pose.vec.z)
    trajectory = Trajectory()
    for t_s, x, y, yaw in zip(sample_times, xs, ys, yaws, strict=True):
        trajectory.poses.append(
            PoseAtTime(
                timestamp_us=plan.created_time_us + int(round(float(t_s) * 1_000_000)),
                pose=Pose(
                    vec=Vec3(x=float(x), y=float(y), z=cur_z),
                    quat=quat_from_yaw(float(yaw)),
                ),
            )
        )
    return trajectory


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
    n_points = max(2, int(math.ceil(horizon_s * frequency_hz)) + 1)

    trajectory = Trajectory()
    for i in range(n_points):
        t_s = i * dt_s
        trajectory.poses.append(
            PoseAtTime(
                timestamp_us=time_now_us + int(round(t_s * 1_000_000)),
                pose=Pose(
                    vec=Vec3(
                        x=float(start_pose.pose.vec.x + dx * speed_mps * t_s),
                        y=float(start_pose.pose.vec.y + dy * speed_mps * t_s),
                        z=float(start_pose.pose.vec.z),
                    ),
                    quat=start_pose.pose.quat,
                ),
            )
        )
    return trajectory


def _headings_from_positions(positions_xy: np.ndarray) -> np.ndarray:
    prev = np.vstack((positions_xy[0:1], positions_xy[:-1]))
    deltas = positions_xy - prev
    return np.arctan2(deltas[:, 1], deltas[:, 0])
