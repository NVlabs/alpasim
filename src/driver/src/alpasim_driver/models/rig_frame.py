# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Re-expressing driver inputs in the rig frame at a model's t0."""

from __future__ import annotations

import numpy as np
from alpasim_grpc.v0.common_pb2 import PoseAtTime
from alpasim_grpc.v0.egodriver_pb2 import Route
from alpasim_utils.geometry import Pose, Trajectory


def trajectory_from_pose_messages(poses: list[PoseAtTime]) -> Trajectory:
    """Convert pose messages into a trajectory, carrying their frame along.

    Args:
        poses: List of PoseAtTime messages.  Each must have .timestamp_us (int),
            .pose.vec.{x,y,z} and .pose.quat.{x,y,z,w}, with a unit-length
            quaternion (validated by the Trajectory constructor).

    Returns:
        The poses as a trajectory, ordered by timestamp, in whichever frame the
        messages were expressed in.
    """
    pose_data = sorted([(p.timestamp_us, p.pose) for p in poses], key=lambda x: x[0])
    return Trajectory(
        np.array([t for t, _ in pose_data], dtype=np.uint64),
        np.array([[p.vec.x, p.vec.y, p.vec.z] for _, p in pose_data], dtype=np.float32),
        np.array(
            [[p.quat.x, p.quat.y, p.quat.z, p.quat.w] for _, p in pose_data],
            dtype=np.float32,
        ),
    )


def route_in_rig_at_t0(
    route: Route,
    ego_pose_history: list[PoseAtTime],
    pose_local_to_rig_t0: Pose,
) -> np.ndarray:
    """Re-express route waypoints in the rig frame at t0.

    A route carries the rig frame of its own timestamp, which trails t0 (the
    newest camera frame the model conditions on) whenever routes and camera
    frames arrive on different schedules.  Hopping through the local frame takes
    the ego motion in between back out.

    Args:
        route: Route with waypoints in the rig frame at ``route.timestamp_us``.
            NaN waypoints, which pad a route shorter than the requested
            lookahead, stay NaN.
        ego_pose_history: Rig poses in the local frame (``pose_local_to_rig`` at
            each ``timestamp_us``), spanning the route timestamp.
        pose_local_to_rig_t0: Pose of the rig at t0 in the local frame.

    Returns:
        Waypoints of shape ``(N, 3)`` in the rig frame at t0.
    """
    waypoints_in_rig_route = np.array(
        [[w.x, w.y, w.z] for w in route.waypoints], dtype=np.float32
    ).reshape(-1, 3)
    rig_route_in_local = trajectory_from_pose_messages(
        ego_pose_history
    ).interpolate_pose(np.uint64(route.timestamp_us))
    rig_route_in_rig_t0 = (
        pose_local_to_rig_t0.inverse().as_se3() @ rig_route_in_local.as_se3()
    )
    return (
        waypoints_in_rig_route @ rig_route_in_rig_t0[:3, :3].T
        + rig_route_in_rig_t0[:3, 3]
    )
