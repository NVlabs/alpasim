# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Geometry primitives: Pose, Polyline, Trajectory and gRPC conversions.

This module consolidates spatial types and their gRPC conversion utilities.
Core types (Pose, Polyline, Trajectory) are re-exported from the Rust utils_rs
extension.

Quaternions use scipy convention (x, y, z, w) internally.
"""

from __future__ import annotations

from typing import NamedTuple

try:
    import csaps
except ImportError:
    csaps = None

import numpy as np
from alpasim_grpc.v0 import common_pb2 as grpc_types
from alpasim_grpc.v0 import egodriver_pb2 as ego_grpc

# Re-export Rust types directly - no Python wrappers needed
from utils_rs import Polyline, Pose, Trajectory

__all__ = [
    # Pose
    "Pose",
    "pose_to_grpc",
    "pose_from_grpc",
    "pose_to_grpc_at_time",
    # Polyline
    "Polyline",
    "ProjectionResult",
    "polyline_from_grpc",
    "polyline_to_grpc_route",
    # Trajectory
    "Trajectory",
    "trajectory_from_grpc",
    "trajectory_to_grpc",
    "trajectory_velocities_cubic",
    "trajectory_accelerations_cubic",
    "trajectory_yaw_rates_cubic",
]


# =============================================================================
# Pose
# =============================================================================


def pose_to_grpc(pose: Pose) -> grpc_types.Pose:
    """Convert a Pose to a gRPC Pose message."""
    pos, quat = pose.to_proto()
    return grpc_types.Pose(
        vec=grpc_types.Vec3(x=pos[0], y=pos[1], z=pos[2]),
        quat=grpc_types.Quat(w=quat[0], x=quat[1], y=quat[2], z=quat[3]),
    )


def pose_from_grpc(grpc_pose: grpc_types.Pose) -> Pose:
    """Create a Pose from a gRPC Pose message."""
    q = grpc_pose.quat
    v = grpc_pose.vec
    quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
    return Pose.from_proto(
        np.array([v.x, v.y, v.z], dtype=np.float32),
        quat_wxyz,
    )


def pose_to_grpc_at_time(pose: Pose, timestamp_us: int) -> grpc_types.PoseAtTime:
    """Convert a Pose to a gRPC PoseAtTime message."""
    return grpc_types.PoseAtTime(
        pose=pose_to_grpc(pose),
        timestamp_us=timestamp_us,
    )


# =============================================================================
# Polyline
# =============================================================================


class ProjectionResult(NamedTuple):
    """Result of projecting a point onto a polyline."""

    point: np.ndarray
    segment_idx: int
    distance_along: float


def polyline_from_grpc(grpc_route: ego_grpc.Route) -> Polyline:
    """Construct a Polyline from a gRPC route message."""
    waypoints = [[wp.x, wp.y, wp.z] for wp in grpc_route.waypoints]

    if not waypoints:
        return Polyline.create_empty(dimension=3)

    return Polyline(points=np.array(waypoints, dtype=float))


def polyline_to_grpc_route(polyline: Polyline, timestamp_us: int) -> ego_grpc.Route:
    """Convert a Polyline to a gRPC route message (3D only).

    Raises:
        ValueError: If the polyline is not 3D
    """
    if polyline.dimension != 3:
        raise ValueError("polyline_to_grpc_route is only defined for 3D polylines")

    route = ego_grpc.Route(timestamp_us=timestamp_us)
    for wp in polyline.waypoints:
        route.waypoints.append(
            grpc_types.Vec3(x=float(wp[0]), y=float(wp[1]), z=float(wp[2]))
        )
    return route


# =============================================================================
# Trajectory
# =============================================================================


def trajectory_from_grpc(trajectory: grpc_types.Trajectory) -> Trajectory:
    """Create a Trajectory from a gRPC Trajectory message."""
    if len(trajectory.poses) == 0:
        return Trajectory.create_empty()

    timestamps_us = np.array(
        [p.timestamp_us for p in trajectory.poses], dtype=np.uint64
    )
    poses = [pose_from_grpc(p.pose) for p in trajectory.poses]

    return Trajectory.from_poses(timestamps_us, poses)


def trajectory_to_grpc(trajectory: Trajectory) -> grpc_types.Trajectory:
    """Convert a Trajectory to a gRPC Trajectory message."""
    timestamps = trajectory.timestamps_us
    poses_at_time = []
    for i, ts in enumerate(timestamps):
        pose = trajectory.get_pose(i)
        poses_at_time.append(
            grpc_types.PoseAtTime(
                timestamp_us=int(ts),
                pose=pose_to_grpc(pose),
            )
        )
    return grpc_types.Trajectory(poses=poses_at_time)


# =============================================================================
# Cubic Spline Derivatives (require csaps - not available in Rust)
# =============================================================================


def _cubic_spline_approximation(
    arr: np.ndarray,
    timestamps_us: np.ndarray,
    deriv: int = 1,
    smoothing_factor: float | None = None,
) -> np.ndarray:
    """Computes derivatives using a cubic spline approximation.

    Args:
        arr: [N, D] array of values to differentiate
        timestamps_us: [N] array of time steps in microseconds
        deriv: Order of derivative to compute
        smoothing_factor: Smoothing factor for the cubic spline.
    """
    assert arr.ndim <= 2
    assert arr.shape[0] == timestamps_us.shape[0]
    if csaps is None:
        raise ImportError(
            "csaps is not installed. Please install csaps to use cubic spline approximation."
        )

    if arr.ndim == 1:
        arr = arr[..., None]

    css = csaps.CubicSmoothingSpline(
        timestamps_us / 1e6,
        np.moveaxis(arr, 0, 1),
        normalizedsmooth=True,
        smooth=smoothing_factor,
    )

    return np.moveaxis(css(timestamps_us / 1e6, nu=deriv), 0, 1).squeeze()


def trajectory_velocities_cubic(
    trajectory: Trajectory,
    smoothing_factor: float | None = None,
) -> np.ndarray:
    """Returns velocities in m/s using cubic spline approximation."""
    return _cubic_spline_approximation(
        trajectory.positions,
        trajectory.timestamps_us,
        deriv=1,
        smoothing_factor=smoothing_factor,
    )


def trajectory_accelerations_cubic(
    trajectory: Trajectory,
    smoothing_factor: float | None = None,
) -> np.ndarray:
    """Returns accelerations in m/s^2 using cubic spline approximation."""
    return _cubic_spline_approximation(
        trajectory.positions,
        trajectory.timestamps_us,
        deriv=2,
        smoothing_factor=smoothing_factor,
    )


def trajectory_yaw_rates_cubic(
    trajectory: Trajectory,
    smoothing_factor: float | None = None,
) -> np.ndarray:
    """Returns yaw rates in rad/s using cubic spline approximation."""
    return _cubic_spline_approximation(
        np.unwrap(trajectory.yaws),
        trajectory.timestamps_us,
        deriv=1,
        smoothing_factor=smoothing_factor,
    )
