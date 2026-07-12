# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for gRPC-boundary helpers in the utils_rs extension."""

import numpy as np
import pytest

from utils_rs import (
    build_drive_response_bytes,
    pack_route_array,
    pack_trajectory_arrays,
)


def _require_generated_common_pb2() -> object:
    """Import generated common_pb2, skipping when protos have not been compiled."""
    common_pb2 = pytest.importorskip("alpasim_grpc.v0.common_pb2")
    if not hasattr(common_pb2.Trajectory().poses, "add"):
        pytest.skip("requires generated alpasim protobufs")
    return common_pb2


def _require_generated_egodriver_pb2() -> object:
    """Import generated egodriver_pb2, skipping when protos have not been compiled."""
    egodriver_pb2 = pytest.importorskip("alpasim_grpc.v0.egodriver_pb2")
    if not hasattr(egodriver_pb2.DriveResponse(), "ParseFromString"):
        pytest.skip("requires generated alpasim protobufs")
    return egodriver_pb2


def test_build_drive_response_bytes_rejects_shape_mismatch() -> None:
    """The native encoder rejects arrays that cannot map to trajectory rows."""
    xyz = np.zeros((2, 3), dtype=np.float32)
    quat_wxyz = np.zeros((3, 4), dtype=np.float32)
    dt_us = np.zeros((2,), dtype=np.int64)

    with pytest.raises(ValueError, match="mismatched leading dims"):
        build_drive_response_bytes(1_000, xyz, quat_wxyz, dt_us)


def test_build_drive_response_bytes_matches_empty_generated_proto() -> None:
    """An empty horizon leaves the trajectory field unset."""
    egodriver_pb2 = _require_generated_egodriver_pb2()

    payload = build_drive_response_bytes(
        1_000,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )

    assert payload == egodriver_pb2.DriveResponse().SerializeToString(
        deterministic=True
    )


def test_build_drive_response_bytes_matches_generated_proto_serialization() -> None:
    """Native bytes match generated protobuf serialization exactly."""
    common_pb2 = _require_generated_common_pb2()
    egodriver_pb2 = _require_generated_egodriver_pb2()
    xyz = np.array(
        [
            [1.25, 0.0, -3.5],
            [-0.0, 5.0, 6.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    quat_wxyz = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    dt_us = np.array([0, 100_000, 200_000], dtype=np.int64)

    payload = build_drive_response_bytes(0, xyz, quat_wxyz, dt_us)
    expected = egodriver_pb2.DriveResponse()
    for dt_us_value, xyz_row, quat_row in zip(dt_us, xyz, quat_wxyz, strict=True):
        pose_at_time = common_pb2.PoseAtTime(timestamp_us=int(dt_us_value))
        pose_at_time.pose.vec.x = float(xyz_row[0])
        pose_at_time.pose.vec.y = float(xyz_row[1])
        pose_at_time.pose.vec.z = float(xyz_row[2])
        pose_at_time.pose.quat.w = float(quat_row[0])
        pose_at_time.pose.quat.x = float(quat_row[1])
        pose_at_time.pose.quat.y = float(quat_row[2])
        pose_at_time.pose.quat.z = float(quat_row[3])
        expected.trajectory.poses.append(pose_at_time)

    assert payload == expected.SerializeToString(deterministic=True)


def test_pack_trajectory_arrays_sorts_and_preserves_pose_fields() -> None:
    """Trajectory packing returns timestamp-sorted numeric arrays."""
    common_pb2 = _require_generated_common_pb2()
    trajectory = common_pb2.Trajectory()
    for timestamp_us, x in ((30, 3.0), (10, 1.0), (20, 2.0)):
        pose_at_time = trajectory.poses.add()
        pose_at_time.timestamp_us = timestamp_us
        pose_at_time.pose.vec.x = x
        pose_at_time.pose.vec.y = x + 0.5
        pose_at_time.pose.vec.z = x + 1.0
        pose_at_time.pose.quat.w = 1.0
        pose_at_time.pose.quat.x = x * 0.1

    timestamps_us, xyz, quat_wxyz = pack_trajectory_arrays(trajectory)

    np.testing.assert_array_equal(timestamps_us, np.array([10, 20, 30], dtype=np.int64))
    np.testing.assert_allclose(
        xyz,
        np.array(
            [
                [1.0, 1.5, 2.0],
                [2.0, 2.5, 3.0],
                [3.0, 3.5, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        quat_wxyz,
        np.array(
            [
                [1.0, 0.1, 0.0, 0.0],
                [1.0, 0.2, 0.0, 0.0],
                [1.0, 0.3, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_pack_route_array_preserves_timestamp_and_waypoints() -> None:
    """Route packing returns the route timestamp plus waypoint xyz rows."""
    egodriver_pb2 = _require_generated_egodriver_pb2()
    route = egodriver_pb2.Route(timestamp_us=123)
    for x, y, z in ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)):
        waypoint = route.waypoints.add()
        waypoint.x = x
        waypoint.y = y
        waypoint.z = z

    timestamp_us, waypoints_xyz = pack_route_array(route)

    assert timestamp_us == 123
    np.testing.assert_allclose(
        waypoints_xyz,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
