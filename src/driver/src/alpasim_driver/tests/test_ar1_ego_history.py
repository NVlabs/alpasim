# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for AR1 ego history construction (build_ego_history logic)."""

from __future__ import annotations

import numpy as np
import torch
from alpasim_driver.models.ar1_model import build_ego_history


# ---------------------------------------------------------------------------
# Fake PoseAtTime helpers (mirror the gRPC proto shape)
# ---------------------------------------------------------------------------
class FakeVec:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class FakeQuat:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x, self.y, self.z, self.w = x, y, z, w


class FakePose:
    def __init__(self, vec: FakeVec, quat: FakeQuat) -> None:
        self.vec, self.quat = vec, quat


class FakePoseAtTime:
    def __init__(self, timestamp_us: int, pose: FakePose) -> None:
        self.timestamp_us, self.pose = timestamp_us, pose


# ---------------------------------------------------------------------------
# Constants matching AR1Model
# ---------------------------------------------------------------------------
NUM_HISTORY_STEPS = 16
HISTORY_TIME_STEP = 0.1  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_straight_line_poses(
    n: int = 20, dt_us: int = 100_000
) -> list[FakePoseAtTime]:
    """Create n poses along the x-axis, identity rotation, spaced dt_us apart."""
    poses = []
    for i in range(n):
        ts = i * dt_us
        pose = FakePose(
            vec=FakeVec(float(i), 0.0, 0.0),
            quat=FakeQuat(0.0, 0.0, 0.0, 1.0),  # identity
        )
        poses.append(FakePoseAtTime(ts, pose))
    return poses


def _quat_from_yaw(yaw_rad: float) -> FakeQuat:
    """Quaternion for a rotation about Z by *yaw_rad*."""
    return FakeQuat(
        x=0.0,
        y=0.0,
        z=float(np.sin(yaw_rad / 2)),
        w=float(np.cos(yaw_rad / 2)),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestStraightLineIdentityRotation:
    """Straight-line motion along X with identity quaternion."""

    def test_output_shapes(self) -> None:
        poses = _make_straight_line_poses(n=20)
        current_ts = poses[-1].timestamp_us
        xyz, rot = build_ego_history(poses, current_ts)

        assert xyz.shape == (1, 1, NUM_HISTORY_STEPS, 3)
        assert rot.shape == (1, 1, NUM_HISTORY_STEPS, 3, 3)

    def test_t0_position_is_origin(self) -> None:
        poses = _make_straight_line_poses(n=20)
        current_ts = poses[-1].timestamp_us
        xyz, _ = build_ego_history(poses, current_ts)

        # t0 is the last step — should be [0, 0, 0] in rig frame
        t0_xyz = xyz[0, 0, -1].numpy()
        np.testing.assert_allclose(t0_xyz, [0.0, 0.0, 0.0], atol=1e-5)

    def test_t0_rotation_is_identity(self) -> None:
        poses = _make_straight_line_poses(n=20)
        current_ts = poses[-1].timestamp_us
        _, rot = build_ego_history(poses, current_ts)

        # t0 rotation should be eye(3)
        t0_rot = rot[0, 0, -1].numpy()
        np.testing.assert_allclose(t0_rot, np.eye(3), atol=1e-5)

    def test_earlier_positions_are_behind(self) -> None:
        """In rig frame, past positions should have negative x (behind t0)."""
        poses = _make_straight_line_poses(n=20)
        current_ts = poses[-1].timestamp_us
        xyz, _ = build_ego_history(poses, current_ts)

        # All history steps before t0 should be behind (negative x in rig frame)
        for i in range(NUM_HISTORY_STEPS - 1):
            assert xyz[0, 0, i, 0].item() < xyz[0, 0, -1, 0].item() + 1e-6


class TestTurningPoses:
    """Gradual yaw rotation — checks rotation validity."""

    @staticmethod
    def _make_turning_poses(n: int = 20, dt_us: int = 100_000) -> list[FakePoseAtTime]:
        poses = []
        for i in range(n):
            yaw = float(i) * np.pi / (4 * n)  # gradual turn up to ~45 deg
            radius = 10.0
            pose = FakePose(
                vec=FakeVec(radius * np.sin(yaw), radius * (1 - np.cos(yaw)), 0.0),
                quat=_quat_from_yaw(yaw),
            )
            poses.append(FakePoseAtTime(i * dt_us, pose))
        return poses

    def test_output_shapes(self) -> None:
        poses = self._make_turning_poses()
        current_ts = poses[-1].timestamp_us
        xyz, rot = build_ego_history(poses, current_ts)

        assert xyz.shape == (1, 1, NUM_HISTORY_STEPS, 3)
        assert rot.shape == (1, 1, NUM_HISTORY_STEPS, 3, 3)

    def test_rotation_matrices_orthogonal(self) -> None:
        poses = self._make_turning_poses()
        current_ts = poses[-1].timestamp_us
        _, rot = build_ego_history(poses, current_ts)

        for t in range(NUM_HISTORY_STEPS):
            R = rot[0, 0, t].numpy()
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)


class TestAntipodalQuaternions:
    """Consecutive quaternions in different hemispheres (q and -q).

    This is the scenario that scipy Slerp can fail on without
    _adjust_orientation; utils_rs handles it natively.
    """

    @staticmethod
    def _make_antipodal_poses(
        n: int = 20, dt_us: int = 100_000
    ) -> list[FakePoseAtTime]:
        poses = []
        for i in range(n):
            yaw = float(i) * 0.01  # tiny rotation
            q = _quat_from_yaw(yaw)
            # Flip every other quaternion to opposite hemisphere
            if i % 2 == 1:
                q = FakeQuat(-q.x, -q.y, -q.z, -q.w)
            pose = FakePose(
                vec=FakeVec(float(i) * 0.5, 0.0, 0.0),
                quat=q,
            )
            poses.append(FakePoseAtTime(i * dt_us, pose))
        return poses

    def test_no_explosion(self) -> None:
        """Positions should stay finite and bounded."""
        poses = self._make_antipodal_poses()
        current_ts = poses[-1].timestamp_us
        xyz, rot = build_ego_history(poses, current_ts)

        assert torch.isfinite(xyz).all()
        assert torch.isfinite(rot).all()

    def test_rotation_matrices_valid(self) -> None:
        poses = self._make_antipodal_poses()
        current_ts = poses[-1].timestamp_us
        _, rot = build_ego_history(poses, current_ts)

        for t in range(NUM_HISTORY_STEPS):
            R = rot[0, 0, t].numpy()
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
            np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)

    def test_output_shapes(self) -> None:
        poses = self._make_antipodal_poses()
        current_ts = poses[-1].timestamp_us
        xyz, rot = build_ego_history(poses, current_ts)

        assert xyz.shape == (1, 1, NUM_HISTORY_STEPS, 3)
        assert rot.shape == (1, 1, NUM_HISTORY_STEPS, 3, 3)
