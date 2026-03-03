# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the utils_rs Rust extension module."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils_rs import Pose, Trajectory, version


class TestVersion:
    """Tests for module version."""

    def test_version_returns_string(self) -> None:
        v = version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_version_is_semver(self) -> None:
        v = version()
        parts = v.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])


def _pose(position: list, quaternion: list) -> Pose:
    """Helper to create a Pose from Python lists."""
    return Pose(
        np.array(position, dtype=np.float32),
        np.array(quaternion, dtype=np.float32),
    )


def _trajectory(
    timestamps: list[int],
    positions: list[list[float]],
    quaternions: list[list[float]] | None = None,
) -> Trajectory:
    """Helper to create a Trajectory from Python lists.

    Args:
        timestamps: List of timestamps in microseconds
        positions: List of [x, y, z] positions
        quaternions: List of [x, y, z, w] quaternions. If None, uses identity.
    """
    n = len(timestamps)
    if quaternions is None:
        quaternions = [[0.0, 0.0, 0.0, 1.0]] * n
    return Trajectory(
        np.array(timestamps, dtype=np.uint64),
        np.array(positions, dtype=np.float32),
        np.array(quaternions, dtype=np.float32),
    )


class TestPose:
    """Tests for the Rust Pose type."""

    def test_create_identity(self) -> None:
        p = Pose.identity()
        assert_allclose(p.vec3, [0.0, 0.0, 0.0])
        assert_allclose(p.quat, [0.0, 0.0, 0.0, 1.0])

    def test_create_from_arrays(self) -> None:
        p = _pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        assert_allclose(p.vec3, [1.0, 2.0, 3.0])
        assert_allclose(p.quat, [0.0, 0.0, 0.0, 1.0])

    def test_inverse(self) -> None:
        p = _pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        inv = p.inverse()
        assert_allclose(inv.vec3, [-1.0, -2.0, -3.0])

    def test_matmul_identity(self) -> None:
        p = _pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        result = p @ Pose.identity()
        assert_allclose(result.vec3, p.vec3)

    def test_matmul_inverse_gives_identity(self) -> None:
        p = _pose([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        result = p @ p.inverse()
        assert_allclose(result.vec3, [0.0, 0.0, 0.0], atol=1e-6)

    def test_yaw(self) -> None:
        # 90 degree rotation around Z
        yaw = math.pi / 2
        quat = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
        p = _pose([0.0, 0.0, 0.0], quat)
        assert_allclose(p.yaw(), yaw, atol=1e-6)

    def test_to_from_proto(self) -> None:
        # Use a unit quaternion with distinct components so "w first" is asserted
        quat_scipy = [0.1, 0.2, 0.3, 0.927374]  # x,y,z,w; normalized
        p = _pose([1.0, 2.0, 3.0], quat_scipy)
        pos, quat_proto = p.to_proto()
        # Proto quaternion is (w, x, y, z)
        assert_allclose(pos, [1.0, 2.0, 3.0])
        assert_allclose(quat_proto[0], 0.927374, atol=2e-5)  # w first (f32 round-trip)

        p2 = Pose.from_proto(
            np.array(pos, dtype=np.float32),
            np.array(quat_proto, dtype=np.float32),
        )
        assert_allclose(p2.vec3, p.vec3)
        assert_allclose(p2.quat, p.quat, atol=1e-6)  # f32 round-trip


class TestPoseDtypeHandling:
    """Tests for Pose dtype handling - accepts float32 and float64."""

    def test_accepts_float32(self) -> None:
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        p = Pose(pos, quat)
        assert_allclose(p.vec3, [1.0, 2.0, 3.0])
        assert_allclose(p.quat, [0.0, 0.0, 0.0, 1.0])

    def test_accepts_float64(self) -> None:
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        p = Pose(pos, quat)
        assert_allclose(p.vec3, [1.0, 2.0, 3.0])
        assert_allclose(p.quat, [0.0, 0.0, 0.0, 1.0])

    def test_accepts_mixed_dtypes(self) -> None:
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        p = Pose(pos, quat)
        assert_allclose(p.vec3, [1.0, 2.0, 3.0])
        assert_allclose(p.quat, [0.0, 0.0, 0.0, 1.0])

    def test_accepts_default_float_dtype(self) -> None:
        # np.array([...]) without dtype creates float64 by default
        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        assert pos.dtype == np.float64  # sanity check
        p = Pose(pos, quat)
        assert_allclose(p.vec3, [1.0, 2.0, 3.0])

    def test_accepts_float32_plus_float64_result(self) -> None:
        # This is the common case that was previously failing:
        # pose.vec3 (float32) + np.array([...]) (float64) -> float64
        base_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        offset = np.array([0.0, 1.0, 0.0])  # float64 by default
        combined = base_pos + offset  # promotes to float64
        assert combined.dtype == np.float64

        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        p = Pose(combined, quat)
        assert_allclose(p.vec3, [1.0, 3.0, 3.0])

    def test_rejects_int_dtype_with_clear_error(self) -> None:
        pos = np.array([1, 2, 3], dtype=np.int64)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        with pytest.raises(TypeError, match=r"position.*float32 or float64.*int64"):
            Pose(pos, quat)

    def test_rejects_int_quaternion_with_clear_error(self) -> None:
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        quat = np.array([0, 0, 0, 1], dtype=np.int32)
        with pytest.raises(TypeError, match=r"quaternion.*float32 or float64.*int32"):
            Pose(pos, quat)

    def test_wrong_position_length(self) -> None:
        pos = np.array([1.0, 2.0], dtype=np.float32)  # 2 elements instead of 3
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        with pytest.raises(ValueError, match=r"position must have 3 elements"):
            Pose(pos, quat)

    def test_wrong_quaternion_length(self) -> None:
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        quat = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # 3 elements instead of 4
        with pytest.raises(ValueError, match=r"quaternion must have 4 elements"):
            Pose(pos, quat)

    def test_from_proto_accepts_float64(self) -> None:
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z
        p = Pose.from_proto(pos, quat_wxyz)
        assert_allclose(p.vec3, [1.0, 2.0, 3.0])

    def test_from_proto_rejects_int_with_clear_error(self) -> None:
        pos = np.array([1, 2, 3], dtype=np.int64)
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(TypeError, match=r"position.*float32 or float64.*int64"):
            Pose.from_proto(pos, quat_wxyz)


class TestQuaternionNormalization:
    """Tests for quaternion unit-length validation and normalization."""

    def test_exact_unit_quaternion_accepted(self) -> None:
        """Quaternions that are exactly unit length are accepted."""
        p = _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        assert_allclose(p.quat, [0.0, 0.0, 0.0, 1.0])

    def test_slightly_off_quaternion_normalized(self) -> None:
        """Quaternions close to unit length are silently normalized."""
        # Use a quaternion with length² within 1e-4 of 1.0 (tolerance)
        q = np.array([0.0, 0.0, 0.0, 1.00005], dtype=np.float32)
        p = Pose(np.zeros(3, dtype=np.float32), q)
        result_norm = np.linalg.norm(p.quat)
        assert_allclose(result_norm, 1.0, atol=1e-6)

    def test_f64_to_f32_drift_handled(self) -> None:
        """Quaternions with small drift from f64->f32 conversion are normalized."""
        # Simulate a quaternion that's unit in f64 but slightly off in f32
        q_f64 = np.array([0.1, 0.2, 0.3, 0.0], dtype=np.float64)
        q_f64[3] = np.sqrt(1.0 - q_f64[0] ** 2 - q_f64[1] ** 2 - q_f64[2] ** 2)
        assert_allclose(np.linalg.norm(q_f64), 1.0, atol=1e-15)  # exact in f64

        # Convert to f32 (may introduce small drift)
        p = Pose(np.zeros(3, dtype=np.float32), q_f64.astype(np.float32))
        result_norm = np.linalg.norm(p.quat)
        assert_allclose(result_norm, 1.0, atol=1e-6)

    def test_far_from_unit_quaternion_rejected(self) -> None:
        """Quaternions far from unit length raise ValueError."""
        q = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)  # length ~0.548
        with pytest.raises(ValueError, match="not unit-length"):
            Pose(np.zeros(3, dtype=np.float32), q)

    def test_zero_quaternion_rejected(self) -> None:
        """Zero quaternions raise ValueError."""
        q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="near-zero"):
            Pose(np.zeros(3, dtype=np.float32), q)

    def test_near_zero_quaternion_rejected(self) -> None:
        """Near-zero quaternions raise ValueError."""
        q = np.array([1e-8, 1e-8, 1e-8, 1e-8], dtype=np.float32)
        with pytest.raises(ValueError, match="near-zero"):
            Pose(np.zeros(3, dtype=np.float32), q)

    def test_from_proto_validates_quaternion(self) -> None:
        """from_proto also validates unit quaternion."""
        pos = np.zeros(3, dtype=np.float32)
        q_wxyz = np.array([0.4, 0.1, 0.2, 0.3], dtype=np.float32)  # w,x,y,z - not unit
        with pytest.raises(ValueError, match="not unit-length"):
            Pose.from_proto(pos, q_wxyz)

    def test_trajectory_validates_quaternions(self) -> None:
        """Trajectory constructor validates quaternions."""
        ts = np.array([100, 200], dtype=np.uint64)
        pos = np.zeros((2, 3), dtype=np.float32)
        quats = np.array(
            [[0.0, 0.0, 0.0, 1.0], [0.1, 0.2, 0.3, 0.4]],  # second is not unit
            dtype=np.float32,
        )
        with pytest.raises(ValueError, match="not unit-length"):
            Trajectory(ts, pos, quats)

    def test_trajectory_normalizes_close_to_unit(self) -> None:
        """Trajectory normalizes quaternions that are close to unit."""
        ts = np.array([100], dtype=np.uint64)
        pos = np.zeros((1, 3), dtype=np.float32)
        quats = np.array([[0.0, 0.0, 0.0, 1.00005]], dtype=np.float32)
        traj = Trajectory(ts, pos, quats)
        result_norm = np.linalg.norm(traj.quaternions[0])
        assert_allclose(result_norm, 1.0, atol=1e-6)

    def test_from_denormalized_quat_normalizes(self) -> None:
        """from_denormalized_quat accepts non-unit quaternion and normalizes it."""
        pos = np.zeros(3, dtype=np.float32)
        q = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)  # not unit
        p = Pose.from_denormalized_quat(pos, q)
        assert_allclose(np.linalg.norm(p.quat), 1.0, atol=1e-6)

    def test_from_denormalized_quat_rejects_zero(self) -> None:
        """from_denormalized_quat rejects zero quaternion."""
        pos = np.zeros(3, dtype=np.float32)
        q = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="near-zero"):
            Pose.from_denormalized_quat(pos, q)


class TestTrajectoryTimestampDtype:
    """Tests that Trajectory rejects signed integer dtypes for timestamps."""

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_rejects_signed_timestamps(self, dtype: np.dtype) -> None:
        ts = np.array([100, 200, 300], dtype=dtype)
        pos = np.zeros((3, 3), dtype=np.float32)
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)).astype(np.float32)
        with pytest.raises(TypeError, match=r"signed dtype.*not allowed.*wrap"):
            Trajectory(ts, pos, quat)

    @pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
    def test_accepts_unsigned_timestamps(self, dtype: np.dtype) -> None:
        ts = np.array([100, 200, 300], dtype=dtype)
        pos = np.zeros((3, 3), dtype=np.float32)
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)).astype(np.float32)
        traj = Trajectory(ts, pos, quat)
        assert len(traj) == 3

    def test_error_message_suggests_astype(self) -> None:
        ts = np.array([100, 200], dtype=np.int64)
        pos = np.zeros((2, 3), dtype=np.float32)
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)).astype(np.float32)
        with pytest.raises(TypeError, match=r"astype\(np\.uint64\)"):
            Trajectory(ts, pos, quat)


class TestTrajectoryBasic:
    """Basic tests for Trajectory creation and properties."""

    def test_create_empty(self) -> None:
        traj = Trajectory.create_empty()
        assert len(traj) == 0
        assert traj.is_empty()
        assert traj.time_range_us == range(0, 0)

    def test_create_from_arrays(self) -> None:
        traj = _trajectory(
            [100, 200, 300],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        )

        assert len(traj) == 3
        assert not traj.is_empty()
        assert traj.time_range_us == range(100, 301)

    def test_timestamps_us_getter(self) -> None:
        traj = _trajectory(
            [100, 200, 300],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

        result = traj.timestamps_us
        assert_allclose(result, [100, 200, 300])
        assert result.dtype == np.uint64

    def test_positions_getter(self) -> None:
        traj = _trajectory(
            [100, 200],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )

        result = traj.positions
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        assert_allclose(result, expected)
        assert result.dtype == np.float32
        assert result.shape == (2, 3)

    def test_quaternions_getter(self) -> None:
        # First quat: distinct unit quaternion (x,y,z,w)
        quat0 = [0.1, 0.2, 0.3, 0.927374]
        traj = _trajectory(
            [100, 200],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [quat0, [0.0, 0.0, 0.0, 1.0]],
        )

        result = traj.quaternions
        expected = np.array([quat0, [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        assert_allclose(result, expected, atol=2e-5)  # f32 round-trip
        assert result.dtype == np.float32
        assert result.shape == (2, 4)

    def test_invalid_positions_shape(self) -> None:
        timestamps = np.array([100, 200], dtype=np.uint64)
        positions = np.zeros((3, 3), dtype=np.float32)  # Should be (2, 3)
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32)

        with pytest.raises(ValueError, match="positions"):
            Trajectory(timestamps, positions, quaternions)

    def test_invalid_quaternions_shape(self) -> None:
        timestamps = np.array([100, 200], dtype=np.uint64)
        positions = np.zeros((2, 3), dtype=np.float32)
        quaternions = np.zeros((2, 3), dtype=np.float32)  # Should be (2, 4)

        with pytest.raises(ValueError, match="quaternions must have shape"):
            Trajectory(timestamps, positions, quaternions)


def _pose_xyz(
    x: float,
    y: float,
    z: float,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
) -> Pose:
    """Helper to create a Pose from position and quaternion components."""
    return Pose(
        np.array([x, y, z], dtype=np.float32),
        np.array([qx, qy, qz, qw], dtype=np.float32),
    )


class TestUpdateAbsolute:
    """Tests for update_absolute method."""

    def test_update_empty_trajectory(self) -> None:
        traj = Trajectory.create_empty()
        traj.update_absolute(100, _pose_xyz(1.0, 2.0, 3.0))

        assert len(traj) == 1
        assert_allclose(traj.positions[0], [1.0, 2.0, 3.0])

    def test_update_multiple(self) -> None:
        traj = Trajectory.create_empty()
        traj.update_absolute(100, _pose_xyz(1.0, 0.0, 0.0))
        traj.update_absolute(200, _pose_xyz(2.0, 0.0, 0.0))
        traj.update_absolute(300, _pose_xyz(3.0, 0.0, 0.0))

        assert len(traj) == 3
        assert_allclose(traj.positions[:, 0], [1.0, 2.0, 3.0])

    def test_update_requires_increasing_timestamp(self) -> None:
        traj = Trajectory.create_empty()
        traj.update_absolute(200, _pose_xyz(1.0, 0.0, 0.0))

        with pytest.raises(ValueError, match="must be greater than"):
            traj.update_absolute(100, _pose_xyz(2.0, 0.0, 0.0))

    def test_update_rejects_equal_timestamp(self) -> None:
        traj = Trajectory.create_empty()
        traj.update_absolute(100, _pose_xyz(1.0, 0.0, 0.0))

        with pytest.raises(ValueError, match="must be greater than"):
            traj.update_absolute(100, _pose_xyz(2.0, 0.0, 0.0))

    def test_last_pose(self) -> None:
        traj = Trajectory.create_empty()

        traj.update_absolute(100, _pose_xyz(1.0, 2.0, 3.0))
        last = traj.get_pose(-1)
        assert_allclose(last.vec3, [1.0, 2.0, 3.0])

        traj.update_absolute(200, _pose_xyz(4.0, 5.0, 6.0))
        last = traj.get_pose(-1)
        assert_allclose(last.vec3, [4.0, 5.0, 6.0])

    def test_first_pose(self) -> None:
        traj = Trajectory.create_empty()

        traj.update_absolute(100, _pose_xyz(1.0, 2.0, 3.0))
        first = traj.get_pose(0)
        assert_allclose(first.vec3, [1.0, 2.0, 3.0])

        traj.update_absolute(200, _pose_xyz(4.0, 5.0, 6.0))
        first = traj.get_pose(0)
        # First should still be the same
        assert_allclose(first.vec3, [1.0, 2.0, 3.0])


class TestUpdateRelative:
    """Tests for update_relative method."""

    def test_update_relative_no_rotation(self) -> None:
        """Moving forward without rotation."""
        traj = Trajectory.create_empty()
        traj.update_absolute(100, _pose_xyz(0.0, 0.0, 0.0))
        traj.update_relative(200, _pose_xyz(1.0, 0.0, 0.0))

        last = traj.get_pose(-1)
        assert_allclose(last.vec3, [1.0, 0.0, 0.0])

    def test_update_relative_with_rotation(self) -> None:
        """Moving forward after 90-degree yaw rotation."""
        traj = Trajectory.create_empty()

        # Start at origin with 90-degree rotation around Z
        yaw = math.pi / 2
        traj.update_absolute(
            100,
            _pose_xyz(0.0, 0.0, 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
        )

        # Move 1 unit in local X (should become global Y)
        traj.update_relative(200, _pose_xyz(1.0, 0.0, 0.0))

        pos = traj.get_pose(-1).vec3
        assert_allclose(pos, [0.0, 1.0, 0.0], atol=1e-6)

    def test_update_relative_fails_on_empty(self) -> None:
        traj = Trajectory.create_empty()

        with pytest.raises(ValueError, match="empty trajectory"):
            traj.update_relative(100, _pose_xyz(1.0, 0.0, 0.0))

    def test_update_relative_chain(self) -> None:
        """Chain of relative updates."""
        traj = Trajectory.create_empty()
        traj.update_absolute(0, _pose_xyz(0.0, 0.0, 0.0))

        for i in range(5):
            traj.update_relative((i + 1) * 100, _pose_xyz(1.0, 0.0, 0.0))

        assert len(traj) == 6
        assert_allclose(traj.get_pose(-1).vec3, [5.0, 0.0, 0.0])


class TestInterpolation:
    """Tests for interpolate method."""

    @pytest.fixture
    def linear_trajectory(self) -> Trajectory:
        """Simple linear trajectory from 0 to 20 in X."""
        return _trajectory(
            [0, 100, 200],
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        )

    def test_interpolate_at_keyframes(self, linear_trajectory: Trajectory) -> None:
        targets = np.array([0, 100, 200], dtype=np.uint64)
        interp = linear_trajectory.interpolate(targets)

        assert_allclose(interp.positions[:, 0], [0.0, 10.0, 20.0])

    def test_interpolate_at_midpoints(self, linear_trajectory: Trajectory) -> None:
        targets = np.array([50, 150], dtype=np.uint64)
        interp = linear_trajectory.interpolate(targets)

        assert_allclose(interp.positions[:, 0], [5.0, 15.0])

    def test_interpolate_arbitrary_points(self, linear_trajectory: Trajectory) -> None:
        targets = np.array([25, 75, 125, 175], dtype=np.uint64)
        interp = linear_trajectory.interpolate(targets)

        assert_allclose(interp.positions[:, 0], [2.5, 7.5, 12.5, 17.5])

    def test_interpolate_single_point(self, linear_trajectory: Trajectory) -> None:
        targets = np.array([50], dtype=np.uint64)
        interp = linear_trajectory.interpolate(targets)

        assert interp.positions.shape == (1, 3)
        assert interp.quaternions.shape == (1, 4)
        assert_allclose(interp.positions[0, 0], 5.0)

    def test_interpolate_out_of_range_low(self, linear_trajectory: Trajectory) -> None:
        targets = np.array([0], dtype=np.uint64)  # At start is OK
        interp = linear_trajectory.interpolate(targets)
        assert_allclose(interp.positions[0, 0], 0.0)

        # But past end is not OK
        with pytest.raises(ValueError, match="outside range"):
            targets = np.array([201], dtype=np.uint64)
            linear_trajectory.interpolate(targets)

    def test_interpolate_out_of_range_high(self, linear_trajectory: Trajectory) -> None:
        targets = np.array([201], dtype=np.uint64)

        with pytest.raises(ValueError, match="outside range"):
            linear_trajectory.interpolate(targets)

    def test_interpolate_empty_trajectory(self) -> None:
        traj = Trajectory.create_empty()
        targets = np.array([100], dtype=np.uint64)

        with pytest.raises(ValueError, match="empty trajectory"):
            traj.interpolate(targets)

    def test_interpolate_output_dtype(self, linear_trajectory: Trajectory) -> None:
        targets = np.array([50], dtype=np.uint64)
        interp = linear_trajectory.interpolate(targets)

        assert interp.positions.dtype == np.float32
        assert interp.quaternions.dtype == np.float32

    def test_interpolate_returns_trajectory(
        self, linear_trajectory: Trajectory
    ) -> None:
        targets = np.array([50, 150], dtype=np.uint64)
        interp = linear_trajectory.interpolate(targets)

        assert isinstance(interp, Trajectory)
        assert len(interp) == 2
        assert_allclose(interp.timestamps_us, targets)


class TestQuaternionSlerp:
    """Tests for quaternion spherical interpolation."""

    def test_slerp_matches_scipy(self) -> None:
        """Verify Rust slerp matches scipy Slerp."""
        # Create rotation from 0 to 90 degrees around Z
        r0 = R.from_euler("z", 0, degrees=True)
        r1 = R.from_euler("z", 90, degrees=True)

        q0 = r0.as_quat().astype(np.float32)
        q1 = r1.as_quat().astype(np.float32)

        traj = _trajectory(
            [0, 100],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [q0.tolist(), q1.tolist()],
        )

        # Interpolate at midpoint
        targets = np.array([50], dtype=np.uint64)
        interp = traj.interpolate(targets)

        # Compare with scipy
        scipy_slerp = Slerp(
            [0, 100], R.from_quat(np.array([q0, q1]).astype(np.float64))
        )
        scipy_quat = scipy_slerp(50).as_quat()

        assert_allclose(interp.quaternions[0], scipy_quat, atol=1e-5)

    def test_slerp_45_degrees(self) -> None:
        """Interpolate to 45 degrees between 0 and 90."""
        r0 = R.from_euler("z", 0, degrees=True)
        r1 = R.from_euler("z", 90, degrees=True)

        q0 = r0.as_quat().astype(np.float32)
        q1 = r1.as_quat().astype(np.float32)

        traj = _trajectory(
            [0, 100],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [q0.tolist(), q1.tolist()],
        )

        targets = np.array([50], dtype=np.uint64)
        interp = traj.interpolate(targets)

        # Convert to Euler and check
        euler = R.from_quat(interp.quaternions[0]).as_euler("xyz", degrees=True)
        assert_allclose(euler[2], 45.0, atol=0.1)

    def test_slerp_multiple_rotations(self) -> None:
        """Test slerp with multiple rotation keyframes."""
        rotations = [
            R.from_euler("z", 0, degrees=True),
            R.from_euler("z", 90, degrees=True),
            R.from_euler("z", 180, degrees=True),
        ]

        quats = [r.as_quat().astype(np.float32).tolist() for r in rotations]

        traj = _trajectory(
            [0, 100, 200],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            quats,
        )

        # Interpolate at 25%, 50%, 75%
        targets = np.array([50, 100, 150], dtype=np.uint64)
        interp = traj.interpolate(targets)

        eulers = [
            R.from_quat(q).as_euler("xyz", degrees=True)[2] for q in interp.quaternions
        ]
        assert_allclose(eulers, [45.0, 90.0, 135.0], atol=0.1)


class TestSinglePoseTrajectory:
    """Tests for edge case of single-pose trajectory."""

    def test_single_pose_interpolate_at_same_time(self) -> None:
        traj = _trajectory([100], [[5.0, 6.0, 7.0]])

        targets = np.array([100], dtype=np.uint64)
        interp = traj.interpolate(targets)

        assert_allclose(interp.positions[0], [5.0, 6.0, 7.0])

    def test_single_pose_interpolate_wrong_time(self) -> None:
        traj = _trajectory([100], [[5.0, 6.0, 7.0]])

        targets = np.array([50], dtype=np.uint64)

        with pytest.raises(ValueError, match="outside range"):
            traj.interpolate(targets)


class TestTrajectoryTransform:
    """Tests for transform method."""

    def test_transform_left_multiply(self) -> None:
        """Test left multiplication (transform @ poses)."""
        traj = _trajectory(
            [100, 200],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )

        # Translate by (10, 0, 0)
        transform = _pose([10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        transformed = traj.transform(transform, False)

        assert_allclose(transformed.positions[:, 0], [10.0, 11.0])

    def test_transform_right_multiply(self) -> None:
        """Test right multiplication (poses @ transform)."""
        # 90 degree rotation around Z
        yaw = math.pi / 2
        quat_90z = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]

        traj = _trajectory(
            [100],
            [[0.0, 0.0, 0.0]],
            [quat_90z],
        )

        # Move 1 unit in local X (should become global Y after rotation)
        transform = _pose([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        transformed = traj.transform(transform, True)

        assert_allclose(transformed.positions[0], [0.0, 1.0, 0.0], atol=1e-6)


class TestTrajectorySliceFilter:
    """Tests for slice and filter methods."""

    def test_slice(self) -> None:
        traj = _trajectory(
            [100, 200, 300, 400],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        )

        sliced = traj.slice(1, 3)
        assert len(sliced) == 2
        assert_allclose(sliced.positions[:, 0], [2.0, 3.0])

    def test_filter(self) -> None:
        traj = _trajectory(
            [100, 200, 300, 400],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        )

        mask = np.array([True, False, True, False])
        filtered = traj.filter(mask)
        assert len(filtered) == 2
        assert_allclose(filtered.positions[:, 0], [1.0, 3.0])

    def test_get_pose(self) -> None:
        traj = _trajectory(
            [100, 200, 300],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        )

        p0 = traj.get_pose(0)
        assert_allclose(p0.vec3, [1.0, 0.0, 0.0])

        p_last = traj.get_pose(-1)
        assert_allclose(p_last.vec3, [3.0, 0.0, 0.0])
