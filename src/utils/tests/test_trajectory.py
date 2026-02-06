# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_utils.geometry import (
    Pose,
    Trajectory,
    pose_from_grpc,
    pose_to_grpc,
    trajectory_accelerations_cubic,
    trajectory_velocities_cubic,
    trajectory_yaw_rates_cubic,
)
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation

# =============================================================================
# Pose Fixtures
# =============================================================================


@pytest.fixture
def pose1() -> Pose:
    return Pose(
        np.array([1.75440159, 0.00758505, 0.01048692], dtype=np.float32),
        np.array(
            [1.09579759e-03, 6.86108488e-04, 1.21788308e-04, 9.99999157e-01],
            dtype=np.float32,
        ),
    )


@pytest.fixture
def pose2() -> Pose:
    return Pose(
        np.array([2.6517036, 0.01063591, -0.00657906], dtype=np.float32),
        np.array(
            [1.38583916e-04, 1.07102027e-03, 1.13202496e-04, 9.99999410e-01],
            dtype=np.float32,
        ),
    )


@pytest.fixture
def pose_offset_x() -> Pose:
    return Pose(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


# =============================================================================
# Pose Tests
# =============================================================================


def test_pose_inverse_lr(pose1: Pose):
    """Check if pose @ inv(pose) gives identity"""
    result = pose1 @ pose1.inverse()
    assert_almost_equal(result.vec3, np.zeros(3), decimal=5)
    assert_almost_equal(result.quat, np.array([0.0, 0.0, 0.0, 1.0]), decimal=5)


def test_pose_inverse_rl(pose1: Pose):
    """Check if inv(pose) @ pose gives identity"""
    result = pose1.inverse() @ pose1
    assert_almost_equal(result.vec3, np.zeros(3), decimal=5)
    assert_almost_equal(result.quat, np.array([0.0, 0.0, 0.0, 1.0]), decimal=5)


def test_pose_se3_roundtrip(pose1: Pose, pose2: Pose):
    """Check if pose multiplication is equivalent to matrix multiplication."""
    direct_mul = pose1 @ pose2
    via_se3 = Pose.from_se3(pose1.as_se3() @ pose2.as_se3())

    assert_almost_equal(direct_mul.vec3, via_se3.vec3, decimal=5)
    assert_almost_equal(direct_mul.quat, via_se3.quat, decimal=5)


def test_pose_grpc_roundtrip(pose1: Pose):
    """Check if pose->grpc and back results in unchanged pose"""
    grpc_pose = pose_to_grpc(pose1)
    returned = pose_from_grpc(grpc_pose)
    assert_almost_equal(pose1.vec3, returned.vec3, decimal=5)
    assert_almost_equal(pose1.quat, returned.quat, decimal=5)


def test_pose_identity():
    """Test identity pose construction"""
    identity = Pose.identity()
    assert_almost_equal(identity.vec3, np.zeros(3))
    assert_almost_equal(identity.quat, np.array([0.0, 0.0, 0.0, 1.0]))


def test_pose_yaw():
    """Test yaw extraction"""
    # 90 degree rotation around Z
    quat = Rotation.from_euler("z", np.pi / 2).as_quat().astype(np.float32)
    pose = Pose(np.zeros(3, dtype=np.float32), quat)
    assert pose.yaw() == pytest.approx(np.pi / 2, abs=1e-5)


# =============================================================================
# Trajectory Fixtures (using Pose)
# =============================================================================


@pytest.fixture
def traj_len_0():
    return Trajectory.create_empty()


@pytest.fixture
def traj_len_1(pose1: Pose):
    return Trajectory.from_poses(
        np.array([10], dtype=np.uint64),
        [pose1],
    )


@pytest.fixture
def traj_len_2(pose1: Pose, pose2: Pose):
    return Trajectory.from_poses(
        np.array([10, 20], dtype=np.uint64),
        [pose1, pose2],
    )


# =============================================================================
# Trajectory Interpolation Tests
# =============================================================================


def test_interpolation_len_0(traj_len_0: Trajectory):
    with pytest.raises(ValueError):
        traj_len_0.interpolate_pose(0)
    with pytest.raises(ValueError):
        traj_len_0.interpolate_pose(10)


def test_interpolation_len_1(traj_len_1: Trajectory, pose1: Pose):
    with pytest.raises(ValueError):
        traj_len_1.interpolate_pose(0)

    interpolated = traj_len_1.interpolate_pose(10)

    assert_almost_equal(interpolated.vec3, pose1.vec3, decimal=5)
    assert_almost_equal(interpolated.quat, pose1.quat, decimal=5)

    with pytest.raises(ValueError):
        traj_len_1.interpolate_pose(21)


def test_interpolation_len_2(traj_len_2: Trajectory, pose1: Pose, pose2: Pose):
    with pytest.raises(ValueError):
        traj_len_2.interpolate_pose(0)

    # should match the first pose
    interp_start = traj_len_2.interpolate_pose(10)
    assert_almost_equal(interp_start.vec3, pose1.vec3, decimal=5)
    assert_almost_equal(interp_start.quat, pose1.quat, decimal=5)

    # should match the second pose
    interp_end = traj_len_2.interpolate_pose(20)
    assert_almost_equal(interp_end.vec3, pose2.vec3, decimal=5)
    assert_almost_equal(interp_end.quat, pose2.quat, decimal=5)

    with pytest.raises(ValueError):
        traj_len_2.interpolate_pose(21)


# =============================================================================
# Trajectory Clip Tests
# =============================================================================


def test_clip_inside_range(traj_len_2: Trajectory):
    clipped = traj_len_2.clip(12, 18)
    assert clipped.time_range_us == range(12, 18)


def test_clip_overlapping_range_left(traj_len_2: Trajectory):
    clipped = traj_len_2.clip(5, 18)
    assert clipped.time_range_us == range(10, 18)


def test_clip_overlapping_range_right(traj_len_2: Trajectory):
    clipped = traj_len_2.clip(12, 22)
    assert clipped.time_range_us == range(12, 21)


def test_clip_outside_range(traj_len_2: Trajectory):
    clipped = traj_len_2.clip(25, 30)
    assert clipped.is_empty()


# =============================================================================
# Trajectory Transform Tests
# =============================================================================


def test_transform_offset(traj_len_2: Trajectory, pose_offset_x: Pose) -> None:
    last_pose_base = traj_len_2.last_pose
    last_pose_traj_transform = traj_len_2.transform(pose_offset_x).last_pose

    vec3_trans_manual = last_pose_base.vec3 + pose_offset_x.vec3
    vec3_trans_traj = last_pose_traj_transform.vec3

    assert np.isclose(vec3_trans_manual, vec3_trans_traj).all()
    assert np.isclose(
        last_pose_base.quat, last_pose_traj_transform.quat
    ).all()  # no difference here


def test_transform_general(traj_len_2: Trajectory, pose1: Pose) -> None:
    """Compares Trajectory.transform to Pose multiply"""
    last_pose_base = traj_len_2.last_pose
    last_pose_traj_transform = traj_len_2.transform(pose1).last_pose

    last_pose_manual = pose1 @ last_pose_base

    assert np.isclose(
        last_pose_traj_transform.vec3, last_pose_manual.vec3, atol=1e-5
    ).all()
    assert np.isclose(
        last_pose_traj_transform.quat, last_pose_manual.quat, atol=1e-5
    ).all()


def test_transform_relative() -> None:
    quat_unrotated = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat_rotated_half_pi = (
        Rotation.from_euler("xyz", np.array([0.0, 0.0, np.pi / 2.0]))
        .as_quat()
        .astype(np.float32)
    )

    transform = Pose(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        quat_unrotated,
    )

    pose_a = Pose(np.array([1.0, 0.0, 0.0], dtype=np.float32), quat_unrotated)
    pose_b = Pose(np.array([0.0, 1.0, 0.0], dtype=np.float32), quat_rotated_half_pi)

    traj = Trajectory.from_poses(
        np.array([0, 1], dtype=np.uint64),
        [pose_a, pose_b],
    )

    transformed = traj.transform(transform, is_relative=True)
    assert_almost_equal(transformed.positions[0], np.array([2.0, 0.0, 0.0]))
    assert_almost_equal(transformed.positions[1], np.array([0.0, 2.0, 0.0]))


# =============================================================================
# Trajectory Derivative Tests (using free functions)
# =============================================================================


def _make_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    """Helper to create a Pose from x, y, z position and yaw angle."""
    quat = (
        Rotation.from_euler("xyz", np.array([0.0, 0.0, yaw]))
        .as_quat()
        .astype(np.float32)
    )
    return Pose(np.array([x, y, z], dtype=np.float32), quat)


@pytest.fixture
def traj_len_5():
    return Trajectory.from_poses(
        np.array([1e6, 2e6, 3e6, 4e6, 5e6], dtype=np.uint64),
        [
            _make_pose(0.0, 0.0, 0.0, 0.0),
            _make_pose(1.0, 0.5, 0.0, np.pi * 1 / 8),
            _make_pose(2.0, 1.0, 0.0, np.pi * 2 / 8),
            _make_pose(3.0, 2.0, 0.0, np.pi * 3 / 8),
            _make_pose(4.0, 4.0, 0.0, np.pi * -1 / 8 + 2 * np.pi),
        ],
    )


@pytest.fixture
def traj_len_5_constant():
    return Trajectory.from_poses(
        np.array([1e6, 2e6, 3e6, 4e6, 5e6], dtype=np.uint64),
        [
            _make_pose(0.0, 1.0, 0.0, 0.0),
            _make_pose(0.0, 1.0, 0.0, 0.0),
            _make_pose(0.0, 1.0, 0.0, 0.0),
            _make_pose(0.0, 1.0, 0.0, 0.0),
            _make_pose(0.0, 1.0, 0.0, 0.0),
        ],
    )


def test_trajectory_velocities_centered(traj_len_5: Trajectory) -> None:
    velocities = traj_len_5.velocities("centered")
    assert velocities.shape == (5, 3)
    assert velocities == pytest.approx(
        np.array(
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 0.75, 0.0],
                [1.0, 1.5, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )
    )


def test_trajectory_accelerations_centered(traj_len_5: Trajectory) -> None:
    accelerations = traj_len_5.accelerations("centered")
    assert accelerations.shape == (5, 3)
    assert accelerations == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.125, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.625, 0.0],
                [0.0, 0.5, 0.0],
            ]
        ),
    )


def test_trajectory_jerk_centered(traj_len_5: Trajectory) -> None:
    jerk = traj_len_5.jerk("centered")
    assert jerk.shape == (5, 3)
    assert jerk == pytest.approx(
        np.array(
            [
                [0.0, 0.125, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -0.125, 0.0],
            ]
        ),
    )


def test_trajectory_yaw_rates_centered(traj_len_5: Trajectory) -> None:
    yaw_rates = traj_len_5.yaw_rates("centered")
    assert yaw_rates.shape == (5,)
    assert yaw_rates == pytest.approx(
        np.pi
        * np.array(
            [
                1 / 8,
                1 / 8,
                1 / 8,
                -1.5 / 8,
                -4 / 8,
            ]
        ),
    )


def test_trajectory_yaw_accelerations_centered(traj_len_5: Trajectory) -> None:
    yaw_accels = traj_len_5.yaw_accelerations("centered")
    assert yaw_accels.shape == (5,)
    # Note: abs tolerance needed due to f32 storage precision
    assert yaw_accels == pytest.approx(
        np.pi * np.array([0.0, 0.0, -1.25 / 8, -2.5 / 8, -2.5 / 8]),
        abs=1e-6,
    )


def test_trajectory_velocities_forward(traj_len_5: Trajectory) -> None:
    velocities = traj_len_5.velocities("forward")
    assert velocities.shape == (5, 3)
    assert velocities == pytest.approx(
        np.array(
            [
                [1.0, 0.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )
    )


def test_trajectory_yaw_rates_forward(traj_len_5: Trajectory) -> None:
    yaw_rates = traj_len_5.yaw_rates("forward")
    assert yaw_rates.shape == (5,)
    assert yaw_rates == pytest.approx(
        np.pi * np.array([1 / 8, 1 / 8, 1 / 8, -4 / 8, -4 / 8]),
    )


def test_trajectory_accelerations_forward(traj_len_5: Trajectory) -> None:
    accelerations = traj_len_5.accelerations("forward")
    assert accelerations.shape == (5, 3)
    assert accelerations == pytest.approx(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_trajectory_velocities_cubic(traj_len_5_constant: Trajectory) -> None:
    velocities = trajectory_velocities_cubic(traj_len_5_constant)
    assert velocities.shape == (5, 3)
    assert velocities == pytest.approx(np.zeros((5, 3)))


def test_trajectory_accelerations_cubic(traj_len_5_constant: Trajectory) -> None:
    accelerations = trajectory_accelerations_cubic(traj_len_5_constant)
    assert accelerations.shape == (5, 3)
    assert accelerations == pytest.approx(np.zeros((5, 3)))


def test_trajectory_yaw_rates_cubic(traj_len_5_constant: Trajectory) -> None:
    yaw_rates = trajectory_yaw_rates_cubic(traj_len_5_constant)
    assert yaw_rates.shape == (5,)
    assert yaw_rates == pytest.approx(np.zeros((5,)))


# =============================================================================
# Trajectory Append Tests
# =============================================================================


def test_trajectory_append_no_overlap(
    traj_len_2: Trajectory, pose1: Pose, pose2: Pose
) -> None:
    """Test concatenating two trajectories with no overlap."""
    second_traj = Trajectory.from_poses(
        np.array([30, 40], dtype=np.uint64),
        [pose1, pose2],
    )

    concatenated = traj_len_2.append(second_traj)

    assert len(concatenated) == 4
    assert_almost_equal(
        concatenated.timestamps_us, np.array([10, 20, 30, 40], dtype=np.uint64)
    )
    assert_almost_equal(concatenated.positions[0], pose1.vec3, decimal=5)
    assert_almost_equal(concatenated.positions[1], pose2.vec3, decimal=5)
    assert_almost_equal(concatenated.positions[2], pose1.vec3, decimal=5)
    assert_almost_equal(concatenated.positions[3], pose2.vec3, decimal=5)


def test_trajectory_append_overlap_fails_with_unequal_poses(
    traj_len_2: Trajectory, pose1: Pose, pose2: Pose
) -> None:
    """Test concatenating two trajectories with one overlapping timestamp but different poses."""
    # Create trajectory with same timestamps but different order (pose1, pose2 vs pose2, pose1)
    second_traj = Trajectory.from_poses(
        np.array([20, 30], dtype=np.uint64),
        [pose1, pose2],  # pose1 != pose2 from traj_len_2
    )
    with pytest.raises(ValueError):
        traj_len_2.append(second_traj)


def test_trajectory_append_overlap_succeeds(
    traj_len_2: Trajectory, pose1: Pose, pose2: Pose
) -> None:
    """Test concatenating two trajectories with one overlapping timestamp and matching pose."""
    # The overlap timestamp (20) should have pose2 from traj_len_2
    second_traj = Trajectory.from_poses(
        np.array([20, 30], dtype=np.uint64),
        [pose2, pose1],  # pose2 matches traj_len_2's last pose
    )

    concatenated = traj_len_2.append(second_traj)

    # Check length (should be 3, not 4, because of the overlap)
    assert len(concatenated) == 3

    assert_almost_equal(
        concatenated.timestamps_us, np.array([10, 20, 30], dtype=np.uint64)
    )
    assert_almost_equal(concatenated.positions[0], pose1.vec3, decimal=5)
    assert_almost_equal(concatenated.positions[1], pose2.vec3, decimal=5)
    assert_almost_equal(concatenated.positions[2], pose1.vec3, decimal=5)


def test_trajectory_append_empty(
    traj_len_2: Trajectory, traj_len_0: Trajectory
) -> None:
    """Test concatenating with an empty trajectory."""
    concatenated1 = traj_len_0.append(traj_len_2)
    assert len(concatenated1) == len(traj_len_2)
    assert_almost_equal(concatenated1.timestamps_us, traj_len_2.timestamps_us)

    concatenated2 = traj_len_2.append(traj_len_0)
    assert len(concatenated2) == len(traj_len_2)
    assert_almost_equal(concatenated2.timestamps_us, traj_len_2.timestamps_us)


# =============================================================================
# Trajectory set_pose Tests
# =============================================================================


def test_set_pose_positive_index(traj_len_2: Trajectory, pose2: Pose) -> None:
    """set_pose with a positive index replaces the correct pose."""
    new_pose = Pose.identity()
    traj_len_2.set_pose(0, new_pose)

    assert_almost_equal(traj_len_2.get_pose(0).vec3, np.zeros(3), decimal=5)
    assert_almost_equal(
        traj_len_2.get_pose(0).quat, np.array([0.0, 0.0, 0.0, 1.0]), decimal=5
    )
    # Second pose should be unchanged
    assert_almost_equal(traj_len_2.get_pose(1).vec3, pose2.vec3, decimal=5)


def test_set_pose_negative_index(traj_len_2: Trajectory, pose1: Pose) -> None:
    """set_pose with a negative index replaces from the end."""
    new_pose = Pose.identity()
    traj_len_2.set_pose(-1, new_pose)

    assert_almost_equal(traj_len_2.get_pose(-1).vec3, np.zeros(3), decimal=5)
    assert_almost_equal(
        traj_len_2.get_pose(-1).quat, np.array([0.0, 0.0, 0.0, 1.0]), decimal=5
    )
    # First pose should be unchanged
    assert_almost_equal(traj_len_2.get_pose(0).vec3, pose1.vec3, decimal=5)


def test_set_pose_single_element(traj_len_1: Trajectory) -> None:
    """set_pose works on a single-element trajectory."""
    new_pose = Pose(
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    traj_len_1.set_pose(0, new_pose)

    assert_almost_equal(traj_len_1.last_pose.vec3, new_pose.vec3, decimal=5)
    assert_almost_equal(traj_len_1.last_pose.quat, new_pose.quat, decimal=5)


def test_set_pose_out_of_bounds(traj_len_2: Trajectory) -> None:
    """set_pose raises IndexError for out-of-bounds indices."""
    dummy = Pose.identity()
    with pytest.raises(IndexError):
        traj_len_2.set_pose(2, dummy)
    with pytest.raises(IndexError):
        traj_len_2.set_pose(-3, dummy)


def test_set_pose_preserves_timestamps(traj_len_2: Trajectory) -> None:
    """set_pose does not alter the trajectory timestamps."""
    original_ts = traj_len_2.timestamps_us.copy()
    traj_len_2.set_pose(0, Pose.identity())
    assert_almost_equal(traj_len_2.timestamps_us, original_ts)
