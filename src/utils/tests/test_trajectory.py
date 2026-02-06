# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation


@pytest.fixture
def qv1():
    return QVec(
        vec3=np.array([1.75440159, 0.00758505, 0.01048692]),
        quat=np.array([1.09579759e-03, 6.86108488e-04, 1.21788308e-04, 9.99999157e-01]),
    )


@pytest.fixture
def qv2():
    return QVec(
        vec3=np.array([2.6517036, 0.01063591, -0.00657906]),
        quat=np.array([1.38583916e-04, 1.07102027e-03, 1.13202496e-04, 9.99999410e-01]),
    )


def test_qv_inverse_lr(qv1: QVec):
    """
    Check if qvec @ inv(qvec) gives identity
    """
    id = qv1 @ qv1.inverse()
    assert_almost_equal(id.vec3, np.zeros(3))
    assert_almost_equal(id.quat, np.array([0.0, 0.0, 0.0, 1.0]))


def test_qv_inverse_rl(qv1: QVec):
    """
    check if inv(qvec) @ qvec gives identity
    """
    id = qv1.inverse() @ qv1
    assert_almost_equal(id.vec3, np.zeros(3))
    assert_almost_equal(id.quat, np.array([0.0, 0.0, 0.0, 1.0]))


def test_equivalent_se3_mul(qv1: QVec, qv2: QVec):
    """
    Check if qvec multiplication is equivalent to matrix multiplication
    """
    direct_mul = qv1 @ qv2
    via_se3 = QVec.from_se3(qv1.as_se3() @ qv2.as_se3())

    assert_almost_equal(direct_mul.vec3, via_se3.vec3)
    assert_almost_equal(direct_mul.quat, direct_mul.quat)


def test_grpc_cycle(qv1: QVec):
    """
    Check if qvec->grpc and back results in unchanged qvec
    """
    return_qv = QVec.from_grpc_pose(qv1.as_grpc_pose())
    assert_almost_equal(qv1.vec3, return_qv.vec3)
    assert_almost_equal(qv1.quat, return_qv.quat)


def test_append_qvec_empty_batch_size(qv1: QVec) -> None:
    appended = qv1.append(
        QVec(
            vec3=np.array([1.0, 2.0, 3.0]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    )
    assert appended.batch_size == (2,)
    assert_almost_equal(appended.vec3[0], qv1.vec3)
    assert_almost_equal(appended.quat[0], qv1.quat)
    assert_almost_equal(appended.vec3[1], np.array([1.0, 2.0, 3.0]))
    assert_almost_equal(appended.quat[1], np.array([0.0, 0.0, 0.0, 1.0]))


def test_append_qvec_batch_size_1(qv1: QVec) -> None:
    appended = qv1[None].append(
        QVec(
            vec3=np.array([[1.0, 2.0, 3.0]]),
            quat=np.array([[0.0, 0.0, 0.0, 1.0]]),
        )
    )
    assert appended.batch_size == (2,)
    assert_almost_equal(appended.vec3[0], qv1.vec3)
    assert_almost_equal(appended.quat[0], qv1.quat)
    assert_almost_equal(appended.vec3[1], np.array([1.0, 2.0, 3.0]))
    assert_almost_equal(appended.quat[1], np.array([0.0, 0.0, 0.0, 1.0]))


@pytest.fixture
def traj_len_0():
    return Trajectory(
        timestamps_us=np.array([], dtype=np.uint64), poses=QVec.create_empty()
    )


@pytest.fixture
def traj_len_1(qv1):
    return Trajectory(
        timestamps_us=np.array([10], dtype=np.uint64),
        poses=QVec.stack([qv1]),
    )


@pytest.fixture
def traj_len_2(qv1, qv2):
    return Trajectory(
        timestamps_us=np.array([10, 20], dtype=np.uint64),
        poses=QVec.stack([qv1, qv2]),
    )


def test_interpolation_len_0(traj_len_0: Trajectory):
    with pytest.raises(ValueError):
        traj_len_0.interpolate_pose(0)
    with pytest.raises(ValueError):
        traj_len_0.interpolate_pose(10)


def test_interpolation_len_1(traj_len_1: Trajectory, qv1: QVec):
    with pytest.raises(ValueError):
        traj_len_1.interpolate_pose(0)

    interpolated = traj_len_1.interpolate_pose(10)

    assert_almost_equal(interpolated.vec3, qv1.vec3)
    assert_almost_equal(interpolated.quat, qv1.quat)

    with pytest.raises(ValueError):
        traj_len_1.interpolate_pose(21)


def test_interpolation_len_2(traj_len_2: Trajectory, qv1: QVec, qv2: QVec):
    with pytest.raises(ValueError):
        traj_len_2.interpolate_pose(0)

    # should match the first qvec
    interp_start = traj_len_2.interpolate_pose(10)
    assert_almost_equal(interp_start.vec3, qv1.vec3)
    assert_almost_equal(interp_start.quat, qv1.quat)

    # should match the second qvec
    interp_end = traj_len_2.interpolate_pose(20)
    assert_almost_equal(interp_end.vec3, qv2.vec3)
    assert_almost_equal(interp_end.quat, qv2.quat)

    with pytest.raises(ValueError):
        traj_len_2.interpolate_pose(21)


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


@pytest.fixture
def qv_offset_x() -> QVec:
    return QVec(
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        vec3=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )


def test_transform_offset(traj_len_2: Trajectory, qv_offset_x: QVec) -> None:
    last_pose_base = traj_len_2.last_pose
    last_pose_traj_transform = traj_len_2.transform(qv_offset_x).last_pose

    vec3_trans_manual = last_pose_base.vec3 + qv_offset_x.vec3
    vec3_trans_traj = last_pose_traj_transform.vec3

    assert np.isclose(vec3_trans_manual, vec3_trans_traj).all()
    assert np.isclose(
        last_pose_base.quat, last_pose_traj_transform.quat
    ).all()  # no difference here


def test_transform_general(traj_len_2: Trajectory, qv1: QVec) -> None:
    """Compares Trajectory.transform to QVec multiply"""
    last_pose_base = traj_len_2.last_pose
    last_pose_traj_transform = traj_len_2.transform(qv1).last_pose

    last_pose_qvec_transform = qv1 @ last_pose_base

    assert np.isclose(
        last_pose_traj_transform.vec3, last_pose_qvec_transform.vec3
    ).all()
    assert np.isclose(
        last_pose_traj_transform.quat, last_pose_qvec_transform.quat
    ).all()


def test_transform_relative() -> None:
    quat_unrotated = np.array([0.0, 0.0, 0.0, 1.0])
    quat_rotated_half_pi = Rotation.from_euler(
        "xyz", np.array([0.0, 0.0, np.pi / 2.0])
    ).as_quat()

    transform = QVec(vec3=np.array([1.0, 0.0, 0.0]), quat=quat_unrotated)

    traj = Trajectory(
        timestamps_us=np.array([0, 1], dtype=np.uint64),
        poses=QVec.stack(
            [
                QVec(vec3=np.array([1.0, 0.0, 0.0]), quat=quat_unrotated),
                QVec(vec3=np.array([0.0, 1.0, 0.0]), quat=quat_rotated_half_pi),
            ]
        ),
    )

    transformed = traj.transform(transform, is_relative=True)
    assert_almost_equal(transformed.poses.vec3[0], np.array([2.0, 0.0, 0.0]))
    assert_almost_equal(transformed.poses.vec3[1], np.array([0.0, 2.0, 0.0]))


@pytest.fixture
def traj_len_5():
    def y2q(yaw):
        return Rotation.from_euler("xyz", np.array([0.0, 0.0, yaw])).as_quat()

    return Trajectory(
        timestamps_us=np.array([1e6, 2e6, 3e6, 4e6, 5e6], dtype=np.uint64),
        poses=QVec.stack(
            [
                QVec(vec3=np.array([0.0, 0.0, 0.0]), quat=y2q(0.0)),
                QVec(vec3=np.array([1.0, 0.5, 0.0]), quat=y2q(np.pi * 1 / 8)),
                QVec(vec3=np.array([2.0, 1.0, 0.0]), quat=y2q(np.pi * 2 / 8)),
                QVec(vec3=np.array([3.0, 2.0, 0.0]), quat=y2q(np.pi * 3 / 8)),
                QVec(
                    vec3=np.array([4.0, 4.0, 0.0]), quat=y2q(np.pi * -1 / 8 + 2 * np.pi)
                ),
            ]
        ),
    )


@pytest.fixture
def traj_len_5_constant():
    def y2q(yaw):
        return Rotation.from_euler("xyz", np.array([0.0, 0.0, yaw])).as_quat()

    return Trajectory(
        timestamps_us=np.array([1e6, 2e6, 3e6, 4e6, 5e6], dtype=np.uint64),
        poses=QVec.stack(
            [
                QVec(vec3=np.array([0.0, 1.0, 0.0]), quat=y2q(0.0)),
                QVec(vec3=np.array([0.0, 1.0, 0.0]), quat=y2q(0.0)),
                QVec(vec3=np.array([0.0, 1.0, 0.0]), quat=y2q(0.0)),
                QVec(vec3=np.array([0.0, 1.0, 0.0]), quat=y2q(0.0)),
                QVec(vec3=np.array([0.0, 1.0, 0.0]), quat=y2q(0.0)),
            ]
        ),
    )


def test_trajectory_velocities_centered(traj_len_5: Trajectory) -> None:
    velocities = traj_len_5.velocities(method="centered")
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
    accelerations = traj_len_5.accelerations(method="centered")
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
    jerk = traj_len_5.jerk(method="centered")
    print(jerk)
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
    yaw_rates = traj_len_5.yaw_rates(method="centered")
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
    yaw_rates = traj_len_5.yaw_accelerations(method="centered")
    assert yaw_rates.shape == (5,)
    assert yaw_rates == pytest.approx(
        np.pi * np.array([0.0, 0.0, -1.25 / 8, -2.5 / 8, -2.5 / 8]),
    )


def test_trajectory_velocities_forward(traj_len_5: Trajectory) -> None:
    velocities = traj_len_5.velocities(method="forward")
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
    yaw_rates = traj_len_5.yaw_rates(method="forward")
    assert yaw_rates.shape == (5,)
    assert yaw_rates == pytest.approx(
        np.pi * np.array([1 / 8, 1 / 8, 1 / 8, -4 / 8, -4 / 8]),
    )


def test_trajectory_accelerations_forward(traj_len_5: Trajectory) -> None:
    accelerations = traj_len_5.accelerations(method="forward")
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
    velocities = traj_len_5_constant.velocities(method="cubic")
    assert velocities.shape == (5, 3)
    assert velocities == pytest.approx(np.zeros((5, 3)))


def test_trajectory_accelerations_cubic(traj_len_5_constant: Trajectory) -> None:
    accelerations = traj_len_5_constant.accelerations(method="cubic")
    assert accelerations.shape == (5, 3)
    assert accelerations == pytest.approx(np.zeros((5, 3)))


def test_trajectory_yaw_rates_cubic(traj_len_5_constant: Trajectory) -> None:
    yaw_rates = traj_len_5_constant.yaw_rates(method="cubic")
    assert yaw_rates.shape == (5,)
    assert yaw_rates == pytest.approx(np.zeros((5,)))


def test_trajectory_append_no_overlap(traj_len_2: Trajectory) -> None:
    """Test concatenating two trajectories with no overlap."""
    # Create a second trajectory that starts after the first one ends

    second_traj = Trajectory(
        timestamps_us=np.array([30, 40], dtype=np.uint64),
        poses=traj_len_2.poses,
    )

    # Concatenate the trajectories
    concatenated = traj_len_2.append(second_traj)

    # Check length
    assert len(concatenated) == 4

    # Check timestamps
    assert_almost_equal(
        concatenated.timestamps_us, np.array([10, 20, 30, 40], dtype=np.uint64)
    )

    # Check poses
    assert_almost_equal(concatenated.poses.vec3[0], traj_len_2.poses.vec3[0])
    assert_almost_equal(concatenated.poses.vec3[1], traj_len_2.poses.vec3[1])
    assert_almost_equal(concatenated.poses.vec3[2], second_traj.poses.vec3[0])
    assert_almost_equal(concatenated.poses.vec3[3], second_traj.poses.vec3[1])


def test_trajectory_append_overlap_fails_with_unequal_poses(
    traj_len_2: Trajectory,
) -> None:
    """Test concatenating two trajectories with one overlapping timestamp."""

    second_traj = Trajectory(
        timestamps_us=np.array([20, 30], dtype=np.uint64),
        poses=traj_len_2.poses,
    )
    # Concatenate the trajectories
    with pytest.raises(ValueError):
        traj_len_2.append(second_traj)


def test_trajectory_append_overlap_succeeds(traj_len_2: Trajectory) -> None:
    """Test concatenating two trajectories with one overlapping timestamp."""

    second_traj = Trajectory(
        timestamps_us=np.array([20, 30], dtype=np.uint64),
        poses=traj_len_2.poses[1:].append(traj_len_2.poses[0:1]),
    )
    print(traj_len_2.poses.vec3)
    print(second_traj.poses.vec3)
    # Concatenate the trajectories
    concatenated = traj_len_2.append(second_traj)

    # Check length (should be 3, not 4, because of the overlap)
    assert len(concatenated) == 3

    # Check timestamps
    assert_almost_equal(
        concatenated.timestamps_us, np.array([10, 20, 30], dtype=np.uint64)
    )

    # Check poses
    assert_almost_equal(concatenated.poses.vec3[0], traj_len_2.poses.vec3[0])
    assert_almost_equal(concatenated.poses.vec3[1], traj_len_2.poses.vec3[1])
    assert_almost_equal(concatenated.poses.vec3[2], second_traj.poses.vec3[1])


def test_trajectory_append_empty(
    traj_len_2: Trajectory, traj_len_0: Trajectory
) -> None:
    """Test concatenating with an empty trajectory."""
    # Concatenate with an empty trajectory (first)
    concatenated1 = traj_len_0.append(traj_len_2)
    assert len(concatenated1) == len(traj_len_2)
    assert_almost_equal(concatenated1.timestamps_us, traj_len_2.timestamps_us)

    # Concatenate with an empty trajectory (second)
    concatenated2 = traj_len_2.append(traj_len_0)
    assert len(concatenated2) == len(traj_len_2)
    assert_almost_equal(concatenated2.timestamps_us, traj_len_2.timestamps_us)
