# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2
from alpasim_utils.geometry import (
    DynamicTrajectory,
    Pose,
    Trajectory,
    array_to_dynamic_states,
    dynamic_state_to_array,
    dynamic_states_to_array,
)
from numpy.testing import assert_allclose

# =============================================================================
# Helpers
# =============================================================================

_IDENTITY_QUAT = np.array([0, 0, 0, 1], dtype=np.float32)  # scipy xyzw


def _make_pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Pose:
    return Pose(np.array([x, y, z], dtype=np.float32), _IDENTITY_QUAT)


def _make_dynamic_state(
    lv: tuple[float, float, float] = (0.0, 0.0, 0.0),
    av: tuple[float, float, float] = (0.0, 0.0, 0.0),
    la: tuple[float, float, float] = (0.0, 0.0, 0.0),
    aa: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> common_pb2.DynamicState:
    return common_pb2.DynamicState(
        linear_velocity=common_pb2.Vec3(x=lv[0], y=lv[1], z=lv[2]),
        angular_velocity=common_pb2.Vec3(x=av[0], y=av[1], z=av[2]),
        linear_acceleration=common_pb2.Vec3(x=la[0], y=la[1], z=la[2]),
        angular_acceleration=common_pb2.Vec3(x=aa[0], y=aa[1], z=aa[2]),
    )


def _dynamics_row(
    lv: tuple[float, float, float] = (0.0, 0.0, 0.0),
    av: tuple[float, float, float] = (0.0, 0.0, 0.0),
    la: tuple[float, float, float] = (0.0, 0.0, 0.0),
    aa: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    return np.array([*lv, *av, *la, *aa], dtype=np.float64)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def two_entry_dt() -> DynamicTrajectory:
    """Two-entry DynamicTrajectory at ts=100, ts=200."""
    timestamps = np.array([100, 200], dtype=np.uint64)
    poses = [_make_pose(x=1.0), _make_pose(x=2.0)]
    dynamics = np.array(
        [
            _dynamics_row(lv=(1.0, 0.0, 0.0), av=(0.0, 0.0, 0.1)),
            _dynamics_row(lv=(3.0, 2.0, 0.0), av=(0.0, 0.0, 0.3)),
        ],
        dtype=np.float64,
    )
    traj = Trajectory.from_poses(timestamps, poses)
    return DynamicTrajectory.from_trajectory_and_dynamics(traj, dynamics)


# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    def test_from_trajectory_and_dynamics(self) -> None:
        ts = np.array([0, 100_000], dtype=np.uint64)
        poses = [_make_pose(), _make_pose(x=1.0)]
        traj = Trajectory.from_poses(ts, poses)
        dyn = np.zeros((2, 12), dtype=np.float64)
        dt = DynamicTrajectory.from_trajectory_and_dynamics(traj, dyn)
        assert len(dt) == 2

    def test_new_constructor(self) -> None:
        ts = np.array([0, 100], dtype=np.uint64)
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        quat = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
        dyn = np.zeros((2, 12), dtype=np.float64)
        dt = DynamicTrajectory(ts, pos, quat, dyn)
        assert len(dt) == 2

    def test_create_empty(self) -> None:
        dt = DynamicTrajectory.create_empty()
        assert len(dt) == 0
        assert dt.is_empty()

    def test_mismatched_lengths_raises(self) -> None:
        poses = [_make_pose()]
        traj = Trajectory.from_poses(np.array([0], dtype=np.uint64), poses)
        dyn = np.zeros((2, 12), dtype=np.float64)
        with pytest.raises(ValueError, match="dynamics has 2 rows"):
            DynamicTrajectory.from_trajectory_and_dynamics(traj, dyn)

    def test_non_monotonic_timestamps_raises(self) -> None:
        ts = np.array([200, 100], dtype=np.uint64)
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        quat = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32)
        dyn = np.zeros((2, 12), dtype=np.float64)
        with pytest.raises(ValueError, match="strictly increasing"):
            DynamicTrajectory(ts, pos, quat, dyn)


# =============================================================================
# Properties
# =============================================================================


class TestProperties:
    def test_timestamps_us(self, two_entry_dt: DynamicTrajectory) -> None:
        ts = two_entry_dt.timestamps_us
        assert_allclose(ts, [100, 200])

    def test_positions(self, two_entry_dt: DynamicTrajectory) -> None:
        pos = two_entry_dt.positions
        assert pos.shape == (2, 3)
        assert_allclose(pos[0, 0], 1.0, atol=1e-5)
        assert_allclose(pos[1, 0], 2.0, atol=1e-5)

    def test_quaternions(self, two_entry_dt: DynamicTrajectory) -> None:
        quat = two_entry_dt.quaternions
        assert quat.shape == (2, 4)

    def test_last_pose(self, two_entry_dt: DynamicTrajectory) -> None:
        p = two_entry_dt.last_pose
        assert_allclose(p.vec3[0], 2.0, atol=1e-5)

    def test_first_pose(self, two_entry_dt: DynamicTrajectory) -> None:
        p = two_entry_dt.first_pose
        assert_allclose(p.vec3[0], 1.0, atol=1e-5)

    def test_len(self, two_entry_dt: DynamicTrajectory) -> None:
        assert len(two_entry_dt) == 2

    def test_is_empty(self) -> None:
        assert DynamicTrajectory.create_empty().is_empty()

    def test_dynamics_property(self, two_entry_dt: DynamicTrajectory) -> None:
        dyn = two_entry_dt.dynamics
        assert dyn.shape == (2, 12)
        assert_allclose(dyn[0, 0], 1.0)  # lv.x at ts=100
        assert_allclose(dyn[1, 0], 3.0)  # lv.x at ts=200

    def test_get_pose(self, two_entry_dt: DynamicTrajectory) -> None:
        p0 = two_entry_dt.get_pose(0)
        assert_allclose(p0.vec3[0], 1.0, atol=1e-5)
        pm1 = two_entry_dt.get_pose(-1)
        assert_allclose(pm1.vec3[0], 2.0, atol=1e-5)

    def test_get_pose_out_of_range(self, two_entry_dt: DynamicTrajectory) -> None:
        with pytest.raises(IndexError):
            two_entry_dt.get_pose(5)


# =============================================================================
# Interpolation (ported from test_dynamic_state_history.py)
# =============================================================================


class TestInterpolateExact:
    def test_at_exact_timestamps(self, two_entry_dt: DynamicTrajectory) -> None:
        ts = np.array([100, 200], dtype=np.uint64)
        result = two_entry_dt.interpolate_dynamics(ts)
        assert result.shape == (2, 12)
        assert_allclose(result[0, 0], 1.0)  # lv.x
        assert_allclose(result[0, 1], 0.0)  # lv.y
        assert_allclose(result[0, 5], 0.1)  # av.z
        assert_allclose(result[1, 0], 3.0)
        assert_allclose(result[1, 1], 2.0)
        assert_allclose(result[1, 5], 0.3)

    def test_single_exact(self, two_entry_dt: DynamicTrajectory) -> None:
        ts = np.array([100], dtype=np.uint64)
        result = two_entry_dt.interpolate_dynamics(ts)
        assert result.shape == (1, 12)
        assert_allclose(result[0, 0], 1.0)


class TestInterpolateBetween:
    def test_midpoint(self, two_entry_dt: DynamicTrajectory) -> None:
        ts = np.array([150], dtype=np.uint64)
        result = two_entry_dt.interpolate_dynamics(ts)
        assert_allclose(result[0, 0], 2.0)  # midpoint (1.0, 3.0)
        assert_allclose(result[0, 1], 1.0)  # midpoint (0.0, 2.0)
        assert_allclose(result[0, 5], 0.2)  # midpoint (0.1, 0.3)

    def test_quarter_point(self, two_entry_dt: DynamicTrajectory) -> None:
        ts = np.array([125], dtype=np.uint64)
        result = two_entry_dt.interpolate_dynamics(ts)
        assert_allclose(result[0, 0], 1.5)  # 25% of (1.0, 3.0)


class TestClampingBehavior:
    def test_before_range_clamps_to_first(
        self, two_entry_dt: DynamicTrajectory
    ) -> None:
        ts = np.array([50], dtype=np.uint64)
        result = two_entry_dt.interpolate_dynamics(ts)
        assert_allclose(result[0, 0], 1.0)
        assert_allclose(result[0, 5], 0.1)

    def test_after_range_clamps_to_last(self, two_entry_dt: DynamicTrajectory) -> None:
        ts = np.array([300], dtype=np.uint64)
        result = two_entry_dt.interpolate_dynamics(ts)
        assert_allclose(result[0, 0], 3.0)
        assert_allclose(result[0, 5], 0.3)


class TestAllFieldsInterpolated:
    def test_all_four_fields(self) -> None:
        ts = np.array([0, 100], dtype=np.uint64)
        poses = [_make_pose(), _make_pose(x=1.0)]
        dynamics = np.array(
            [
                _dynamics_row(
                    lv=(1.0, 2.0, 3.0),
                    av=(4.0, 5.0, 6.0),
                    la=(7.0, 8.0, 9.0),
                    aa=(10.0, 11.0, 12.0),
                ),
                _dynamics_row(
                    lv=(3.0, 4.0, 5.0),
                    av=(6.0, 7.0, 8.0),
                    la=(9.0, 10.0, 11.0),
                    aa=(12.0, 13.0, 14.0),
                ),
            ],
            dtype=np.float64,
        )
        traj = Trajectory.from_poses(ts, poses)
        dt = DynamicTrajectory.from_trajectory_and_dynamics(traj, dynamics)

        result = dt.interpolate_dynamics(np.array([50], dtype=np.uint64))
        r = result[0]
        # All midpoints
        assert_allclose(r[0], 2.0)  # lv.x
        assert_allclose(r[1], 3.0)  # lv.y
        assert_allclose(r[2], 4.0)  # lv.z
        assert_allclose(r[3], 5.0)  # av.x
        assert_allclose(r[4], 6.0)  # av.y
        assert_allclose(r[5], 7.0)  # av.z
        assert_allclose(r[6], 8.0)  # la.x
        assert_allclose(r[7], 9.0)  # la.y
        assert_allclose(r[8], 10.0)  # la.z
        assert_allclose(r[9], 11.0)  # aa.x
        assert_allclose(r[10], 12.0)  # aa.y
        assert_allclose(r[11], 13.0)  # aa.z


class TestMultipleQueryTimestamps:
    def test_many_queries(self, two_entry_dt: DynamicTrajectory) -> None:
        ts = np.array([100, 125, 150, 175, 200], dtype=np.uint64)
        result = two_entry_dt.interpolate_dynamics(ts)
        assert result.shape == (5, 12)
        expected_lv_x = [1.0, 1.5, 2.0, 2.5, 3.0]
        assert_allclose(result[:, 0], expected_lv_x)


# =============================================================================
# Trajectory extraction
# =============================================================================


class TestTrajectoryExtraction:
    def test_trajectory_returns_plain_trajectory(
        self, two_entry_dt: DynamicTrajectory
    ) -> None:
        traj = two_entry_dt.trajectory()
        assert isinstance(traj, Trajectory)
        assert len(traj) == 2
        assert_allclose(traj.timestamps_us, [100, 200])

    def test_trajectory_poses_match(self, two_entry_dt: DynamicTrajectory) -> None:
        traj = two_entry_dt.trajectory()
        assert traj.first_pose == two_entry_dt.first_pose
        assert traj.last_pose == two_entry_dt.last_pose


# =============================================================================
# Mutation
# =============================================================================


class TestUpdateAbsolute:
    def test_append_entry(self) -> None:
        dt = DynamicTrajectory.create_empty()
        p = _make_pose(x=5.0)
        dyn = np.zeros(12, dtype=np.float64)
        dyn[0] = 10.0  # lv.x
        dt.update_absolute(100, p, dyn)
        assert len(dt) == 1
        assert_allclose(dt.dynamics[0, 0], 10.0)

    def test_timestamp_must_be_increasing(self) -> None:
        dt = DynamicTrajectory.create_empty()
        p = _make_pose()
        dyn = np.zeros(12, dtype=np.float64)
        dt.update_absolute(200, p, dyn)
        with pytest.raises(ValueError, match="must be greater"):
            dt.update_absolute(100, p, dyn)

    def test_dynamics_wrong_length_raises(self) -> None:
        dt = DynamicTrajectory.create_empty()
        p = _make_pose()
        with pytest.raises(ValueError, match="12 elements"):
            dt.update_absolute(100, p, np.zeros(5, dtype=np.float64))


# =============================================================================
# Concat
# =============================================================================


class TestConcat:
    def test_concat_non_overlapping(self) -> None:
        ts1 = np.array([100, 200], dtype=np.uint64)
        ts2 = np.array([300, 400], dtype=np.uint64)
        poses1 = [_make_pose(x=1.0), _make_pose(x=2.0)]
        poses2 = [_make_pose(x=3.0), _make_pose(x=4.0)]
        dyn1 = np.ones((2, 12), dtype=np.float64)
        dyn2 = np.ones((2, 12), dtype=np.float64) * 2.0

        t1 = Trajectory.from_poses(ts1, poses1)
        t2 = Trajectory.from_poses(ts2, poses2)
        dt1 = DynamicTrajectory.from_trajectory_and_dynamics(t1, dyn1)
        dt2 = DynamicTrajectory.from_trajectory_and_dynamics(t2, dyn2)

        result = dt1.concat(dt2)
        assert len(result) == 4
        assert_allclose(result.timestamps_us, [100, 200, 300, 400])
        assert_allclose(result.dynamics[0, 0], 1.0)
        assert_allclose(result.dynamics[2, 0], 2.0)

    def test_concat_overlapping_raises(self) -> None:
        ts = np.array([100], dtype=np.uint64)
        p = [_make_pose()]
        dyn = np.zeros((1, 12), dtype=np.float64)
        t = Trajectory.from_poses(ts, p)
        dt1 = DynamicTrajectory.from_trajectory_and_dynamics(t, dyn)
        dt2 = DynamicTrajectory.from_trajectory_and_dynamics(t, dyn)
        with pytest.raises(ValueError, match="Cannot concat"):
            dt1.concat(dt2)

    def test_concat_with_empty(self) -> None:
        ts = np.array([100], dtype=np.uint64)
        p = [_make_pose()]
        dyn = np.zeros((1, 12), dtype=np.float64)
        t = Trajectory.from_poses(ts, p)
        dt = DynamicTrajectory.from_trajectory_and_dynamics(t, dyn)
        empty = DynamicTrajectory.create_empty()

        assert len(dt.concat(empty)) == 1
        assert len(empty.concat(dt)) == 1


# =============================================================================
# Append
# =============================================================================


class TestAppend:
    def test_append_overlapping_endpoint(self) -> None:
        p_shared = _make_pose(x=2.0)
        ts1 = np.array([100, 200], dtype=np.uint64)
        ts2 = np.array([200, 300], dtype=np.uint64)
        poses1 = [_make_pose(x=1.0), p_shared]
        poses2 = [p_shared, _make_pose(x=3.0)]
        dyn1 = np.ones((2, 12), dtype=np.float64)
        dyn2 = np.ones((2, 12), dtype=np.float64) * 2.0

        t1 = Trajectory.from_poses(ts1, poses1)
        t2 = Trajectory.from_poses(ts2, poses2)
        dt1 = DynamicTrajectory.from_trajectory_and_dynamics(t1, dyn1)
        dt2 = DynamicTrajectory.from_trajectory_and_dynamics(t2, dyn2)

        result = dt1.append(dt2)
        assert len(result) == 3  # overlap deduplication
        assert_allclose(result.timestamps_us, [100, 200, 300])

    def test_append_gap(self) -> None:
        ts1 = np.array([100], dtype=np.uint64)
        ts2 = np.array([300], dtype=np.uint64)
        t1 = Trajectory.from_poses(ts1, [_make_pose()])
        t2 = Trajectory.from_poses(ts2, [_make_pose(x=1.0)])
        dyn1 = np.zeros((1, 12), dtype=np.float64)
        dyn2 = np.zeros((1, 12), dtype=np.float64)
        dt1 = DynamicTrajectory.from_trajectory_and_dynamics(t1, dyn1)
        dt2 = DynamicTrajectory.from_trajectory_and_dynamics(t2, dyn2)

        result = dt1.append(dt2)
        assert len(result) == 2


# =============================================================================
# Transform
# =============================================================================


class TestTransform:
    def test_transform_changes_poses_not_dynamics(self) -> None:
        ts = np.array([100], dtype=np.uint64)
        poses = [_make_pose(x=1.0)]
        dynamics = np.array([[1.0] * 12], dtype=np.float64)
        t = Trajectory.from_poses(ts, poses)
        dt = DynamicTrajectory.from_trajectory_and_dynamics(t, dynamics)

        offset = _make_pose(x=10.0)
        result = dt.transform(offset)

        # Position changed
        assert_allclose(result.first_pose.vec3[0], 11.0, atol=1e-5)
        # Dynamics unchanged
        assert_allclose(result.dynamics, dynamics)


# =============================================================================
# Protobuf conversion helpers
# =============================================================================


class TestProtobufConversion:
    def test_dynamic_state_to_array(self) -> None:
        state = _make_dynamic_state(
            lv=(1.0, 2.0, 3.0),
            av=(4.0, 5.0, 6.0),
            la=(7.0, 8.0, 9.0),
            aa=(10.0, 11.0, 12.0),
        )
        arr = dynamic_state_to_array(state)
        assert arr.shape == (12,)
        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_allclose(arr, expected)

    def test_dynamic_states_to_array(self) -> None:
        states = [
            _make_dynamic_state(lv=(1.0, 0.0, 0.0)),
            _make_dynamic_state(lv=(2.0, 0.0, 0.0)),
        ]
        arr = dynamic_states_to_array(states)
        assert arr.shape == (2, 12)
        assert_allclose(arr[0, 0], 1.0)
        assert_allclose(arr[1, 0], 2.0)

    def test_array_to_dynamic_states(self) -> None:
        arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=np.float64)
        states = array_to_dynamic_states(arr)
        assert len(states) == 1
        s = states[0]
        assert_allclose(s.linear_velocity.x, 1.0)
        assert_allclose(s.angular_velocity.y, 5.0)
        assert_allclose(s.linear_acceleration.z, 9.0)
        assert_allclose(s.angular_acceleration.z, 12.0)

    def test_roundtrip(self) -> None:
        original = _make_dynamic_state(
            lv=(1.5, 2.5, 3.5),
            av=(4.5, 5.5, 6.5),
            la=(7.5, 8.5, 9.5),
            aa=(10.5, 11.5, 12.5),
        )
        arr = dynamic_states_to_array([original])
        recovered = array_to_dynamic_states(arr)
        assert len(recovered) == 1
        r = recovered[0]
        assert_allclose(r.linear_velocity.x, 1.5)
        assert_allclose(r.angular_acceleration.z, 12.5)


# =============================================================================
# Repr
# =============================================================================


class TestRepr:
    def test_repr(self, two_entry_dt: DynamicTrajectory) -> None:
        s = repr(two_entry_dt)
        assert "DynamicTrajectory" in s
        assert "n_poses=2" in s
