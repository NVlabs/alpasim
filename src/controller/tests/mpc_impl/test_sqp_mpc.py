# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for the sequential quadratic programming MPC."""

import numpy as np
import pytest
from alpasim_controller.mpc_controller import ControllerInput, MPCGains
from alpasim_controller.mpc_impl import SQPMPC
from alpasim_utils.geometry import Pose, Trajectory


def _straight_trajectory(num_points: int = 8) -> Trajectory:
    timestamps = np.arange(num_points, dtype=np.uint64) * 100_000
    poses = [
        Pose(
            np.array([float(k), 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        for k in range(num_points)
    ]
    return Trajectory.from_poses(timestamps, poses)


def _controller() -> SQPMPC:
    return SQPMPC(
        n_horizon=5,
        gains=MPCGains(
            idx_start_penalty=0,
            sqp_max_iterations=3,
            sqp_step_tolerance=1e-4,
        ),
    )


def test_rejects_non_positive_sqp_iterations() -> None:
    with pytest.raises(ValueError, match="sqp_max_iterations must be positive"):
        SQPMPC(gains=MPCGains(sqp_max_iterations=0))


def test_rejects_negative_sqp_step_tolerance() -> None:
    with pytest.raises(ValueError, match="sqp_step_tolerance must be non-negative"):
        SQPMPC(gains=MPCGains(sqp_step_tolerance=-1e-3))


def test_rejects_negative_lateral_position_ramp() -> None:
    with pytest.raises(
        ValueError, match="sqp_lat_position_ramp_steps must be non-negative"
    ):
        SQPMPC(gains=MPCGains(sqp_lat_position_ramp_steps=-1))


def test_lateral_position_ramp_softens_the_first_correction() -> None:
    """Offset sideways from a straight reference, the ramp steers back less hard at once."""

    def first_steering(ramp_steps: int) -> float:
        controller = SQPMPC(
            n_horizon=20,
            gains=MPCGains(
                idx_start_penalty=0,
                lat_position_weight=3.0,
                heading_weight=2.0,
                sqp_max_iterations=3,
                sqp_lat_position_ramp_steps=ramp_steps,
            ),
        )
        state = np.array([0.0, 0.5, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        result = controller.compute_control(
            ControllerInput(
                state=state,
                reference_trajectory=_straight_trajectory(num_points=40),
                timestamp_us=0,
            )
        )
        assert result.status.startswith("solved")
        return float(result.control[0])

    full = first_steering(0)
    ramped = first_steering(10)

    # Both steer right, back towards the reference; the ramp steers less.
    assert full < 0.0 and ramped < 0.0
    assert abs(ramped) < abs(full)


def test_reset_warm_start_clears_previous_solution() -> None:
    controller = _controller()
    controller._previous_control = np.array([0.1, 0.2])
    controller._last_qp_solution = np.zeros(controller._num_vars)

    controller.reset_warm_start()

    assert controller._previous_control is None
    assert controller._last_qp_solution is None


def test_compute_control_with_pose_reference_inputs() -> None:
    controller = _controller()
    trajectory = _straight_trajectory()
    state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])

    result = controller.compute_control(
        ControllerInput(
            state=state,
            reference_trajectory=trajectory,
            timestamp_us=0,
        )
    )

    assert result.control.shape == (2,)
    assert result.status.startswith("solved")
    assert np.all(np.isfinite(result.control))
    assert -2.0 <= result.control[0] <= 2.0
    assert -9.0 <= result.control[1] <= 6.0


def test_short_pose_reference_extrapolates_without_steering_snap() -> None:
    controller = SQPMPC(gains=MPCGains())
    timestamps = np.array([0, 49_962, 100_000], dtype=np.uint64)
    poses = [
        Pose(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        Pose(
            np.array([1.1417847, -0.0010713, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -7.6421e-5, 1.0], dtype=np.float32),
        ),
        Pose(
            np.array([2.4264946, -0.0002874, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.9081e-4, 1.0], dtype=np.float32),
        ),
    ]
    trajectory = Trajectory.from_poses(timestamps, poses)
    state = np.array(
        [
            0.0,
            0.0,
            0.0,
            23.94,
            -0.0048,
            0.00054,
            0.0010,
            -3.11,
        ]
    )

    x_ref, _ = controller._build_reference(
        ControllerInput(state=state, reference_trajectory=trajectory, timestamp_us=0)
    )
    result = controller.compute_control(
        ControllerInput(state=state, reference_trajectory=trajectory, timestamp_us=0)
    )

    assert result.status.startswith("solved")
    assert x_ref[10, 0] > 20.0
    assert abs(result.control[0]) < 0.1


def test_reference_dynamic_states_follow_discrete_bicycle_dynamics() -> None:
    controller = _controller()
    trajectory = _straight_trajectory()
    state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    controller_input = ControllerInput(
        state=state,
        reference_trajectory=trajectory,
        timestamp_us=0,
    )

    x_ref, u_ref = controller._build_reference(controller_input)

    for k in range(controller._n_horizon):
        np.testing.assert_allclose(
            x_ref[k + 1, SQPMPC.IVX :],
            controller._evaluate_dynamics(x_ref[k], u_ref[k])[SQPMPC.IVX :],
            atol=1e-10,
        )


def _left_turn_trajectory(
    speed: float = 10.0, radius: float = 40.0, num_points: int = 30
) -> Trajectory:
    timestamps = np.arange(num_points, dtype=np.uint64) * 100_000
    poses = []
    for k in range(num_points):
        angle = speed * k * 0.1 / radius
        poses.append(
            Pose(
                np.array(
                    [radius * np.sin(angle), radius * (1.0 - np.cos(angle)), 0.0],
                    dtype=np.float32,
                ),
                np.array(
                    [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)],
                    dtype=np.float32,
                ),
            )
        )
    return Trajectory.from_poses(timestamps, poses)


def test_reference_poses_match_reference_trajectory_in_turn() -> None:
    controller = SQPMPC(gains=MPCGains(idx_start_penalty=0))
    speed, radius = 10.0, 40.0
    trajectory = _left_turn_trajectory(speed, radius)
    state = np.array([0.0, 0.0, 0.0, speed, 0.0, speed / radius, 0.0, 0.0])
    controller_input = ControllerInput(
        state=state, reference_trajectory=trajectory, timestamp_us=0
    )

    x_ref, u_ref = controller._build_reference(controller_input)

    expected = trajectory.interpolate(controller._horizon_timestamps(controller_input))
    np.testing.assert_allclose(
        x_ref[:, : SQPMPC.IYAW], expected.positions[:, :2], atol=1e-3
    )
    np.testing.assert_allclose(x_ref[:, SQPMPC.IYAW], expected.yaws, atol=1e-3)
    # The pure rollout leaves the (slip-free) reference poses to the outside of the
    # turn; the pose columns must not inherit that drift.
    rollout = x_ref[0].copy()
    for k in range(controller._n_horizon):
        rollout = controller._evaluate_dynamics(rollout, u_ref[k])
    assert rollout[SQPMPC.IY] < x_ref[-1, SQPMPC.IY] - 0.05


@pytest.mark.parametrize("speed", [0.0, 0.001, 0.1, 1.0, 4.99, 5.01, 20.0, 45.0])
def test_steady_turn_reference_matches_vehicle_equilibrium(speed):
    """Reference tire force balance stays valid through crawl and highway speeds."""
    from alpasim_controller.vehicle_model import VehicleModel

    controller = _controller()
    yaw_rate = speed * min(0.01, 1 / (speed**2 + 1))
    vy, steering = controller._steady_turn_state(speed, yaw_rate)
    model = VehicleModel(np.array([speed, 0.0]), 0.0)
    model.state[4:7] = [vy, yaw_rate, steering]
    model.advance(np.array([steering, 0.0]), 0.1)
    np.testing.assert_allclose(model.state[4:6], [vy, yaw_rate], atol=1e-12)
