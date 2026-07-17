# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for LinearMPC controller."""

import numpy as np
from alpasim_controller.mpc_controller import ControllerInput, MPCGains
from alpasim_controller.mpc_impl import LinearMPC
from alpasim_controller.vehicle_model import VehicleModel
from alpasim_utils.geometry import Trajectory


class TestLinearMPCInit:
    """Tests for LinearMPC initialization."""

    def test_default_init(self):
        """LinearMPC should initialize with defaults."""
        controller = LinearMPC()

        assert controller.name == "linear_mpc"

    def test_init_with_params(self):
        """LinearMPC should accept custom vehicle parameters."""
        params = VehicleModel.Parameters(mass=1800.0)
        controller = LinearMPC(vehicle_params=params)

        assert controller._vehicle_params.mass == 1800.0

    def test_init_with_gains(self):
        """LinearMPC should accept custom gains."""
        gains = MPCGains(heading_weight=3.0)
        controller = LinearMPC(gains=gains)

        assert controller._gains.heading_weight == 3.0


class TestLinearMPCComputeControl:
    """Tests for LinearMPC.compute_control()."""

    def test_compute_control_returns_output(self):
        """compute_control should return a ControllerOutput."""
        controller = LinearMPC()
        trajectory = _create_simple_trajectory()

        state = np.zeros(8)
        state[3] = 10.0  # vx

        input = ControllerInput(
            state=state,
            reference_trajectory=trajectory,
            timestamp_us=0,
        )

        output = controller.compute_control(input)

        assert output.control.shape == (2,)
        assert output.solve_time_ms > 0
        assert output.status in ("solved", "solved_inaccurate")

    def test_compute_control_with_lateral_error(self):
        """Controller should command steering to correct lateral error."""
        controller = LinearMPC()
        trajectory = _create_simple_trajectory(velocity=10.0)

        # State with lateral offset
        state = np.zeros(8)
        state[1] = 1.0  # y = 1m offset
        state[3] = 10.0  # vx

        input = ControllerInput(
            state=state,
            reference_trajectory=trajectory,
            timestamp_us=0,
        )

        output = controller.compute_control(input)

        # Should command negative steering to correct positive y error
        assert output.control[0] < 0

    def test_control_is_continuous_at_kinematic_model_threshold(self):
        """Crossing the speed threshold must not double the steering contribution."""
        controller = LinearMPC()
        trajectory = _create_turning_trajectory()
        state = np.array([0.0, 0.0, 0.0, 4.99, 0.666, 0.437, 0.251, 1.313])

        below = controller.compute_control(
            ControllerInput(
                state=state,
                reference_trajectory=trajectory,
                timestamp_us=0,
            )
        )
        state[3] = controller._vehicle_params.kinematic_threshold_speed
        at_threshold = controller.compute_control(
            ControllerInput(
                state=state,
                reference_trajectory=trajectory,
                timestamp_us=0,
            )
        )

        assert below.status in ("solved", "solved_inaccurate")
        assert at_threshold.status in ("solved", "solved_inaccurate")
        assert abs(below.control[0] - at_threshold.control[0]) < 0.01


class TestLinearMPCLinearization:
    """Tests for LinearMPC dynamics linearization."""

    def test_linearize_dynamics_output_shape(self):
        """_linearize_dynamics should return correctly shaped matrices."""
        controller = LinearMPC()

        state = np.zeros(8)
        state[3] = 10.0  # Need non-zero velocity for dynamic model

        A_d, B_d = controller._linearize_dynamics(state)

        assert A_d.shape == (8, 8)
        assert B_d.shape == (8, 2)

    def test_linearize_dynamics_kinematic_model(self):
        """At low speeds, should use kinematic model."""
        controller = LinearMPC()

        # Low speed - should use kinematic model
        state = np.zeros(8)
        state[3] = 1.0  # vx < kinematic_threshold_speed (5.0)

        A_d, B_d = controller._linearize_dynamics(state)

        # Should still produce valid matrices
        assert not np.isnan(A_d).any()
        assert not np.isnan(B_d).any()

    def test_kinematic_model_preserves_steady_turning_manifold(self):
        """Steady low-speed lateral velocity and yaw rate should be preserved."""
        controller = LinearMPC()
        params = controller._vehicle_params
        velocity = params.kinematic_threshold_speed - 0.01
        steering = 0.25

        state = np.zeros(controller.NX)
        state[controller.IVX] = velocity
        state[controller.IVY] = (
            velocity * steering * params.l_rig_to_cg / params.wheelbase
        )
        state[controller.IYAW_RATE] = velocity * steering / params.wheelbase
        state[controller.ISTEERING] = steering
        command = np.array([steering, 0.0])

        A_d, B_d = controller._linearize_dynamics(state)
        next_state = A_d @ state + B_d @ command

        indices = [controller.IVY, controller.IYAW_RATE]
        np.testing.assert_allclose(next_state[indices], state[indices], atol=1e-12)

    def test_linearize_dynamics_dynamic_model(self):
        """At higher speeds, should use dynamic model."""
        controller = LinearMPC()

        # Higher speed - should use dynamic model
        state = np.zeros(8)
        state[3] = 15.0  # vx > kinematic_threshold_speed (5.0)

        A_d, B_d = controller._linearize_dynamics(state)

        # Should produce valid matrices
        assert not np.isnan(A_d).any()
        assert not np.isnan(B_d).any()


def _create_simple_trajectory(
    duration_s: float = 5.0, velocity: float = 10.0, dt_us: int = 100_000
) -> Trajectory:
    """Create a simple straight-line trajectory for testing."""
    num_points = int(duration_s * 1e6 / dt_us) + 1

    vec3_list = []
    quat_list = []

    for i in range(num_points):
        t_s = i * dt_us / 1e6
        x = velocity * t_s
        vec3_list.append(np.array([x, 0.0, 0.0], dtype=np.float32))
        quat_list.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    positions = np.stack(vec3_list, axis=0).astype(np.float32)
    quaternions = np.stack(quat_list, axis=0).astype(np.float32)

    timestamps = np.array([i * dt_us for i in range(num_points)], dtype=np.uint64)
    return Trajectory(timestamps, positions, quaternions)


def _create_turning_trajectory() -> Trajectory:
    """Create a gently curving trajectory that exercises lateral control."""
    timestamps = np.arange(0, 2_100_000, 100_000, dtype=np.uint64)
    x = timestamps.astype(np.float64) * 5.0 / 1e6
    positions = np.stack(
        [x, 0.08 * x**2, np.zeros_like(x)],
        axis=1,
    ).astype(np.float32)
    quaternions = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (len(timestamps), 1),
    )
    return Trajectory(timestamps, positions, quaternions)
