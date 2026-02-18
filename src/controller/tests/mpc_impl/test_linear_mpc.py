# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for LinearMPC controller."""

import numpy as np
from alpasim_controller.mpc_controller import ControllerInput, MPCGains
from alpasim_controller.mpc_impl import LinearMPC
from alpasim_controller.vehicle_model import VehicleModel
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory


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
        vec3_list.append(np.array([x, 0.0, 0.0]))
        quat_list.append(np.array([0.0, 0.0, 0.0, 1.0]))

    poses = QVec(
        vec3=np.stack(vec3_list, axis=0),
        quat=np.stack(quat_list, axis=0),
    )

    timestamps = np.array([i * dt_us for i in range(num_points)], dtype=np.uint64)
    return Trajectory(poses=poses, timestamps_us=timestamps)
