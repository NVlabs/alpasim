# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for NonlinearMPC controller."""

import numpy as np
from alpasim_controller.mpc_controller import ControllerInput, MPCGains
from alpasim_controller.mpc_impl import NonlinearMPC
from alpasim_controller.vehicle_model import VehicleModel
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory


class TestNonlinearMPCInit:
    """Tests for NonlinearMPC initialization."""

    def test_default_init(self):
        """NonlinearMPC should initialize with defaults."""
        controller = NonlinearMPC()

        assert controller.name == "nonlinear_mpc"

    def test_init_with_params(self):
        """NonlinearMPC should accept custom vehicle parameters."""
        params = VehicleModel.Parameters(mass=1800.0)
        controller = NonlinearMPC(vehicle_params=params)

        assert controller._vehicle_params.mass == 1800.0

    def test_init_with_gains(self):
        """NonlinearMPC should accept custom gains."""
        gains = MPCGains(heading_weight=3.0)
        controller = NonlinearMPC(gains=gains)

        assert controller._gains.heading_weight == 3.0

    def test_lazy_mpc_init(self):
        """MPC solver should not be initialized until first call."""
        controller = NonlinearMPC()

        # MPC should be None before first call
        assert controller._mpc is None


class TestNonlinearMPCProperties:
    """Tests for NonlinearMPC properties."""

    def test_name_property(self):
        """name property should return 'nonlinear_mpc'."""
        controller = NonlinearMPC()
        assert controller.name == "nonlinear_mpc"

    def test_horizon_and_timestep(self):
        """Horizon and timestep should match MPCController."""
        assert NonlinearMPC.N_HORIZON == 20
        assert NonlinearMPC.DT_MPC == 0.1


class TestNonlinearMPCComputeControl:
    """Tests for NonlinearMPC.compute_control()."""

    def test_compute_control_returns_output(self):
        """compute_control should return a ControllerOutput."""
        controller = NonlinearMPC()
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
        assert output.status == "solved"

    def test_compute_control_initializes_mpc(self):
        """First compute_control call should initialize MPC."""
        controller = NonlinearMPC()
        trajectory = _create_simple_trajectory()

        assert controller._mpc is None

        state = np.zeros(8)
        state[3] = 10.0

        input = ControllerInput(
            state=state,
            reference_trajectory=trajectory,
            timestamp_us=0,
        )

        controller.compute_control(input)

        assert controller._mpc is not None

    def test_compute_control_output_bounded(self):
        """Control outputs should be within reasonable bounds."""
        controller = NonlinearMPC()
        trajectory = _create_simple_trajectory()

        state = np.zeros(8)
        state[3] = 10.0

        input = ControllerInput(
            state=state,
            reference_trajectory=trajectory,
            timestamp_us=0,
        )

        output = controller.compute_control(input)

        # Steering command should be bounded (typical range)
        assert -1.0 <= output.control[0] <= 1.0
        # Acceleration command should be bounded
        assert -10.0 <= output.control[1] <= 10.0

    def test_compute_control_on_reference(self):
        """When on reference, control should be near zero."""
        controller = NonlinearMPC()
        trajectory = _create_simple_trajectory(velocity=10.0)

        # State exactly on reference
        state = np.zeros(8)
        state[3] = 10.0  # vx matches reference velocity

        input = ControllerInput(
            state=state,
            reference_trajectory=trajectory,
            timestamp_us=0,
        )

        output = controller.compute_control(input)

        # Control should be small when on reference
        assert abs(output.control[0]) < 0.5  # steering
        assert abs(output.control[1]) < 1.0  # accel

    def test_compute_control_with_lateral_error(self):
        """Controller should command steering to correct lateral error."""
        controller = NonlinearMPC()
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


class TestNonlinearMPCModel:
    """Tests for NonlinearMPC model building."""

    def test_model_is_built(self):
        """Model should be built during initialization."""
        controller = NonlinearMPC()

        assert controller._model is not None

    def test_model_has_states(self):
        """Model should have expected state variables."""
        controller = NonlinearMPC()
        model = controller._model

        # Check that model has expected state names
        state_names = list(model.x.keys())
        assert "x_rig_inertial" in state_names
        assert "y_rig_inertial" in state_names
        assert "yaw_angle" in state_names
        assert "v_cg_x" in state_names

    def test_model_has_inputs(self):
        """Model should have expected input variables."""
        controller = NonlinearMPC()
        model = controller._model

        input_names = list(model.u.keys())
        assert "front_steering_angle_cmd" in input_names
        assert "acceleration_cmd" in input_names


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
