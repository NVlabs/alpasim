# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for NonlinearMPC controller."""

import do_mpc
import numpy as np
import pytest
from alpasim_controller.mpc_controller import ControllerInput, MPCGains
from alpasim_controller.mpc_impl import NonlinearMPC
from alpasim_controller.vehicle_model import VehicleModel
from alpasim_utils.geometry import Trajectory


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
        """Name property should return 'nonlinear_mpc'."""
        controller = NonlinearMPC()
        assert controller.name == "nonlinear_mpc"

    def test_timestep_property(self):
        """Default-constructed controller should expose the MPC timestep."""
        controller = NonlinearMPC()
        assert controller.dt_mpc == 0.1


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

    def test_reference_slower_than_ego_brakes_without_steering(self):
        """A straight reference slower than the ego is met by braking, not swerving."""
        controller = NonlinearMPC(n_horizon=15, gains=MPCGains(idx_start_penalty=10))
        state = np.zeros(8)
        state[3] = 22.0

        output = controller.compute_control(
            ControllerInput(
                state=state,
                reference_trajectory=_create_simple_trajectory(velocity=14.0),
                timestamp_us=0,
            )
        )

        assert output.status == "solved"
        assert abs(output.control[0]) < 0.01
        assert output.control[1] < -5.0


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
        vec3_list.append(np.array([x, 0.0, 0.0], dtype=np.float32))
        quat_list.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    positions = np.stack(vec3_list, axis=0).astype(np.float32)
    quaternions = np.stack(quat_list, axis=0).astype(np.float32)

    timestamps = np.array([i * dt_us for i in range(num_points)], dtype=np.uint64)
    return Trajectory(timestamps, positions, quaternions)


@pytest.mark.parametrize("cold_start", ["fresh", "reset", "failed_solve"])
def test_nonlinear_cold_solve_uses_physical_actuators_for_command_penalty(
    monkeypatch, cold_start
):
    controller = NonlinearMPC(n_horizon=5, gains=MPCGains(idx_start_penalty=0))
    request = ControllerInput(
        np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        _create_simple_trajectory(),
        0,
    )
    if cold_start != "fresh":
        assert controller.compute_control(request).status == "solved"
        if cold_start == "reset":
            controller.reset_warm_start()
        else:
            request.state[3] = 60.0
            assert controller.compute_control(request).status.startswith("failed:")
            request.state[3] = 10.0
    request.state[6:8] = [0.12, 1.0]
    guesses = []
    native_step = do_mpc.controller.MPC.make_step

    def make_step(mpc, state):
        guesses.append(
            (
                np.asarray(mpc.opt_x_num["_x", 0, 0, 0]).ravel(),
                np.asarray(mpc.opt_x_num["_u", 0, 0]).ravel(),
            )
        )
        return native_step(mpc, state)

    monkeypatch.setattr(do_mpc.controller.MPC, "make_step", make_step)
    output = controller.compute_control(request)
    assert output.status == "solved"
    np.testing.assert_array_equal(guesses[0][0], request.state)
    np.testing.assert_array_equal(guesses[0][1], request.state[6:8])
    np.testing.assert_array_equal(
        np.asarray(controller._mpc.opt_p_num["_u_prev"]).ravel(),
        request.state[6:8],
    )

    request.state[6:8] = [0.05, -0.2]
    assert controller.compute_control(request).status == "solved"
    np.testing.assert_array_equal(
        np.asarray(controller._mpc.opt_p_num["_u_prev"]).ravel(), output.control
    )


def test_launch_from_standstill_tracks_accelerating_reference():
    """The optimizer must see acceleration sensitivity through a zero-speed horizon."""
    controller = NonlinearMPC()
    times = np.arange(31) * 0.1
    positions = np.zeros((len(times), 3), dtype=np.float32)
    positions[:, 0] = 0.5 * times**2
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1)).astype(np.float32)
    reference = Trajectory((times * 1e6).astype(np.uint64), positions, quaternions)
    model = VehicleModel(np.zeros(2), 0.0)
    stationary = Trajectory(
        (times * 1e6).astype(np.uint64), np.zeros_like(positions), quaternions
    )
    for _ in range(3):
        output = controller.compute_control(
            ControllerInput(
                state=model.state.copy(),
                reference_trajectory=stationary,
                timestamp_us=0,
            )
        )
        model.advance(output.control, 0.1)
    # A stopped handoff can retain braking actuator state and a braking warm start.
    model.state[3:6] = 0.0
    model.state[7] = -0.1
    controller._mpc.opt_x_num["_u"] = np.array([0.0, -0.1])
    output = controller.compute_control(
        ControllerInput(
            state=model.state.copy(), reference_trajectory=reference, timestamp_us=0
        )
    )
    assert controller._mpc.solver_stats["success"]
    assert output.control[1] > 0.5


def test_stop_go_turn_converges_through_brake_release(monkeypatch, tmp_path):
    """A stationary warm start must launch without failed nonlinear solves."""
    from alpasim_controller.mpc_controller import ControllerConfig
    from benchmark.runner import BenchmarkRunner
    from benchmark.trajectories import TrajectoryGenerator

    failures = []
    compute_control = NonlinearMPC.compute_control

    def checked_control(controller, controller_input):
        output = compute_control(controller, controller_input)
        if output.control is None or output.status.startswith("failed:"):
            failures.append(controller_input.timestamp_us)
        return output

    monkeypatch.setattr(NonlinearMPC, "compute_control", checked_control)
    trajectory = TrajectoryGenerator()._generate_stop_then_go_with_turn(
        "stop_go", "Stop then turn", 80.0, 1, 10.0
    )
    result = BenchmarkRunner(
        ControllerConfig(mpc_implementation="nonlinear"), log_dir=tmp_path
    ).run(trajectories=[trajectory])
    assert not failures
    assert result.simulations[0].tracking_rmse_m < 0.15


def test_first_solve_at_maximum_speed_converges():
    """The initial prediction horizon must start at the measured speed."""
    from alpasim_controller.mpc_controller import DEFAULT_MAX_SPEED_MPS

    controller = NonlinearMPC()
    state = np.zeros(8)
    state[3] = DEFAULT_MAX_SPEED_MPS
    output = controller.compute_control(
        ControllerInput(
            state=state,
            reference_trajectory=_create_simple_trajectory(
                velocity=DEFAULT_MAX_SPEED_MPS
            ),
            timestamp_us=0,
        )
    )
    assert controller._mpc.solver_stats["success"]
    assert abs(output.control[1]) < 0.01
