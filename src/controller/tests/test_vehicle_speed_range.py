# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Behavioral checks across the forward-speed operating range."""

import numpy as np
import pytest
from alpasim_controller.vehicle_model import VehicleModel


def test_no_jump_crossing_five_metres_per_second():
    """Identical transient states on either side of 5 m/s evolve continuously."""
    states = []
    accelerations = []
    for speed in [5.0 - 1e-7, 5.0 + 1e-7]:
        model = VehicleModel(np.array([speed, 0.0]), 0.0)
        model.state[4:7] = [0.2, 0.1, 0.08]
        model.advance(np.array([0.08, 0.0]), 0.01)
        states.append(model.state.copy())
        accelerations.append(model.accelerations.copy())
    np.testing.assert_allclose(states[0], states[1], atol=1e-6, rtol=0)
    np.testing.assert_allclose(accelerations[0], accelerations[1], atol=1e-5, rtol=0)


SPEEDS = [0.0, 1e-12, 1e-6, 0.01, 0.1, 1.0, 4.99, 5.0, 5.01, 20.0, 45.0, 80.0]


@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("direction", [-1, 1])
def test_speed_sweep_converges_to_force_balance(speed, direction):
    """Forward dynamics remain bounded and converge to steady tire force balance."""
    model = VehicleModel(np.array([speed, 0.0]), 0.0)
    p = model.parameters
    steering = direction * min(0.15, 2 * p.wheelbase / (speed**2 + 1))
    lr = p.l_rig_to_cg
    lf = p.wheelbase - lr
    understeer = (
        p.mass
        / (2 * p.wheelbase)
        * (lr / p.front_cornering_stiffness - lf / p.rear_cornering_stiffness)
    )
    expected_yaw_rate = speed * steering / (p.wheelbase + understeer * speed**2)
    expected_vy = (
        lr - p.mass * lf * speed**2 / (2 * p.rear_cornering_stiffness * p.wheelbase)
    ) * expected_yaw_rate
    with np.errstate(all="raise"):
        for _ in range(100):
            model.advance(np.array([steering, 0.0]), 0.1)
            assert np.all(np.isfinite(model.state))
            assert np.all(np.isfinite(model.accelerations))
            assert abs(model.state[4]) < 2.0
            assert abs(model.state[5]) < 1.0
    # Normalize crawl states so absolute tolerances cannot hide a zero response.
    scale = speed if speed > 0 else 1.0
    np.testing.assert_allclose(
        model.state[4:6] / scale,
        np.array([expected_vy, expected_yaw_rate]) / scale,
        atol=1e-9,
        rtol=1e-5,
    )


@pytest.mark.parametrize("steering", [-0.5, 0.0, 0.5])
def test_steering_at_standstill_does_not_move_vehicle(steering):
    model = VehicleModel(np.zeros(2), 0.0)
    for _ in range(20):
        model.advance(np.array([steering, -3.0]), 0.1)
    np.testing.assert_array_equal(model.state[:6], np.zeros(6))
    np.testing.assert_array_equal(model.accelerations, np.zeros(3))
    assert model.front_steering_angle == pytest.approx(steering, abs=1e-8)


def test_repeated_stop_go_turns_cross_five_without_drift_at_stop():
    model = VehicleModel(np.zeros(2), 0.0)
    for _ in range(3):
        speeds = []
        for accel, count in [(2.0, 50), (-3.0, 40)]:
            for _ in range(count):
                model.advance(np.array([0.08, accel]), 0.1)
                assert np.all(np.isfinite(model.state))
                assert np.all(np.isfinite(model.accelerations))
                speeds.append(model.state[3])
        assert max(speeds) > 5.0
        np.testing.assert_array_equal(model.state[3:6], np.zeros(3))
        stopped_pose = model.state[:3].copy()
        model.advance(np.array([0.08, -3.0]), 1.0)
        np.testing.assert_array_equal(model.state[:3], stopped_pose)


@pytest.mark.parametrize(
    "speed", [0.1, 1.0, 4.99, 5.0, 5.01, 10.0, 20.0, 40.0, 45.0, 80.0]
)
def test_transient_matches_independent_continuous_tire_model(speed):
    """Compare to a high-accuracy stiff ODE solver using tire forces directly."""
    from scipy.integrate import solve_ivp

    model = VehicleModel(np.array([speed, 0.0]), 0.0)
    p = model.parameters
    steering = min(0.1, 2 * p.wheelbase / (speed**2 + 1))
    command = np.array([steering, 0.0])
    initial = model.state.copy()

    def derivative(t, state):
        yaw, vx, vy, r, delta, accel = state[2:]
        lf = p.wheelbase - p.l_rig_to_cg
        front_force = 2 * p.front_cornering_stiffness * (delta - (vy + lf * r) / vx)
        rear_force = 2 * p.rear_cornering_stiffness * (-(vy - p.l_rig_to_cg * r) / vx)
        rig_vy = vy - p.l_rig_to_cg * r
        return [
            vx * np.cos(yaw) - rig_vy * np.sin(yaw),
            vx * np.sin(yaw) + rig_vy * np.cos(yaw),
            r,
            accel,
            (front_force + rear_force) / p.mass - vx * r,
            (lf * front_force - p.l_rig_to_cg * rear_force) / p.inertia,
            (steering - delta) / p.steering_time_constant,
            -accel / p.acceleration_time_constant,
        ]

    times = np.arange(1, 11) * 0.1
    reference = solve_ivp(
        derivative,
        (0, 1.0),
        initial,
        method="Radau",
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
    )
    assert reference.success
    for expected in reference.y.T:
        model.advance(command, 0.1)
        np.testing.assert_allclose(model.state[:2], expected[:2], atol=0.02, rtol=0)
        np.testing.assert_allclose(model.state[2], expected[2], atol=0.003, rtol=0)
        np.testing.assert_allclose(model.state[4:6], expected[4:6], atol=0.005, rtol=0)


@pytest.mark.parametrize("dt", [-0.1, float("nan"), float("inf")])
def test_invalid_timestep_is_rejected(dt):
    model = VehicleModel(np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="dt"):
        model.advance(np.zeros(2), dt)


@pytest.fixture(scope="module")
def prediction_models():
    import casadi
    from alpasim_controller.mpc_impl import SQPMPC, LinearMPC, NonlinearMPC

    linear = LinearMPC(n_horizon=2)
    sqp = SQPMPC(n_horizon=2)
    nonlinear = NonlinearMPC(n_horizon=2)
    model = nonlinear._model
    nonlinear_rhs = casadi.Function(
        "test_nonlinear_rhs", [model.x.cat, model.u.cat], [model._rhs.cat]
    )
    return linear, sqp, nonlinear_rhs


@pytest.mark.parametrize(
    "speed", [0.0, 1e-9, 0.1, 1.0, 4.999999, 5.0, 5.000001, 20.0, 40.0, 45.0]
)
@pytest.mark.parametrize("accel", [-3.0, 0.0, 2.0])
def test_discrete_predictors_match_simulator_and_have_finite_jacobians(
    prediction_models, speed, accel
):
    linear, sqp, _ = prediction_models
    model = VehicleModel(np.array([speed, 0.0]), 0.0)
    model.state[6] = 0.05
    initial = model.state.copy()
    command = np.array([0.08, accel])
    model.advance(command, 0.1)
    linear_next, linear_a, linear_b = linear._dynamics_linearization(initial, command)
    sqp_a, sqp_b = sqp._dynamics_jacobians(initial, command)
    for prediction in [
        linear_next,
        sqp._evaluate_dynamics(initial, command),
    ]:
        np.testing.assert_allclose(
            np.asarray(prediction).ravel(), model.state, atol=1e-12
        )
    for jacobian in [linear_a, linear_b, sqp_a, sqp_b]:
        assert np.all(np.isfinite(np.asarray(jacobian)))


def test_standstill_jacobian_preserves_launch_sensitivity(prediction_models):
    """The right derivative at zero must survive all integration substeps."""
    _, sqp, _ = prediction_models
    state = np.zeros(8)
    command = np.zeros(2)
    a, b = sqp._dynamics_jacobians(state, command)
    epsilon = 1e-6
    forward_difference = (
        sqp._evaluate_dynamics(state, np.array([0.0, epsilon]))
        - sqp._evaluate_dynamics(state, command)
    ) / epsilon
    np.testing.assert_allclose(np.asarray(b)[:, 1], forward_difference, atol=1e-9)
    assert float(a[3, 3]) == pytest.approx(1.0)
    tau = VehicleModel.Parameters().acceleration_time_constant
    assert float(b[3, 1]) == pytest.approx(0.1 - tau * (1 - np.exp(-0.1 / tau)))


@pytest.mark.parametrize("speed", [0.0, 1e-6, 0.01, 0.1, 1.0, 4.99, 5.01, 20.0, 45.0])
def test_continuous_predictor_matches_simulator(prediction_models, speed):
    """Regularization preserves the stationary limit and short-horizon motion."""
    from scipy.integrate import solve_ivp

    _, _, rhs = prediction_models
    model = VehicleModel(np.array([speed, 0.0]), 0.0)
    command = np.array([min(0.08, 5 / (speed**2 + 1)), 0.0])
    reference = solve_ivp(
        lambda t, x: np.asarray(rhs(x, command)).ravel(),
        (0, 0.1),
        model.state.copy(),
        method="Radau",
        rtol=1e-10,
        atol=1e-12,
    )
    assert reference.success
    model.advance(command, 0.1)
    np.testing.assert_allclose(model.state[:3], reference.y[:3, -1], atol=0.003, rtol=0)
    np.testing.assert_allclose(
        model.state[4:6], reference.y[4:6, -1], atol=0.005, rtol=0
    )
