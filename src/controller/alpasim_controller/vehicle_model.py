# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Vehicle dynamics model using planar dynamic bicycle model.

The VehicleModel class simulates vehicle dynamics and computes relative pose
over an advance call. Used by the System class for vehicle simulation.
"""

import logging
import math
from dataclasses import dataclass

import casadi
import numpy as np
from alpasim_utils.geometry import Pose


class VehicleModel:
    """Planar dynamic bicycle model for vehicle simulation.

    The state is:
        [x, y, yaw, vx_cg, vy_cg, yaw_rate, steering_angle, acceleration]

    Where:
        - x, y, yaw: Position and orientation of rig origin in inertial frame
        - vx_cg, vy_cg: CG velocity components in body frame
        - yaw_rate: Angular velocity in body frame
        - steering_angle: Front wheel steering angle
        - acceleration: Longitudinal acceleration state

    Note: Velocity is measured at the body center of gravity (CG), whereas
    the interface (trajectories, gRPC states) use rig frame velocity.

    Note: Lateral and longitudinal dynamics are decoupled.
    """

    @dataclass
    class Parameters:
        """Vehicle model parameters. Default values from Ford Fusion."""

        mass: float = 2014.4  # Mass [kg]
        inertia: float = 3414.2  # Moment of inertia around z-axis [kg*m^2]
        l_rig_to_cg: float = 1.59  # Distance from rear wheel to CoG [m]
        wheelbase: float = 2.85  # Wheelbase [m]
        front_cornering_stiffness: float = 93534.5  # Front cornering stiffness [N/rad]
        rear_cornering_stiffness: float = 176162.1  # Rear cornering stiffness [N/rad]
        steering_time_constant: float = 0.1  # Steering response time constant [s]
        acceleration_time_constant: float = 0.1  # Accel response time constant [s]

    def __init__(
        self,
        initial_velocity: np.ndarray,
        initial_yaw_rate: float,
    ):
        """Initialize the vehicle model.

        Args:
            initial_velocity: Initial velocity in the rig frame [vx, vy] [m/s]
            initial_yaw_rate: Initial yaw rate in the rig frame [rad/s]
        """
        self._parameters = self.Parameters()

        # Approximate kinematic steering angle from initial conditions
        if initial_velocity[0] > 0.25:
            initial_steering_angle = math.atan(
                initial_yaw_rate / initial_velocity[0] * self._parameters.wheelbase
            )
        else:
            initial_steering_angle = 0.0

        self._state = np.array(
            [
                0.0,  # x
                0.0,  # y
                0.0,  # yaw
                initial_velocity[0],  # vx_cg
                initial_velocity[1],  # vy_cg
                initial_yaw_rate,  # yaw_rate
                initial_steering_angle,  # steering_angle
                0.0,  # acceleration
            ]
        )

        # Final-substep body-velocity derivatives [d_vx_cg, d_vy_cg, d_yaw_rate]
        self._accelerations = np.array([0.0, 0.0, 0.0])

    @property
    def parameters(self) -> "VehicleModel.Parameters":
        """Get vehicle parameters."""
        return self._parameters

    @property
    def state(self) -> np.ndarray:
        """Get current vehicle state."""
        return self._state

    @property
    def front_steering_angle(self) -> float:
        """Get current front steering angle."""
        return self._state[6]

    @property
    def accelerations(self) -> np.ndarray:
        """Get body-velocity derivatives over the final integration substep.

        Returns:
            Array of [d_vx_cg, d_vy_cg, d_yaw_rate] where:
            - d_vx_cg: longitudinal acceleration at CG [m/s²]
            - d_vy_cg: lateral acceleration at CG [m/s²]
            - d_yaw_rate: angular acceleration [rad/s²]
        """
        return self._accelerations

    def reset_origin(self) -> None:
        """Reset position states to origin (x, y, yaw) = 0."""
        self._state[:3] = 0.0

    def coerce_dynamic_state(
        self,
        v_cg_x: float,
        v_cg_y: float,
        yaw_rate: float,
        longitudinal_acceleration: float,
    ) -> None:
        """Replace dynamic state with values estimated from ground truth.

        Args:
            v_cg_x: Longitudinal velocity of the CG [m/s]
            v_cg_y: Lateral velocity of the CG [m/s]
            yaw_rate: Angular velocity in the body frame [rad/s]
            longitudinal_acceleration: Longitudinal acceleration [m/s²]
        """
        self._state[3] = v_cg_x
        self._state[4] = v_cg_y
        self._state[5] = yaw_rate
        if v_cg_x > 0.25:
            self._state[6] = math.atan(yaw_rate / v_cg_x * self._parameters.wheelbase)
        else:
            self._state[6] = 0.0
        self._state[7] = longitudinal_acceleration
        _, self._accelerations = integrate_vehicle_state(
            self._state, self._state[6:8], 0.01, self._parameters
        )

    def advance(self, u: np.ndarray, dt: float) -> Pose:
        """Advance the vehicle model by dt seconds.

        Uses a speed-scaled backward Euler lateral solve with sub-stepping.

        Args:
            u: Control input [steering_cmd, accel_cmd]
            dt: Time step in seconds

        Returns:
            Relative pose (pose_rig_t0_to_rig_t1) if reset_origin was called
            prior, otherwise absolute pose in inertial frame.
        """
        self._state, self._accelerations = integrate_vehicle_state(
            self._state, u, dt, self._parameters
        )

        logging.debug("state (after prop): %s", self._state)

        return Pose(
            np.array([self._state[0], self._state[1], 0], dtype=np.float32),
            np.array(
                [0, 0, math.sin(self._state[2] / 2), math.cos(self._state[2] / 2)],
                dtype=np.float32,
            ),
        )


def lateral_dynamics_coefficients(
    params: VehicleModel.Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return C and b for v * z_dot = (C - v**2 E01) z + v b steering."""
    lr = params.l_rig_to_cg
    lf = params.wheelbase - lr
    cf = params.front_cornering_stiffness
    cr = params.rear_cornering_stiffness
    return np.array(
        [
            [-2 * (cf + cr) / params.mass, -2 * (lf * cf - lr * cr) / params.mass],
            [
                -2 * (lf * cf - lr * cr) / params.inertia,
                -2 * (lf * lf * cf + lr * lr * cr) / params.inertia,
            ],
        ]
    ), np.array([2 * cf / params.mass, 2 * lf * cf / params.inertia])


def integrate_vehicle_state(
    state: np.ndarray | casadi.SX,
    control: np.ndarray | casadi.SX,
    dt: float,
    params: VehicleModel.Parameters,
) -> tuple[np.ndarray | casadi.SX, np.ndarray | casadi.SX]:
    """Propagate the forward-only bicycle model, numerically or symbolically.

    The lateral system is z_dot = A(v) z + b delta, z = [vy, yaw_rate].
    Multiplying backward Euler by the end-of-step speed gives
    (v I - h v A(v)) z_next = v z + h v b delta_next. This removes all
    divisions by speed. At v=0 the nonsingular tire matrix enforces z_next=0;
    at small positive speed its equilibrium approaches the no-slip limit.

    Actuator states use their exact constant-command solution. Pose uses
    trapezoidal velocities and midpoint heading, with the same rear-axle
    velocity conversion at every speed. Linear tires and small steering/slip
    angles are assumed; this is not a standstill friction or tire saturation model.

    Returns:
        State after dt and body-velocity derivatives over the final substep.
        The discrete update serves the simulator, linear MPC, and SQP MPC.
    """
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and non-negative")
    symbolic = isinstance(state, casadi.SX)
    ops = casadi if symbolic else np
    if dt == 0:
        return state, casadi.SX.zeros(3) if symbolic else np.zeros(3)

    num_steps = max(1, math.ceil(dt / 0.01))
    h = dt / num_steps
    lr = params.l_rig_to_cg
    coefficients, steering_coefficients = lateral_dynamics_coefficients(params)
    c00, c01_base, c10, c11 = coefficients.ravel()
    b0, b1 = steering_coefficients
    steering_decay = math.exp(-h / params.steering_time_constant)
    accel_decay = math.exp(-h / params.acceleration_time_constant)

    for _ in range(num_steps):
        steering = control[0] + (state[6] - control[0]) * steering_decay
        accel = control[1] + (state[7] - control[1]) * accel_decay
        vx_unclamped = (
            state[3]
            + control[1] * h
            + (state[7] - control[1])
            * params.acceleration_time_constant
            * (1 - accel_decay)
        )
        # At rest use the forward (right) derivative. fmax's half derivative
        # at equality compounds across substeps and hides launch acceleration
        # from the optimizer over a stationary prediction horizon.
        vx = (
            casadi.if_else(vx_unclamped >= 0, vx_unclamped, 0.0)
            if symbolic
            else max(0.0, vx_unclamped)
        )
        c01 = c01_base - vx * vx
        m00 = vx - h * c00
        m01 = -h * c01
        m10 = -h * c10
        m11 = vx - h * c11
        rhs0 = vx * (state[4] + h * b0 * steering)
        rhs1 = vx * (state[5] + h * b1 * steering)
        determinant = m00 * m11 - m01 * m10
        vy = (m11 * rhs0 - m01 * rhs1) / determinant
        yaw_rate = (m00 * rhs1 - m10 * rhs0) / determinant

        yaw = state[2] + h * (state[5] + yaw_rate) / 2
        yaw_mid = (state[2] + yaw) / 2
        vx_mid = (state[3] + vx) / 2
        vy_rig_mid = (state[4] + vy - lr * (state[5] + yaw_rate)) / 2
        values = [
            state[0] + h * (vx_mid * ops.cos(yaw_mid) - vy_rig_mid * ops.sin(yaw_mid)),
            state[1] + h * (vx_mid * ops.sin(yaw_mid) + vy_rig_mid * ops.cos(yaw_mid)),
            yaw,
            vx,
            vy,
            yaw_rate,
            steering,
            accel,
        ]
        next_state = casadi.vertcat(*values) if symbolic else np.array(values)
        accelerations = (next_state[3:6] - state[3:6]) / h
        state = next_state

    return state, accelerations
