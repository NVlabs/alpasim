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

import numpy as np
from alpasim_utils import trajectory


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
        kinematic_threshold_speed: float = 5.0  # Speed for kinematic model [m/s]

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

        # Store last computed accelerations [d_vx_cg, d_vy_cg, d_yaw_rate]
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
        """Get last computed accelerations in CG frame.

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

    def set_velocity(self, v_cg_x: float, v_cg_y: float) -> None:
        """Set CG velocity in the rig/body frame.

        Args:
            v_cg_x: Longitudinal velocity of the CG [m/s]
            v_cg_y: Lateral velocity of the CG [m/s]
        """
        self._state[3] = v_cg_x
        self._state[4] = v_cg_y

    def advance(self, u: np.ndarray, dt: float) -> trajectory.QVec:
        """Advance the vehicle model by dt seconds.

        Uses 2nd order Runge-Kutta integration with sub-stepping.

        Args:
            u: Control input [steering_cmd, accel_cmd]
            dt: Time step in seconds

        Returns:
            Relative pose (pose_rig_t0_to_rig_t1) if reset_origin was called
            prior, otherwise absolute pose in inertial frame.
        """
        DT_STEP_MAX = 0.01

        logging.debug("state: %s, u: %s, dt: %s", self._state, u, dt)

        total_time = 0.0
        while total_time < dt:
            step_dt = min(DT_STEP_MAX, dt - total_time)
            total_time += step_dt

            # 2nd order Runge-Kutta
            k1 = step_dt * self._derivs(self._state, u)
            k2 = step_dt * self._derivs(self._state + k1 / 2.0, u)
            self._state += k2
            self._state[3] = max(0.0, self._state[3])  # Ensure non-negative velocity

        # Store final accelerations
        final_derivs = self._derivs(self._state, u)
        self._accelerations = np.array(
            [final_derivs[3], final_derivs[4], final_derivs[5]]
        )

        logging.debug("state (after prop): %s", self._state)

        return trajectory.QVec(
            vec3=np.array([self._state[0], self._state[1], 0]),
            quat=np.array(
                [0, 0, math.sin(self._state[2] / 2), math.cos(self._state[2] / 2)]
            ),
        )

    def _derivs(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivatives.

        Args:
            state: Current state vector
            u: Control input [steering_cmd, accel_cmd]

        Returns:
            State derivative vector
        """
        yaw_angle = state[2]
        v_x = state[3]
        v_y = state[4]
        yaw_rate = state[5]
        front_steering_angle = state[6]
        longitudinal_acceleration = state[7]

        use_kinematic_model = v_x < self._parameters.kinematic_threshold_speed

        if use_kinematic_model:
            # Kinematic model - pull system to no-slip conditions
            steady_state_v_y = (
                v_x
                * front_steering_angle
                * self._parameters.l_rig_to_cg
                / self._parameters.wheelbase
            )
            steady_state_yaw_rate = (
                v_x * front_steering_angle / self._parameters.wheelbase
            )
            GAIN = 10.0
            v_y_rig = 0.0  # No slip at rear
            d_v_y = GAIN * (steady_state_v_y - v_y)
            d_yaw_rate = GAIN * (steady_state_yaw_rate - yaw_rate)
        else:
            # Dynamic bicycle model
            kinetic_mass = self._parameters.mass * v_x
            kinetic_inertia = self._parameters.inertia * v_x
            lf = self._parameters.wheelbase - self._parameters.l_rig_to_cg
            lf_caf = lf * self._parameters.front_cornering_stiffness
            lr_car = (
                self._parameters.l_rig_to_cg * self._parameters.rear_cornering_stiffness
            )
            lf_sq_caf = lf * lf_caf
            lr_sq_car = self._parameters.l_rig_to_cg * lr_car

            a_00 = (
                -2
                * (
                    self._parameters.front_cornering_stiffness
                    + self._parameters.rear_cornering_stiffness
                )
                / kinetic_mass
            )
            a_01 = -v_x - 2 * (lf_caf - lr_car) / kinetic_mass
            a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
            a_11 = -2 * (lf_sq_caf + lr_sq_car) / kinetic_inertia

            b_00 = (
                2 * self._parameters.front_cornering_stiffness / self._parameters.mass
            )
            b_10 = 2 * lf_caf / self._parameters.inertia

            v_y_rig = v_y - state[5] * self._parameters.l_rig_to_cg

            d_v_y = a_00 * v_y + a_01 * yaw_rate + b_00 * front_steering_angle
            d_yaw_rate = a_10 * v_y + a_11 * yaw_rate + b_10 * front_steering_angle

        front_steering_angle_cmd = u[0]
        longitudinal_acceleration_cmd = u[1]

        return np.array(
            [
                v_x * math.cos(yaw_angle) - v_y_rig * math.sin(yaw_angle),
                v_x * math.sin(yaw_angle) + v_y_rig * math.cos(yaw_angle),
                yaw_rate,
                longitudinal_acceleration,
                d_v_y,
                d_yaw_rate,
                (front_steering_angle_cmd - front_steering_angle)
                / self._parameters.steering_time_constant,
                (longitudinal_acceleration_cmd - longitudinal_acceleration)
                / self._parameters.acceleration_time_constant,
            ]
        )
