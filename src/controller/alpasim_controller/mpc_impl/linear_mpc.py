# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Linear MPC controller using OSQP.

Linearizes bicycle dynamics around current state and solves a QP for optimal control.
Uses matrix exponential discretization for accuracy with the dynamic bicycle model.
"""

import logging
import math
import time

import numpy as np
import osqp
import scipy.linalg
import scipy.sparse as sparse
from alpasim_controller.mpc_controller import (
    ControllerInput,
    ControllerOutput,
    MPCController,
    MPCGains,
)
from alpasim_controller.vehicle_model import VehicleModel
from alpasim_utils.geometry import Trajectory


class LinearMPC(MPCController):
    """Linear MPC using OSQP QP solver.

    Linearizes bicycle dynamics around current state, formulates a QP tracking
    problem, and solves with OSQP.
    """

    # State indices
    IX = 0  # x position
    IY = 1  # y position
    IYAW = 2  # yaw angle
    IVX = 3  # longitudinal velocity (cg)
    IVY = 4  # lateral velocity (cg)
    IYAW_RATE = 5  # yaw rate
    ISTEERING = 6  # steering angle
    IACCEL = 7  # acceleration state

    NX = 8  # Number of states
    NU = 2  # Number of controls [steering_cmd, accel_cmd]

    # Solver parameters
    OSQP_EPS_ABS = 1e-4
    OSQP_EPS_REL = 1e-4
    OSQP_MAX_ITER = 500
    QP_REGULARIZATION = 1e-6

    def __init__(
        self,
        vehicle_params: VehicleModel.Parameters | None = None,
        gains: MPCGains | None = None,
    ):
        """Initialize the LinearMPC controller.

        Args:
            vehicle_params: Vehicle parameters for dynamics model. If None,
                uses default parameters.
            gains: Cost function weights. If None, uses default gains.
        """
        self._vehicle_params = vehicle_params or VehicleModel.Parameters()
        self._gains = gains or MPCGains()

        self._qp_solver = osqp.OSQP()

        # Build/cache cost matrices
        self._Q = np.diag(
            [
                self._gains.long_position_weight,  # x
                self._gains.lat_position_weight,  # y
                self._gains.heading_weight,  # yaw
                0.0,  # vx - no tracking
                0.0,  # vy - no tracking
                0.0,  # yaw_rate - no tracking
                0.0,  # steering - no tracking
                self._gains.acceleration_weight,  # accel
            ]
        )
        self._R = np.diag(
            [
                self._gains.rel_front_steering_angle_weight,
                self._gains.rel_acceleration_weight,
            ]
        )
        self._Qf = self._Q.copy()  # Terminal cost same as stage cost

        # State constraints for a subset of the states
        self._constrained_state_indices = [
            self.IYAW,
            self.IVX,
            self.ISTEERING,
            self.IACCEL,
        ]
        self._x_min_constrained = np.array(
            [
                -math.pi / 2,  # yaw (rad) - ±90 degrees over horizon
                0.0,  # vx (m/s) - no reverse
                -math.pi / 4,  # steering (rad) - ±45 degrees
                -8.0,  # accel (m/s²) - braking limit
            ]
        )
        self._x_max_constrained = np.array(
            [
                math.pi / 2,  # yaw (rad)
                35.0,  # vx (m/s) - ~125 km/h
                math.pi / 4,  # steering (rad)
                6.0,  # accel (m/s²) - acceleration limit
            ]
        )
        self._u_min = np.array([-2.0, -9.0])
        self._u_max = np.array([2.0, 6.0])

    @property
    def name(self) -> str:
        return "linear_mpc"

    def compute_control(self, input: ControllerInput) -> ControllerOutput:
        """Compute optimal control using linearized MPC.

        Args:
            input: Current state, reference trajectory, and timestamp

        Returns:
            Optimal control command and solver metadata
        """
        start_time = time.perf_counter()

        x0 = input.state

        # Interpolate reference trajectory at MPC horizon timesteps
        x_ref = self._interpolate_reference(
            input.reference_trajectory,
            input.timestamp_us,
        )

        # Solve QP
        u_opt, status = self._solve_qp(
            x0, x_ref, self._Q, self._R, self._Qf, self._gains.idx_start_penalty
        )

        solve_time_ms = (time.perf_counter() - start_time) * 1000

        return ControllerOutput(
            control=u_opt,
            solve_time_ms=solve_time_ms,
            status=status,
        )

    def _interpolate_reference(
        self,
        ref_trajectory: Trajectory,
        timestamp_us: int,
    ) -> np.ndarray:
        """Interpolate reference trajectory at MPC horizon timesteps.

        Args:
            ref_trajectory: Reference trajectory in rig frame
            timestamp_us: Current timestamp in microseconds

        Returns:
            Reference state array (N+1, NX) with x, y, yaw populated
        """
        x_ref = np.zeros((self.N_HORIZON + 1, self.NX))

        dt_us = int(self.DT_MPC * 1e6)
        timestamps = np.array(
            [timestamp_us + k * dt_us for k in range(self.N_HORIZON + 1)],
            dtype=np.uint64,
        )

        # Clamp to reference trajectory range
        ref_range = ref_trajectory.time_range_us
        timestamps = np.clip(timestamps, ref_range.start, ref_range.stop - 1)

        # Batch interpolate all poses at once using Trajectory
        ref_interp = ref_trajectory.interpolate(timestamps.astype(np.uint64))

        # Extract x, y, yaw for all horizon steps
        for k in range(self.N_HORIZON + 1):
            pose = ref_interp.get_pose(k)
            x_ref[k, self.IX] = pose.vec3[0]
            x_ref[k, self.IY] = pose.vec3[1]
            x_ref[k, self.IYAW] = pose.yaw()

        return x_ref

    def _linearize_dynamics(self, x_op: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearize the bicycle model around operating point.

        Returns discrete-time matrices (A_d, B_d) such that:
            x_{k+1} = A_d @ x_k + B_d @ u_k

        Args:
            x_op: Operating point state vector (8,)

        Returns:
            A_d: Discrete-time state matrix (NX, NX)
            B_d: Discrete-time input matrix (NX, NU)
        """
        params = self._vehicle_params
        dt = self.DT_MPC

        # Extract operating point values
        yaw = x_op[self.IYAW]
        v_cg_x = x_op[self.IVX]
        v_cg_y = x_op[self.IVY]
        yaw_rate = x_op[self.IYAW_RATE]
        steering = x_op[self.ISTEERING]

        # Use kinematic model at low speeds
        use_kinematic = v_cg_x < params.kinematic_threshold_speed

        # Initialize Jacobians
        A = np.zeros((self.NX, self.NX))
        B = np.zeros((self.NX, self.NU))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        if use_kinematic:
            # Kinematic model linearization
            A[self.IX, self.IYAW] = -v_cg_x * sin_yaw
            A[self.IX, self.IVX] = cos_yaw

            A[self.IY, self.IYAW] = v_cg_x * cos_yaw
            A[self.IY, self.IVX] = sin_yaw

            A[self.IYAW, self.IYAW_RATE] = 1.0
            A[self.IVX, self.IACCEL] = 1.0

            GAIN = 10.0
            l_r = params.l_rig_to_cg
            L = params.wheelbase
            A[self.IVY, self.IVX] = GAIN * steering * l_r / L
            A[self.IVY, self.IVY] = -GAIN
            A[self.IVY, self.ISTEERING] = GAIN * v_cg_x * l_r / L

            A[self.IYAW_RATE, self.IVX] = GAIN * steering / L
            A[self.IYAW_RATE, self.ISTEERING] = GAIN * v_cg_x / L
            A[self.IYAW_RATE, self.IYAW_RATE] = -GAIN

        else:
            # Dynamic model linearization
            kinetic_mass = params.mass * v_cg_x
            kinetic_inertia = params.inertia * v_cg_x

            lf = params.wheelbase - params.l_rig_to_cg
            lr = params.l_rig_to_cg
            Caf = params.front_cornering_stiffness
            Car = params.rear_cornering_stiffness

            lf_caf = lf * Caf
            lr_car = lr * Car

            a_00 = -2 * (Caf + Car) / kinetic_mass
            a_01 = -v_cg_x - 2 * (lf_caf - lr_car) / kinetic_mass
            a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
            a_11 = -2 * (lf * lf_caf + lr * lr_car) / kinetic_inertia

            b_00 = 2 * Caf / params.mass
            b_10 = 2 * lf_caf / params.inertia

            v_rig_y = v_cg_y - lr * yaw_rate

            A[self.IX, self.IYAW] = -v_cg_x * sin_yaw - v_rig_y * cos_yaw
            A[self.IX, self.IVX] = cos_yaw
            A[self.IX, self.IVY] = -sin_yaw
            A[self.IX, self.IYAW_RATE] = lr * sin_yaw

            A[self.IY, self.IYAW] = v_cg_x * cos_yaw - v_rig_y * sin_yaw
            A[self.IY, self.IVX] = sin_yaw
            A[self.IY, self.IVY] = cos_yaw
            A[self.IY, self.IYAW_RATE] = -lr * cos_yaw

            A[self.IYAW, self.IYAW_RATE] = 1.0
            A[self.IVX, self.IACCEL] = 1.0

            A[self.IVY, self.IVY] = a_00
            A[self.IVY, self.IYAW_RATE] = a_01
            A[self.IVY, self.ISTEERING] = b_00

            A[self.IYAW_RATE, self.IVY] = a_10
            A[self.IYAW_RATE, self.IYAW_RATE] = a_11
            A[self.IYAW_RATE, self.ISTEERING] = b_10

        # Steering dynamics
        tau_s = params.steering_time_constant
        A[self.ISTEERING, self.ISTEERING] = -1.0 / tau_s
        B[self.ISTEERING, 0] = 1.0 / tau_s

        # Acceleration dynamics
        tau_a = params.acceleration_time_constant
        A[self.IACCEL, self.IACCEL] = -1.0 / tau_a
        B[self.IACCEL, 1] = 1.0 / tau_a

        # Discretize using matrix exponential
        nx = self.NX
        nu = self.NU
        M = np.zeros((nx + nu, nx + nu))
        M[:nx, :nx] = A * dt
        M[:nx, nx:] = B * dt

        expM = scipy.linalg.expm(M)
        A_d = expM[:nx, :nx]
        B_d = expM[:nx, nx:]

        return A_d, B_d

    def _solve_qp(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        idx_start_penalty: int,
    ) -> tuple[np.ndarray, str]:
        """Solve the MPC QP problem using condensed formulation.

        Cost: J = sum_k (x_k - x_ref_k)' Q (x_k - x_ref_k) + u_k' R u_k
        Subject to:
            x_{k+1} = A @ x_k + B @ u_k  (dynamics)
            u_min <= u_k <= u_max        (input constraints)
            x_min <= x_k <= x_max        (state constraints)

        Args:
            x0: Initial state (NX,)
            x_ref: Reference trajectory (N+1, NX)
            Q: State cost matrix (NX, NX)
            R: Control cost matrix (NU, NU)
            Qf: Terminal cost matrix (NX, NX)
            idx_start_penalty: Horizon index where tracking starts

        Returns:
            u_opt: Optimal first control input (NU,)
            status: Solver status string
        """
        N = self.N_HORIZON
        nx = self.NX
        nu = self.NU

        # Linearize around current state
        A_d, B_d = self._linearize_dynamics(x0)

        # Build condensed prediction matrices
        # x_k = S_x[k] @ x_0 + S_u[k] @ U
        S_x = np.zeros(((N + 1) * nx, nx))
        S_u = np.zeros(((N + 1) * nx, N * nu))

        A_pow = np.eye(nx)
        for k in range(N + 1):
            S_x[k * nx : (k + 1) * nx, :] = A_pow
            if k < N:
                A_pow = A_d @ A_pow

        for j in range(N):
            Psi = B_d
            for k in range(j, N):
                S_u[(k + 1) * nx : (k + 2) * nx, j * nu : (j + 1) * nu] = Psi
                Psi = A_d @ Psi

        # Build block-diagonal cost matrices
        Q_blk = np.zeros(((N + 1) * nx, (N + 1) * nx))
        for k in range(N):
            if k >= idx_start_penalty:
                Q_blk[k * nx : (k + 1) * nx, k * nx : (k + 1) * nx] = Q
        if N >= idx_start_penalty:
            Q_blk[N * nx :, N * nx :] = Qf

        R_blk = np.zeros((N * nu, N * nu))
        for k in range(N):
            R_blk[k * nu : (k + 1) * nu, k * nu : (k + 1) * nu] = R

        # Flatten reference trajectory
        x_ref_flat = x_ref.flatten()

        # Predicted state with zero control
        x_pred_free = S_x @ x0

        # Condensed QP cost: min 0.5 * U' H U + g' U
        H = S_u.T @ Q_blk @ S_u + R_blk
        g = S_u.T @ Q_blk @ (x_pred_free - x_ref_flat)

        # Make H symmetric and add regularization
        H = 0.5 * (H + H.T) + self.QP_REGULARIZATION * np.eye(N * nu)
        H_sparse = sparse.csc_matrix(H)

        # Build constraints
        A_input = sparse.eye(N * nu, format="csc")

        # State constraints for constrained states only
        constrained_rows = []
        for k in range(1, N + 1):
            for idx in self._constrained_state_indices:
                constrained_rows.append(k * nx + idx)

        S_u_for_states = S_u[constrained_rows, :]
        A_state = sparse.csc_matrix(S_u_for_states)
        A_ineq = sparse.vstack([A_input, A_state], format="csc")

        # Bounds
        l_input = np.tile(self._u_min, N)
        u_input = np.tile(self._u_max, N)

        x_pred_from_x0 = x_pred_free[constrained_rows]
        l_state = np.tile(self._x_min_constrained, N) - x_pred_from_x0
        u_state = np.tile(self._x_max_constrained, N) - x_pred_from_x0

        l_ineq = np.concatenate([l_input, l_state])
        u_ineq = np.concatenate([u_input, u_state])

        # Setup and solve
        self._qp_solver.setup(
            P=H_sparse,
            q=g,
            A=A_ineq,
            l=l_ineq,
            u=u_ineq,
            verbose=False,
            eps_abs=self.OSQP_EPS_ABS,
            eps_rel=self.OSQP_EPS_REL,
            max_iter=self.OSQP_MAX_ITER,
            polish=False,
        )
        result = self._qp_solver.solve()

        if result.info.status not in ("solved", "solved_inaccurate"):
            logging.debug(f"OSQP solver status: {result.info.status}")
            return np.zeros(nu), result.info.status

        return result.x[:nu], result.info.status
