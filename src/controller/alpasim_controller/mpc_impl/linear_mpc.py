# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Linear MPC controller using OSQP.

Linearizes bicycle dynamics around current state and solves a QP for optimal control.
Uses an affine linearization of the shared discrete bicycle model.
"""

import logging
import math

import casadi
import numpy as np
import osqp
from alpasim_controller.mpc_controller import (
    DEFAULT_DT_MPC,
    DEFAULT_MAX_SPEED_MPS,
    DEFAULT_N_HORIZON,
    ControllerInput,
    MPCController,
    MPCGains,
)
from alpasim_controller.vehicle_model import VehicleModel, integrate_vehicle_state
from alpasim_utils.geometry import Trajectory
from scipy import sparse


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
        n_horizon: int = DEFAULT_N_HORIZON,
        dt_mpc: float = DEFAULT_DT_MPC,
    ):
        """Initialize the LinearMPC controller.

        Args:
            vehicle_params: Vehicle parameters for dynamics model. If None,
                uses default parameters.
            gains: Cost function weights. If None, uses default gains.
            n_horizon: Prediction horizon length.
            dt_mpc: MPC timestep in seconds.
        """
        self._vehicle_params = vehicle_params or VehicleModel.Parameters()
        self._gains = gains or MPCGains()
        self._n_horizon = n_horizon
        self._dt_mpc = dt_mpc

        state = casadi.SX.sym("state", self.NX)
        control = casadi.SX.sym("control", self.NU)
        next_state, _ = integrate_vehicle_state(
            state, control, self._dt_mpc, self._vehicle_params
        )
        self._dynamics_linearization = casadi.Function(
            "linear_mpc_dynamics",
            [state, control],
            [
                next_state,
                casadi.jacobian(next_state, state),
                casadi.jacobian(next_state, control),
            ],
        )
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
                DEFAULT_MAX_SPEED_MPS,  # vx (m/s)
                math.pi / 4,  # steering (rad)
                6.0,  # accel (m/s²) - acceleration limit
            ]
        )
        self._u_min = np.array([-2.0, -9.0])
        self._u_max = np.array([2.0, 6.0])

    @property
    def name(self) -> str:
        return "linear_mpc"

    def reset_warm_start(self) -> None:
        """Each solve replaces the native workspace, so no warm state is reused."""
        return

    def _compute_control(self, input: ControllerInput) -> tuple[np.ndarray | None, str]:
        """Compute optimal control using linearized MPC.

        Args:
            input: Current state, reference trajectory, and timestamp

        Returns:
            Optimal control command and solver metadata
        """
        x0 = input.state

        # Interpolate reference trajectory at MPC horizon timesteps
        x_ref = self._interpolate_reference(
            input.reference_trajectory,
            input.timestamp_us,
        )

        # Solve QP
        try:
            return self._solve_qp(
                x0, x_ref, self._Q, self._R, self._Qf, self._gains.idx_start_penalty
            )
        except osqp.OSQPException as exc:
            return None, str(exc)

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
        x_ref = np.zeros((self._n_horizon + 1, self.NX))

        dt_us = int(self._dt_mpc * 1e6)
        timestamps = np.array(
            [timestamp_us + k * dt_us for k in range(self._n_horizon + 1)],
            dtype=np.uint64,
        )

        # Clamp to reference trajectory range
        ref_range = ref_trajectory.time_range_us
        timestamps = np.clip(timestamps, ref_range.start, ref_range.stop - 1)

        # Batch interpolate all poses at once using Trajectory
        ref_interp = ref_trajectory.interpolate(timestamps.astype(np.uint64))

        # Extract x, y, yaw for all horizon steps
        for k in range(self._n_horizon + 1):
            pose = ref_interp.get_pose(k)
            x_ref[k, self.IX] = pose.vec3[0]
            x_ref[k, self.IY] = pose.vec3[1]
            x_ref[k, self.IYAW] = pose.yaw()

        return x_ref

    def _linearize_dynamics(
        self, x_op: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return A, B, c for x_next = A x + B u + c about zero command.

        The affine residual preserves the operating-point prediction when
        differentiating products such as speed times steering.
        """
        next_state, A, B = self._dynamics_linearization(x_op, np.zeros(self.NU))
        A = np.asarray(A)
        B = np.asarray(B)
        residual = np.asarray(next_state).reshape(self.NX) - A @ x_op
        return A, B, residual

    def _solve_qp(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        idx_start_penalty: int,
    ) -> tuple[np.ndarray | None, str]:
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
        N = self._n_horizon
        nx = self.NX
        nu = self.NU

        # Linearize around current state
        A_d, B_d, residual = self._linearize_dynamics(x0)

        # Build condensed prediction matrices
        # x_k = S_x[k] @ x_0 + S_c[k] + S_u[k] @ U
        S_x = np.zeros(((N + 1) * nx, nx))
        S_u = np.zeros(((N + 1) * nx, N * nu))

        S_c = np.zeros((N + 1, nx))
        A_pow = np.eye(nx)
        for k in range(N + 1):
            S_x[k * nx : (k + 1) * nx, :] = A_pow
            if k < N:
                A_pow = A_d @ A_pow
                S_c[k + 1] = A_d @ S_c[k] + residual

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
        x_pred_free = S_x @ x0 + S_c.ravel()

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
            polishing=False,
        )
        result = self._qp_solver.solve(raise_error=False)

        if result.info.status not in ("solved", "solved inaccurate"):
            logging.debug(f"OSQP solver status: {result.info.status}")
            return None, result.info.status

        return result.x[:nu], result.info.status
