# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Sequential quadratic programming MPC using CasADi Jacobians and OSQP."""

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
from scipy import sparse

logger = logging.getLogger(__name__)


class SQPMPC(MPCController):
    """MPC solved by repeated linearization of the nonlinear bicycle model."""

    IX = 0
    IY = 1
    IYAW = 2
    IVX = 3
    IVY = 4
    IYAW_RATE = 5
    ISTEERING = 6
    IACCEL = 7

    NX = 8
    NU = 2

    OSQP_EPS_ABS = 1e-4
    OSQP_EPS_REL = 1e-4
    OSQP_MAX_ITER = 1000
    MIN_REFERENCE_SPEED = 0.1
    OSQP_ACCEPTED_STATUSES = {"solved", "solved inaccurate"}

    def __init__(
        self,
        vehicle_params: VehicleModel.Parameters | None = None,
        gains: MPCGains | None = None,
        n_horizon: int = DEFAULT_N_HORIZON,
        dt_mpc: float = DEFAULT_DT_MPC,
    ):
        self._vehicle_params = vehicle_params or VehicleModel.Parameters()
        self._gains = gains or MPCGains()
        if self._gains.sqp_max_iterations <= 0:
            raise ValueError("sqp_max_iterations must be positive")
        if self._gains.sqp_step_tolerance < 0.0:
            raise ValueError("sqp_step_tolerance must be non-negative")
        if self._gains.max_curvature_abs <= 0.0:
            raise ValueError("max_curvature_abs must be positive")
        if self._gains.max_curvature_speed_floor <= 0.0:
            raise ValueError("max_curvature_speed_floor must be positive")
        if self._gains.sqp_lat_position_ramp_steps < 0:
            raise ValueError("sqp_lat_position_ramp_steps must be non-negative")
        self._n_horizon = n_horizon
        self._dt_mpc = dt_mpc
        self._previous_control: np.ndarray | None = None
        steering_bound = min(
            math.pi / 4,
            math.atan(self._gains.max_curvature_abs * self._vehicle_params.wheelbase),
        )

        # State weights per horizon step, (n_horizon + 1, NX); only the lateral position
        # weight varies, ramping in over sqp_lat_position_ramp_steps.
        self._Q = np.tile(
            [
                self._gains.long_position_weight,
                self._gains.lat_position_weight,
                self._gains.heading_weight,
                self._gains.velocity_weight,
                self._gains.lateral_velocity_weight,
                self._gains.yaw_rate_weight,
                self._gains.steering_state_weight,
                self._gains.acceleration_weight,
            ],
            (n_horizon + 1, 1),
        )
        ramp_steps = self._gains.sqp_lat_position_ramp_steps
        if ramp_steps > 0:
            steps_tracked = np.arange(n_horizon + 1) - self._gains.idx_start_penalty
            self._Q[:, self.IY] *= np.clip(steps_tracked / ramp_steps, 0.0, 1.0)
        self._S = np.diag(
            [
                self._gains.rel_front_steering_angle_weight,
                self._gains.rel_acceleration_weight,
            ]
        )

        self._x_min = np.array(
            [-500.0, -20.0, -math.pi / 2, -5.0, -10.0, -3.0, -steering_bound, -8.0]
        )
        self._x_max = np.array(
            [
                500.0,
                20.0,
                math.pi / 2,
                DEFAULT_MAX_SPEED_MPS,
                10.0,
                3.0,
                steering_bound,
                6.0,
            ]
        )
        self._u_min = np.array([-steering_bound, -9.0])
        self._u_max = np.array([steering_bound, 6.0])

        self._discrete_dynamics, self._dynamics_jacobians = (
            self._build_discrete_dynamics()
        )
        self._horizon_linearization = self._build_horizon_linearization()
        self._num_x_vars = (self._n_horizon + 1) * self.NX
        self._num_vars = self._num_x_vars + self._n_horizon * self.NU
        self._num_eq_rows = self.NX * (self._n_horizon + 1)
        self._num_curvature_rows = 2 * self._n_horizon
        self._curvature_row_base = self._num_eq_rows + self._num_vars
        self._num_constraints = (
            self._num_eq_rows + self._num_vars + self._num_curvature_rows
        )
        self._qp_solver = osqp.OSQP()
        self._last_qp_solution: np.ndarray | None = None
        self._setup_qp_workspace()

    @property
    def name(self) -> str:
        return "sqp_mpc"

    def reset_warm_start(self) -> None:
        """Clear control and QP warm starts after a discontinuous state reset."""
        self._previous_control = None
        self._last_qp_solution = None
        self._setup_qp_workspace()

    def _compute_control(self, input: ControllerInput) -> tuple[np.ndarray | None, str]:
        """Compute a control command using stabilized SQP iterations."""
        x_ref, u_ref = self._build_reference(input)
        x_nom = x_ref.copy()
        u_nom = u_ref.copy()
        x_nom[0] = input.state

        if self._previous_control is None:
            self._previous_control = input.state[[self.ISTEERING, self.IACCEL]].copy()

        quality = "solved"
        iterations = "max sqp iterations"
        for iteration in range(self._gains.sqp_max_iterations):
            try:
                step_x, step_u, qp_status = self._solve_qp_step(
                    input.state,
                    x_ref,
                    x_nom,
                    u_nom,
                    self._previous_control,
                )
            except osqp.OSQPException as exc:
                return None, str(exc)
            if qp_status not in self.OSQP_ACCEPTED_STATUSES:
                return None, qp_status
            if qp_status == "solved inaccurate":
                quality = "solved inaccurate"

            x_nom += step_x
            u_nom += step_u
            x_nom[0] = input.state

            step_norm = max(
                float(np.max(np.abs(step_x))),
                float(np.max(np.abs(step_u))),
            )
            if step_norm <= self._gains.sqp_step_tolerance:
                iterations = f"{iteration + 1} sqp iterations"
                break
        control = np.clip(u_nom[0], self._u_min, self._u_max)
        self._previous_control = control.copy()
        return control, f"{quality} ({iterations})"

    def _build_discrete_dynamics(
        self,
    ) -> tuple[casadi.Function, casadi.Function]:
        """Build the shared discrete model and its exact CasADi Jacobians."""
        x = casadi.SX.sym("x", self.NX)
        u = casadi.SX.sym("u", self.NU)
        x_next = self._discrete_dynamics_expression(x, u)

        dynamics = casadi.Function("sqp_discrete_dynamics", [x, u], [x_next])
        jacobians = casadi.Function(
            "sqp_dynamics_jacobians",
            [x, u],
            [casadi.jacobian(x_next, x), casadi.jacobian(x_next, u)],
        )
        return dynamics, jacobians

    def _discrete_dynamics_expression(self, x: casadi.SX, u: casadi.SX) -> casadi.SX:
        """Use the simulator's low-speed-stable discrete dynamics."""
        return integrate_vehicle_state(x, u, self._dt_mpc, self._vehicle_params)[0]

    def _build_horizon_linearization(self) -> casadi.Function:
        """Compile one CasADi function for all horizon Jacobians and defects."""
        X = casadi.SX.sym("X", self.NX, self._n_horizon + 1)
        U = casadi.SX.sym("U", self.NU, self._n_horizon)

        a_blocks = []
        b_blocks = []
        defects = []
        for k in range(self._n_horizon):
            x_next = self._discrete_dynamics_expression(X[:, k], U[:, k])
            a_blocks.append(
                casadi.reshape(casadi.jacobian(x_next, X[:, k]), self.NX * self.NX, 1)
            )
            b_blocks.append(
                casadi.reshape(casadi.jacobian(x_next, U[:, k]), self.NX * self.NU, 1)
            )
            defects.append(x_next - X[:, k + 1])

        return casadi.Function(
            "sqp_horizon_linearization",
            [X, U],
            [
                casadi.vertcat(*a_blocks),
                casadi.vertcat(*b_blocks),
                casadi.vertcat(*defects),
            ],
        )

    def _build_reference(self, input: ControllerInput) -> tuple[np.ndarray, np.ndarray]:
        """Create a full-state reference from the reference poses.

        Speed, acceleration and curvature are differentiated from the reference poses.
        Velocity, lateral velocity, yaw rate, steering, and acceleration references
        are rolled out through the discrete bicycle model from those inputs.
        Position and yaw references are taken from the reference poses themselves:
        the rollout carries rear-axle slip that the (path-tangent) reference poses do
        not, so re-integrated positions drift to the outside of turns over the horizon.
        """
        horizon_timestamps = self._horizon_timestamps(input)
        pose_ref = self._reference_pose_array(input, horizon_timestamps)

        curvature, acceleration = self._reference_inputs(pose_ref)
        pose_speed = self._reference_speed_from_pose(pose_ref)
        speed = np.empty(self._n_horizon + 1)
        speed[0] = np.clip(pose_speed[0], 0.0, self._x_max[self.IVX])
        for k in range(self._n_horizon):
            speed[k + 1] = np.clip(
                speed[k] + self._dt_mpc * acceleration[k], 0.0, self._x_max[self.IVX]
            )

        steering = np.empty(self._n_horizon + 1)
        lateral_velocity = np.empty(self._n_horizon + 1)
        yaw_rate = speed * curvature
        for k in range(self._n_horizon + 1):
            lateral_velocity[k], steering[k] = self._steady_turn_state(
                speed[k], yaw_rate[k]
            )

        steering_rate = np.gradient(steering, self._dt_mpc)
        acceleration_rate = np.gradient(acceleration, self._dt_mpc)
        u_ref = np.column_stack(
            [
                steering + self._vehicle_params.steering_time_constant * steering_rate,
                acceleration
                + self._vehicle_params.acceleration_time_constant * acceleration_rate,
            ]
        )[: self._n_horizon]
        u_ref = np.clip(u_ref, self._u_min, self._u_max)

        # x, y and yaw are known exactly: they are the reference poses. The other
        # five states are not sent by the driver, so they are rolled out through the
        # bicycle model from the pose-derived inputs. The poses are not
        # rolled out as well because the model's rear-axle slip is absent from the
        # path-tangent reference yaw, so re-integrated positions drift to the outside
        # of turns over the horizon.
        x_ref = np.zeros((self._n_horizon + 1, self.NX))
        x_ref[:, [self.IX, self.IY, self.IYAW]] = pose_ref
        rollout = np.array(
            [
                pose_ref[0, 0],
                pose_ref[0, 1],
                pose_ref[0, 2],
                speed[0],
                lateral_velocity[0],
                yaw_rate[0],
                steering[0],
                acceleration[0],
            ]
        )
        x_ref[0, self.IVX :] = rollout[self.IVX :]
        for k in range(self._n_horizon):
            rollout = self._evaluate_dynamics(rollout, u_ref[k])
            x_ref[k + 1, self.IVX :] = rollout[self.IVX :]
        return x_ref, u_ref

    def _horizon_timestamps(self, input: ControllerInput) -> np.ndarray:
        dt_us = int(self._dt_mpc * 1e6)
        return np.array(
            [input.timestamp_us + k * dt_us for k in range(self._n_horizon + 1)],
            dtype=np.uint64,
        )

    def _reference_pose_array(
        self, input: ControllerInput, horizon_timestamps: np.ndarray
    ) -> np.ndarray:
        """Interpolate reference poses and extrapolate when the horizon outlives it."""
        trajectory = input.reference_trajectory
        source_timestamps = trajectory.timestamps_us.astype(np.float64)
        if len(source_timestamps) == 0:
            raise ValueError("reference trajectory must contain at least one pose")

        first_ts = source_timestamps[0]
        last_ts = source_timestamps[-1]
        clipped_timestamps = np.clip(horizon_timestamps, first_ts, last_ts).astype(
            np.uint64
        )
        poses = trajectory.interpolate(clipped_timestamps)

        pose_ref = np.zeros((self._n_horizon + 1, 3))
        for k in range(self._n_horizon + 1):
            pose = poses.get_pose(k)
            pose_ref[k] = [pose.vec3[0], pose.vec3[1], pose.yaw()]
        pose_ref[:, 2] = np.unwrap(pose_ref[:, 2])

        future_mask = horizon_timestamps.astype(np.float64) > last_ts
        if not np.any(future_mask):
            return pose_ref

        positions = trajectory.positions.astype(np.float64)
        yaws = np.unwrap(trajectory.yaws.astype(np.float64))
        if (
            len(source_timestamps) >= 2
            and source_timestamps[-1] > source_timestamps[-2]
        ):
            terminal_dt = (source_timestamps[-1] - source_timestamps[-2]) * 1e-6
            terminal_velocity = (positions[-1, :2] - positions[-2, :2]) / terminal_dt
            terminal_yaw_rate = (yaws[-1] - yaws[-2]) / terminal_dt
        else:
            terminal_speed = max(float(input.state[self.IVX]), 0.0)
            terminal_velocity = terminal_speed * np.array(
                [math.cos(pose_ref[-1, 2]), math.sin(pose_ref[-1, 2])]
            )
            terminal_yaw_rate = float(input.state[self.IYAW_RATE])

        future_dt = (
            horizon_timestamps[future_mask].astype(np.float64) - last_ts
        ) * 1e-6
        pose_ref[future_mask, :2] = (
            positions[-1, :2] + future_dt[:, None] * terminal_velocity
        )
        pose_ref[future_mask, 2] = yaws[-1] + future_dt * terminal_yaw_rate
        return pose_ref

    def _reference_inputs(self, pose_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yaw_rate = np.gradient(pose_ref[:, 2], self._dt_mpc)
        speed = self._reference_speed_from_pose(pose_ref)
        curvature = np.divide(
            yaw_rate,
            speed,
            out=np.zeros_like(yaw_rate),
            where=np.abs(speed) >= self.MIN_REFERENCE_SPEED,
        )
        acceleration = np.gradient(speed, self._dt_mpc)

        return curvature, acceleration

    def _reference_speed_from_pose(self, pose_ref: np.ndarray) -> np.ndarray:
        """Estimate longitudinal reference speed from pose samples."""
        dx = np.gradient(pose_ref[:, 0], self._dt_mpc)
        dy = np.gradient(pose_ref[:, 1], self._dt_mpc)
        return dx * np.cos(pose_ref[:, 2]) + dy * np.sin(pose_ref[:, 2])

    def _steady_turn_state(
        self, speed: float, desired_yaw_rate: float
    ) -> tuple[float, float]:
        params = self._vehicle_params
        # Steady force/moment balance, with a continuous no-slip limit.
        # A finite curvature at standstill implies desired_yaw_rate == 0.
        curvature = desired_yaw_rate / speed if speed > 0.0 else 0.0
        lf = params.wheelbase - params.l_rig_to_cg
        lr = params.l_rig_to_cg
        lateral_velocity = (
            lr
            - params.mass
            * lf
            * speed**2
            / (2 * params.rear_cornering_stiffness * params.wheelbase)
        ) * desired_yaw_rate
        understeer = (
            params.mass
            / (2 * params.wheelbase)
            * (
                lr / params.front_cornering_stiffness
                - lf / params.rear_cornering_stiffness
            )
        )
        steering = (params.wheelbase + understeer * speed**2) * curvature
        return float(lateral_velocity), float(steering)

    def _evaluate_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.asarray(self._discrete_dynamics(x, u)).reshape(self.NX)

    def _x_slice(self, k: int) -> slice:
        return slice(k * self.NX, (k + 1) * self.NX)

    def _u_slice(self, k: int) -> slice:
        start = self._num_x_vars + k * self.NU
        return slice(start, start + self.NU)

    def _setup_qp_workspace(self) -> None:
        """Create a fresh native workspace with fixed Hessian and constraint sparsity."""
        P = self._build_qp_hessian()
        A = self._build_constraint_matrix(
            np.ones((self._n_horizon, self.NX, self.NX)),
            np.ones((self._n_horizon, self.NX, self.NU)),
        )
        self._A_indices = A.indices.copy()
        self._A_indptr = A.indptr.copy()

        self._qp_solver.setup(
            P=P,
            q=np.zeros(self._num_vars),
            A=A,
            l=np.zeros(self._num_constraints),
            u=np.zeros(self._num_constraints),
            verbose=False,
            eps_abs=self.OSQP_EPS_ABS,
            eps_rel=self.OSQP_EPS_REL,
            max_iter=self.OSQP_MAX_ITER,
            polishing=False,
            warm_starting=True,
        )

    def _build_qp_hessian(self) -> sparse.csc_matrix:
        """Build the constant SQP Hessian for the delta-state/control QP."""
        P = sparse.lil_matrix((self._num_vars, self._num_vars))

        for k in range(self._gains.idx_start_penalty, self._n_horizon + 1):
            xs = self._x_slice(k)
            P[xs, xs] += 2.0 * np.diag(self._Q[k])

        for k in range(self._n_horizon):
            current = self._u_slice(k)
            P[current, current] += 2.0 * self._S
            if k > 0:
                previous = self._u_slice(k - 1)
                P[previous, previous] += 2.0 * self._S
                P[current, previous] += -2.0 * self._S
                P[previous, current] += -2.0 * self._S

        P[: self._num_x_vars, : self._num_x_vars] += (
            2.0 * self._gains.sqp_state_step_weight * sparse.eye(self._num_x_vars)
        )
        P[self._num_x_vars :, self._num_x_vars :] += (
            2.0
            * self._gains.sqp_control_step_weight
            * sparse.eye(self._n_horizon * self.NU)
        )

        return sparse.triu(P.tocsc(), format="csc")

    def _linearize_horizon(
        self, x_nom: np.ndarray, u_nom: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate all horizon dynamics Jacobians and defects in one CasADi call."""
        a_flat, b_flat, defect_flat = self._horizon_linearization(
            x_nom.T,
            u_nom.T,
        )
        A_blocks = np.asarray(a_flat).reshape(self._n_horizon, self.NX * self.NX)
        A_blocks = np.stack(
            [
                A_blocks[k].reshape(self.NX, self.NX, order="F")
                for k in range(self._n_horizon)
            ]
        )
        B_blocks = np.asarray(b_flat).reshape(self._n_horizon, self.NX * self.NU)
        B_blocks = np.stack(
            [
                B_blocks[k].reshape(self.NX, self.NU, order="F")
                for k in range(self._n_horizon)
            ]
        )
        defects = np.asarray(defect_flat).reshape(self._n_horizon, self.NX)
        return A_blocks, B_blocks, defects

    def _build_constraint_matrix(
        self,
        A_blocks: np.ndarray,
        B_blocks: np.ndarray,
    ) -> sparse.csc_matrix:
        """Build a fixed-pattern OSQP constraint matrix for current dynamics."""
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        # Initial condition: delta_x_0 = measured_state - x_nom_0.
        for i in range(self.NX):
            rows.append(i)
            cols.append(i)
            data.append(1.0)

        # Linearized dynamics:
        # delta_x_{k+1} - A_k delta_x_k - B_k delta_u_k = defect_k.
        for k in range(self._n_horizon):
            row_base = (k + 1) * self.NX
            x_col_base = k * self.NX
            x_next_col_base = (k + 1) * self.NX
            u_col_base = self._num_x_vars + k * self.NU

            for i in range(self.NX):
                rows.append(row_base + i)
                cols.append(x_next_col_base + i)
                data.append(1.0)

            for i in range(self.NX):
                for j in range(self.NX):
                    rows.append(row_base + i)
                    cols.append(x_col_base + j)
                    data.append(float(-A_blocks[k, i, j]))

            for i in range(self.NX):
                for j in range(self.NU):
                    rows.append(row_base + i)
                    cols.append(u_col_base + j)
                    data.append(float(-B_blocks[k, i, j]))

        # Box bounds on every delta decision variable.
        for v in range(self._num_vars):
            rows.append(self._num_eq_rows + v)
            cols.append(v)
            data.append(1.0)

        # Dynamic curvature bound on predicted yaw rate:
        # |yaw_rate| <= kappa_max * max(speed_floor, abs(vx_nom)).
        #
        # The speed term is frozen at the SQP linearization point, so these remain
        # linear rows in the delta-QP.  Using abs(vx_nom) with a small floor avoids
        # the zero-speed degeneracy of |yaw_rate| <= kappa * vx.
        for k in range(1, self._n_horizon + 1):
            x_col_base = k * self.NX
            row_base = self._curvature_row_base + 2 * (k - 1)

            rows.append(row_base)
            cols.append(x_col_base + self.IYAW_RATE)
            data.append(1.0)

            rows.append(row_base + 1)
            cols.append(x_col_base + self.IYAW_RATE)
            data.append(-1.0)

        return sparse.csc_matrix(
            (data, (rows, cols)),
            shape=(self._num_constraints, self._num_vars),
        )

    def _build_qp_linear_term(
        self,
        x_ref: np.ndarray,
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        previous_control: np.ndarray,
    ) -> np.ndarray:
        """Build the linear QP cost term for the current SQP iterate."""
        q = np.zeros(self._num_vars)

        for k in range(self._gains.idx_start_penalty, self._n_horizon + 1):
            ref = x_ref[k].copy()
            ref[self.IYAW] = x_nom[k, self.IYAW] - math.atan2(
                math.sin(x_nom[k, self.IYAW] - ref[self.IYAW]),
                math.cos(x_nom[k, self.IYAW] - ref[self.IYAW]),
            )
            xs = self._x_slice(k)
            q[xs] += 2.0 * self._Q[k] * (x_nom[k] - ref)

        for k in range(self._n_horizon):
            current = self._u_slice(k)
            nominal_difference = (
                u_nom[k] - previous_control if k == 0 else u_nom[k] - u_nom[k - 1]
            )
            q[current] += 2.0 * self._S @ nominal_difference
            if k > 0:
                previous = self._u_slice(k - 1)
                q[previous] += -2.0 * self._S @ nominal_difference

        return q

    def _build_qp_bounds(
        self,
        measured_state: np.ndarray,
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        defects: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build equality and box bounds for the delta QP."""
        b_eq = np.zeros(self._num_eq_rows)
        b_eq[: self.NX] = measured_state - x_nom[0]
        b_eq[self.NX :] = defects.reshape(self._n_horizon * self.NX)

        lower = np.full(self._num_vars, -np.inf)
        upper = np.full(self._num_vars, np.inf)
        for k in range(1, self._n_horizon + 1):
            lower[self._x_slice(k)] = self._x_min - x_nom[k]
            upper[self._x_slice(k)] = self._x_max - x_nom[k]
        for k in range(self._n_horizon):
            lower[self._u_slice(k)] = self._u_min - u_nom[k]
            upper[self._u_slice(k)] = self._u_max - u_nom[k]

        curvature_lower = np.full(self._num_curvature_rows, -np.inf)
        curvature_upper = np.empty(self._num_curvature_rows)
        for k in range(1, self._n_horizon + 1):
            row = 2 * (k - 1)
            nominal_yaw_rate = x_nom[k, self.IYAW_RATE]
            nominal_speed = max(
                self._gains.max_curvature_speed_floor,
                abs(float(x_nom[k, self.IVX])),
            )
            max_yaw_rate = self._gains.max_curvature_abs * nominal_speed
            curvature_upper[row] = max_yaw_rate - nominal_yaw_rate
            curvature_upper[row + 1] = max_yaw_rate + nominal_yaw_rate

        return np.concatenate([b_eq, lower, curvature_lower]), np.concatenate(
            [b_eq, upper, curvature_upper]
        )

    def _solve_qp_step(
        self,
        measured_state: np.ndarray,
        x_ref: np.ndarray,
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        previous_control: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        A_blocks, B_blocks, defects = self._linearize_horizon(x_nom, u_nom)
        q = self._build_qp_linear_term(x_ref, x_nom, u_nom, previous_control)
        constraint_lower, constraint_upper = self._build_qp_bounds(
            measured_state,
            x_nom,
            u_nom,
            defects,
        )
        A_constraints = self._build_constraint_matrix(A_blocks, B_blocks)
        if not (
            np.array_equal(A_constraints.indices, self._A_indices)
            and np.array_equal(A_constraints.indptr, self._A_indptr)
        ):
            raise RuntimeError("SQP OSQP constraint sparsity pattern changed")

        self._qp_solver.update(
            q=q,
            l=constraint_lower,
            u=constraint_upper,
            Ax=A_constraints.data,
        )
        if self._last_qp_solution is not None:
            self._qp_solver.warm_start(x=self._last_qp_solution)
        result = self._qp_solver.solve(raise_error=False)
        if result.info.status not in self.OSQP_ACCEPTED_STATUSES:
            logger.debug("SQP QP solver status: %s", result.info.status)
            return (
                np.zeros_like(x_nom),
                np.zeros_like(u_nom),
                result.info.status,
            )

        if not np.all(np.isfinite(result.x)):
            return np.zeros_like(x_nom), np.zeros_like(u_nom), "nonfinite QP step"

        self._last_qp_solution = result.x.copy()
        step_x = result.x[: self._num_x_vars].reshape(self._n_horizon + 1, self.NX)
        step_u = result.x[self._num_x_vars :].reshape(self._n_horizon, self.NU)
        return step_x, step_u, result.info.status
