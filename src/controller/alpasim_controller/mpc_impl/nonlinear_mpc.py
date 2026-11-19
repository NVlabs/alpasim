# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Nonlinear MPC controller using do_mpc/CasADi.

Full nonlinear bicycle dynamics solved with interior point method.
More accurate but slower than LinearMPC.
"""

import warnings

# Suppress do_mpc warnings
warnings.filterwarnings("ignore", message="The ONNX feature is not available")
warnings.filterwarnings("ignore", message="The opcua feature is not available")
warnings.filterwarnings("ignore", message='"is" with a literal', category=SyntaxWarning)

import casadi  # noqa: E402
import do_mpc  # noqa: E402
import numpy as np  # noqa: E402
from alpasim_controller.mpc_controller import (  # noqa: E402
    DEFAULT_DT_MPC,
    DEFAULT_MAX_SPEED_MPS,
    DEFAULT_N_HORIZON,
    ControllerInput,
    MPCController,
    MPCGains,
)
from alpasim_controller.vehicle_model import (  # noqa: E402
    VehicleModel,
    lateral_dynamics_coefficients,
)


class NonlinearMPC(MPCController):
    """Nonlinear MPC using do_mpc/CasADi.

    Full nonlinear bicycle dynamics, solved with interior point method.
    More accurate but slower than LinearMPC.
    """

    def __init__(
        self,
        vehicle_params: VehicleModel.Parameters | None = None,
        gains: MPCGains | None = None,
        n_horizon: int = DEFAULT_N_HORIZON,
        dt_mpc: float = DEFAULT_DT_MPC,
    ):
        """Initialize the NonlinearMPC controller.

        Args:
            vehicle_params: Vehicle parameters for dynamics model. If None,
                uses default parameters.
            gains: Cost function weights. If None, uses default gains.
            n_horizon: Prediction horizon length.
            dt_mpc: MPC timestep in seconds.
        """
        self._vehicle_params = vehicle_params or VehicleModel.Parameters()
        self._gains = gains or MPCGains()
        if self._gains.max_curvature_abs <= 0.0:
            raise ValueError("max_curvature_abs must be positive")
        if self._gains.max_curvature_speed_floor <= 0.0:
            raise ValueError("max_curvature_speed_floor must be positive")
        if self._gains.max_lateral_acceleration <= 0.0:
            raise ValueError("max_lateral_acceleration must be positive")
        self._n_horizon = n_horizon
        self._dt_mpc = dt_mpc
        self._model = self._build_model()
        self._mpc = None  # Lazy init on first call

        # Store reference trajectory for tvp_fun callback
        self._current_reference: ControllerInput | None = None
        self._first_reference_pose_vec3: np.ndarray = np.array([0, 0, 0])
        self._first_reference_pose_yaw: float = 0.0

    @property
    def name(self) -> str:
        return "nonlinear_mpc"

    def _compute_control(self, input: ControllerInput) -> tuple[np.ndarray | None, str]:
        """Compute optimal control using nonlinear MPC.

        Args:
            input: Current state, reference trajectory, and timestamp

        Returns:
            Optimal control command and solver metadata
        """
        # Store reference for tvp_fun callback (must be set before setup/solve)
        self._current_reference = input

        # Setup MPC on first call
        if self._mpc is None:
            self._setup_mpc()

        # Prepare state vector (do_mpc expects column vector)
        x0 = input.state.reshape(-1, 1)
        if not self._mpc.flags["initial_run"]:
            self._mpc.x0 = x0
            self._mpc.u0 = input.state[6:8].reshape(-1, 1)
            self._mpc.set_initial_guess()

        # Solve
        try:
            u_opt = self._mpc.make_step(x0)
        except RuntimeError as exc:
            return None, str(exc)
        stats = self._mpc.solver_stats
        if not stats["success"]:
            return None, stats["return_status"]
        return u_opt.flatten(), "solved"

    def reset_warm_start(self) -> None:
        """Discard primal, dual and previous-control state after a physical-state jump."""
        if self._mpc is None:
            return
        self._mpc.x0 = np.zeros((8, 1))
        self._mpc.u0 = np.zeros((2, 1))
        self._mpc.z0 = np.zeros((self._model.n_z, 1))
        self._mpc.t0 = np.zeros((1,))
        self._mpc.opt_x_num.master = casadi.DM.zeros(self._mpc.opt_x_num.cat.shape)
        self._mpc.opt_x_num_unscaled.master = casadi.DM.zeros(
            self._mpc.opt_x_num.cat.shape
        )
        self._mpc.lam_x_num = casadi.DM.zeros(self._mpc.opt_x_num.cat.shape)
        self._mpc.lam_g_num = casadi.DM.zeros(self._mpc.nlp_cons_lb.shape)
        self._mpc.set_initial_guess()
        self._mpc.flags["initial_run"] = False

    def _build_model(self) -> do_mpc.model.Model:
        """Build the do_mpc model with bicycle dynamics."""
        params = self._vehicle_params

        model = do_mpc.model.Model("continuous", "SX")

        # State variables
        model.set_variable(var_type="_x", var_name="x_rig_inertial", shape=(1, 1))
        model.set_variable(var_type="_x", var_name="y_rig_inertial", shape=(1, 1))
        yaw_angle = model.set_variable(
            var_type="_x", var_name="yaw_angle", shape=(1, 1)
        )
        v_cg_x = model.set_variable(var_type="_x", var_name="v_cg_x", shape=(1, 1))
        v_cg_y = model.set_variable(var_type="_x", var_name="v_cg_y", shape=(1, 1))
        yaw_rate = model.set_variable(var_type="_x", var_name="yaw_rate", shape=(1, 1))
        front_steering_angle = model.set_variable(
            var_type="_x", var_name="front_steering_angle", shape=(1, 1)
        )
        acceleration = model.set_variable(
            var_type="_x", var_name="acceleration", shape=(1, 1)
        )

        # Input variables
        front_steering_angle_cmd = model.set_variable(
            var_type="_u", var_name="front_steering_angle_cmd"
        )
        acceleration_cmd = model.set_variable(
            var_type="_u", var_name="acceleration_cmd"
        )

        # Dynamics coefficients
        EPS_SPEED = 1.0e-2
        # Smooth positive speed keeps the continuous predictor finite at rest.
        # Scaling steering by v / smooth_v preserves the no-slip zero-speed
        # equilibrium instead of creating lateral force while stationary.
        smooth_v = casadi.sqrt(v_cg_x**2 + EPS_SPEED**2)
        coefficients, steering_coefficients = lateral_dynamics_coefficients(params)
        a_00 = coefficients[0, 0] / smooth_v
        a_01 = (coefficients[0, 1] - v_cg_x**2) / smooth_v
        a_10 = coefficients[1, 0] / smooth_v
        a_11 = coefficients[1, 1] / smooth_v
        b_00 = steering_coefficients[0] * v_cg_x / smooth_v
        b_10 = steering_coefficients[1] * v_cg_x / smooth_v

        v_rig_y = v_cg_y - params.l_rig_to_cg * yaw_rate
        model.set_rhs(
            "x_rig_inertial",
            v_cg_x * casadi.cos(yaw_angle) - v_rig_y * casadi.sin(yaw_angle),
        )
        model.set_rhs(
            "y_rig_inertial",
            v_cg_x * casadi.sin(yaw_angle) + v_rig_y * casadi.cos(yaw_angle),
        )
        model.set_rhs("yaw_angle", yaw_rate)
        model.set_rhs("v_cg_x", acceleration)
        model.set_rhs(
            "v_cg_y",
            a_00 * v_cg_y + a_01 * yaw_rate + b_00 * front_steering_angle,
        )
        model.set_rhs(
            "yaw_rate",
            a_10 * v_cg_y + a_11 * yaw_rate + b_10 * front_steering_angle,
        )

        # Actuator dynamics
        model.set_rhs(
            "front_steering_angle",
            (front_steering_angle_cmd - front_steering_angle)
            / params.steering_time_constant,
        )
        model.set_rhs(
            "acceleration",
            (acceleration_cmd - acceleration) / params.acceleration_time_constant,
        )

        # Time-varying parameters for reference trajectory
        model.set_variable(var_type="_tvp", var_name="x_ref")
        model.set_variable(var_type="_tvp", var_name="y_ref")
        model.set_variable(var_type="_tvp", var_name="heading_ref")
        model.set_variable(var_type="_tvp", var_name="tracking_enabled")

        model.setup()
        return model

    def _setup_mpc(self) -> None:
        """Setup the do_mpc MPC controller."""
        self._mpc = do_mpc.controller.MPC(self._model)

        # MPC settings
        self._mpc.settings.n_horizon = self._n_horizon
        self._mpc.settings.t_step = self._dt_mpc
        self._mpc.settings.n_robust = 0
        self._mpc.settings.open_loop = 0
        self._mpc.settings.state_discretization = "collocation"
        self._mpc.settings.collocation_type = "radau"
        self._mpc.settings.collocation_deg = 2
        self._mpc.settings.collocation_ni = 1
        self._mpc.settings.store_full_solution = False
        self._mpc.settings.nlpsol_opts = {"ipopt.max_iter": 30}
        self._mpc.settings.supress_ipopt_output()

        # Cost function
        term = self._model.tvp["tracking_enabled"] * (
            self._gains.long_position_weight
            * (self._model.x["x_rig_inertial"] - self._model.tvp["x_ref"]) ** 2
            + self._gains.lat_position_weight
            * (self._model.x["y_rig_inertial"] - self._model.tvp["y_ref"]) ** 2
            + self._gains.acceleration_weight * (self._model.x["acceleration"] ** 2)
            + self._gains.heading_weight
            * casadi.atan2(
                casadi.sin(self._model.x["yaw_angle"] - self._model.tvp["heading_ref"]),
                casadi.cos(self._model.x["yaw_angle"] - self._model.tvp["heading_ref"]),
            )
            ** 2
        )
        self._mpc.set_objective(mterm=term, lterm=term)

        self._mpc.set_rterm(
            front_steering_angle_cmd=self._gains.rel_front_steering_angle_weight,
            acceleration_cmd=self._gains.rel_acceleration_weight,
        )

        self._mpc.set_tvp_fun(self._tvp_fun)

        # State bounds
        steering_bound = min(
            0.785,
            float(
                np.arctan(
                    self._gains.max_curvature_abs * self._vehicle_params.wheelbase
                )
            ),
        )
        self._mpc.bounds["lower", "_x", "x_rig_inertial"] = -500
        self._mpc.bounds["upper", "_x", "x_rig_inertial"] = 500
        self._mpc.bounds["lower", "_x", "y_rig_inertial"] = -20
        self._mpc.bounds["upper", "_x", "y_rig_inertial"] = 20
        self._mpc.bounds["lower", "_x", "yaw_angle"] = -0.78
        self._mpc.bounds["upper", "_x", "yaw_angle"] = 0.78
        # Small reverse-velocity slack keeps low-speed handoff states feasible;
        # it is not intended to model parking or reverse-driving maneuvers.
        self._mpc.bounds["lower", "_x", "v_cg_x"] = -5.0
        self._mpc.bounds["upper", "_x", "v_cg_x"] = DEFAULT_MAX_SPEED_MPS
        self._mpc.bounds["lower", "_x", "v_cg_y"] = -10
        self._mpc.bounds["upper", "_x", "v_cg_y"] = 10
        self._mpc.bounds["lower", "_x", "yaw_rate"] = -3
        self._mpc.bounds["upper", "_x", "yaw_rate"] = 3
        self._mpc.bounds["lower", "_x", "front_steering_angle"] = -steering_bound
        self._mpc.bounds["upper", "_x", "front_steering_angle"] = steering_bound

        # Input bounds
        self._mpc.bounds["lower", "_u", "front_steering_angle_cmd"] = -steering_bound
        self._mpc.bounds["upper", "_u", "front_steering_angle_cmd"] = steering_bound
        self._mpc.bounds["lower", "_u", "acceleration_cmd"] = -9.0
        self._mpc.bounds["upper", "_u", "acceleration_cmd"] = 6.0

        speed_for_curvature = casadi.fmax(
            self._gains.max_curvature_speed_floor,
            casadi.fabs(self._model.x["v_cg_x"]),
        )
        max_yaw_rate = self._gains.max_curvature_abs * speed_for_curvature
        self._mpc.set_nl_cons(
            "yaw_rate_upper_curvature",
            self._model.x["yaw_rate"] - max_yaw_rate,
            ub=0.0,
        )
        self._mpc.set_nl_cons(
            "yaw_rate_lower_curvature",
            -self._model.x["yaw_rate"] - max_yaw_rate,
            ub=0.0,
        )
        # Tire grip: without it, a sharp steer at speed is the cheapest way to
        # fall back onto a reference slower than the ego, and the car spins.
        lateral_acceleration_cmd = (
            self._model.x["v_cg_x"] ** 2
            * casadi.tan(self._model.u["front_steering_angle_cmd"])
            / self._vehicle_params.wheelbase
        )
        self._mpc.set_nl_cons(
            "lateral_acceleration_upper",
            lateral_acceleration_cmd,
            ub=self._gains.max_lateral_acceleration,
        )
        self._mpc.set_nl_cons(
            "lateral_acceleration_lower",
            -lateral_acceleration_cmd,
            ub=self._gains.max_lateral_acceleration,
        )

        self._mpc.setup()
        self._mpc.x0 = np.zeros((8, 1))
        self._mpc.set_initial_guess()

    def _tvp_fun(self, _t_now: float) -> dict:
        """Time-varying parameter function for do_mpc.

        Provides reference trajectory at each horizon step.
        """
        tvp_template = self._mpc.get_tvp_template()
        n_horizon = self._mpc.settings.n_horizon

        if self._current_reference is None:
            raise RuntimeError(
                "NonlinearMPC._tvp_fun called without current_reference set. "
                "This is an internal error."
            )

        ref_traj = self._current_reference.reference_trajectory
        timestamp_us = self._current_reference.timestamp_us
        idx_start = self._gains.idx_start_penalty

        dt_us = int(self._dt_mpc * 1e6)
        timestamps = np.array(
            [timestamp_us + k * dt_us for k in range(n_horizon + 1)],
            dtype=np.uint64,
        )

        # Clamp to reference trajectory range
        ref_range = ref_traj.time_range_us
        timestamps = np.clip(timestamps, ref_range.start, ref_range.stop - 1)

        # Batch interpolate using Trajectory
        ref_interp = ref_traj.interpolate(timestamps.astype(np.uint64))

        for k in range(n_horizon + 1):
            pose = ref_interp.get_pose(k)
            if k == 0:
                self._first_reference_pose_vec3 = pose.vec3
                self._first_reference_pose_yaw = pose.yaw()

            tvp_template["_tvp", k, "x_ref"] = pose.vec3[0]
            tvp_template["_tvp", k, "y_ref"] = pose.vec3[1]
            tvp_template["_tvp", k, "heading_ref"] = pose.yaw()
            tvp_template["_tvp", k, "tracking_enabled"] = 1.0 if k >= idx_start else 0.0

        return tvp_template

    @property
    def first_reference_pose_vec3(self) -> np.ndarray:
        """Get the first reference pose position (for logging)."""
        return self._first_reference_pose_vec3

    @property
    def first_reference_pose_yaw(self) -> float:
        """Get the first reference pose yaw (for logging)."""
        return self._first_reference_pose_yaw
