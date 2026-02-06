# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Nonlinear MPC controller using do_mpc/CasADi.

Full nonlinear bicycle dynamics solved with interior point method.
More accurate but slower than LinearMPC.
"""

import logging
import time
import warnings

# Suppress do_mpc warnings
warnings.filterwarnings("ignore", message="The ONNX feature is not available")
warnings.filterwarnings("ignore", message="The opcua feature is not available")
warnings.filterwarnings("ignore", message='"is" with a literal', category=SyntaxWarning)

import casadi  # noqa: E402
import do_mpc  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

from alpasim_controller.mpc_controller import (  # noqa: E402
    ControllerInput,
    ControllerOutput,
    MPCController,
    MPCGains,
)
from alpasim_controller.vehicle_model import VehicleModel  # noqa: E402


class NonlinearMPC(MPCController):
    """Nonlinear MPC using do_mpc/CasADi.

    Full nonlinear bicycle dynamics, solved with interior point method.
    More accurate but slower than LinearMPC.
    """

    def __init__(
        self,
        vehicle_params: VehicleModel.Parameters | None = None,
        gains: MPCGains | None = None,
    ):
        """Initialize the NonlinearMPC controller.

        Args:
            vehicle_params: Vehicle parameters for dynamics model. If None,
                uses default parameters.
            gains: Cost function weights. If None, uses default gains.
        """
        self._vehicle_params = vehicle_params or VehicleModel.Parameters()
        self._gains = gains or MPCGains()
        self._model = self._build_model()
        self._mpc = None  # Lazy init on first call

        # Store reference trajectory for tvp_fun callback
        self._current_reference: ControllerInput | None = None
        self._first_reference_pose_vec3: np.ndarray = np.array([0, 0, 0])
        self._first_reference_pose_yaw: float = 0.0

    @property
    def name(self) -> str:
        return "nonlinear_mpc"

    def compute_control(self, input: ControllerInput) -> ControllerOutput:
        """Compute optimal control using nonlinear MPC.

        Args:
            input: Current state, reference trajectory, and timestamp

        Returns:
            Optimal control command and solver metadata
        """
        start_time = time.perf_counter()

        # Store reference for tvp_fun callback (must be set before setup/solve)
        self._current_reference = input

        # Setup MPC on first call
        if self._mpc is None:
            self._setup_mpc()

        # Prepare state vector (do_mpc expects column vector)
        x0 = input.state.reshape(-1, 1)

        # Solve
        try:
            u_opt = self._mpc.make_step(x0)
            status = "solved"
        except Exception as e:
            u_opt = np.zeros((2, 1))
            status = f"failed: {e}"
            logger.warning("Nonlinear MPC solver failed: %s", e)

        solve_time_ms = (time.perf_counter() - start_time) * 1000

        return ControllerOutput(
            control=u_opt.flatten(),
            solve_time_ms=solve_time_ms,
            status=status,
        )

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
        kinetic_mass = params.mass * casadi.fmax(v_cg_x, EPS_SPEED)
        kinetic_inertia = params.inertia * casadi.fmax(v_cg_x, EPS_SPEED)
        lf = params.wheelbase - params.l_rig_to_cg
        lf_caf = lf * params.front_cornering_stiffness
        lr_car = params.l_rig_to_cg * params.rear_cornering_stiffness
        lf_sq_caf = lf * lf_caf
        lr_sq_car = params.l_rig_to_cg * lr_car

        a_00 = (
            -2
            * (params.front_cornering_stiffness + params.rear_cornering_stiffness)
            / kinetic_mass
        )
        a_01 = -v_cg_x - 2 * (lf_caf - lr_car) / kinetic_mass
        a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
        a_11 = -2 * (lf_sq_caf + lr_sq_car) / kinetic_inertia

        b_00 = 2 * params.front_cornering_stiffness / params.mass
        b_10 = 2 * lf_caf / params.inertia

        # Model switching (kinematic vs dynamic)
        use_kinematic_model = casadi.lt(v_cg_x, params.kinematic_threshold_speed)

        # Position dynamics
        model.set_rhs(
            "x_rig_inertial",
            casadi.if_else(
                use_kinematic_model,
                v_cg_x * casadi.cos(yaw_angle),
                v_cg_x * casadi.cos(yaw_angle)
                - (v_cg_y - params.l_rig_to_cg * yaw_rate) * casadi.sin(yaw_angle),
            ),
        )
        model.set_rhs(
            "y_rig_inertial",
            casadi.if_else(
                use_kinematic_model,
                v_cg_x * casadi.sin(yaw_angle),
                v_cg_x * casadi.sin(yaw_angle)
                + (v_cg_y - params.l_rig_to_cg * yaw_rate) * casadi.cos(yaw_angle),
            ),
        )
        model.set_rhs("yaw_angle", yaw_rate)
        model.set_rhs("v_cg_x", acceleration)

        # Lateral dynamics
        yaw_rate_kinematic = v_cg_x * front_steering_angle / params.wheelbase
        v_cg_y_kinematic = (
            v_cg_x * front_steering_angle * params.l_rig_to_cg / params.wheelbase
        )
        GAIN = 10.0

        model.set_rhs(
            "v_cg_y",
            casadi.if_else(
                use_kinematic_model,
                GAIN * (v_cg_y_kinematic - v_cg_y),
                a_00 * v_cg_y + a_01 * yaw_rate + b_00 * front_steering_angle,
            ),
        )
        model.set_rhs(
            "yaw_rate",
            casadi.if_else(
                use_kinematic_model,
                GAIN * (yaw_rate_kinematic - yaw_rate),
                a_10 * v_cg_y + a_11 * yaw_rate + b_10 * front_steering_angle,
            ),
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
        self._mpc.settings.n_horizon = self.N_HORIZON
        self._mpc.settings.t_step = self.DT_MPC
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
        self._mpc.bounds["lower", "_x", "x_rig_inertial"] = -500
        self._mpc.bounds["upper", "_x", "x_rig_inertial"] = 500
        self._mpc.bounds["lower", "_x", "y_rig_inertial"] = -20
        self._mpc.bounds["upper", "_x", "y_rig_inertial"] = 20
        self._mpc.bounds["lower", "_x", "yaw_angle"] = -0.78
        self._mpc.bounds["upper", "_x", "yaw_angle"] = 0.78
        self._mpc.bounds["lower", "_x", "v_cg_x"] = 0.0
        self._mpc.bounds["upper", "_x", "v_cg_x"] = 35
        self._mpc.bounds["lower", "_x", "v_cg_y"] = -10
        self._mpc.bounds["upper", "_x", "v_cg_y"] = 10
        self._mpc.bounds["lower", "_x", "yaw_rate"] = -3
        self._mpc.bounds["upper", "_x", "yaw_rate"] = 3
        self._mpc.bounds["lower", "_x", "front_steering_angle"] = -0.785
        self._mpc.bounds["upper", "_x", "front_steering_angle"] = 0.785

        # Input bounds
        self._mpc.bounds["lower", "_u", "front_steering_angle_cmd"] = -2
        self._mpc.bounds["upper", "_u", "front_steering_angle_cmd"] = 2
        self._mpc.bounds["lower", "_u", "acceleration_cmd"] = -9.0
        self._mpc.bounds["upper", "_u", "acceleration_cmd"] = 6.0

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

        dt_us = int(self.DT_MPC * 1e6)
        timestamps = np.array(
            [timestamp_us + k * dt_us for k in range(n_horizon + 1)],
            dtype=np.uint64,
        )

        # Clamp to reference trajectory range
        ref_range = ref_traj.time_range_us
        timestamps = np.clip(timestamps, ref_range.start, ref_range.stop - 1)

        # Batch interpolate
        ref_poses = ref_traj.interpolate_to_timestamps(timestamps)

        for k in range(n_horizon + 1):
            pose = ref_poses.poses[k]

            if k == 0:
                self._first_reference_pose_vec3 = pose.vec3
                self._first_reference_pose_yaw = pose.yaw

            tvp_template["_tvp", k, "x_ref"] = pose.vec3[0]
            tvp_template["_tvp", k, "y_ref"] = pose.vec3[1]
            tvp_template["_tvp", k, "heading_ref"] = pose.yaw
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
