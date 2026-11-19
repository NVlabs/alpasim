# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""MPC Controller interface and shared dataclasses.

This module defines the abstract interface for MPC controllers, allowing
different implementations (LinearMPC, NonlinearMPC, SQPMPC) to be chosen at
runtime.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from alpasim_utils.geometry import Trajectory

# Module-level defaults — single source of truth for ControllerConfig and MPC subclasses.
DEFAULT_N_HORIZON: int = 20
DEFAULT_DT_MPC: float = 0.1
DEFAULT_MAX_SPEED_MPS: float = 45.0


class MPCImplementation(StrEnum):
    """Available MPC implementations.

    Using StrEnum allows direct use with argparse choices and string comparisons.
    """

    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    SQP = "sqp"


@dataclass
class MPCGains:
    """MPC cost function weights - shared across all controller types.

    Attributes:
        long_position_weight: Penalty on longitudinal position error
        lat_position_weight: Penalty on lateral position error
        heading_weight: Penalty on heading error
        acceleration_weight: Penalty on acceleration state
        rel_front_steering_angle_weight: Regularization on steering command changes
        rel_acceleration_weight: Regularization on acceleration command changes
        velocity_weight: Penalty on longitudinal velocity tracking error
        lateral_velocity_weight: Penalty on lateral velocity tracking error
        yaw_rate_weight: Penalty on yaw-rate tracking error
        steering_state_weight: Penalty on steering-state tracking error
        sqp_state_step_weight: Proximal penalty on state changes between SQP iterations
        sqp_control_step_weight: Proximal penalty on control changes between SQP iterations
        sqp_max_iterations: Maximum QP linearization iterations per control update
        sqp_step_tolerance: Infinity-norm threshold for SQP step convergence
        max_curvature_abs: Hard SQP curvature bound [1/m]
        max_curvature_speed_floor: Speed floor for SQP yaw-rate curvature bound [m/s]
        max_lateral_acceleration: Nonlinear only. Bound on the lateral acceleration
            v^2 * tan(steering command) / wheelbase that a steering command asks of
            the tires [m/s^2]
        idx_start_penalty: Tracking costs are ignored before this horizon index
        sqp_lat_position_ramp_steps: SQP only. The lateral position weight rises linearly
            from 0 at idx_start_penalty to lat_position_weight this many steps later, so
            near-term lateral error is corrected through heading rather than an immediate
            steering spike; 0 applies the full weight from idx_start_penalty on
    """

    long_position_weight: float = 2.0
    lat_position_weight: float = 1.0
    heading_weight: float = 1.0
    acceleration_weight: float = 0.1
    rel_front_steering_angle_weight: float = 5.0
    rel_acceleration_weight: float = 1.0
    velocity_weight: float = 1.0
    lateral_velocity_weight: float = 0.1
    yaw_rate_weight: float = 0.5
    steering_state_weight: float = 0.1
    sqp_state_step_weight: float = 1e-3
    sqp_control_step_weight: float = 1e-2
    sqp_max_iterations: int = 5
    sqp_step_tolerance: float = 1e-3
    max_curvature_abs: float = 0.35
    max_curvature_speed_floor: float = 0.1
    max_lateral_acceleration: float = 6.0
    idx_start_penalty: int = 10
    sqp_lat_position_ramp_steps: int = 0


@dataclass
class ControllerInput:
    """Input to the MPC controller.

    Attributes:
        state: Current vehicle state vector (8,) containing:
            [x, y, yaw, vx, vy, yaw_rate, steering, accel]
        reference_trajectory: Reference trajectory to track (in rig frame)
        timestamp_us: Current timestamp in microseconds
    """

    state: np.ndarray
    reference_trajectory: Trajectory
    timestamp_us: int


@dataclass
class ControllerOutput:
    """Output from the MPC controller.

    Attributes:
        control: Accepted [steering_cmd, accel_cmd], or None after a failed solve.
        solve_time_ms: Time spent in solver (milliseconds)
        status: Native outcome or SQP iteration diagnostic; failures start with "failed:".
    """

    control: np.ndarray | None
    solve_time_ms: float
    status: str


@dataclass
class ControllerConfig:
    """Top-level controller configuration loaded from YAML."""

    mpc_implementation: str = "linear"
    n_horizon: int = DEFAULT_N_HORIZON
    dt_mpc: float = DEFAULT_DT_MPC
    gains: MPCGains = field(default_factory=MPCGains)


class MPCController(ABC):
    """Abstract interface for MPC controllers.

    Implementations:
        - LinearMPC: Uses OSQP QP solver with linearized dynamics
        - NonlinearMPC: Uses do_mpc/CasADi with full nonlinear dynamics
        - SQPMPC: Repeatedly linearizes nonlinear dynamics along a trajectory

    Subclasses are responsible to populate ``_dt_mpc`` in ``__init__`` implementations
    """

    @property
    def dt_mpc(self) -> float:
        return self._dt_mpc

    def compute_control(self, input: ControllerInput) -> ControllerOutput:
        """Accept a finite backend command, or reset without a command on failure."""
        start_time = time.perf_counter()
        control, status = self._compute_control(input)
        if control is not None:
            if control.shape != (2,):
                raise ValueError("Control must have shape (2,)")
            if not np.all(np.isfinite(control)):
                control, status = None, "nonfinite control"
        if control is None:
            self.reset_warm_start()
            status = f"failed: {status}"
            logging.getLogger(__name__).warning("%s solver %s", self.name, status)
        return ControllerOutput(
            control=control,
            solve_time_ms=(time.perf_counter() - start_time) * 1000.0,
            status=status,
        )

    @abstractmethod
    def _compute_control(self, input: ControllerInput) -> tuple[np.ndarray | None, str]:
        """Return an accepted command and status, or (None, native failure reason)."""
        ...

    @abstractmethod
    def reset_warm_start(self) -> None:
        """Discard reusable optimizer state (native iterates and previous controls)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller name for logging/identification."""
        ...
