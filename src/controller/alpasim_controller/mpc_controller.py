# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
MPC Controller interface and shared dataclasses.

This module defines the abstract interface for MPC controllers, allowing
different implementations (LinearMPC, NonlinearMPC) to be chosen at runtime.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from alpasim_utils import trajectory


class MPCImplementation(StrEnum):
    """Available MPC implementations.

    Using StrEnum allows direct use with argparse choices and string comparisons.
    """

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


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
        idx_start_penalty: Tracking costs are ignored before this horizon index
    """

    long_position_weight: float = 2.0
    lat_position_weight: float = 1.0
    heading_weight: float = 1.0
    acceleration_weight: float = 0.1
    rel_front_steering_angle_weight: float = 5.0
    rel_acceleration_weight: float = 1.0
    idx_start_penalty: int = 10


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
    reference_trajectory: trajectory.Trajectory
    timestamp_us: int


@dataclass
class ControllerOutput:
    """Output from the MPC controller.

    Attributes:
        control: Optimal control command [steering_cmd, accel_cmd]
        solve_time_ms: Time spent in solver (milliseconds)
        status: Solver status string ("solved", "max_iter", etc.)
    """

    control: np.ndarray
    solve_time_ms: float
    status: str


class MPCController(ABC):
    """Abstract interface for MPC controllers.

    Implementations:
        - LinearMPC: Uses OSQP QP solver with linearized dynamics
        - NonlinearMPC: Uses do_mpc/CasADi with full nonlinear dynamics
    """

    DT_MPC: float = 0.1  # MPC timestep in seconds
    N_HORIZON: int = 20  # Prediction horizon

    @abstractmethod
    def compute_control(self, input: ControllerInput) -> ControllerOutput:
        """Compute optimal control given current state and reference.

        Args:
            input: Current state, reference trajectory, and timestamp

        Returns:
            Optimal control command and solver metadata
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller name for logging/identification."""
        ...
