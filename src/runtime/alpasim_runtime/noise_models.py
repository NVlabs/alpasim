# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from alpasim_runtime.config import EgomotionNoiseModelConfig
from alpasim_utils.geometry import Pose


@dataclass
class EgomotionNoiseModel:
    """
    The EgomotionNoiseModel class provides noise dynamics for the pose
    of the ego vehicle. It uses error dynamics as described in:
    https://docs.google.com/presentation/d/1KcC632ZD4ze4ehN7KmLmhpQb5ajiPaNz8ZI6JA6aAcQ/edit#slide=id.g3259455a657_0_66

    Note: the steady state covariance of the error will be approximated by
          P = cov * dt**2 / (2 * tau)
          This can be used to help choose the terms of the covariance matrices for
          a desired time constant
    """

    # noise parameters
    covariance_position: np.ndarray = field(
        default_factory=lambda: np.diag([0.05, 0.05, 0.0])
    )  # covariance on position error dynamics
    covariance_orientation: np.ndarray = field(
        default_factory=lambda: np.diag([0.0, 0.0, 0.0007])
    )  # covariance on orientation error dynamics
    time_constant_position: float = 3.0  # time constant for position error dynamics
    time_constant_orientation: float = (
        5.0  # time constant for orientation error dynamics
    )

    # current state
    position_error: np.ndarray = field(init=False, default_factory=lambda: np.zeros(3))
    orientation_error: np.ndarray = field(
        init=False, default_factory=lambda: np.zeros(3)
    )

    def __post_init__(self) -> None:
        """
        Validate the input parameters and initialize the noise
        """
        if self.covariance_position.shape != (3, 3):
            raise ValueError("The covariance_position matrix must be 3x3")
        if self.covariance_orientation.shape != (3, 3):
            raise ValueError("The covariance_orientation matrix must be 3x3")
        if self.time_constant_position <= 0.0:
            raise ValueError("The time_constant_position must be positive")
        if self.time_constant_orientation <= 0.0:
            raise ValueError("The time_constant_orientation must be positive")

    @staticmethod
    def from_config(config: EgomotionNoiseModelConfig) -> EgomotionNoiseModel | None:
        """
        Create an EgomotionNoiseModel from the given config.
        """
        if not config.enabled:
            return None
        cov_position = np.diag(np.array([config.cov_x, config.cov_y, config.cov_z]))
        time_constant_position = config.time_constant_position
        cov_orientation = np.diag(
            np.array(
                [
                    config.cov_orientation_x,
                    config.cov_orientation_y,
                    config.cov_orientation_z,
                ]
            )
        )
        time_constant_orientation = config.time_constant_orientation

        return EgomotionNoiseModel(
            cov_position,
            cov_orientation,
            time_constant_position,
            time_constant_orientation,
        )

    def update(self, dt: float) -> Pose:
        """
        Update the noise model for the given time step.
        Args:
            dt: The time step in seconds
        Returns:
            The noise in the form of a Pose
        """
        a_position = math.exp(-dt / self.time_constant_position)
        a_orientation = math.exp(-dt / self.time_constant_orientation)

        self.position_error = a_position * self.position_error + (
            1.0 - a_position
        ) * np.random.multivariate_normal(np.zeros(3), self.covariance_position * dt)
        self.orientation_error = a_orientation * self.orientation_error + (
            1.0 - a_orientation
        ) * np.random.multivariate_normal(np.zeros(3), self.covariance_orientation * dt)

        qw = math.sqrt(1.0 - self.orientation_error.dot(self.orientation_error) / 4.0)

        return Pose(
            self.position_error.astype(np.float32),
            np.array(
                [
                    self.orientation_error[0] / 2.0,
                    self.orientation_error[1] / 2.0,
                    self.orientation_error[2] / 2.0,
                    qw,
                ],
                dtype=np.float32,
            ),
        )
