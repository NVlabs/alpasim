# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Dependency-free PDMClosed-style planner model adapter."""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..schema import ModelConfig
from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput
from .pdm_bridge import PDMClosedLoopPlanner, build_pdm_planner_input

logger = logging.getLogger(__name__)


class PDMModel(BaseTrajectoryModel):
    """PDMClosed-style planner backend for driver model switching.

    This implementation uses the compact runtime planner_context as its formal
    input source and does not depend on camera frames for inference. Camera
    validation is intentionally kept to preserve compatibility with the driver
    model interface and model-switch safety checks.
    """

    TRAJECTORY_HORIZON_S = 4.0
    DEFAULT_CONTEXT_LENGTH = 1
    DEFAULT_CRUISE_SPEED_MPS = 5.0
    MIN_SPEED_MPS = 0.0
    MAX_SPEED_MPS = 20.0
    MIN_TURN_RADIUS_M = 12.0
    MAX_ACCEL_MPS2 = 4.0

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "PDMModel":
        del model_cfg, device
        return cls(
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
            output_frequency_hz=output_frequency_hz,
        )

    def __init__(
        self,
        camera_ids: list[str],
        context_length: int,
        output_frequency_hz: int,
    ) -> None:
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        self._planner = PDMClosedLoopPlanner(
            horizon_s=self.TRAJECTORY_HORIZON_S,
            output_frequency_hz=output_frequency_hz,
            min_turn_radius_m=self.MIN_TURN_RADIUS_M,
            max_accel_mps2=self.MAX_ACCEL_MPS2,
            max_speed_mps=self.MAX_SPEED_MPS,
        )
        logger.info(
            "Initialized PDMModel with %d camera(s), context_length=%d, output_frequency=%dHz",
            len(camera_ids),
            context_length,
            output_frequency_hz,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz

    def _encode_command(self, command: DriveCommand) -> int:
        return int(command)

    def _normalize_speed(self, speed_mps: float) -> float:
        if speed_mps < self.MIN_SPEED_MPS:
            return self.DEFAULT_CRUISE_SPEED_MPS
        return float(np.clip(speed_mps, self.MIN_SPEED_MPS, self.MAX_SPEED_MPS))

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)

        planner_input = build_pdm_planner_input(
            speed_mps=self._normalize_speed(prediction_input.speed),
            acceleration_mps2=prediction_input.acceleration,
            planner_context=prediction_input.planner_context,
            fallback_command=prediction_input.command,
        )
        result = self._planner.plan(planner_input)
        debug_metadata = {"planner_backend": "pdm_closed", **result.debug_metadata}
        return ModelPrediction(
            trajectory_xy=result.trajectory_xy,
            headings=result.headings,
            debug_metadata=debug_metadata,
        )
