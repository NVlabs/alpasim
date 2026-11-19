# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Model abstraction layer for trajectory prediction models."""

from .base import (
    BaseTrajectoryModel,
    CameraFrame,
    CameraImages,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)

__all__ = [
    "BaseTrajectoryModel",
    "CameraFrame",
    "CameraImages",
    "DriveCommand",
    "ModelPrediction",
    "PredictionInput",
]
