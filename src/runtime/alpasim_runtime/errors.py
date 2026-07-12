# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared runtime exception types."""

from alpasim_grpc.v0 import runtime_pb2


class UnknownSceneError(ValueError):
    """Raised when a scene_id cannot be resolved to a known data source."""

    def __init__(self, scene_id: str):
        super().__init__(f"No data source found for scene_id: {scene_id}")
        self.scene_id = scene_id


class RolloutError(Exception):
    """Base for rollout failures that carry a RolloutErrorCode for SimulationReturn."""

    error_code: int = runtime_pb2.ROLLOUT_ERROR_CODE_UNSPECIFIED


class InvalidSceneError(RolloutError):
    """Raised when scene data is permanently unsuitable for a rollout."""

    error_code = runtime_pb2.ROLLOUT_ERROR_CODE_INVALID_SCENE

    def __init__(self, scene_id: str, detail: str):
        super().__init__(f"INVALID_SCENE: Scene {scene_id!r} is invalid: {detail}")
        self.scene_id = scene_id
        self.detail = detail
