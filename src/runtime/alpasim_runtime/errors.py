# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared runtime exception types."""

from alpasim_grpc.v0 import runtime_pb2


class UnknownClipError(ValueError):
    """Raised when a clip ID cannot be resolved to a known data source."""

    def __init__(self, clip_id: str):
        super().__init__(f"No data source found for clip_id: {clip_id}")
        self.clip_id = clip_id


class UnknownSceneError(ValueError):
    """Raised when a structured scene address is not configured."""

    def __init__(self, source_id: str, policy_start_us: int | str):
        super().__init__(
            "No configured scene found for "
            f"source_id={source_id!r}, policy_start_us={policy_start_us!r}"
        )
        self.source_id = source_id
        self.policy_start_us = policy_start_us


class RolloutError(Exception):
    """Base for rollout failures that carry a RolloutErrorCode for SimulationReturn."""

    error_code: int = runtime_pb2.ROLLOUT_ERROR_CODE_UNSPECIFIED


class InvalidSceneError(RolloutError):
    """Raised when scene data is permanently unsuitable for a rollout."""

    error_code = runtime_pb2.ROLLOUT_ERROR_CODE_INVALID_SCENE

    def __init__(self, scene_key: str, detail: str):
        super().__init__(f"INVALID_SCENE: Scene {scene_key!r} is invalid: {detail}")
        self.scene_key = scene_key
        self.detail = detail
