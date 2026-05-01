# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared runtime exception types."""


class UnknownSceneError(ValueError):
    """Raised when a scene_id cannot be resolved to a known data source."""

    def __init__(self, scene_id: str):
        super().__init__(f"No data source found for scene_id: {scene_id}")
        self.scene_id = scene_id
