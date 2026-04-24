# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Exception classes for runtime daemon."""


class InvalidRequestError(ValueError):
    """Raised when a simulation request contains invalid parameters."""

    pass


class UnknownSceneError(InvalidRequestError):
    """Raised when a simulation request references a scene_id with no known data source."""

    def __init__(self, scene_id: str):
        super().__init__(f"No data source found for scene_id: {scene_id}")
        self.scene_id = scene_id
