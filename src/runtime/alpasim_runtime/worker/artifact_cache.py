# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import functools
from typing import Callable

from alpasim_utils.artifact import Artifact


def make_artifact_loader(
    smooth_trajectories: bool,
    max_cache_size: int | None = None,
) -> Callable[[str, str], Artifact]:
    """Create a cached artifact loader function.

    Uses :func:`functools.lru_cache` for LRU eviction.

    Args:
        smooth_trajectories: Passed through to :class:`Artifact`.
        max_cache_size: Maximum number of cached artifacts.
            ``None`` means unlimited, ``0`` disables caching.

    Returns:
        A ``(scene_id, artifact_path) -> Artifact`` callable with LRU caching.
    """

    @functools.lru_cache(maxsize=max_cache_size)
    def load_artifact(scene_id: str, artifact_path: str) -> Artifact:
        del scene_id  # used only as a cache key component
        return Artifact(
            source=artifact_path,
            _smooth_trajectories=smooth_trajectories,
        )

    return load_artifact
