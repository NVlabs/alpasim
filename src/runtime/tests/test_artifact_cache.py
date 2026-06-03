# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from alpasim_runtime.worker.artifact_cache import make_artifact_loader


def test_artifact_loader_reuses_cached_instances() -> None:
    """Repeated loads for same (scene_id, path) return the same cached object."""
    load = make_artifact_loader(smooth_trajectories=True)

    first = load("scene-a", "/tmp/scene-a.usdz")
    second = load("scene-a", "/tmp/scene-a.usdz")

    assert first is second


def test_artifact_loader_loads_distinct_scenes() -> None:
    """Different scene IDs produce different Artifact objects."""
    load = make_artifact_loader(smooth_trajectories=True)

    scene_a = load("scene-a", "/tmp/scene-a.usdz")
    scene_b = load("scene-b", "/tmp/scene-b.usdz")

    assert scene_a is not scene_b


def test_artifact_loader_evicts_when_capacity_exceeded() -> None:
    """When full, the cache should evict entries to satisfy max_cache_size."""
    load = make_artifact_loader(smooth_trajectories=True, max_cache_size=2)

    scene_a_first = load("scene-a", "/tmp/scene-a.usdz")
    load("scene-b", "/tmp/scene-b.usdz")
    load("scene-c", "/tmp/scene-c.usdz")

    assert load("scene-b", "/tmp/scene-b.usdz") is not None

    # scene-a should have been evicted; a fresh load returns a new object.
    scene_a_second = load("scene-a", "/tmp/scene-a.usdz")
    assert scene_a_second is not scene_a_first


def test_artifact_loader_cache_info() -> None:
    """The returned callable exposes lru_cache's cache_info."""
    load = make_artifact_loader(smooth_trajectories=True, max_cache_size=4)

    load("s1", "/tmp/s1.usdz")
    load("s1", "/tmp/s1.usdz")  # hit
    load("s2", "/tmp/s2.usdz")

    info = load.cache_info()  # type: ignore[attr-defined]
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 4


def test_artifact_loader_disabled_cache() -> None:
    """max_cache_size=0 disables caching; every call creates a new Artifact."""
    load = make_artifact_loader(smooth_trajectories=True, max_cache_size=0)

    first = load("scene-a", "/tmp/scene-a.usdz")
    second = load("scene-a", "/tmp/scene-a.usdz")

    assert first is not second
