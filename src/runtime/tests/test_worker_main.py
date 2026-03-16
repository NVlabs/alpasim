# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for worker main artifact loading helpers."""

from __future__ import annotations

from multiprocessing import Queue
from unittest.mock import MagicMock

import pytest
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import ServiceAddress
from alpasim_runtime.worker.artifact_cache import make_artifact_loader
from alpasim_runtime.worker.ipc import (
    SHUTDOWN_SENTINEL,
    AssignedRolloutJob,
    JobResult,
    ServiceEndpoints,
)
from alpasim_runtime.worker.main import run_worker_loop


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


@pytest.mark.asyncio
async def test_run_worker_loop_uses_parent_version_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_worker_loop should pass parent-provided version IDs into rollouts."""
    seen_version_ids = None

    async def _fake_run_single_rollout(
        job,
        user_config,
        artifacts,
        camera_catalog,
        version_ids,
        rollouts_dir,
        eval_config,
    ) -> JobResult:
        del user_config, artifacts, camera_catalog, rollouts_dir, eval_config
        nonlocal seen_version_ids
        seen_version_ids = version_ids
        return JobResult(
            request_id=job.request_id,
            job_id=job.job_id,
            rollout_spec_index=job.rollout_spec_index,
            success=True,
            error=None,
            error_traceback=None,
            rollout_uuid="rollout-uuid",
        )

    monkeypatch.setattr(
        "alpasim_runtime.worker.main.run_single_rollout",
        _fake_run_single_rollout,
    )

    endpoints = ServiceEndpoints(
        driver=ServiceAddress("localhost:10001", skip=False),
        sensorsim=ServiceAddress("localhost:10002", skip=False),
        physics=ServiceAddress("localhost:10003", skip=False),
        trafficsim=ServiceAddress("localhost:10004", skip=False),
        controller=ServiceAddress("localhost:10005", skip=False),
    )
    job = AssignedRolloutJob(
        request_id="req-1",
        job_id="job-1",
        scene_id="scene-1",
        rollout_spec_index=0,
        artifact_path="/tmp/scene-1.usdz",
        endpoints=endpoints,
    )

    job_queue: Queue = Queue()
    result_queue: Queue = Queue()
    job_queue.put(job)
    job_queue.put(SHUTDOWN_SENTINEL)

    parent_version_ids = RolloutMetadata.VersionIds(
        runtime_version=VersionId(version_id="0.3.0", git_hash="abc"),
    )
    user_config = MagicMock()
    user_config.endpoints.startup_timeout_s = 1

    completed = await run_worker_loop(
        worker_id=0,
        job_queue=job_queue,
        result_queue=result_queue,
        num_consumers=1,
        user_config=user_config,
        smooth_trajectories=False,
        artifact_cache_size=0,
        camera_catalog=MagicMock(),
        version_ids=parent_version_ids,
        rollouts_dir="/tmp",
        eval_config=MagicMock(),
        parent_pid=None,
    )

    result = result_queue.get(timeout=1)
    assert completed == 1
    assert result.request_id == "req-1"
    assert result.job_id == "job-1"
    assert result.rollout_spec_index == 0
    assert result.success is True
    assert seen_version_ids is parent_version_ids
