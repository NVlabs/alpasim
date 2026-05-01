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
from alpasim_runtime.worker.ipc import (
    SHUTDOWN_SENTINEL,
    AssignedRolloutJob,
    JobResult,
    ServiceEndpoints,
)
from alpasim_runtime.worker.main import run_worker_loop


@pytest.mark.asyncio
async def test_run_worker_loop_uses_parent_version_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_worker_loop should pass parent-provided version IDs into rollouts."""
    seen_version_ids = None

    async def _fake_run_single_rollout(
        job,
        user_config,
        data_source,
        camera_catalog,
        version_ids,
        rollouts_dir,
        eval_config,
        eval_executor=None,
    ) -> JobResult:
        del (
            user_config,
            data_source,
            camera_catalog,
            rollouts_dir,
            eval_config,
            eval_executor,
        )
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
    scene_loader = MagicMock()
    scene_loader.get_data_source.return_value = MagicMock()

    completed = await run_worker_loop(
        worker_id=0,
        job_queue=job_queue,
        result_queue=result_queue,
        num_consumers=1,
        user_config=user_config,
        scene_loader=scene_loader,
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
