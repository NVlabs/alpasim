# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio

import pytest
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.daemon.scheduler import DaemonScheduler
from alpasim_runtime.worker.ipc import JobResult, PendingRolloutJob


def _make_pools(capacity_per_service: int) -> dict[str, AddressPool]:
    return {
        "driver": AddressPool(["driver:50051"], capacity_per_service, skip=False),
        "renderer": AddressPool(["sensorsim:50052"], capacity_per_service, skip=False),
        "physics": AddressPool(["physics:50053"], capacity_per_service, skip=False),
        "trafficsim": AddressPool(
            ["trafficsim:50054"], capacity_per_service, skip=False
        ),
        "controller": AddressPool(
            ["controller:50055"], capacity_per_service, skip=False
        ),
    }


def _pending(
    job_id: str,
    scene_id: str = "scene-a",
    rollout_spec_index: int = 0,
) -> PendingRolloutJob:
    return PendingRolloutJob(
        job_id=job_id,
        scene_id=scene_id,
        rollout_spec_index=rollout_spec_index,
    )


def _result(
    request_id: str,
    job_id: str,
    *,
    success: bool = True,
    error: str | None = None,
    retryable: bool = False,
) -> JobResult:
    return JobResult(
        request_id=request_id,
        job_id=job_id,
        rollout_spec_index=0,
        success=success,
        error=error,
        error_traceback=None,
        rollout_uuid=f"uuid-{job_id}",
        retryable=retryable,
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.submitted_job_ids: list[str] = []
        self.submitted_jobs = []

    def submit_assigned_job(self, job) -> None:
        self.submitted_job_ids.append(job.job_id)
        self.submitted_jobs.append(job)

    async def poll_result(self) -> JobResult | None:
        await asyncio.sleep(0.01)
        return None

    def check_for_crashes(self) -> None:
        return None


@pytest.mark.asyncio
async def test_scheduler_uses_global_fifo_queue() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
    )
    await scheduler.submit_request("req-a", [_pending("a1"), _pending("a2")])
    await scheduler.submit_request("req-b", [_pending("b1")])

    await scheduler.dispatch_once()
    assert set(scheduler._in_flight) == {"a1"}
    assert runtime.submitted_job_ids == ["a1"]

    scheduler.on_result(_result("req-a", "a1"))
    await scheduler.dispatch_once()

    assert set(scheduler._in_flight) == {"a2"}
    assert runtime.submitted_job_ids == ["a1", "a2"]

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_routes_results_to_request_completion() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=2),
        runtime=runtime,
    )
    await scheduler.submit_request("req-1", [_pending("j1"), _pending("j2")])

    await scheduler.dispatch_once()
    scheduler.on_result(_result("req-1", "j1"))
    scheduler.on_result(_result("req-1", "j2"))

    completion = await scheduler.wait_request("req-1")
    assert [result.request_id for result in completion] == ["req-1", "req-1"]
    assert [result.job_id for result in completion] == ["j1", "j2"]

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_uses_per_request_driver_pool() -> None:
    """When a per-request driver_pool is provided, dispatch uses it instead of the base pool."""
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
    )

    custom_driver = AddressPool(["custom-driver:9999"], n_concurrent=1, skip=False)
    await scheduler.submit_request(
        "req-custom",
        [_pending("j1")],
        driver_pool=custom_driver,
    )

    assert runtime.submitted_job_ids == ["j1"]

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_per_request_pool_releases_correctly() -> None:
    """Per-request driver pool slots are released back to the correct pool on result."""
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
    )

    # Custom pool with capacity=1, so second job can only dispatch after first completes
    custom_driver = AddressPool(["custom-driver:9999"], n_concurrent=1, skip=False)
    await scheduler.submit_request(
        "req-custom",
        [_pending("j1"), _pending("j2")],
        driver_pool=custom_driver,
    )

    # Only j1 dispatched (capacity=1)
    assert runtime.submitted_job_ids == ["j1"]

    # Complete j1 — should release slot and dispatch j2
    scheduler.on_result(_result("req-custom", "j1"))
    await scheduler.dispatch_once()

    assert runtime.submitted_job_ids == ["j1", "j2"]

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_retries_retryable_infrastructure_failure_once() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
    )

    await scheduler.submit_request("req-retry", [_pending("j1"), _pending("j2")])
    assert runtime.submitted_job_ids == ["j1"]

    scheduler.on_result(
        _result(
            "req-retry",
            "j1",
            success=False,
            error="batch_render_rgb failed for camera(s): camera_cross_left_120fov",
            retryable=True,
        )
    )
    await scheduler.dispatch_once()

    assert len(runtime.submitted_jobs) == 2
    assert runtime.submitted_job_ids == ["j1", "j2"]

    scheduler.on_result(_result("req-retry", "j2"))
    await scheduler.dispatch_once()

    assert len(runtime.submitted_jobs) == 3
    retry_job = runtime.submitted_jobs[2]
    assert retry_job.job_id != "j1"
    assert retry_job.scene_id == "scene-a"
    assert retry_job.rollout_spec_index == 0
    assert retry_job.retry_attempt == 1
    assert retry_job.max_retry_attempts == 1
    assert retry_job.session_uuid == ""

    scheduler.on_result(_result("req-retry", retry_job.job_id))
    completion = await scheduler.wait_request("req-retry")

    assert [result.job_id for result in completion] == ["j2", retry_job.job_id]
    assert all(result.success for result in completion)

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_records_retryable_failure_after_retry_budget_exhausted() -> (
    None
):
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
    )

    await scheduler.submit_request(
        "req-exhausted",
        [
            PendingRolloutJob(
                job_id="j1",
                scene_id="scene-a",
                rollout_spec_index=0,
                retry_attempt=1,
                max_retry_attempts=1,
            )
        ],
    )

    scheduler.on_result(
        _result(
            "req-exhausted",
            "j1",
            success=False,
            error="still bad",
            retryable=True,
        )
    )
    completion = await scheduler.wait_request("req-exhausted")

    assert runtime.submitted_job_ids == ["j1"]
    assert len(completion) == 1
    assert completion[0].success is False
    assert completion[0].job_id == "j1"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_assigns_active_renderer_endpoint() -> None:
    runtime = _FakeRuntime()
    pools = _make_pools(capacity_per_service=1)
    pools["renderer"] = AddressPool(["video-model:50056"], 1, skip=False)
    scheduler = DaemonScheduler(
        pools=pools,
        runtime=runtime,
    )

    await scheduler.submit_request("req-renderer", [_pending("j1")])

    job = runtime.submitted_jobs[0]
    assert job.endpoints.renderer.address == "video-model:50056"
    assert not hasattr(job.endpoints, "sensorsim")

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_does_not_acquire_inactive_renderer_pool() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
    )

    await scheduler.submit_request("req-renderer", [_pending("j1")])

    assert runtime.submitted_job_ids == ["j1"]
    assert runtime.submitted_jobs[0].endpoints.renderer.address == "sensorsim:50052"

    await scheduler.shutdown(reason="test cleanup")
