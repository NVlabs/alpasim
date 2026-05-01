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
        "sensorsim": AddressPool(["sensorsim:50052"], capacity_per_service, skip=False),
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


def _result(request_id: str, job_id: str) -> JobResult:
    return JobResult(
        request_id=request_id,
        job_id=job_id,
        rollout_spec_index=0,
        success=True,
        error=None,
        error_traceback=None,
        rollout_uuid=f"uuid-{job_id}",
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.submitted_job_ids: list[str] = []

    def submit_assigned_job(self, job) -> None:
        self.submitted_job_ids.append(job.job_id)

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
