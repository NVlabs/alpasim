# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio

import pytest
from alpasim_runtime.daemon.request_store import RequestStore
from alpasim_runtime.worker.ipc import JobResult


def _make_result(request_id: str, job_id: str) -> JobResult:
    return JobResult(
        request_id=request_id,
        job_id=job_id,
        rollout_spec_index=0,
        success=True,
        error=None,
        error_traceback=None,
        rollout_uuid=f"uuid-{job_id}",
    )


@pytest.mark.asyncio
async def test_request_store_completes_when_all_results_arrive() -> None:
    store = RequestStore()
    await store.register_request("req-1", expected_jobs=2)

    store.record_result(_make_result("req-1", "job-1"))
    store.record_result(_make_result("req-1", "job-2"))

    final = await store.wait_for_completion("req-1")
    assert len(final) == 2


@pytest.mark.asyncio
async def test_request_store_tracks_multiple_requests_independently() -> None:
    store = RequestStore()
    await store.register_request("req-a", expected_jobs=1)
    await store.register_request("req-b", expected_jobs=1)

    store.record_result(_make_result("req-b", "job-b1"))
    store.record_result(_make_result("req-a", "job-a1"))

    done_a = await store.wait_for_completion("req-a")
    done_b = await store.wait_for_completion("req-b")
    assert [result.request_id for result in done_a] == ["req-a"]
    assert [result.request_id for result in done_b] == ["req-b"]


@pytest.mark.asyncio
async def test_request_store_unknown_result_request_raises() -> None:
    store = RequestStore()

    with pytest.raises(RuntimeError, match="Unknown request_id"):
        store.record_result(_make_result("req-missing", "job-1"))


@pytest.mark.asyncio
async def test_request_store_fail_request_sets_completion_exception() -> None:
    store = RequestStore()
    await store.register_request("req-1", expected_jobs=1)

    store.fail_request("req-1", "request failed")

    with pytest.raises(RuntimeError, match="request failed"):
        await store.wait_for_completion("req-1")


@pytest.mark.asyncio
async def test_request_store_fail_all_requests_allows_waiter_cleanup() -> None:
    store = RequestStore()
    await store.register_request("req-1", expected_jobs=1)

    waiter = asyncio.create_task(store.wait_for_completion("req-1"))
    await asyncio.sleep(0)
    store.fail_all_requests("daemon stopping")

    with pytest.raises(RuntimeError, match="daemon stopping"):
        await waiter


@pytest.mark.asyncio
async def test_request_store_too_many_results_raises() -> None:
    store = RequestStore()
    await store.register_request("req-1", expected_jobs=1)

    store.record_result(_make_result("req-1", "job-1"))

    with pytest.raises(RuntimeError, match="Too many results recorded"):
        store.record_result(_make_result("req-1", "job-2"))


@pytest.mark.asyncio
async def test_request_store_reap_abandoned_cleans_cancelled_waiters() -> None:
    store = RequestStore()
    await store.register_request("req-1", expected_jobs=1)

    waiter = asyncio.create_task(store.wait_for_completion("req-1"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    # Request state is still tracked (the leak scenario).
    assert store.active_request_count == 1

    # Complete the jobs so the future is done.
    store.record_result(_make_result("req-1", "job-1"))

    # Reap should clean up the completed-but-unreaped entry.
    reaped = store.reap_abandoned()
    assert reaped == 1
    assert store.active_request_count == 0


@pytest.mark.asyncio
async def test_request_store_reap_abandoned_ignores_pending() -> None:
    store = RequestStore()
    await store.register_request("req-1", expected_jobs=1)

    # Request is still pending (no results yet) — should not be reaped.
    reaped = store.reap_abandoned()
    assert reaped == 0
    assert store.active_request_count == 1
