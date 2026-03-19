# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import suppress
from typing import Protocol

from alpasim_runtime.address_pool import (
    AddressPool,
    ServiceAddress,
    release_all,
    try_acquire_all,
)
from alpasim_runtime.daemon.request_store import RequestStore
from alpasim_runtime.worker.ipc import (
    AssignedRolloutJob,
    JobResult,
    PendingRolloutJob,
    ServiceEndpoints,
)

logger = logging.getLogger(__name__)


class DaemonUnavailableError(RuntimeError):
    """Raised when a request cannot be served because the daemon is shutting down."""

    pass


class WorkerRuntimeProtocol(Protocol):
    """Minimal interface the scheduler requires from a worker runtime."""

    def submit_assigned_job(self, job: AssignedRolloutJob) -> None: ...

    async def poll_result(self) -> JobResult | None: ...

    def check_for_crashes(self) -> None: ...


class DaemonScheduler:
    """Job scheduler that manages dispatch of simulation jobs to workers.

    Maintains a global pending queue and uses a greedy acquire-all strategy:
    for each pending job, it attempts to acquire one slot from every service
    pool simultaneously.  If any pool is exhausted, dispatch pauses until a
    running job completes and releases its slots.

    Requests may optionally override specific pools (e.g. a per-request driver
    pool) via ``submit_request``.
    """

    def __init__(
        self,
        *,
        pools: dict[str, AddressPool],
        runtime: WorkerRuntimeProtocol,
    ) -> None:
        self._pools = pools
        self._runtime = runtime
        self._request_store = RequestStore()
        self._global_pending: deque[tuple[str, PendingRolloutJob]] = deque()
        self._in_flight: dict[
            str, tuple[dict[str, AddressPool], dict[str, ServiceAddress]]
        ] = {}
        self._request_pools: dict[str, dict[str, AddressPool]] = {}
        self._accepting_requests = True
        self._dispatch_loop_task = asyncio.create_task(self._dispatch_loop())

    async def submit_request(
        self,
        request_id: str,
        jobs: list[PendingRolloutJob],
        *,
        driver_pool: AddressPool | None = None,
    ) -> None:
        """Register a new simulation request and enqueue its jobs for dispatch.

        If *driver_pool* is provided, it overrides the global driver pool for
        all jobs belonging to this request.  After enqueuing, immediately
        attempts to dispatch as many jobs as possible.

        Raises:
            DaemonUnavailableError: If the scheduler has stopped accepting requests.
        """
        if not self._accepting_requests:
            raise DaemonUnavailableError("daemon is not accepting new requests")

        if driver_pool is not None:
            self._request_pools[request_id] = {**self._pools, "driver": driver_pool}

        await self._request_store.register_request(request_id, expected_jobs=len(jobs))
        for job in jobs:
            self._global_pending.append((request_id, job))
        await self.dispatch_once()

    async def wait_request(self, request_id: str) -> list[JobResult]:
        results = await self._request_store.wait_for_completion(request_id)
        self._request_pools.pop(request_id, None)
        return results

    async def shutdown(self, *, reason: str) -> None:
        """Stop accepting requests, fail pending jobs, and cancel the dispatch loop.

        Only queued jobs that have not yet been assigned to workers are failed
        immediately.  Jobs already in-flight are **not** drained: the dispatch
        loop is cancelled, so any results arriving after this point will not be
        recorded and their pool slots will not be released.  The caller is
        expected to stop the worker runtime shortly after, making in-flight
        result processing unnecessary.
        """
        self._accepting_requests = False

        pending_request_ids = {request_id for request_id, _ in self._global_pending}
        self._global_pending.clear()
        for request_id in pending_request_ids:
            self._request_pools.pop(request_id, None)
            self._request_store.fail_request(request_id, reason)

        self._dispatch_loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._dispatch_loop_task

    async def dispatch_once(self) -> None:
        """Greedily dispatch as many pending jobs as possible.

        Peeks at the head of the queue, attempts to acquire all required
        service slots, and submits the job to the worker runtime.  Repeats
        until the queue is empty or no service slots are available.
        """
        while self._global_pending:
            request_id, pending_job = self._global_pending[0]  # peek
            pools = self._request_pools.get(request_id, self._pools)
            acquired = try_acquire_all(pools)
            if acquired is None:
                return

            self._global_pending.popleft()  # consume after successful acquire

            assigned = AssignedRolloutJob(
                request_id=request_id,
                job_id=pending_job.job_id,
                scene_id=pending_job.scene_id,
                rollout_spec_index=pending_job.rollout_spec_index,
                data_source=pending_job.data_source,
                endpoints=ServiceEndpoints(
                    driver=acquired["driver"],
                    sensorsim=acquired["sensorsim"],
                    physics=acquired["physics"],
                    trafficsim=acquired["trafficsim"],
                    controller=acquired["controller"],
                ),
            )
            self._runtime.submit_assigned_job(assigned)
            self._in_flight[assigned.job_id] = (pools, acquired)

    def on_result(self, result: JobResult) -> None:
        entry = self._in_flight.pop(result.job_id, None)
        if entry is None:
            raise RuntimeError(f"Unknown job_id in result queue: {result.job_id}")
        pools, acquired = entry
        release_all(pools, acquired)
        self._request_store.record_result(result)

        reaped = self._request_store.reap_abandoned()
        if reaped:
            logger.info("Reaped %d abandoned request(s)", reaped)

    async def _dispatch_loop(self) -> None:
        """Background loop that processes completed jobs and re-dispatches.

        Polls the worker runtime for results, releases service slots, records
        results in the request store, and triggers another dispatch round.
        If an unexpected error occurs, all pending requests are failed.
        """
        try:
            while True:
                result = await self._runtime.poll_result()
                self._runtime.check_for_crashes()
                if result is None:
                    continue
                self.on_result(result)
                await self.dispatch_once()
        except Exception as exc:
            self._request_store.fail_all_requests(str(exc))
            logger.exception("Result pump failed")
            raise
