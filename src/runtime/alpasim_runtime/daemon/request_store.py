# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from alpasim_runtime.worker.ipc import JobResult


@dataclass
class RequestState:
    """Tracks completion progress for a single simulation request."""

    expected_jobs: int
    pending_jobs: int
    results: list[JobResult]
    completion: asyncio.Future[list[JobResult]]
    abandoned: bool = False


class RequestStore:
    """Future-based tracker for in-flight simulation requests.

    Each request is registered with an expected job count.  As results arrive,
    they are recorded and a completion future is resolved once all expected
    results are in.  Callers await the future via ``wait_for_completion``.

    All operations assume a single event-loop (no internal locking).
    """

    def __init__(self) -> None:
        self._requests: dict[str, RequestState] = {}

    async def register_request(self, request_id: str, expected_jobs: int) -> None:
        """Register a new request and create its completion future.

        If *expected_jobs* is zero the future is resolved immediately with an
        empty result list.
        """
        if expected_jobs < 0:
            raise ValueError(f"expected_jobs must be non-negative, got {expected_jobs}")
        if request_id in self._requests:
            raise RuntimeError(f"Request already registered: {request_id}")

        completion = asyncio.get_running_loop().create_future()
        self._requests[request_id] = RequestState(
            expected_jobs=expected_jobs,
            pending_jobs=expected_jobs,
            results=[],
            completion=completion,
        )

        if expected_jobs == 0:
            completion.set_result([])

    def record_result(self, result: JobResult) -> None:
        """Record a job result and auto-complete the request when all results arrive."""
        state = self._requests.get(result.request_id)
        if state is None:
            raise RuntimeError(f"Unknown request_id in result: {result.request_id}")

        state.results.append(result)
        state.pending_jobs -= 1

        if state.pending_jobs < 0:
            raise RuntimeError(
                f"Too many results recorded for request: {result.request_id}"
            )

        if state.pending_jobs == 0:
            state.completion.set_result(state.results)

    async def wait_for_completion(self, request_id: str) -> list[JobResult]:
        """Await completion of a request and return its results.

        The request's state is cleaned up after the future resolves on success
        or failure.  If the waiter is *cancelled* (e.g. client disconnect),
        the state is **kept** so that in-flight results can still be recorded
        without crashing the scheduler's dispatch loop.  The caller is
        responsible for explicitly cancelling/draining abandoned requests.
        """
        state = self._requests.get(request_id)
        if state is None:
            raise RuntimeError(f"Unknown request_id: {request_id}")
        try:
            return await state.completion
        except asyncio.CancelledError:
            # Keep request state alive — jobs may still complete.  Mark it
            # abandoned so the reaper can distinguish it from a request whose
            # waiter simply hasn't resumed yet after the future resolved.
            state.abandoned = True
            raise
        finally:
            if state.completion.done():
                self._requests.pop(request_id, None)

    def fail_request(self, request_id: str, message: str) -> None:
        state = self._requests.get(request_id)
        if state is None:
            raise RuntimeError(f"Unknown request_id: {request_id}")
        if not state.completion.done():
            state.completion.set_exception(RuntimeError(message))

    def reap_abandoned(self) -> int:
        """Remove completed requests whose waiters were cancelled.

        Only entries explicitly marked ``abandoned`` by a cancelled
        ``wait_for_completion`` are reaped.  Requests whose future is merely
        ``done()`` are left alone: the waiter may still be scheduled to
        resume on the event loop, or the caller may not have called
        ``wait_for_completion`` yet.

        Returns the number of reaped entries.  Safe to call periodically
        (e.g. after each simulation step) to bound memory growth from
        client disconnects.
        """
        abandoned = [
            rid
            for rid, state in self._requests.items()
            if state.abandoned and state.completion.done()
        ]
        for rid in abandoned:
            del self._requests[rid]
        return len(abandoned)

    @property
    def active_request_count(self) -> int:
        """Number of requests currently tracked (pending + abandoned)."""
        return len(self._requests)

    def fail_all_requests(self, message: str) -> None:
        for state in self._requests.values():
            if not state.completion.done():
                state.completion.set_exception(RuntimeError(message))
        self._requests.clear()
