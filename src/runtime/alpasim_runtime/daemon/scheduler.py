# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from alpasim_runtime.address_pool import (
    AddressPool,
    ServiceAddress,
    release_all,
    try_acquire_all,
)
from alpasim_runtime.config import BASE_SERVICE_NAMES
from alpasim_runtime.daemon.request_store import RequestStore
from alpasim_runtime.nre_introspection import get_loaded_scenes
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


@dataclass
class _InFlightEntry:
    """Bookkeeping for a dispatched job awaiting its result."""

    scene_id: str
    pools: dict[str, AddressPool]
    acquired: dict[str, ServiceAddress]


class WorkerRuntimeProtocol(Protocol):
    """Minimal interface the scheduler requires from a worker runtime."""

    def submit_assigned_job(self, job: AssignedRolloutJob) -> None: ...

    async def poll_result(self) -> JobResult | None: ...

    def check_for_crashes(self) -> None: ...


# ---------------------------------------------------------------------------
# Dispatch strategy interface + implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingReservation:
    """Outcome of a dispatch strategy's ``try_reserve()`` call.

    Holds a pre-acquired renderer slot and the selected job.  Must be
    finalized via ``commit()`` or the renderer slot must be released by
    the caller.
    """

    request_id: str
    job: PendingRolloutJob
    renderer_slot: ServiceAddress
    is_affine_hit: bool


class DispatchStrategy(Protocol):
    """Narrow interface for job-selection and pending-job bookkeeping.

    Two concrete implementations exist: ``FifoDispatch`` (strict submission
    order) and ``SceneAffineDispatch`` (three-tier cache-aware priority).

    Dispatch follows a two-phase reservation/commit protocol:

    1. ``try_reserve()`` selects a job **and** pre-acquires a renderer slot.
    2. The caller acquires the remaining service pools.
    3. ``commit(reservation)`` atomically removes the job from pending
       and records it as in-flight.

    If step 2 fails, the caller releases the renderer slot itself and
    the job remains pending for the next round.
    """

    @property
    def pending_count(self) -> int: ...

    def add_pending(self, request_id: str, job: PendingRolloutJob) -> None: ...

    def try_reserve(self) -> PendingReservation | None:
        """Select the best pending job and pre-acquire a renderer slot.

        Returns ``None`` when no jobs are pending or no renderer slot is free.
        """
        ...

    def commit(self, reservation: PendingReservation) -> None:
        """Finalize a reservation: remove from pending and record in-flight."""
        ...

    def on_result(self, scene_id: str, renderer_address: str, success: bool) -> None:
        """Called when a job completes."""
        ...

    def drain_pending_request_ids(self) -> set[str]:
        """Extract all pending request IDs and clear pending storage."""
        ...

    async def shutdown(self) -> None:
        """Cancel background tasks and log summary statistics."""
        ...


class FifoDispatch:
    """Strict submission-order dispatch -- identical to pre-affine behavior."""

    def __init__(self, *, renderer_pool: AddressPool) -> None:
        self._renderer_pool = renderer_pool
        self._queue: deque[tuple[str, PendingRolloutJob]] = deque()

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def add_pending(self, request_id: str, job: PendingRolloutJob) -> None:
        self._queue.append((request_id, job))

    def try_reserve(self) -> PendingReservation | None:
        if not self._queue:
            return None
        slot = self._renderer_pool.try_acquire()
        if slot is None:
            return None
        request_id, job = self._queue[0]
        return PendingReservation(request_id, job, slot, is_affine_hit=False)

    def commit(self, reservation: PendingReservation) -> None:
        self._queue.popleft()

    def on_result(self, scene_id: str, renderer_address: str, success: bool) -> None:
        return

    def drain_pending_request_ids(self) -> set[str]:
        ids = {req_id for req_id, _ in self._queue}
        self._queue.clear()
        return ids

    async def shutdown(self) -> None:
        pass


class SceneAffineDispatch:
    """Three-tier cache-aware dispatch strategy.

    Tier 1 -- Affine: pick a job whose scene is already cached *or
    in-flight* on a free renderer GPU.  Likely warm-cache hit.

    Tier 2 -- New scene: pick a job for a scene not yet cached by
    any GPU.  Maximises cache diversity across GPUs.

    Tier 3 -- Fallback: pick any pending job.
    """

    def __init__(
        self,
        *,
        renderer_pool: AddressPool,
        cache_refresh_interval_s: float | None = 5.0,
    ) -> None:
        self._renderer_pool = renderer_pool
        self._cache_refresh_interval_s = cache_refresh_interval_s

        self._pending_by_scene: dict[str, deque[tuple[str, PendingRolloutJob]]] = {}
        # In-flight scene<->address tracking so the dispatch tiers can
        # see what's been dispatched but not yet released.
        # Counts (not sets) because multiple jobs for the same scene
        # can be on the same address (n_concurrent > 1).
        self._inflight_addr_scenes: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # address --> {scene --> count}

        # Per-address set of known cached scenes (written only by sync_scene_cache).
        self._address_scenes: dict[str, set[str]] = {}

        self._affine_hits = 0
        self._total_dispatched = 0

        self._cache_refresh_task: asyncio.Task[None] | None = None

    def sync_scene_cache(self, address: str, scene_ids: list[str]) -> None:
        """Overwrite the scene set for *address* with the server's authoritative list."""
        self._address_scenes[address] = set(scene_ids)

    def cached_scenes(self, address: str) -> set[str]:
        """Return the scenes known to be cached on *address*."""
        return self._address_scenes.get(address, set())

    def is_scene_cached(self, scene_id: str) -> bool:
        """True if at least one address has *scene_id* in its cache."""
        return any(scene_id in scenes for scenes in self._address_scenes.values())

    async def warm_start(self) -> None:
        """Perform initial cache sync from all NRE addresses, then start refresh loop.

        Raises ``IntrospectionNotSupportedError`` if any NRE server returns
        UNIMPLEMENTED, allowing fast failure at startup when the NRE image is
        incompatible with scene-affine dispatch.
        """
        if self._renderer_pool.skip or not self._renderer_pool.all_addresses():
            return

        snapshot = await self._refresh_cache_once(raise_on_unimplemented=True)
        if snapshot:
            total_scenes = sum(len(s) for s in snapshot.values())
            logger.info(
                "Warm-started %d renderer address(es) with %d cached scene(s)",
                len(snapshot),
                total_scenes,
            )

        if self._cache_refresh_interval_s is not None:
            self._cache_refresh_task = asyncio.create_task(self._cache_refresh_loop())

    # -- pending-job bookkeeping --

    @property
    def pending_count(self) -> int:
        return sum(len(q) for q in self._pending_by_scene.values())

    def add_pending(self, request_id: str, job: PendingRolloutJob) -> None:
        scene = job.scene_id
        if scene not in self._pending_by_scene:
            self._pending_by_scene[scene] = deque()
        self._pending_by_scene[scene].append((request_id, job))

    def _pop_pending(self, scene_id: str) -> tuple[str, PendingRolloutJob]:
        q = self._pending_by_scene[scene_id]
        entry = q.popleft()
        if not q:
            del self._pending_by_scene[scene_id]
        return entry

    # -- job selection (three-tier priority) --

    def _is_scene_inflight(self, scene_id: str) -> bool:
        """True if *scene_id* is in-flight on any renderer address."""
        return any(scene_id in scenes for scenes in self._inflight_addr_scenes.values())

    def try_reserve(self) -> PendingReservation | None:
        # Tier 1: free GPU slot whose cached *or in-flight* scene has a
        # pending job.  Iterates free addresses (tiny, ~1-8) x scenes per
        # address (small), NOT pending scenes (could be thousands).
        for address in self._renderer_pool.free_addresses():
            scenes: set[str] = set(self.cached_scenes(address))
            inflight = self._inflight_addr_scenes.get(address)
            if inflight:
                scenes.update(inflight)
            for scene in scenes:
                if scene in self._pending_by_scene:
                    slot = self._renderer_pool.try_acquire_for_address(address)
                    if slot is None:
                        break
                    request_id, job = self._pending_by_scene[scene][0]
                    return PendingReservation(request_id, job, slot, is_affine_hit=True)

        # Tier 2: pending scene not yet cached or in-flight on any GPU.
        for scene in self._pending_by_scene:
            if not self.is_scene_cached(scene) and not self._is_scene_inflight(scene):
                slot = self._renderer_pool.try_acquire()
                if slot is None:
                    return None
                request_id, job = self._pending_by_scene[scene][0]
                return PendingReservation(request_id, job, slot, is_affine_hit=False)

        # Tier 3: fallback -- any pending job, FIFO renderer slot.
        if not self._pending_by_scene:
            return None
        slot = self._renderer_pool.try_acquire()
        if slot is None:
            return None
        scene_deque = next(iter(self._pending_by_scene.values()))
        request_id, job = scene_deque[0]
        return PendingReservation(request_id, job, slot, is_affine_hit=False)

    # -- dispatch / result hooks --

    def commit(self, reservation: PendingReservation) -> None:
        self._pop_pending(reservation.job.scene_id)
        self._inflight_addr_scenes[reservation.renderer_slot.address][
            reservation.job.scene_id
        ] += 1
        if reservation.is_affine_hit:
            self._affine_hits += 1
        self._total_dispatched += 1

    def on_result(self, scene_id: str, renderer_address: str, success: bool) -> None:
        addr_scenes = self._inflight_addr_scenes.get(renderer_address)
        if addr_scenes:
            addr_scenes[scene_id] -= 1
            if addr_scenes[scene_id] <= 0:
                del addr_scenes[scene_id]
            if not addr_scenes:
                del self._inflight_addr_scenes[renderer_address]

    def drain_pending_request_ids(self) -> set[str]:
        ids = {req_id for jobs in self._pending_by_scene.values() for req_id, _ in jobs}
        self._pending_by_scene.clear()
        return ids

    # -- lifecycle --

    async def shutdown(self) -> None:
        if self._cache_refresh_task is not None:
            self._cache_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cache_refresh_task

        if self._total_dispatched > 0:
            pct = self._affine_hits / self._total_dispatched * 100
            logger.info(
                "Scene-affine dispatch summary: %d/%d affine hits (%.1f%%)",
                self._affine_hits,
                self._total_dispatched,
                pct,
            )

    async def _refresh_cache_once(
        self, *, raise_on_unimplemented: bool = False
    ) -> dict[str, frozenset[str]]:
        """Query all renderer addresses and sync the local cache mirror.

        Returns a mapping of address → frozenset of cached scene IDs for each
        address that responded successfully.
        """
        unique_addresses = sorted(self._renderer_pool.all_addresses())
        snapshot: dict[str, frozenset[str]] = {}
        for address in unique_addresses:
            loaded = await get_loaded_scenes(
                address, raise_on_unimplemented=raise_on_unimplemented
            )
            if loaded is None:
                continue
            scene_ids = list(loaded.keys())
            self.sync_scene_cache(address, scene_ids)
            snapshot[address] = frozenset(scene_ids)
        return snapshot

    async def _cache_refresh_loop(self) -> None:
        """Periodically re-sync the scene cache from NRE."""
        assert self._cache_refresh_interval_s is not None
        num_addresses = len(self._renderer_pool.all_addresses())
        logger.info(
            "Cache refresh loop started: %d address(es), interval=%.1fs",
            num_addresses,
            self._cache_refresh_interval_s,
        )
        prev_snapshots: dict[str, frozenset[str]] = {}
        while True:
            await asyncio.sleep(self._cache_refresh_interval_s)
            current_snapshots = await self._refresh_cache_once()
            for address, current in current_snapshots.items():
                prev = prev_snapshots.get(address, frozenset())
                if current != prev:
                    added = current - prev
                    removed = prev - current
                    logger.info(
                        "Cache changed on %s: +%d scene(s) %s, -%d scene(s) %s",
                        address,
                        len(added),
                        sorted(added) if added else "[]",
                        len(removed),
                        sorted(removed) if removed else "[]",
                    )
            prev_snapshots = current_snapshots


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class DaemonScheduler:
    """Job scheduler that manages dispatch of simulation jobs to workers.

    Maintains pending jobs and uses a greedy acquire-all strategy: for each
    dispatch round it selects the best pending job via a pluggable
    ``DispatchStrategy``, acquires one slot from every service pool, and
    submits the job to the worker runtime.

    Requests may optionally override specific pools (e.g. a per-request
    driver pool) via ``submit_request``.
    """

    def __init__(
        self,
        *,
        pools: dict[str, AddressPool],
        runtime: WorkerRuntimeProtocol,
        scene_affine_dispatch: bool = True,
        cache_refresh_interval_s: float | None = 5.0,
    ) -> None:
        self._pools = pools
        self._runtime = runtime
        self._required_service_names = (*BASE_SERVICE_NAMES, "renderer")
        self._request_store = RequestStore()

        renderer_pool = pools["renderer"]
        if scene_affine_dispatch:
            self._strategy: DispatchStrategy = SceneAffineDispatch(
                renderer_pool=renderer_pool,
                cache_refresh_interval_s=cache_refresh_interval_s,
            )
            logger.info("Scene-affine dispatch ENABLED for renderer")
        else:
            self._strategy = FifoDispatch(renderer_pool=renderer_pool)
            logger.info("Scene-affine dispatch DISABLED")

        self._in_flight: dict[str, _InFlightEntry] = {}
        self._request_pools: dict[str, dict[str, AddressPool]] = {}
        self._accepting_requests = True
        self._dispatch_loop_task = asyncio.create_task(self._dispatch_loop())

    async def warm_start(self) -> None:
        """Seed the dispatch strategy's cache from NRE servers.

        Only meaningful for SceneAffineDispatch; no-op for FifoDispatch.
        Raises ``IntrospectionNotSupportedError`` if the NRE image does not
        support GetLoadedScenes.
        """
        if isinstance(self._strategy, SceneAffineDispatch):
            await self._strategy.warm_start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_request(
        self,
        request_id: str,
        jobs: list[PendingRolloutJob],
        *,
        driver_pool: AddressPool | None = None,
    ) -> None:
        """Register a new simulation request and enqueue its jobs for dispatch.

        Jobs are grouped by scene_id before enqueuing so that consecutive
        dispatches for the same scene benefit from renderer cache affinity.

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
            self._strategy.add_pending(request_id, job)

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

        pending_request_ids = self._strategy.drain_pending_request_ids()
        for request_id in pending_request_ids:
            self._request_pools.pop(request_id, None)
            self._request_store.fail_request(request_id, reason)

        self._dispatch_loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._dispatch_loop_task

        await self._strategy.shutdown()

    async def dispatch_once(self) -> None:
        """Greedily dispatch pending jobs via the active strategy.

        The strategy reserves the best job *and* pre-acquires the renderer
        slot.  ``try_acquire_all`` then acquires the remaining pools,
        rolling back the renderer slot on failure.
        """
        while self._strategy.pending_count > 0:
            reservation = self._strategy.try_reserve()
            if reservation is None:
                return

            pools = self._request_pools.get(reservation.request_id, self._pools)
            required_pools = {
                name: pools[name] for name in self._required_service_names
            }

            acquired = try_acquire_all(
                required_pools, renderer_slot=reservation.renderer_slot
            )
            if acquired is None:
                # try_acquire_all already released renderer_slot.
                return

            self._strategy.commit(reservation)

            assigned = AssignedRolloutJob(
                request_id=reservation.request_id,
                job_id=reservation.job.job_id,
                scene_id=reservation.job.scene_id,
                rollout_spec_index=reservation.job.rollout_spec_index,
                endpoints=ServiceEndpoints(
                    driver=acquired["driver"],
                    renderer=acquired["renderer"],
                    physics=acquired["physics"],
                    trafficsim=acquired["trafficsim"],
                    controller=acquired["controller"],
                ),
                session_uuid=reservation.job.session_uuid,
            )
            self._runtime.submit_assigned_job(assigned)
            self._in_flight[assigned.job_id] = _InFlightEntry(
                scene_id=reservation.job.scene_id,
                pools=required_pools,
                acquired=acquired,
            )

    def on_result(self, result: JobResult) -> None:
        entry = self._in_flight.pop(result.job_id, None)
        if entry is None:
            raise RuntimeError(f"Unknown job_id in result queue: {result.job_id}")

        renderer_addr = entry.acquired["renderer"].address
        self._strategy.on_result(entry.scene_id, renderer_addr, result.success)

        release_all(entry.pools, entry.acquired)
        self._request_store.record_result(result)

        try:
            reaped = self._request_store.reap_abandoned()
        except Exception:
            logger.exception("Failed to reap abandoned requests")
            reaped = 0
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
