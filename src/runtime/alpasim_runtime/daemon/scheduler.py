# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import logging
import shutil
from collections import Counter, defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import uuid4

from alpasim_runtime.address_pool import (
    AddressPool,
    ServiceAddress,
    release_all,
    try_acquire_all,
)
from alpasim_runtime.config import BASE_SERVICE_NAMES, SceneAffineDispatchConfig
from alpasim_runtime.daemon.request_store import RequestStore
from alpasim_runtime.nre_introspection import get_loaded_scenes
from alpasim_runtime.worker.ipc import (
    AssignedRolloutJob,
    DispatchKind,
    JobResult,
    PendingRolloutJob,
    ServiceEndpoints,
)

logger = logging.getLogger(__name__)


class DaemonUnavailableError(RuntimeError):
    """Raised when a request cannot be served because the daemon is shutting down."""

    pass


def _clear_retry_rollout_dir(pending_job: PendingRolloutJob, rollouts_dir: str) -> None:
    """Remove artifacts that would otherwise be reused by a retry."""
    if not pending_job.session_uuid:
        return

    if (
        Path(pending_job.scene_id).name != pending_job.scene_id
        or Path(pending_job.session_uuid).name != pending_job.session_uuid
    ):
        raise ValueError(
            "Retry rollout scene and session identifiers must each be one path "
            f"component: scene_id={pending_job.scene_id!r} "
            f"session_uuid={pending_job.session_uuid!r}"
        )

    rollouts_root = Path(rollouts_dir).resolve()
    scene_dir = (rollouts_root / pending_job.scene_id).resolve()
    rollout_dir = scene_dir / pending_job.session_uuid
    resolved_rollout_dir = rollout_dir.resolve()

    if scene_dir.parent != rollouts_root or resolved_rollout_dir.parent != scene_dir:
        raise ValueError(
            "Retry rollout directory must be a direct child of the rollouts root "
            f"and scene directory: scene_id={pending_job.scene_id!r} "
            f"session_uuid={pending_job.session_uuid!r}"
        )
    if rollout_dir.is_symlink():
        raise ValueError(f"Refusing to remove retry rollout symlink: {rollout_dir}")
    if rollout_dir.exists():
        logger.info("Removing artifacts from failed rollout attempt: %s", rollout_dir)
        shutil.rmtree(rollout_dir)


@dataclass
class _InFlightEntry:
    """Bookkeeping for a dispatched job awaiting its result."""

    pending_job: PendingRolloutJob
    dispatch_kind: DispatchKind  # For telemetry
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

    job: PendingRolloutJob
    renderer_slot: ServiceAddress
    dispatch_kind: DispatchKind


class DispatchStrategy(Protocol):
    """Narrow interface for job-selection and pending-job bookkeeping.

    Two concrete implementations exist: ``FifoDispatch`` (strict submission
    order) and ``SceneAffineDispatch`` (bounded cache-aware replication).

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

    def add_pending(self, job: PendingRolloutJob) -> None: ...

    def try_reserve(self) -> PendingReservation | None:
        """Select the best pending job and pre-acquire a renderer slot.

        Must only be called while ``pending_count > 0``.  Returns ``None``
        when no renderer slot is free for any pending job.
        """
        ...

    def commit(self, reservation: PendingReservation) -> None:
        """Finalize a reservation: remove from pending and record in-flight."""
        ...

    def on_result(
        self,
        scene_id: str,
        renderer_address: str,
        dispatch_kind: DispatchKind,
        success: bool,
    ) -> None:
        """Called when a job completes."""
        ...

    def drain_pending_request_ids(self) -> set[str]:
        """Extract all pending request IDs and clear pending storage."""
        ...

    async def shutdown(self) -> None:
        """Log summary statistics."""
        ...


class FifoDispatch:
    """Strict submission-order dispatch -- identical to pre-affine behavior."""

    def __init__(self, *, renderer_pool: AddressPool) -> None:
        self._renderer_pool = renderer_pool
        self._queue: deque[PendingRolloutJob] = deque()

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def add_pending(self, job: PendingRolloutJob) -> None:
        self._queue.append(job)

    def try_reserve(self) -> PendingReservation | None:
        slot = self._renderer_pool.try_acquire()
        if slot is None:
            return None
        return PendingReservation(
            self._queue[0],
            slot,
            dispatch_kind="fifo",
        )

    def commit(self, reservation: PendingReservation) -> None:
        self._queue.popleft()

    def on_result(
        self,
        scene_id: str,
        renderer_address: str,
        dispatch_kind: DispatchKind,
        success: bool,
    ) -> None:
        return

    def drain_pending_request_ids(self) -> set[str]:
        ids = {job.request_id for job in self._queue}
        self._queue.clear()
        return ids

    async def shutdown(self) -> None:
        pass


class SceneAffineDispatch:
    """Cache-aware dispatch with bounded per-scene renderer replication.

    Pending jobs are queued per scene, and the strategy tracks the scene
    "locations" of each renderer address: cache contents confirmed by NRE
    introspection snapshots (``sync_scene_cache``) plus cold loads committed
    by this strategy that are still awaiting confirmation.

    ``try_reserve`` picks the next dispatch in two priority tiers:

    1. ``cached_affine``: a renderer whose confirmed cache holds a pending
       scene, so the rollout skips the cold load.  Among matches, the address
       with the most free slots wins, spreading work across replicas.
    2. cold load: walk pending scenes in arrival order and cold-load the
       first one with fewer than ``max_renderers_per_scene`` locations onto
       a renderer where it is absent.  The dispatch kind records whether the
       load creates the scene's first location (``cold_initial``) or an
       additional one (``cold_replica``).  Earlier-queued scenes replicate
       up to the bound before later scenes get their first location, so a
       later scene waits at most ``max_renderers_per_scene`` cold loads per
       scene queued ahead of it.

    Cold loads land on the least-loaded eligible renderer: fewest active
    scenes (loading or with in-flight rollouts) first, most free slots as
    tie-break.  Renderers with ``max_scenes_per_renderer`` active scenes
    are ineligible for cold dispatch, which keeps each renderer's working
    set within its NRE backend cache.
    """

    def __init__(
        self,
        *,
        renderer_pool: AddressPool,
        max_renderers_per_scene: int,
        max_scenes_per_renderer: int | None,
    ) -> None:
        if max_renderers_per_scene <= 0:
            raise ValueError("max_renderers_per_scene must be positive")
        if max_scenes_per_renderer is not None and max_scenes_per_renderer <= 0:
            raise ValueError("max_scenes_per_renderer must be positive")
        self._renderer_pool = renderer_pool
        self._max_renderers_per_scene = max_renderers_per_scene
        self._max_scenes_per_renderer = max_scenes_per_renderer

        self._pending_by_scene: dict[str, deque[PendingRolloutJob]] = {}

        # Successful introspection snapshots and committed cold loads respectively.
        self._address_scenes: dict[str, set[str]] = {}
        self._loading_addr_scenes: dict[str, set[str]] = defaultdict(set)

        # Last snapshot per address, kept only to log cache-change diffs.
        self._prev_snapshots: dict[str, frozenset[str]] = {}

        # In-flight rollout counts per renderer address and scene.  A scene is
        # "active" on an address while it is loading or has in-flight rollouts;
        # idle cached scenes are evictable by NRE and do not count.
        self._inflight_addr_scenes: dict[str, Counter[str]] = defaultdict(Counter)

        self._dispatch_counts: Counter[DispatchKind] = Counter()

    def sync_scene_cache(self, address: str, scene_ids: list[str]) -> bool:
        """Apply one authoritative cache snapshot from a renderer address.

        Replaces the cached-scene set for *address* with the snapshot; scenes
        marked as loading that appear in it are confirmed and unmarked, while
        loading scenes absent from it stay marked (the cold load is still in
        flight — introspection only reports completed loads).

        Returns whether scheduler-relevant state changed: the cached set
        differs, or a load was confirmed (which frees active-scene capacity
        even when the cached set is unchanged), so the caller knows to run a
        dispatch pass.
        """
        scenes = set(scene_ids)
        previous = self._address_scenes.get(address, set())
        loading = self._loading_addr_scenes.get(address)
        confirmed_loading = set() if loading is None else loading & scenes

        self._address_scenes[address] = scenes
        if loading is not None:
            loading.difference_update(confirmed_loading)
            if not loading:
                del self._loading_addr_scenes[address]

        return previous != scenes or bool(confirmed_loading)

    def cached_scenes(self, address: str) -> set[str]:
        """Return the scenes known to be cached on *address*."""
        return self._address_scenes.get(address, set())

    def is_scene_cached(self, scene_id: str) -> bool:
        """True if at least one address has *scene_id* in its cache."""
        return any(scene_id in scenes for scenes in self._address_scenes.values())

    def _scene_locations(self, scene_id: str) -> set[str]:
        return {
            address
            for address in self._renderer_pool.address_order
            if scene_id in self._address_scenes.get(address, set())
            or scene_id in self._loading_addr_scenes.get(address, set())
        }

    def _active_scene_counts(self) -> dict[str, int]:
        """Distinct scenes loading or executing per renderer address."""
        return {
            address: len(
                set(self._inflight_addr_scenes.get(address, {}))
                | self._loading_addr_scenes.get(address, set())
            )
            for address in self._renderer_pool.address_order
        }

    async def refresh_cache(self, *, raise_on_unimplemented: bool = False) -> bool:
        """Query all renderer addresses and sync the local cache mirror.

        Returns whether scheduler-relevant state changed.  Raises
        ``IntrospectionNotSupportedError`` when *raise_on_unimplemented* is
        set and an NRE server returns UNIMPLEMENTED, allowing fast failure
        at startup when the NRE image is incompatible with scene-affine
        dispatch.
        """
        changed = False
        for address in sorted(self._renderer_pool.address_order):
            loaded = await get_loaded_scenes(
                address, raise_on_unimplemented=raise_on_unimplemented
            )
            if loaded is None:
                continue
            scene_ids = list(loaded.keys())
            current = frozenset(scene_ids)
            previous = self._prev_snapshots.get(address, frozenset())
            if current != previous:
                added = sorted(current - previous)
                removed = sorted(previous - current)
                logger.info(
                    "Cache changed on %s: +%d scene(s) %s, -%d scene(s) %s",
                    address,
                    len(added),
                    added or "[]",
                    len(removed),
                    removed or "[]",
                )
            self._prev_snapshots[address] = current
            changed = self.sync_scene_cache(address, scene_ids) or changed
        return changed

    # -- pending-job bookkeeping --

    @property
    def pending_count(self) -> int:
        return sum(len(q) for q in self._pending_by_scene.values())

    def add_pending(self, job: PendingRolloutJob) -> None:
        scene = job.scene_id
        if scene not in self._pending_by_scene:
            self._pending_by_scene[scene] = deque()
        self._pending_by_scene[scene].append(job)

    def _pop_pending(self, scene_id: str) -> PendingRolloutJob:
        q = self._pending_by_scene[scene_id]
        job = q.popleft()
        if not q:
            del self._pending_by_scene[scene_id]
        return job

    # -- job selection (two-tier priority) --

    def _best_affine_match(
        self,
        address_scenes: dict[str, set[str]],
        free_slot_counts: dict[str, int],
    ) -> tuple[str, str] | None:
        """Pair a pending scene with an address that already holds it.

        Candidates are addresses that have a free slot and hold at least one
        scene with pending jobs.  Among those, the address with the most free
        slots wins (spreading affine work across replicas); its scene is the
        earliest-queued pending scene it holds.

        Args:
            address_scenes: Scene sets per renderer address to match pending
                jobs against, e.g. ``_address_scenes`` for cached-scene
                affinity.
            free_slot_counts: Free slots per address, from ``try_reserve``.

        Returns:
            The chosen ``(address, scene_id)`` pair, or ``None`` if no
            address with a free slot holds a pending scene.
        """
        matching_addresses = [
            address
            for address in self._renderer_pool.address_order
            if free_slot_counts[address] > 0
            and any(
                scene in self._pending_by_scene
                for scene in address_scenes.get(address, {})
            )
        ]
        if not matching_addresses:
            return None
        address = max(matching_addresses, key=free_slot_counts.__getitem__)
        scenes = address_scenes.get(address, {})
        scene = next(scene for scene in self._pending_by_scene if scene in scenes)
        return address, scene

    def _reserve_affine(
        self,
        address_scenes: dict[str, set[str]],
        free_slot_counts: dict[str, int],
    ) -> PendingReservation | None:
        """Reserve a slot for the best affine match in *address_scenes*.

        Returns a ``cached_affine`` reservation for the head job of the
        matched scene's queue, or ``None`` when no match exists.
        """
        match = self._best_affine_match(address_scenes, free_slot_counts)
        if match is None:
            return None
        address, scene = match
        slot = self._renderer_pool.try_acquire_for_address(address)
        # _best_affine_match only returns addresses with free slots.
        assert slot is not None
        return PendingReservation(
            self._pending_by_scene[scene][0],
            slot,
            dispatch_kind="cached_affine",
        )

    def _reserve_non_affine(
        self,
        job: PendingRolloutJob,
        dispatch_kind: DispatchKind,
        cold_candidates: list[str],
        excluded_addresses: set[str],
    ) -> PendingReservation | None:
        """Reserve a slot for *job* on a renderer that will cold-load its scene.

        Args:
            job: Pending job to reserve a slot for.
            dispatch_kind: Recorded on the reservation, e.g. ``cold_initial``
                or ``cold_replica``.
            cold_candidates: Cold-load-eligible addresses from
                ``try_reserve``, ranked best-first.
            excluded_addresses: Addresses to skip, typically ones that already
                hold or are loading the job's scene.

        Returns:
            The reservation with an acquired slot, or ``None`` if every
            candidate is excluded.
        """
        address = next(
            (a for a in cold_candidates if a not in excluded_addresses), None
        )
        if address is None:
            return None
        slot = self._renderer_pool.try_acquire_for_address(address)
        assert slot is not None  # candidates are filtered on free slots
        return PendingReservation(
            job,
            slot,
            dispatch_kind=dispatch_kind,
        )

    def try_reserve(self) -> PendingReservation | None:
        # Slot and active-scene state only changes through commit/on_result,
        # so one snapshot serves the whole call.
        free_slot_counts = self._renderer_pool.free_slot_counts()
        if not any(free_slot_counts.values()):
            return None

        # Tier 1: confirmed cache affinity.
        reservation = self._reserve_affine(self._address_scenes, free_slot_counts)
        if reservation is not None:
            return reservation

        # Cold-load candidates: a free slot and active-scene headroom.  Ranked
        # once for all cold-load attempts — fewest active scenes first (keeping
        # each renderer's NRE cache working set small); most free slots and
        # configured order break ties.
        active_scene_counts = self._active_scene_counts()
        cold_candidates = sorted(
            (
                address
                for address in self._renderer_pool.address_order
                if free_slot_counts[address] > 0
                and (
                    self._max_scenes_per_renderer is None
                    or active_scene_counts[address] < self._max_scenes_per_renderer
                )
            ),
            key=lambda addr: (active_scene_counts[addr], -free_slot_counts[addr]),
        )
        if not cold_candidates:
            return None

        # Tier 2: cold-load scenes in arrival order, adding a location on a
        # renderer where the scene is absent, up to the per-scene bound.
        for scene, jobs in self._pending_by_scene.items():
            locations = self._scene_locations(scene)
            if len(locations) >= self._max_renderers_per_scene:
                continue
            reservation = self._reserve_non_affine(
                jobs[0],
                dispatch_kind="cold_initial" if not locations else "cold_replica",
                cold_candidates=cold_candidates,
                excluded_addresses=locations,
            )
            if reservation is not None:
                return reservation
        return None

    # -- dispatch / result hooks --

    def commit(self, reservation: PendingReservation) -> None:
        self._pop_pending(reservation.job.scene_id)
        self._inflight_addr_scenes[reservation.renderer_slot.address][
            reservation.job.scene_id
        ] += 1
        if reservation.dispatch_kind in {"cold_initial", "cold_replica"}:
            self._loading_addr_scenes[reservation.renderer_slot.address].add(
                reservation.job.scene_id
            )
        self._dispatch_counts[reservation.dispatch_kind] += 1

    def on_result(
        self,
        scene_id: str,
        renderer_address: str,
        dispatch_kind: DispatchKind,
        success: bool,
    ) -> None:
        inflight = self._inflight_addr_scenes[renderer_address]
        inflight[scene_id] -= 1
        if inflight[scene_id] <= 0:
            del inflight[scene_id]
        if not inflight:
            del self._inflight_addr_scenes[renderer_address]

        if dispatch_kind not in {"cold_initial", "cold_replica"}:
            return

        loading = self._loading_addr_scenes.get(renderer_address)
        if loading is not None and scene_id in loading:
            loading.remove(scene_id)
            if not loading:
                del self._loading_addr_scenes[renderer_address]
        if success:
            self._address_scenes.setdefault(renderer_address, set()).add(scene_id)

    def drain_pending_request_ids(self) -> set[str]:
        ids = {
            job.request_id for jobs in self._pending_by_scene.values() for job in jobs
        }
        self._pending_by_scene.clear()
        return ids

    # -- lifecycle --

    async def shutdown(self) -> None:
        total_dispatched = self._dispatch_counts.total()
        if total_dispatched > 0:
            affine_hits = self._dispatch_counts["cached_affine"]
            pct = affine_hits / total_dispatched * 100
            logger.info(
                "Scene-affine dispatch summary: %d/%d affine hits (%.1f%%)",
                affine_hits,
                total_dispatched,
                pct,
            )


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
        scene_affine_dispatch: SceneAffineDispatchConfig,
        max_rollout_retries: int = 2,
        rollouts_dir: str | None = None,
    ) -> None:
        if max_rollout_retries < 0:
            raise ValueError(
                f"max_rollout_retries must be non-negative, got {max_rollout_retries}"
            )

        self._pools = pools
        self._runtime = runtime
        self._max_rollout_retries = max_rollout_retries
        self._rollouts_dir = rollouts_dir
        self._dispatch_lock = asyncio.Lock()
        self._required_service_names = (*BASE_SERVICE_NAMES, "renderer")
        self._request_store = RequestStore()

        renderer_pool = pools["renderer"]
        if scene_affine_dispatch.enabled:
            self._strategy: DispatchStrategy = SceneAffineDispatch(
                renderer_pool=renderer_pool,
                max_renderers_per_scene=scene_affine_dispatch.max_renderers_per_scene,
                max_scenes_per_renderer=scene_affine_dispatch.max_scenes_per_renderer,
            )
            logger.info("Scene-affine dispatch ENABLED for renderer")
        else:
            self._strategy = FifoDispatch(renderer_pool=renderer_pool)
            logger.info("Scene-affine dispatch DISABLED")

        self._cache_refresh_interval_s = scene_affine_dispatch.cache_refresh_interval_s
        self._cache_refresh_task: asyncio.Task[None] | None = None
        self._in_flight: dict[str, _InFlightEntry] = {}
        self._request_pools: dict[str, dict[str, AddressPool]] = {}
        self._accepting_requests = True
        self._dispatch_loop_task = asyncio.create_task(self._dispatch_loop())

    async def warm_start(self) -> None:
        """Seed the strategy's cache from NRE servers and start the refresh loop.

        Only meaningful for SceneAffineDispatch; no-op for FifoDispatch.
        Raises ``IntrospectionNotSupportedError`` if the NRE image does not
        support GetLoadedScenes.
        """
        if not isinstance(self._strategy, SceneAffineDispatch):
            return
        if not self._pools["renderer"].address_order:
            return

        await self._strategy.refresh_cache(raise_on_unimplemented=True)

        if self._cache_refresh_interval_s is not None:
            self._cache_refresh_task = asyncio.create_task(self._cache_refresh_loop())

    async def _cache_refresh_loop(self) -> None:
        """Periodically re-sync the scene cache from NRE and dispatch on changes.

        Cache changes (e.g. NRE evictions) can unblock pending jobs that no
        request or job-result event would re-dispatch, so a changed sync
        triggers a dispatch round.
        """
        assert isinstance(self._strategy, SceneAffineDispatch)
        assert self._cache_refresh_interval_s is not None
        logger.info(
            "Cache refresh loop started: interval=%.1fs",
            self._cache_refresh_interval_s,
        )
        while True:
            await asyncio.sleep(self._cache_refresh_interval_s)
            if await self._strategy.refresh_cache():
                await self.dispatch_once()

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
            self._strategy.add_pending(job)

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

        if self._cache_refresh_task is not None:
            self._cache_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cache_refresh_task

        self._dispatch_loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._dispatch_loop_task

        await self._strategy.shutdown()

        async with self._dispatch_lock:
            pending_request_ids = self._strategy.drain_pending_request_ids()
            for request_id in pending_request_ids:
                self._request_pools.pop(request_id, None)
                self._request_store.fail_request(request_id, reason)

    async def dispatch_once(self) -> None:
        """Dispatch all currently eligible jobs, serialized across callers."""
        async with self._dispatch_lock:
            await self._dispatch_once_locked()

    async def _dispatch_once_locked(self) -> None:
        """Greedily dispatch pending jobs via the active strategy.

        The strategy reserves the best job *and* pre-acquires the renderer
        slot.  ``try_acquire_all`` then acquires the remaining pools,
        rolling back the renderer slot on failure.
        """
        while self._strategy.pending_count > 0:
            reservation = self._strategy.try_reserve()
            if reservation is None:
                return

            job = reservation.job
            pools = self._request_pools.get(job.request_id, self._pools)
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
                request_id=job.request_id,
                job_id=job.job_id,
                scene_id=job.scene_id,
                rollout_spec_index=job.rollout_spec_index,
                endpoints=ServiceEndpoints(
                    driver=acquired["driver"],
                    renderer=acquired["renderer"],
                    physics=acquired["physics"],
                    trafficsim=acquired["trafficsim"],
                    controller=acquired["controller"],
                ),
                dispatch_kind=reservation.dispatch_kind,
                scheduler_wait_seconds=max(0.0, monotonic() - job.enqueued_at),
                session_uuid=job.session_uuid,
            )
            self._runtime.submit_assigned_job(assigned)
            self._in_flight[assigned.job_id] = _InFlightEntry(
                pending_job=job,
                dispatch_kind=reservation.dispatch_kind,
                pools=required_pools,
                acquired=acquired,
            )

    def on_result(self, result: JobResult) -> None:
        entry = self._in_flight.pop(result.job_id, None)
        if entry is None:
            raise RuntimeError(f"Unknown job_id in result queue: {result.job_id}")

        pending_job = entry.pending_job
        renderer_addr = entry.acquired["renderer"].address
        self._strategy.on_result(
            pending_job.scene_id,
            renderer_addr,
            entry.dispatch_kind,
            result.success,
        )

        release_all(entry.pools, entry.acquired)

        if not result.success and pending_job.retry_attempt < self._max_rollout_retries:
            retry_job = replace(
                pending_job,
                job_id=uuid4().hex,
                retry_attempt=pending_job.retry_attempt + 1,
            )
            if self._rollouts_dir is not None:
                try:
                    _clear_retry_rollout_dir(retry_job, self._rollouts_dir)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed to clear artifacts before retrying rollout; "
                        "recording the failed attempt instead"
                    )
                    self._request_store.record_result(result)
                    return
            logger.warning(
                "Retrying failed rollout: scene=%s failed_job=%s retry_job=%s "
                "retry=%d/%d error=%s",
                retry_job.scene_id,
                result.job_id,
                retry_job.job_id,
                retry_job.retry_attempt,
                self._max_rollout_retries,
                result.error,
            )
            self._strategy.add_pending(retry_job)
            return

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
