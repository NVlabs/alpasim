# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.daemon.scheduler import DaemonScheduler, SceneAffineDispatch
from alpasim_runtime.worker.ipc import JobResult, PendingRolloutJob


def _affine_strategy(scheduler: DaemonScheduler) -> SceneAffineDispatch:
    """Return the scheduler's strategy with proper typing for test assertions."""
    assert isinstance(scheduler._strategy, SceneAffineDispatch)
    return scheduler._strategy


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


def _make_pools_multi_gpu(
    n_concurrent: int = 1,
) -> dict[str, AddressPool]:
    """Create pools with 2 renderer GPUs for affine tests."""
    return {
        "driver": AddressPool(["driver:50051"], n_concurrent=2, skip=False),
        "renderer": AddressPool(
            ["gpu-0:50052", "gpu-1:50052"],
            n_concurrent=n_concurrent,
            skip=False,
        ),
        "physics": AddressPool(["physics:50053"], n_concurrent=2, skip=False),
        "trafficsim": AddressPool(["trafficsim:50054"], n_concurrent=2, skip=False),
        "controller": AddressPool(["controller:50055"], n_concurrent=2, skip=False),
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
        self.submitted_jobs = []

    def submit_assigned_job(self, job) -> None:
        self.submitted_job_ids.append(job.job_id)
        self.submitted_jobs.append(job)

    async def poll_result(self) -> JobResult | None:
        await asyncio.sleep(0.01)
        return None

    def check_for_crashes(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Basic scheduling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_dispatches_jobs() -> None:
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


# ---------------------------------------------------------------------------
# Scene-affine dispatch: tier 1/2/3 job selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier1_prefers_job_for_cached_scene() -> None:
    """When a free GPU has scene-A cached, a pending scene-A job is preferred."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(pools=pools, runtime=runtime)

    # Simulate gRPC sync: gpu-0 has scene-A cached.
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-A"])

    # Submit jobs for scene-B (new) and scene-A (cached on gpu-0).
    await scheduler.submit_request(
        "req-1",
        [_pending("j2", scene_id="scene-B"), _pending("j3", scene_id="scene-A")],
    )

    # Both dispatched (2 GPUs available). j3 (scene-A) should go to gpu-0.
    assert len(runtime.submitted_jobs) == 2
    j3_gpu = next(
        j.endpoints.renderer.address for j in runtime.submitted_jobs if j.job_id == "j3"
    )
    assert j3_gpu == "gpu-0:50052"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_tier2_prefers_new_scene_over_cached_elsewhere() -> None:
    """When no free GPU has a matching cache, prefer a scene not cached anywhere.

    Setup: gpu-0 has scene-A cached, gpu-1 has scene-B cached (via sync).
    Block gpu-1 with work, leaving gpu-0 (with A) free.
    Submit jobs for scene-B (cached on busy gpu-1) and scene-C (new).
    Tier 1 fails (gpu-0 has A, not B or C).
    Tier 2 should prefer scene-C (not cached anywhere) over scene-B.
    """
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(pools=pools, runtime=runtime)

    # Seed cache via sync (simulating gRPC introspection).
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-A"])
    _affine_strategy(scheduler).sync_scene_cache("gpu-1:50052", ["scene-B"])

    # Block gpu-1 with work.
    await scheduler.submit_request(
        "req-block", [_pending("jblock", scene_id="scene-X")]
    )
    # jblock acquires gpu-0 (FIFO front — tier 1 hit on scene-A? No, scene-X not cached).
    # Actually tier 2: scene-X not cached → FIFO slot → gpu-0.
    first_gpu = runtime.submitted_jobs[0].endpoints.renderer.address

    # Block the other GPU too if first went to gpu-0.
    if first_gpu == "gpu-0:50052":
        # gpu-1 is free with scene-B cached. Block it.
        await scheduler.submit_request(
            "req-block2", [_pending("jblock2", scene_id="scene-Y")]
        )
        # Release gpu-0 so it's available.
        scheduler.on_result(_result("req-block", "jblock"))
    else:
        # gpu-0 is free with scene-A cached. Good — gpu-1 is busy.
        pass

    # Now we need gpu-0 free (with scene-A) and gpu-1 busy.
    # Simpler approach: just release and re-block deterministically.
    await scheduler.shutdown(reason="test cleanup")

    # Re-create with deterministic setup.
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(pools=pools, runtime=runtime)
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-A"])
    _affine_strategy(scheduler).sync_scene_cache("gpu-1:50052", ["scene-B"])

    # Acquire gpu-1 slot directly to block it, then submit.
    slot = pools["renderer"].try_acquire_for_address("gpu-1:50052")
    assert slot is not None  # gpu-1 is now busy

    # Submit scene-B (cached on busy gpu-1) and scene-C (new).
    await scheduler.submit_request(
        "req-2",
        [_pending("j3", scene_id="scene-B"), _pending("j4", scene_id="scene-C")],
    )

    # Only gpu-0 is free (has A cached, not B or C → tier 1 fails).
    # Tier 2: scene-B is cached (gpu-1), scene-C is not cached → pick scene-C.
    dispatched_job = runtime.submitted_jobs[0]
    assert dispatched_job.job_id == "j4"
    assert dispatched_job.scene_id == "scene-C"

    pools["renderer"].release(slot)
    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_affine_same_scene_returns_to_same_gpu() -> None:
    """A sync-seeded cache directs repeat jobs to the same GPU."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(pools=pools, runtime=runtime)

    # Simulate gRPC sync reporting scene-A on gpu-0.
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-A"])

    # Submit scene-A job — should hit gpu-0 (tier-1).
    await scheduler.submit_request("req-1", [_pending("j1", scene_id="scene-A")])
    first_gpu = runtime.submitted_jobs[0].endpoints.renderer.address
    assert first_gpu == "gpu-0:50052"

    # Complete the job.
    scheduler.on_result(_result("req-1", "j1"))

    # Submit another scene-A job — should still hit gpu-0 (cache still seeded).
    await scheduler.submit_request("req-2", [_pending("j2", scene_id="scene-A")])
    second_gpu = runtime.submitted_jobs[1].endpoints.renderer.address
    assert second_gpu == first_gpu

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_different_scenes_get_different_gpus() -> None:
    """Different scenes are routed to different GPUs when possible."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(pools=pools, runtime=runtime)

    # Simulate gRPC sync: gpu-0 has scene-A, gpu-1 has scene-B.
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-A"])
    _affine_strategy(scheduler).sync_scene_cache("gpu-1:50052", ["scene-B"])

    # Dispatch scene-A — should hit gpu-0 (tier-1).
    await scheduler.submit_request("req-1", [_pending("j1", scene_id="scene-A")])
    gpu_a = runtime.submitted_jobs[0].endpoints.renderer.address
    assert gpu_a == "gpu-0:50052"
    scheduler.on_result(_result("req-1", "j1"))

    # Dispatch scene-B — should hit gpu-1 (tier-1).
    await scheduler.submit_request("req-2", [_pending("j2", scene_id="scene-B")])
    gpu_b = runtime.submitted_jobs[1].endpoints.renderer.address
    assert gpu_b == "gpu-1:50052"
    scheduler.on_result(_result("req-2", "j2"))

    # Dispatch scene-A again — should still get gpu-0.
    await scheduler.submit_request("req-3", [_pending("j3", scene_id="scene-A")])
    gpu_a2 = runtime.submitted_jobs[2].endpoints.renderer.address
    assert gpu_a2 == gpu_a

    # Dispatch scene-B again — should still get gpu-1.
    scheduler.on_result(_result("req-3", "j3"))
    await scheduler.submit_request("req-4", [_pending("j4", scene_id="scene-B")])
    gpu_b2 = runtime.submitted_jobs[3].endpoints.renderer.address
    assert gpu_b2 == gpu_b

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_sync_seeded_cache_preserves_diversity() -> None:
    """Sync-seeded caches route each scene to its designated GPU.

    gpu-0 has scene-A, gpu-1 has scene-B.  Sequential dispatches with
    one GPU free at a time should always route to the correct GPU.
    """
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(pools=pools, runtime=runtime)

    # Seed via sync.
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["A"])
    _affine_strategy(scheduler).sync_scene_cache("gpu-1:50052", ["B"])

    # Block gpu-1 so only gpu-0 is free.
    gpu1_slot = pools["renderer"].try_acquire_for_address("gpu-1:50052")
    assert gpu1_slot is not None

    # Submit scene-A — should route to gpu-0 (tier-1 hit).
    await scheduler.submit_request("req-1", [_pending("a1", scene_id="A")])
    assert runtime.submitted_jobs[0].endpoints.renderer.address == "gpu-0:50052"
    scheduler.on_result(_result("req-1", "a1"))

    # Free gpu-1, block gpu-0.
    pools["renderer"].release(gpu1_slot)
    gpu0_slot = pools["renderer"].try_acquire_for_address("gpu-0:50052")
    assert gpu0_slot is not None

    # Submit scene-B — should route to gpu-1 (tier-1 hit).
    await scheduler.submit_request("req-2", [_pending("b1", scene_id="B")])
    assert runtime.submitted_jobs[1].endpoints.renderer.address == "gpu-1:50052"
    scheduler.on_result(_result("req-2", "b1"))

    # Free gpu-0, submit scene-A again — still hits gpu-0.
    pools["renderer"].release(gpu0_slot)
    gpu1_slot = pools["renderer"].try_acquire_for_address("gpu-1:50052")
    assert gpu1_slot is not None
    await scheduler.submit_request("req-3", [_pending("a2", scene_id="A")])
    assert runtime.submitted_jobs[2].endpoints.renderer.address == "gpu-0:50052"

    pools["renderer"].release(gpu1_slot)
    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_affine_disabled_uses_plain_fifo() -> None:
    """With scene_affine_dispatch=False, dispatch is pure FIFO."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(
        pools=pools, runtime=runtime, scene_affine_dispatch=False
    )

    # Seed: dispatch scene-A on gpu-0, complete it.
    await scheduler.submit_request("req-1", [_pending("j1", scene_id="scene-A")])
    scheduler.on_result(_result("req-1", "j1"))

    # Submit scene-B and scene-A. With affine off, the scheduler just picks
    # any job — no preference for scene-A on the cached GPU.
    await scheduler.submit_request(
        "req-2",
        [_pending("j2", scene_id="scene-B"), _pending("j3", scene_id="scene-A")],
    )

    # Both should dispatch (2 GPUs), but order is not scene-aware.
    assert len(runtime.submitted_jobs) == 3

    # FifoDispatch has no affine tracking at all.
    assert not hasattr(scheduler._strategy, "_affine_hits")

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_warm_started_pool_produces_tier1_hit() -> None:
    """A warm-started pool should produce tier-1 hits on the very first dispatch."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    scheduler = DaemonScheduler(pools=pools, runtime=runtime)

    # Pre-seed the strategy's cache as SceneAffineDispatch.warm_start() would.
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-X"])
    _affine_strategy(scheduler).sync_scene_cache("gpu-1:50052", ["scene-Y"])

    # Submit scene-X — should hit gpu-0 (tier-1: cached + free).
    await scheduler.submit_request("req-1", [_pending("j1", scene_id="scene-X")])
    assert runtime.submitted_jobs[0].endpoints.renderer.address == "gpu-0:50052"
    assert _affine_strategy(scheduler)._affine_hits == 1

    # Complete and submit scene-Y — should hit gpu-1 (tier-1).
    scheduler.on_result(_result("req-1", "j1"))
    await scheduler.submit_request("req-2", [_pending("j2", scene_id="scene-Y")])
    assert runtime.submitted_jobs[1].endpoints.renderer.address == "gpu-1:50052"
    assert _affine_strategy(scheduler)._affine_hits == 2

    await scheduler.shutdown(reason="test cleanup")


# ---------------------------------------------------------------------------
# warm_start integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_start_seeds_cache_and_starts_refresh() -> None:
    """warm_start() should query NRE, seed the cache, and start the refresh loop."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    responses = {
        "gpu-0:50052": {"scene-A": 2},
        "gpu-1:50052": {"scene-B": 1},
    }

    async def _fake_get(address: str, **kwargs):
        return responses.get(address, {})

    with patch("alpasim_runtime.daemon.scheduler.get_loaded_scenes", _fake_get):
        scheduler = DaemonScheduler(
            pools=pools,
            runtime=runtime,
            cache_refresh_interval_s=60.0,
        )
        # Before warm_start, no cache and no refresh task.
        assert not _affine_strategy(scheduler).is_scene_cached("scene-A")
        assert _affine_strategy(scheduler)._cache_refresh_task is None

        await scheduler.warm_start()

        # After warm_start, cache is seeded and refresh task is running.
        strategy = _affine_strategy(scheduler)
        assert strategy.is_scene_cached("scene-A")
        assert strategy.is_scene_cached("scene-B")
        assert strategy._cache_refresh_task is not None
        assert not strategy._cache_refresh_task.done()

        await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_warm_start_raises_on_unimplemented() -> None:
    """warm_start() should raise IntrospectionNotSupportedError on UNIMPLEMENTED."""
    from alpasim_runtime.nre_introspection import IntrospectionNotSupportedError

    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    async def _fake_get(address: str, **kwargs):
        if kwargs.get("raise_on_unimplemented"):
            raise IntrospectionNotSupportedError(
                f"NRE at {address} does not support GetLoadedScenes"
            )
        return None

    with patch("alpasim_runtime.daemon.scheduler.get_loaded_scenes", _fake_get):
        scheduler = DaemonScheduler(
            pools=pools,
            runtime=runtime,
            cache_refresh_interval_s=5.0,
        )
        with pytest.raises(IntrospectionNotSupportedError):
            await scheduler.warm_start()

        # Refresh task should NOT have been started on failure.
        assert _affine_strategy(scheduler)._cache_refresh_task is None

        await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_warm_start_partial_failure_continues() -> None:
    """warm_start() should continue if some addresses return None (transient failure)."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    async def _fake_get(address: str, **kwargs):
        if address == "gpu-0:50052":
            return {"scene-A": 1}
        return None  # gpu-1 is unreachable

    with patch("alpasim_runtime.daemon.scheduler.get_loaded_scenes", _fake_get):
        scheduler = DaemonScheduler(
            pools=pools,
            runtime=runtime,
            cache_refresh_interval_s=60.0,
        )
        await scheduler.warm_start()

        # Only gpu-0's scenes should be cached.
        assert _affine_strategy(scheduler).is_scene_cached("scene-A")
        assert not _affine_strategy(scheduler).is_scene_cached("scene-B")
        # Refresh task should still start.
        assert _affine_strategy(scheduler)._cache_refresh_task is not None

        await scheduler.shutdown(reason="test cleanup")


# ---------------------------------------------------------------------------
# Periodic cache refresh tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_refresh_updates_routing() -> None:
    """The periodic refresh should update the cache so the next dispatch benefits."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    call_count = 0
    responses = {
        "gpu-0:50052": {"scene-Z": 2},
        "gpu-1:50052": {"scene-W": 1},
    }

    async def _fake_get(address: str, **kwargs):
        nonlocal call_count
        call_count += 1
        return responses.get(address, {})

    with patch("alpasim_runtime.daemon.scheduler.get_loaded_scenes", _fake_get):
        scheduler = DaemonScheduler(
            pools=pools,
            runtime=runtime,
            cache_refresh_interval_s=0.05,
        )
        # warm_start seeds the cache and starts the refresh loop.
        await scheduler.warm_start()
        # Let the refresh loop fire at least once after warm_start.
        await asyncio.sleep(0.15)

        # The strategy should now know about scene-Z on gpu-0 and scene-W on gpu-1.
        assert _affine_strategy(scheduler).is_scene_cached("scene-Z")
        assert _affine_strategy(scheduler).is_scene_cached("scene-W")
        assert call_count >= 2  # at least one full cycle (2 addresses)

        # Dispatch scene-Z — should hit gpu-0 (tier-1).
        await scheduler.submit_request("req-1", [_pending("j1", scene_id="scene-Z")])
        assert runtime.submitted_jobs[0].endpoints.renderer.address == "gpu-0:50052"
        assert _affine_strategy(scheduler)._affine_hits == 1

        await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_cache_refresh_disabled_when_none() -> None:
    """cache_refresh_interval_s=None should not create a refresh task."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    scheduler = DaemonScheduler(
        pools=pools,
        runtime=runtime,
        cache_refresh_interval_s=None,
    )
    assert _affine_strategy(scheduler)._cache_refresh_task is None
    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_cache_refresh_disabled_when_affine_off() -> None:
    """With scene_affine_dispatch=False, no refresh task should be created."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    scheduler = DaemonScheduler(
        pools=pools,
        runtime=runtime,
        scene_affine_dispatch=False,
        cache_refresh_interval_s=30.0,
    )
    # FifoDispatch has no cache refresh mechanism.
    assert not hasattr(scheduler._strategy, "_cache_refresh_task")
    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_shutdown_cancels_refresh_task() -> None:
    """shutdown() should cleanly cancel the refresh task."""
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()

    mock_get = AsyncMock(return_value={"scene-A": 1})

    with patch("alpasim_runtime.daemon.scheduler.get_loaded_scenes", mock_get):
        scheduler = DaemonScheduler(
            pools=pools,
            runtime=runtime,
            cache_refresh_interval_s=60.0,
        )
        strategy = _affine_strategy(scheduler)
        # Refresh task is not started until warm_start.
        assert strategy._cache_refresh_task is None

        await scheduler.warm_start()
        assert strategy._cache_refresh_task is not None
        assert not strategy._cache_refresh_task.done()

        await scheduler.shutdown(reason="test cleanup")

        assert strategy._cache_refresh_task.done()
