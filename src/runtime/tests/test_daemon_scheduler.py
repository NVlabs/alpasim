# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.config import SceneAffineDispatchConfig
from alpasim_runtime.daemon.scheduler import (
    DaemonScheduler,
    SceneAffineDispatch,
    _clear_retry_rollout_dir,
)
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
    num_renderers: int = 2,
) -> dict[str, AddressPool]:
    """Create pools with multiple renderer GPUs for affine tests."""
    total_capacity = num_renderers * n_concurrent
    return {
        "driver": AddressPool(
            ["driver:50051"], n_concurrent=total_capacity, skip=False
        ),
        "renderer": AddressPool(
            [f"gpu-{i}:50052" for i in range(num_renderers)],
            n_concurrent=n_concurrent,
            skip=False,
        ),
        "physics": AddressPool(
            ["physics:50053"], n_concurrent=total_capacity, skip=False
        ),
        "trafficsim": AddressPool(
            ["trafficsim:50054"], n_concurrent=total_capacity, skip=False
        ),
        "controller": AddressPool(
            ["controller:50055"], n_concurrent=total_capacity, skip=False
        ),
    }


def _pending(
    job_id: str,
    scene_id: str = "scene-a",
    rollout_spec_index: int = 0,
    request_id: str = "req-a",
) -> PendingRolloutJob:
    return PendingRolloutJob(
        request_id=request_id,
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
) -> JobResult:
    return JobResult(
        request_id=request_id,
        job_id=job_id,
        rollout_spec_index=0,
        success=success,
        error=error,
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
async def test_scheduler_per_request_pool_releases_correctly() -> None:
    """Per-request driver pool is used for dispatch and slots release back to it."""
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(enabled=True),
    )

    # Custom pool with capacity=1, so second job can only dispatch after first completes
    custom_driver = AddressPool(["custom-driver:9999"], n_concurrent=1, skip=False)
    await scheduler.submit_request(
        "req-custom",
        [
            _pending("j1", request_id="req-custom"),
            _pending("j2", request_id="req-custom"),
        ],
        driver_pool=custom_driver,
    )

    # Only j1 dispatched (capacity=1), using the custom pool's address.
    assert runtime.submitted_job_ids == ["j1"]
    assert runtime.submitted_jobs[0].endpoints.driver.address == "custom-driver:9999"

    # Complete j1 — should release slot and dispatch j2
    scheduler.on_result(_result("req-custom", "j1"))
    await scheduler.dispatch_once()

    assert runtime.submitted_job_ids == ["j1", "j2"]

    await scheduler.shutdown(reason="test cleanup")


# ---------------------------------------------------------------------------
# Scene-affine dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cold_dispatch_prefers_renderer_with_fewest_active_scenes() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=4),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(enabled=True),
    )
    strategy = _affine_strategy(scheduler)
    strategy.sync_scene_cache("gpu-0:50052", ["scene-A"])
    strategy.sync_scene_cache("gpu-1:50052", ["scene-B", "scene-C"])

    # gpu-0 ends up with 3 rollouts of one scene, gpu-1 with 2 rollouts of
    # two scenes, so gpu-0 has fewer active scenes but fewer free slots.
    await scheduler.submit_request(
        "req-1",
        [
            _pending("a1", scene_id="scene-A", request_id="req-1"),
            _pending("a2", scene_id="scene-A", request_id="req-1"),
            _pending("a3", scene_id="scene-A", request_id="req-1"),
            _pending("b1", scene_id="scene-B", request_id="req-1"),
            _pending("c1", scene_id="scene-C", request_id="req-1"),
        ],
    )
    a_jobs = [job for job in runtime.submitted_jobs if job.scene_id == "scene-A"]
    assert {job.endpoints.renderer.address for job in a_jobs} == {"gpu-0:50052"}

    await scheduler.submit_request(
        "req-2", [_pending("d1", scene_id="scene-D", request_id="req-2")]
    )

    d1 = runtime.submitted_jobs[-1]
    assert d1.job_id == "d1"
    assert d1.dispatch_kind == "cold_initial"
    assert d1.endpoints.renderer.address == "gpu-0:50052"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_retries_any_failure_up_to_configured_limit(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(enabled=False),
        max_rollout_retries=2,
        rollouts_dir=str(tmp_path),
    )
    original = _pending("j1")
    original.session_uuid = "requested-session"
    await scheduler.submit_request("req-retry", [original])

    current_job_id = "j1"
    for retry_attempt in range(1, 3):
        rollout_dir = tmp_path / original.scene_id / original.session_uuid
        rollout_dir.mkdir(parents=True)
        (rollout_dir / "rollout.asl").write_text("failed attempt")
        (rollout_dir / "_complete").touch()

        scheduler.on_result(
            _result(
                "req-retry",
                current_job_id,
                success=False,
                error=f"failure {retry_attempt}",
            )
        )
        assert not rollout_dir.exists()
        await scheduler.dispatch_once()

        retry_job = runtime.submitted_jobs[-1]
        assert retry_job.job_id != current_job_id
        assert retry_job.scene_id == original.scene_id
        assert retry_job.rollout_spec_index == original.rollout_spec_index
        assert retry_job.session_uuid == "requested-session"
        assert (
            scheduler._in_flight[retry_job.job_id].pending_job.retry_attempt
            == retry_attempt
        )
        current_job_id = retry_job.job_id

    scheduler.on_result(
        _result(
            "req-retry",
            current_job_id,
            success=False,
            error="final failure",
        )
    )
    completion = await scheduler.wait_request("req-retry")

    assert len(runtime.submitted_jobs) == 3
    assert len(completion) == 1
    assert completion[0].job_id == current_job_id
    assert completion[0].error == "final failure"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_scheduler_returns_successful_retry_instead_of_initial_failure() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools(capacity_per_service=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(enabled=False),
        max_rollout_retries=2,
    )
    await scheduler.submit_request("req-retry", [_pending("j1")])

    scheduler.on_result(
        _result("req-retry", "j1", success=False, error="transient failure")
    )
    await scheduler.dispatch_once()
    retry_job = runtime.submitted_jobs[-1]
    assert retry_job.session_uuid == ""
    retry_job_id = retry_job.job_id
    scheduler.on_result(_result("req-retry", retry_job_id))

    completion = await scheduler.wait_request("req-retry")
    assert len(runtime.submitted_jobs) == 2
    assert len(completion) == 1
    assert completion[0].success is True
    assert completion[0].job_id == retry_job_id

    await scheduler.shutdown(reason="test cleanup")


def test_scheduler_rejects_negative_retry_limit() -> None:
    with pytest.raises(ValueError, match="max_rollout_retries must be non-negative"):
        DaemonScheduler(
            pools=_make_pools(capacity_per_service=1),
            runtime=_FakeRuntime(),
            scene_affine_dispatch=SceneAffineDispatchConfig(enabled=False),
            max_rollout_retries=-1,
        )


def test_clear_retry_rollout_dir_rejects_path_traversal(tmp_path: Path) -> None:
    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside-session"
    outside_dir.mkdir(exist_ok=True)
    artifact = outside_dir / "rollout.asl"
    artifact.write_text("keep")
    pending_job = _pending("j1")
    pending_job.session_uuid = f"../../{outside_dir.name}"

    with pytest.raises(ValueError, match="one path component"):
        _clear_retry_rollout_dir(pending_job, str(tmp_path))

    assert artifact.read_text() == "keep"


@pytest.mark.asyncio
async def test_max_scenes_per_renderer_defers_cold_dispatch() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=2),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_scenes_per_renderer=1
        ),
    )

    await scheduler.submit_request(
        "req-1",
        [
            _pending("a1", scene_id="scene-A", request_id="req-1"),
            _pending("b1", scene_id="scene-B", request_id="req-1"),
            _pending("c1", scene_id="scene-C", request_id="req-1"),
        ],
    )

    assert [job.job_id for job in runtime.submitted_jobs] == ["a1", "b1"]
    assert scheduler._strategy.pending_count == 1

    # Completing scene-A leaves it cached but idle on gpu-0, so gpu-0 is
    # eligible again: only loading or executing scenes count toward the cap.
    scheduler.on_result(_result("req-1", "a1"))
    await scheduler.dispatch_once()

    c1 = runtime.submitted_jobs[-1]
    assert c1.job_id == "c1"
    assert c1.dispatch_kind == "cold_initial"
    assert c1.endpoints.renderer.address == "gpu-0:50052"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_max_scenes_per_renderer_does_not_block_cached_affinity() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=3, num_renderers=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_scenes_per_renderer=1
        ),
    )
    strategy = _affine_strategy(scheduler)
    strategy.sync_scene_cache("gpu-0:50052", ["scene-A"])

    await scheduler.submit_request(
        "req-1", [_pending("b1", scene_id="scene-B", request_id="req-1")]
    )
    await scheduler.submit_request(
        "req-2", [_pending("a1", scene_id="scene-A", request_id="req-2")]
    )
    await scheduler.submit_request(
        "req-3", [_pending("c1", scene_id="scene-C", request_id="req-3")]
    )

    assert [job.job_id for job in runtime.submitted_jobs] == ["b1", "a1"]
    assert runtime.submitted_jobs[1].dispatch_kind == "cached_affine"
    assert scheduler._strategy.pending_count == 1

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_cached_affinity_chooses_least_loaded_matching_renderer() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=2),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(enabled=True),
    )
    strategy = _affine_strategy(scheduler)
    strategy.sync_scene_cache("gpu-0:50052", ["scene-A", "scene-B"])
    strategy.sync_scene_cache("gpu-1:50052", ["scene-A"])

    await scheduler.submit_request(
        "req-1", [_pending("b1", scene_id="scene-B", request_id="req-1")]
    )
    await scheduler.submit_request(
        "req-2", [_pending("a1", scene_id="scene-A", request_id="req-2")]
    )

    assert runtime.submitted_jobs[0].endpoints.renderer.address == "gpu-0:50052"
    assert runtime.submitted_jobs[1].endpoints.renderer.address == "gpu-1:50052"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_cold_scene_uses_at_most_one_rollout_per_renderer() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=2),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=2
        ),
    )

    await scheduler.submit_request(
        "req-1",
        [_pending(f"a{i}", scene_id="scene-A", request_id="req-1") for i in range(3)],
    )

    assert [job.dispatch_kind for job in runtime.submitted_jobs] == [
        "cold_initial",
        "cold_replica",
    ]
    assert [job.endpoints.renderer.address for job in runtime.submitted_jobs] == [
        "gpu-0:50052",
        "gpu-1:50052",
    ]
    assert scheduler._strategy.pending_count == 1

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_earlier_scene_replicates_to_bound_before_later_scene_starts() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=1, num_renderers=3),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=2
        ),
    )

    await scheduler.submit_request(
        "req-1",
        [
            _pending("a1", scene_id="scene-A", request_id="req-1"),
            _pending("a2", scene_id="scene-A", request_id="req-1"),
            _pending("b1", scene_id="scene-B", request_id="req-1"),
        ],
    )

    # Scene-A arrived first, so it replicates to max_renderers_per_scene
    # before scene-B gets its first location.
    assert [(job.job_id, job.dispatch_kind) for job in runtime.submitted_jobs] == [
        ("a1", "cold_initial"),
        ("a2", "cold_replica"),
        ("b1", "cold_initial"),
    ]

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_cached_scene_full_starts_only_one_cold_replica() -> None:
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu(n_concurrent=2)
    scheduler = DaemonScheduler(
        pools=pools,
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=2
        ),
    )
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-A"])
    gpu0_slots = [
        pools["renderer"].try_acquire_for_address("gpu-0:50052") for _ in range(2)
    ]
    assert all(slot is not None for slot in gpu0_slots)

    await scheduler.submit_request(
        "req-1",
        [_pending(f"a{i}", scene_id="scene-A", request_id="req-1") for i in range(3)],
    )

    assert [(job.job_id, job.dispatch_kind) for job in runtime.submitted_jobs] == [
        ("a0", "cold_replica")
    ]
    assert runtime.submitted_jobs[0].endpoints.renderer.address == "gpu-1:50052"
    assert scheduler._strategy.pending_count == 2

    for slot in gpu0_slots:
        assert slot is not None
        pools["renderer"].release(slot)
    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_failed_cold_rollout_allows_retry() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=1
        ),
    )
    await scheduler.submit_request(
        "req-1",
        [
            _pending("a1", scene_id="scene-A", request_id="req-1"),
            _pending("a2", scene_id="scene-A", request_id="req-1"),
        ],
    )

    failed = _result("req-1", "a1")
    failed.success = False
    scheduler.on_result(failed)
    await scheduler.dispatch_once()

    assert [job.job_id for job in runtime.submitted_jobs] == ["a1", "a2"]
    assert runtime.submitted_jobs[-1].dispatch_kind == "cold_initial"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_successful_cold_rollout_promotes_affinity_without_refresh() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=1, cache_refresh_interval_s=None
        ),
    )
    await scheduler.submit_request(
        "req-1",
        [
            _pending("a1", scene_id="scene-A", request_id="req-1"),
            _pending("a2", scene_id="scene-A", request_id="req-1"),
        ],
    )

    first_address = runtime.submitted_jobs[0].endpoints.renderer.address
    scheduler.on_result(_result("req-1", "a1"))
    await scheduler.dispatch_once()

    assert runtime.submitted_jobs[-1].dispatch_kind == "cached_affine"
    assert runtime.submitted_jobs[-1].endpoints.renderer.address == first_address

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_empty_cache_snapshot_does_not_clear_loading_location() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=2),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=1
        ),
    )
    await scheduler.submit_request(
        "req-1",
        [
            _pending("a1", scene_id="scene-A", request_id="req-1"),
            _pending("a2", scene_id="scene-A", request_id="req-1"),
        ],
    )
    first_address = runtime.submitted_jobs[0].endpoints.renderer.address

    _affine_strategy(scheduler).sync_scene_cache(first_address, [])
    await scheduler.dispatch_once()
    assert [job.job_id for job in runtime.submitted_jobs] == ["a1"]

    _affine_strategy(scheduler).sync_scene_cache(first_address, ["scene-A"])
    await scheduler.dispatch_once()
    assert [job.job_id for job in runtime.submitted_jobs] == ["a1", "a2"]

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_cache_snapshot_removes_optimistic_loaded_location() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=1
        ),
    )
    await scheduler.submit_request(
        "req-1", [_pending("a1", scene_id="scene-A", request_id="req-1")]
    )
    first_address = runtime.submitted_jobs[0].endpoints.renderer.address
    scheduler.on_result(_result("req-1", "a1"))

    _affine_strategy(scheduler).sync_scene_cache(first_address, [])
    await scheduler.submit_request(
        "req-2", [_pending("a2", scene_id="scene-A", request_id="req-2")]
    )

    assert runtime.submitted_jobs[-1].dispatch_kind == "cold_initial"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_failed_service_reservation_does_not_mark_scene_loading() -> None:
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu(n_concurrent=1)
    blocked_drivers = [pools["driver"].try_acquire() for _ in range(2)]
    assert all(slot is not None for slot in blocked_drivers)
    scheduler = DaemonScheduler(
        pools=pools,
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=1
        ),
    )

    await scheduler.submit_request(
        "req-1", [_pending("a1", scene_id="scene-A", request_id="req-1")]
    )
    assert runtime.submitted_jobs == []

    for slot in blocked_drivers:
        assert slot is not None
        pools["driver"].release(slot)
    await scheduler.dispatch_once()
    assert runtime.submitted_jobs[0].dispatch_kind == "cold_initial"

    await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_earlier_scene_cached_on_busy_renderer_replicates_first() -> None:
    """Cold dispatch follows scene arrival order, not cache coverage.

    Setup: gpu-0 has scene-A cached, gpu-1 has scene-B cached (via sync).
    Block gpu-1, leaving gpu-0 (with A) free.
    Submit jobs for scene-B (cached on busy gpu-1) and scene-C (new).
    Tier 1 fails (gpu-0 has A, not B or C).
    Scene-B arrived first, so it replicates onto gpu-0 before scene-C
    gets its first location.
    """
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu()
    scheduler = DaemonScheduler(
        pools=pools,
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(enabled=True),
    )
    _affine_strategy(scheduler).sync_scene_cache("gpu-0:50052", ["scene-A"])
    _affine_strategy(scheduler).sync_scene_cache("gpu-1:50052", ["scene-B"])

    # Acquire gpu-1 slot directly to block it, then submit.
    slot = pools["renderer"].try_acquire_for_address("gpu-1:50052")
    assert slot is not None  # gpu-1 is now busy

    # Submit scene-B (cached on busy gpu-1) and scene-C (new).
    await scheduler.submit_request(
        "req-2",
        [
            _pending("j3", scene_id="scene-B", request_id="req-2"),
            _pending("j4", scene_id="scene-C", request_id="req-2"),
        ],
    )

    dispatched_job = runtime.submitted_jobs[0]
    assert dispatched_job.job_id == "j3"
    assert dispatched_job.dispatch_kind == "cold_replica"
    assert dispatched_job.endpoints.renderer.address == "gpu-0:50052"
    assert scheduler._strategy.pending_count == 1

    pools["renderer"].release(slot)
    await scheduler.shutdown(reason="test cleanup")


# ---------------------------------------------------------------------------
# warm_start integration tests
# ---------------------------------------------------------------------------


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
            scene_affine_dispatch=SceneAffineDispatchConfig(
                enabled=True, cache_refresh_interval_s=5.0
            ),
        )
        with pytest.raises(IntrospectionNotSupportedError):
            await scheduler.warm_start()

        # Refresh task should NOT have been started on failure.
        assert scheduler._cache_refresh_task is None

        await scheduler.shutdown(reason="test cleanup")


# ---------------------------------------------------------------------------
# Periodic cache refresh tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_refresh_wakes_pending_affine_dispatch() -> None:
    runtime = _FakeRuntime()
    pools = _make_pools_multi_gpu(n_concurrent=2)
    refresh_started = False

    async def _fake_get(address: str, **kwargs):
        nonlocal refresh_started
        del kwargs
        if not refresh_started:
            return {}
        if address == "gpu-0:50052":
            return {"scene-A": 1}
        return {}

    with patch("alpasim_runtime.daemon.scheduler.get_loaded_scenes", _fake_get):
        scheduler = DaemonScheduler(
            pools=pools,
            runtime=runtime,
            scene_affine_dispatch=SceneAffineDispatchConfig(
                enabled=True, cache_refresh_interval_s=0.01, max_renderers_per_scene=1
            ),
        )
        await scheduler.warm_start()
        await scheduler.submit_request(
            "req-1",
            [
                _pending("a1", scene_id="scene-A", request_id="req-1"),
                _pending("a2", scene_id="scene-A", request_id="req-1"),
            ],
        )
        assert [job.job_id for job in runtime.submitted_jobs] == ["a1"]

        refresh_started = True
        for _ in range(20):
            if len(runtime.submitted_jobs) == 2:
                break
            await asyncio.sleep(0.01)

        assert [job.job_id for job in runtime.submitted_jobs] == ["a1", "a2"]
        assert runtime.submitted_jobs[-1].dispatch_kind == "cached_affine"
        assert runtime.submitted_jobs[-1].endpoints.renderer.address == "gpu-0:50052"

        await scheduler.shutdown(reason="test cleanup")


@pytest.mark.asyncio
async def test_warm_cache_above_replica_limit_remains_usable() -> None:
    runtime = _FakeRuntime()
    scheduler = DaemonScheduler(
        pools=_make_pools_multi_gpu(n_concurrent=1),
        runtime=runtime,
        scene_affine_dispatch=SceneAffineDispatchConfig(
            enabled=True, max_renderers_per_scene=1
        ),
    )
    strategy = _affine_strategy(scheduler)
    strategy.sync_scene_cache("gpu-0:50052", ["scene-A"])
    strategy.sync_scene_cache("gpu-1:50052", ["scene-A"])

    await scheduler.submit_request(
        "req-1",
        [
            _pending("a1", scene_id="scene-A", request_id="req-1"),
            _pending("a2", scene_id="scene-A", request_id="req-1"),
        ],
    )

    assert {job.endpoints.renderer.address for job in runtime.submitted_jobs} == {
        "gpu-0:50052",
        "gpu-1:50052",
    }
    assert all(job.dispatch_kind == "cached_affine" for job in runtime.submitted_jobs)

    await scheduler.shutdown(reason="test cleanup")
