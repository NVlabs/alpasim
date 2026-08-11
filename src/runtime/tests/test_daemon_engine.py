# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import gc
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from alpasim_grpc.v0 import runtime_pb2
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.config import RendererConfig, RendererKind
from alpasim_runtime.daemon.engine import (
    DaemonEngine,
    InvalidRequestError,
    build_simulation_return,
)
from alpasim_runtime.daemon.scheduler import DaemonUnavailableError
from alpasim_runtime.scene_loader import SceneLoader, _build_artifact_scene_provider
from alpasim_runtime.worker.ipc import JobResult

from eval.data import AggregationType, MetricReturn
from eval.scenario_evaluator import ScenarioEvalResult


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        user=SimpleNamespace(
            nr_workers=1,
            renderer=RendererConfig(kind=RendererKind.sensorsim),
            smooth_trajectories=True,
            scenes=[SimpleNamespace(scene_id="clipgt-a")],
            endpoints=SimpleNamespace(startup_timeout_s=1),
            scene_affine_dispatch=SimpleNamespace(
                enabled=False,
                cache_refresh_interval_s=5.0,
                max_renderers_per_scene=2,
                max_scenes_per_renderer=None,
            ),
            max_rollout_retries=2,
        ),
        network=SimpleNamespace(),
    )


def _write_usdz_metadata(path: Path) -> None:
    metadata = {
        "scene_id": "clipgt-a",
        "version_string": "nre-1",
        "training_date": "2024-11-11",
        "dataset_hash": "hash-a",
        "uuid": "uuid-a",
        "is_resumable": False,
        "sensors": {
            "camera_ids": ["camera_front", "camera_left"],
            "lidar_ids": ["lidar_top"],
        },
        "logger": {
            "name": "wandb",
            "run_id": "run-a",
            "run_url": "https://example.invalid/run-a",
        },
        "time_range": {
            "start": 1648604351509172,
            "end": 1648604371509172,
        },
        "training_step_outputs": {
            "psnr": 28.9,
        },
    }
    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("metadata.yaml", yaml.safe_dump(metadata))


@pytest.mark.asyncio
async def test_engine_get_runtime_info_reports_capacity_scenes_and_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _make_failing_artifact_loader(*args, **kwargs):
        del args, kwargs

        def _load_artifact(scene_id: str, artifact_path: str):
            pytest.fail(
                "get_runtime_info should not load artifacts through the runtime cache"
            )

        return _load_artifact

    monkeypatch.setattr(
        "alpasim_runtime.scene_loader.make_artifact_loader",
        _make_failing_artifact_loader,
    )
    artifact_path = tmp_path / "scene.usdz"
    _write_usdz_metadata(artifact_path)
    scene_loader = SceneLoader(
        _build_artifact_scene_provider(
            user_config=SimpleNamespace(smooth_trajectories=True),
            usdz_provider_config=SimpleNamespace(
                data_dir=str(artifact_path),
                artifact_cache_size=None,
            ),
        )
    )
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir=str(log_dir),
    )
    engine._started = True
    engine._version_ids = _version_ids()
    engine._runtime_context = SimpleNamespace(
        max_in_flight=2,
        config=SimpleNamespace(
            user=SimpleNamespace(
                nr_workers=3,
                renderer=RendererConfig(kind=RendererKind.sensorsim),
            ),
        ),
        pools={
            "driver": AddressPool(["driver-a:50051"], n_concurrent=2, skip=False),
            "renderer": AddressPool([], n_concurrent=0, skip=True),
        },
        scene_loader=scene_loader,
    )

    info = await engine.get_runtime_info()

    assert info.max_supported_concurrent_rollouts == 2
    assert info.nr_workers == 3
    assert info.renderer_type == "sensorsim"
    assert info.runtime_version.version_id == "runtime"
    assert info.video_model_version.version_id == "video-model"
    assert [scene.scene_id for scene in info.scenes] == ["clipgt-a"]
    assert info.scenes[0].provider_kind == "usdz"
    assert info.scenes[0].metadata.camera_ids == ["camera_front", "camera_left"]
    assert info.scenes[0].metadata.start_time_us == 1648604351509172
    capacities = {
        capacity.service_name: capacity for capacity in info.service_capacities
    }
    assert capacities["driver"].total_capacity == 2
    assert capacities["driver"].skipped is False
    assert capacities["renderer"].total_capacity == 0
    assert capacities["renderer"].skipped is True


@pytest.mark.asyncio
async def test_engine_startup_gathers_versions_and_validates_scenes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config()
    version_ids = _version_ids()
    eval_config = MagicMock()
    worker_runtime = SimpleNamespace(stop=AsyncMock())

    class _FakeScheduler:
        def __init__(self, *, runtime, **kwargs) -> None:
            assert runtime is worker_runtime
            assert kwargs["max_rollout_retries"] == 2
            assert kwargs["rollouts_dir"] == "/tmp/log/rollouts"

        async def shutdown(self, *, reason: str) -> None:
            del reason

    def _fake_start_worker_runtime(*args, **kwargs):
        del args, kwargs
        return worker_runtime

    async def _fake_build_runtime_context(*args, **kwargs):
        del args
        assert kwargs["validate_config_scenes"] is True
        return SimpleNamespace(
            config=config,
            eval_config=eval_config,
            version_ids=version_ids,
            scene_loader=MagicMock(),
            pools={"driver": AddressPool(["driver-a:50051"], 1, skip=False)},
            max_in_flight=1,
        )

    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.build_runtime_context",
        _fake_build_runtime_context,
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.compute_num_consumers_per_worker",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.start_worker_runtime", _fake_start_worker_runtime
    )
    monkeypatch.setattr("alpasim_runtime.daemon.engine.DaemonScheduler", _FakeScheduler)
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )

    await engine.startup()
    assert engine.version_ids is version_ids
    await engine.shutdown()


@pytest.mark.asyncio
async def test_engine_startup_skips_config_scene_validation_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config()
    version_ids = _version_ids()
    eval_config = MagicMock()
    worker_runtime = SimpleNamespace(stop=AsyncMock())

    class _FakeScheduler:
        def __init__(self, *, runtime, **kwargs) -> None:
            assert runtime is worker_runtime

        async def shutdown(self, *, reason: str) -> None:
            del reason

    def _fake_start_worker_runtime(*args, **kwargs):
        del args, kwargs
        return worker_runtime

    async def _fake_build_runtime_context(*args, **kwargs):
        del args
        assert kwargs["validate_config_scenes"] is False
        return SimpleNamespace(
            config=config,
            eval_config=eval_config,
            version_ids=version_ids,
            scene_loader=MagicMock(),
            pools={"driver": AddressPool(["driver-a:50051"], 1, skip=False)},
            max_in_flight=1,
        )

    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.build_runtime_context",
        _fake_build_runtime_context,
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.compute_num_consumers_per_worker",
        lambda **_: 1,
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.start_worker_runtime", _fake_start_worker_runtime
    )
    monkeypatch.setattr("alpasim_runtime.daemon.engine.DaemonScheduler", _FakeScheduler)
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
        validate_config_scenes=False,
    )

    await engine.startup()
    await engine.shutdown()


@pytest.mark.asyncio
async def test_engine_simulate_submits_without_global_request_lock() -> None:
    entered_wait = asyncio.Event()
    release_wait = asyncio.Event()
    submitted_request_ids: list[str] = []

    class _FakeScheduler:
        async def submit_request(
            self, request_id: str, jobs, *, driver_pool=None
        ) -> None:
            del jobs, driver_pool
            submitted_request_ids.append(request_id)

        async def wait_request(self, request_id: str):
            entered_wait.set()
            await release_wait.wait()
            del request_id
            return []

    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    # Mock SceneLoader with a fake data source
    mock_scene_loader = MagicMock()
    mock_scene_loader.get_data_source.return_value = MagicMock()
    engine._scene_loader = mock_scene_loader
    engine._version_ids = RolloutMetadata.VersionIds(
        runtime_version=VersionId(version_id="runtime", git_hash="a"),
        sensorsim_version=VersionId(version_id="sensorsim", git_hash="b"),
        physics_version=VersionId(version_id="physics", git_hash="c"),
        egodriver_version=VersionId(version_id="driver", git_hash="d"),
        traffic_version=VersionId(version_id="traffic", git_hash="e"),
    )
    engine._scheduler = _FakeScheduler()

    request = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1)]
    )

    first = asyncio.create_task(engine.simulate(request))
    await entered_wait.wait()

    second = asyncio.create_task(engine.simulate(request))
    await asyncio.sleep(0)

    assert len(submitted_request_ids) == 2

    release_wait.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_engine_simulate_passes_driver_pool_when_request_has_available_drivers() -> (
    None
):
    """When SimulationRequest has available_drivers, engine creates a driver pool override."""
    captured_driver_pool: list[AddressPool | None] = []

    class _CapturingScheduler:
        async def submit_request(self, request_id, jobs, *, driver_pool=None):
            captured_driver_pool.append(driver_pool)

        async def wait_request(self, request_id):
            return []

    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    # Mock SceneLoader with a fake data source
    mock_scene_loader = MagicMock()
    mock_scene_loader.get_data_source.return_value = MagicMock()
    engine._scene_loader = mock_scene_loader
    engine._version_ids = RolloutMetadata.VersionIds(
        runtime_version=VersionId(version_id="runtime", git_hash="a"),
        sensorsim_version=VersionId(version_id="sensorsim", git_hash="b"),
        physics_version=VersionId(version_id="physics", git_hash="c"),
        egodriver_version=VersionId(version_id="driver", git_hash="d"),
        traffic_version=VersionId(version_id="traffic", git_hash="e"),
    )
    engine._scheduler = _CapturingScheduler()

    request = runtime_pb2.SimulationRequest(
        available_drivers=[
            runtime_pb2.SimulationRequest.DriverAddress(
                ip="10.0.0.1",
                port=50051,
            ),
        ],
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1)],
        n_concurrent_per_driver=2,
    )

    await engine.simulate(request)

    assert len(captured_driver_pool) == 1
    pool = captured_driver_pool[0]
    assert pool is not None
    # 1 address * 2 concurrent = 2 total capacity
    assert pool.total_capacity == 2
    assert pool.skip is False


@pytest.mark.asyncio
async def test_engine_simulate_rejects_offset_for_video_model_renderer() -> None:
    """A start offset is rejected up front when the renderer is the video model."""
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    engine._renderer_kind = RendererKind.video_model

    request = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(
                scenario_id="clipgt-a", nr_rollouts=1, start_time_offset_us=1_000_000
            )
        ]
    )

    with pytest.raises(InvalidRequestError, match="video_model"):
        await engine.simulate(request)


@pytest.mark.asyncio
async def test_engine_simulate_no_driver_pool_when_request_has_no_available_drivers() -> (
    None
):
    """When SimulationRequest has no available_drivers, driver_pool should be None."""
    captured_driver_pool: list[AddressPool | None] = []

    class _CapturingScheduler:
        async def submit_request(self, request_id, jobs, *, driver_pool=None):
            captured_driver_pool.append(driver_pool)

        async def wait_request(self, request_id):
            return []

    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    # Mock SceneLoader with a fake data source
    mock_scene_loader = MagicMock()
    mock_scene_loader.get_data_source.return_value = MagicMock()
    engine._scene_loader = mock_scene_loader
    engine._version_ids = RolloutMetadata.VersionIds(
        runtime_version=VersionId(version_id="runtime", git_hash="a"),
        sensorsim_version=VersionId(version_id="sensorsim", git_hash="b"),
        physics_version=VersionId(version_id="physics", git_hash="c"),
        egodriver_version=VersionId(version_id="driver", git_hash="d"),
        traffic_version=VersionId(version_id="traffic", git_hash="e"),
    )
    engine._scheduler = _CapturingScheduler()

    request = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1)],
    )

    await engine.simulate(request)

    assert len(captured_driver_pool) == 1
    assert captured_driver_pool[0] is None


@pytest.mark.asyncio
async def test_engine_prefetch_scene_coalesces_duplicate_requests(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Concurrent requests for one scene share a single NRE backend load."""
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    prefetch_calls: list[tuple[str, str]] = []

    async def _prefetch_scene(address: str, scene_id: str) -> None:
        prefetch_calls.append((address, scene_id))
        load_started.set()
        await release_load.wait()

    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    engine._scene_loader = MagicMock()
    engine._scene_loader.has_scene.return_value = True
    engine._runtime_context = SimpleNamespace(
        config=SimpleNamespace(
            user=SimpleNamespace(renderer=RendererConfig(kind=RendererKind.sensorsim))
        ),
        pools={"renderer": AddressPool(["nre.local:50052"], 4, skip=False)},
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.prefetch_nre_scene", _prefetch_scene
    )

    with caplog.at_level("INFO"):
        first = asyncio.create_task(engine.prefetch_scene("scene-A"))
        await load_started.wait()
        second = asyncio.create_task(engine.prefetch_scene("scene-A"))
        await asyncio.sleep(0)
        release_load.set()
        await asyncio.gather(first, second)

    assert prefetch_calls == [("nre.local:50052", "scene-A")]
    assert "coalesced" in caplog.text


@pytest.mark.asyncio
async def test_engine_prefetch_scene_requires_one_renderer_address() -> None:
    """Prefetch rejects ambiguous multi-renderer routing."""
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    engine._scene_loader = MagicMock()
    engine._scene_loader.has_scene.return_value = True
    engine._runtime_context = SimpleNamespace(
        config=SimpleNamespace(
            user=SimpleNamespace(renderer=RendererConfig(kind=RendererKind.sensorsim))
        ),
        pools={
            "renderer": AddressPool(
                ["nre-0.local:50052", "nre-1.local:50052"], 4, skip=False
            )
        },
    )

    with pytest.raises(ValueError, match="exactly one renderer address"):
        await engine.prefetch_scene("scene-A")


@pytest.mark.asyncio
async def test_engine_shutdown_rejects_new_scene_prefetches(tmp_path: Path) -> None:
    """Shutdown stops admitting prefetches before awaiting component cleanup."""
    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()

    async def _shutdown(*, reason: str) -> None:
        del reason
        shutdown_started.set()
        await release_shutdown.wait()

    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir=str(tmp_path / "log"),
    )
    engine._started = True
    engine._scheduler = SimpleNamespace(shutdown=_shutdown)
    engine._worker_runtime = SimpleNamespace(stop=AsyncMock())

    shutdown = asyncio.create_task(engine.shutdown())
    await shutdown_started.wait()

    with pytest.raises(DaemonUnavailableError, match="daemon is shutting down"):
        await engine.prefetch_scene("scene-A")

    release_shutdown.set()
    await shutdown


@pytest.mark.asyncio
async def test_engine_shutdown_stops_worker_when_scheduler_shutdown_fails() -> None:
    """Worker cleanup and reference cleanup still run after scheduler failures."""
    worker_runtime = SimpleNamespace(stop=AsyncMock())
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    engine._runtime_context = SimpleNamespace()
    engine._scene_loader = MagicMock()
    engine._scheduler = SimpleNamespace(
        shutdown=AsyncMock(side_effect=RuntimeError("scheduler failed"))
    )
    engine._worker_runtime = worker_runtime

    with pytest.raises(RuntimeError, match="scheduler failed"):
        await engine.shutdown()

    worker_runtime.stop.assert_awaited_once()
    assert engine._scheduler is None
    assert engine._worker_runtime is None
    assert engine._runtime_context is None
    assert engine._scene_loader is None
    assert engine._started is False
    assert engine._shutting_down is False
    await engine.shutdown()


@pytest.mark.asyncio
async def test_engine_prefetch_retrieves_failure_after_waiter_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shielded warmup failure is observed even if the last waiter was canceled."""
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    loop = asyncio.get_running_loop()
    unhandled_contexts: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unhandled_contexts.append(context))

    async def _prefetch_scene(address: str, scene_id: str) -> None:
        del address, scene_id
        load_started.set()
        await release_load.wait()
        raise RuntimeError("prefetch failed")

    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )
    engine._started = True
    engine._scene_loader = MagicMock()
    engine._scene_loader.has_scene.return_value = True
    engine._runtime_context = SimpleNamespace(
        config=SimpleNamespace(
            user=SimpleNamespace(renderer=RendererConfig(kind=RendererKind.sensorsim))
        ),
        pools={"renderer": AddressPool(["nre.local:50052"], 4, skip=False)},
    )
    monkeypatch.setattr(
        "alpasim_runtime.daemon.engine.prefetch_nre_scene", _prefetch_scene
    )

    try:
        waiter = asyncio.create_task(engine.prefetch_scene("scene-A"))
        await load_started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        release_load.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert unhandled_contexts == []


# ---------------------------------------------------------------------------
# build_simulation_return tests
# ---------------------------------------------------------------------------


def _version_ids() -> RolloutMetadata.VersionIds:
    return RolloutMetadata.VersionIds(
        runtime_version=VersionId(version_id="runtime", git_hash="a"),
        sensorsim_version=VersionId(version_id="sensorsim", git_hash="b"),
        physics_version=VersionId(version_id="physics", git_hash="c"),
        egodriver_version=VersionId(version_id="driver", git_hash="d"),
        traffic_version=VersionId(version_id="traffic", git_hash="e"),
        video_model_version=VersionId(version_id="video-model", git_hash="f"),
    )


def _make_eval_result() -> ScenarioEvalResult:
    return ScenarioEvalResult(
        timestep_metrics=[
            MetricReturn(
                name="collision_any",
                timestamps_us=[1000, 2000, 3000],
                values=[0.0, 1.0, 0.0],
                valid=[True, True, True],
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="offroad",
                timestamps_us=[1000, 2000],
                values=[False, True],
                valid=[True, True],
                time_aggregation=AggregationType.MEAN,
            ),
        ],
        aggregated_metrics={"collision_any": 1.0, "offroad": 0.5},
    )


def test_build_simulation_return_without_eval_results() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="s1", nr_rollouts=1)]
    )
    result = JobResult(
        request_id="r1",
        job_id="j1",
        rollout_spec_index=0,
        success=True,
        error=None,
        error_traceback=None,
        rollout_uuid="uuid-1",
        eval_result=None,
    )

    ret = build_simulation_return(
        request=request, version_ids=_version_ids(), results=[result]
    )

    assert ret.video_model_version.version_id == "video-model"
    assert len(ret.rollout_returns) == 1
    rr = ret.rollout_returns[0]
    assert rr.success is True
    assert rr.rollout_uuid == "uuid-1"
    assert len(rr.timestep_metrics) == 0
    assert dict(rr.aggregated_metrics) == {}


def test_build_simulation_return_with_eval_results() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="s1", nr_rollouts=1)]
    )
    eval_result = _make_eval_result()
    result = JobResult(
        request_id="r1",
        job_id="j1",
        rollout_spec_index=0,
        success=True,
        error=None,
        error_traceback=None,
        rollout_uuid="uuid-1",
        eval_result=eval_result,
    )

    ret = build_simulation_return(
        request=request, version_ids=_version_ids(), results=[result]
    )

    assert len(ret.rollout_returns) == 1
    rr = ret.rollout_returns[0]

    # Aggregated metrics
    assert dict(rr.aggregated_metrics) == {"collision_any": 1.0, "offroad": 0.5}

    # Timestep metrics
    assert len(rr.timestep_metrics) == 2

    collision_metric = rr.timestep_metrics[0]
    assert collision_metric.name == "collision_any"
    assert list(collision_metric.timestamps_us) == [1000, 2000, 3000]
    assert list(collision_metric.values) == [0.0, 1.0, 0.0]
    assert list(collision_metric.valid) == [True, True, True]
    assert collision_metric.time_aggregation == runtime_pb2.TIME_AGGREGATION_MAX

    offroad_metric = rr.timestep_metrics[1]
    assert offroad_metric.name == "offroad"
    assert list(offroad_metric.timestamps_us) == [1000, 2000]
    # bool values should be converted to floats
    assert list(offroad_metric.values) == [0.0, 1.0]
    assert list(offroad_metric.valid) == [True, True]
    assert offroad_metric.time_aggregation == runtime_pb2.TIME_AGGREGATION_MEAN


def test_build_simulation_return_mixed_results_with_and_without_eval() -> None:
    """One result has eval metrics, the other does not (e.g. failed rollout)."""
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="s1", nr_rollouts=2)]
    )
    results = [
        JobResult(
            request_id="r1",
            job_id="j1",
            rollout_spec_index=0,
            success=True,
            error=None,
            error_traceback=None,
            rollout_uuid="uuid-1",
            eval_result=_make_eval_result(),
        ),
        JobResult(
            request_id="r1",
            job_id="j2",
            rollout_spec_index=0,
            success=False,
            error="invalid scene",
            error_traceback=None,
            rollout_uuid="uuid-2",
            error_code=runtime_pb2.ROLLOUT_ERROR_CODE_INVALID_SCENE,
            eval_result=None,
        ),
    ]

    ret = build_simulation_return(
        request=request, version_ids=_version_ids(), results=results
    )

    assert len(ret.rollout_returns) == 2

    # First rollout: has metrics
    rr0 = ret.rollout_returns[0]
    assert rr0.success is True
    assert rr0.error_code == runtime_pb2.ROLLOUT_ERROR_CODE_NONE
    assert len(rr0.timestep_metrics) == 2
    assert dict(rr0.aggregated_metrics) == {"collision_any": 1.0, "offroad": 0.5}

    # Second rollout: no metrics
    rr1 = ret.rollout_returns[1]
    assert rr1.success is False
    assert rr1.error == "invalid scene"
    assert rr1.error_code == runtime_pb2.ROLLOUT_ERROR_CODE_INVALID_SCENE
    assert len(rr1.timestep_metrics) == 0
    assert dict(rr1.aggregated_metrics) == {}
