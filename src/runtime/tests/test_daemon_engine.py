# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.daemon.engine import DaemonEngine, build_simulation_return
from alpasim_runtime.worker.ipc import JobResult

from eval.data import AggregationType, MetricReturn
from eval.scenario_evaluator import ScenarioEvalResult


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        user=SimpleNamespace(
            nr_workers=1,
            smooth_trajectories=True,
            scenes=[SimpleNamespace(scene_id="clipgt-a")],
            endpoints=SimpleNamespace(startup_timeout_s=1),
        ),
        network=SimpleNamespace(),
    )


@pytest.mark.asyncio
async def test_engine_startup_gathers_versions_and_validates_scenes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config()
    version_ids = MagicMock()
    eval_config = MagicMock()
    worker_runtime = SimpleNamespace(stop=AsyncMock())

    class _FakeScheduler:
        def __init__(
            self,
            *,
            pools,
            runtime,
            request_store=None,
        ) -> None:
            del pools, request_store
            assert runtime is worker_runtime

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
            scene_id_to_idx={"clipgt-a": 0},
            pools={"driver": MagicMock()},
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
    version_ids = MagicMock()
    eval_config = MagicMock()
    worker_runtime = SimpleNamespace(stop=AsyncMock())

    class _FakeScheduler:
        def __init__(
            self,
            *,
            pools,
            runtime,
            request_store=None,
        ) -> None:
            del pools, request_store
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
            scene_id_to_idx={"clipgt-a": 0},
            pools={"driver": MagicMock()},
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
            error="sim crashed",
            error_traceback=None,
            rollout_uuid="uuid-2",
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
    assert len(rr0.timestep_metrics) == 2
    assert dict(rr0.aggregated_metrics) == {"collision_any": 1.0, "offroad": 0.5}

    # Second rollout: no metrics
    rr1 = ret.rollout_returns[1]
    assert rr1.success is False
    assert rr1.error == "sim crashed"
    assert len(rr1.timestep_metrics) == 0
    assert dict(rr1.aggregated_metrics) == {}
