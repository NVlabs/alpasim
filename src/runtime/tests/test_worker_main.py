# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for worker main artifact loading helpers."""

from __future__ import annotations

import logging
from multiprocessing import Queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import ServiceAddress
from alpasim_runtime.config import RendererConfig, RendererKind, VideoModelConfig
from alpasim_runtime.errors import InvalidSceneError
from alpasim_runtime.worker.ipc import (
    SHUTDOWN_SENTINEL,
    AssignedRolloutJob,
    JobResult,
    ServiceEndpoints,
)
from alpasim_runtime.worker.main import run_single_rollout, run_worker_loop
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene
from prometheus_client import generate_latest


def _scene(clip_id: str) -> Scene:
    return Scene(ClipDataSource(clip_id, f"uuid-for-{clip_id}"), T0_AUTO)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (
            InvalidSceneError("bad-scene", "broken map"),
            runtime_pb2.ROLLOUT_ERROR_CODE_INVALID_SCENE,
        ),
        (
            RuntimeError("rollout failed"),
            runtime_pb2.ROLLOUT_ERROR_CODE_UNSPECIFIED,
        ),
    ],
)
async def test_run_single_rollout_reports_error_code(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_code: int,
) -> None:
    create_kwargs: dict[str, object] = {}
    overrides = {"bad-scene": object()}
    for service in [
        "DriverService",
        "PhysicsService",
        "TrafficService",
        "ControllerService",
        "SensorsimService",
    ]:
        monkeypatch.setattr(
            f"alpasim_runtime.worker.main.{service}",
            lambda *args, **kwargs: SimpleNamespace(),
        )

    def _raise(**kwargs):
        create_kwargs.update(kwargs)
        raise error

    monkeypatch.setattr(
        "alpasim_runtime.worker.main.UnboundRollout.create",
        _raise,
    )

    result = await run_single_rollout(
        job=AssignedRolloutJob(
            request_id="req-1",
            job_id="job-1",
            scene=_scene("bad-scene"),
            rollout_spec_index=0,
            endpoints=ServiceEndpoints(
                driver=ServiceAddress("localhost:10001", skip=False),
                renderer=ServiceAddress("localhost:10002", skip=False),
                physics=ServiceAddress("localhost:10003", skip=False),
                trafficsim=ServiceAddress("localhost:10004", skip=False),
                controller=ServiceAddress("localhost:10005", skip=False),
            ),
            dispatch_kind="fifo",
            scheduler_wait_seconds=0.0,
            ego_history_bytes=b"serialized-ego-history",
        ),
        user_config=SimpleNamespace(
            renderer=RendererConfig(kind=RendererKind.sensorsim),
            simulation_config=MagicMock(),
        ),
        data_source_loader=MagicMock(),
        camera_catalog=MagicMock(),
        version_ids=MagicMock(),
        rollouts_dir="/tmp",
        eval_config=MagicMock(),
        eval_executor=MagicMock(),
        ego_mask_overrides=overrides,
    )

    assert result.success is False
    assert result.error_code == expected_code
    assert create_kwargs["ego_history_bytes"] == b"serialized-ego-history"
    assert create_kwargs["ego_mask_overrides"] is overrides


@pytest.mark.asyncio
async def test_run_worker_loop_uses_parent_version_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_worker_loop should pass parent-provided version IDs into rollouts."""
    seen_version_ids = None

    async def _fake_run_single_rollout(
        job,
        user_config,
        data_source_loader,
        camera_catalog,
        version_ids,
        rollouts_dir,
        eval_config,
        eval_executor=None,
        force_gt_frame_cache=None,
        ego_mask_overrides=None,
        nav_routes=None,
    ) -> JobResult:
        del (
            user_config,
            data_source_loader,
            camera_catalog,
            rollouts_dir,
            eval_config,
            eval_executor,
            force_gt_frame_cache,
        )
        nonlocal seen_version_ids
        seen_version_ids = version_ids
        return JobResult(
            request_id=job.request_id,
            job_id=job.job_id,
            rollout_spec_index=job.rollout_spec_index,
            success=True,
            error=None,
            error_traceback=None,
            rollout_uuid="rollout-uuid",
        )

    monkeypatch.setattr(
        "alpasim_runtime.worker.main.run_single_rollout",
        _fake_run_single_rollout,
    )

    endpoints = ServiceEndpoints(
        driver=ServiceAddress("localhost:10001", skip=False),
        renderer=ServiceAddress("localhost:10002", skip=False),
        physics=ServiceAddress("localhost:10003", skip=False),
        trafficsim=ServiceAddress("localhost:10004", skip=False),
        controller=ServiceAddress("localhost:10005", skip=False),
    )
    job = AssignedRolloutJob(
        request_id="req-1",
        job_id="job-1",
        scene=_scene("scene-1"),
        rollout_spec_index=0,
        endpoints=endpoints,
        dispatch_kind="fifo",
        scheduler_wait_seconds=0.5,
    )

    job_queue: Queue = Queue()
    result_queue: Queue = Queue()
    job_queue.put(job)
    job_queue.put(SHUTDOWN_SENTINEL)

    parent_version_ids = RolloutMetadata.VersionIds(
        runtime_version=VersionId(version_id="0.3.0", git_hash="abc"),
    )
    user_config = MagicMock()
    user_config.endpoints.startup_timeout_s = 1
    user_config.simulation_config.force_gt_frame_cache.enabled = False
    data_source_loader = MagicMock()
    data_source_loader.get_data_source.return_value = MagicMock()

    completed = await run_worker_loop(
        worker_id=0,
        job_queue=job_queue,
        result_queue=result_queue,
        num_consumers=1,
        user_config=user_config,
        data_source_loader=data_source_loader,
        camera_catalog=MagicMock(),
        version_ids=parent_version_ids,
        rollouts_dir="/tmp",
        eval_config=MagicMock(),
        parent_pid=None,
    )

    result = result_queue.get(timeout=1)
    assert completed == 1
    assert result.request_id == "req-1"
    assert result.job_id == "job-1"
    assert result.rollout_spec_index == 0
    assert result.success is True
    assert seen_version_ids is parent_version_ids


@pytest.mark.asyncio
async def test_run_worker_loop_reports_materialization_failure_without_teardown() -> (
    None
):
    loader_error = OSError("temporary artifact read failure")
    job = AssignedRolloutJob(
        request_id="req-1",
        job_id="job-1",
        scene=_scene("scene-1"),
        rollout_spec_index=0,
        endpoints=ServiceEndpoints(
            driver=ServiceAddress("driver", skip=False),
            renderer=ServiceAddress("renderer", skip=False),
            physics=ServiceAddress("physics", skip=False),
            trafficsim=ServiceAddress("trafficsim", skip=False),
            controller=ServiceAddress("controller", skip=False),
        ),
        dispatch_kind="fifo",
        scheduler_wait_seconds=0.0,
    )
    job_queue: Queue = Queue()
    result_queue: Queue = Queue()
    job_queue.put(job)
    job_queue.put(SHUTDOWN_SENTINEL)
    data_source_loader = MagicMock()
    data_source_loader.get_data_source.side_effect = loader_error
    user_config = MagicMock()
    user_config.simulation_config.force_gt_frame_cache.enabled = False

    completed = await run_worker_loop(
        worker_id=0,
        job_queue=job_queue,
        result_queue=result_queue,
        num_consumers=1,
        user_config=user_config,
        data_source_loader=data_source_loader,
        camera_catalog=MagicMock(),
        version_ids=MagicMock(),
        rollouts_dir="/tmp",
        eval_config=MagicMock(),
        parent_pid=None,
    )

    result = result_queue.get(timeout=1)
    assert completed == 1
    assert result.request_id == job.request_id
    assert result.job_id == job.job_id
    assert result.success is False
    assert result.error_code == runtime_pb2.ROLLOUT_ERROR_CODE_UNSPECIFIED
    assert str(loader_error) in result.error
    data_source_loader.get_data_source.assert_called_once_with("scene-1")


@pytest.mark.asyncio
async def test_run_worker_loop_clears_active_metric_when_rollout_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raise(**kwargs) -> JobResult:
        del kwargs
        raise RuntimeError("rollout failed before producing a result")

    monkeypatch.setattr("alpasim_runtime.worker.main.run_single_rollout", _raise)

    from alpasim_runtime.telemetry.telemetry_context import TelemetryContext

    telemetry = TelemetryContext(port=9200, worker_id=4)
    monkeypatch.setattr(
        "alpasim_runtime.worker.main.try_get_context", lambda: telemetry
    )
    job_queue: Queue = Queue()
    job_queue.put(
        AssignedRolloutJob(
            request_id="req-1",
            job_id="job-1",
            scene=_scene("scene-1"),
            rollout_spec_index=0,
            endpoints=ServiceEndpoints(
                driver=ServiceAddress("driver", skip=False),
                renderer=ServiceAddress("renderer", skip=False),
                physics=ServiceAddress("physics", skip=False),
                trafficsim=ServiceAddress("trafficsim", skip=False),
                controller=ServiceAddress("controller", skip=False),
            ),
            dispatch_kind="cold_replica",
            scheduler_wait_seconds=1.25,
        )
    )

    with pytest.raises(ExceptionGroup):  # noqa: F821
        await run_worker_loop(
            worker_id=4,
            job_queue=job_queue,
            result_queue=Queue(),
            num_consumers=1,
            user_config=MagicMock(),
            data_source_loader=MagicMock(),
            camera_catalog=MagicMock(),
            version_ids=MagicMock(),
            rollouts_dir="/tmp",
            eval_config=MagicMock(),
            parent_pid=None,
        )

    metrics = generate_latest(telemetry.registry).decode("utf-8")
    assert 'alpasim_renderer_active_rollouts{worker_id="4"} 0.0' in metrics
    assert (
        'alpasim_renderer_rollouts_started_total{kind="cold_replica",worker_id="4"} 1.0'
    ) in metrics
    assert (
        'alpasim_renderer_scheduler_wait_seconds_sum{kind="cold_replica",'
        'worker_id="4"} 1.25'
    ) in metrics


@pytest.mark.asyncio
async def test_run_single_rollout_uses_builtin_video_model_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_renderer_service = object()
    captured: dict[str, object] = {}

    class FakeRolloutRunner:
        async def run(self):
            return "eval-ok"

    monkeypatch.setattr(
        "alpasim_runtime.worker.main.DriverService",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "alpasim_runtime.worker.main.SensorsimService",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "alpasim_runtime.worker.main.PhysicsService",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "alpasim_runtime.worker.main.TrafficService",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "alpasim_runtime.worker.main.ControllerService",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def _fake_video_model_service(
        address,
        config,
        skip=False,
        camera_catalog=None,
    ):
        captured["config"] = config
        captured["address"] = address
        captured["skip"] = skip
        captured["camera_catalog"] = camera_catalog
        return fake_renderer_service

    monkeypatch.setattr(
        "alpasim_runtime.worker.main.VideoModelService",
        _fake_video_model_service,
    )

    def _fake_unbound_create(**kwargs):
        captured["unbound_create_kwargs"] = kwargs
        return SimpleNamespace(
            rollout_uuid="rollout-uuid",
        )

    monkeypatch.setattr(
        "alpasim_runtime.worker.main.UnboundRollout.create",
        _fake_unbound_create,
    )

    def _fake_create_event_rollout(**kwargs):
        captured["rollout_kwargs"] = kwargs
        return FakeRolloutRunner()

    monkeypatch.setattr(
        "alpasim_runtime.worker.main.create_event_rollout",
        _fake_create_event_rollout,
    )

    job = AssignedRolloutJob(
        request_id="req-1",
        job_id="job-1",
        scene=_scene("scene-1"),
        rollout_spec_index=0,
        endpoints=ServiceEndpoints(
            driver=ServiceAddress("localhost:10001", skip=False),
            renderer=ServiceAddress("localhost:10006", skip=False),
            physics=ServiceAddress("localhost:10003", skip=False),
            trafficsim=ServiceAddress("localhost:10004", skip=False),
            controller=ServiceAddress("localhost:10005", skip=False),
        ),
        dispatch_kind="fifo",
        scheduler_wait_seconds=0.5,
    )
    video_model_config = VideoModelConfig(fps=24)
    user_config = SimpleNamespace(
        renderer=RendererConfig(
            kind=RendererKind.video_model,
            video_model_config=video_model_config,
        ),
        simulation_config=MagicMock(force_gt_frame_cache=MagicMock(enabled=False)),
    )

    data_source = MagicMock()
    data_source_loader = MagicMock()
    data_source_loader.get_data_source.return_value = data_source
    result = await run_single_rollout(
        job=job,
        user_config=user_config,
        data_source_loader=data_source_loader,
        camera_catalog=MagicMock(),
        version_ids=MagicMock(),
        rollouts_dir="/tmp",
        eval_config=MagicMock(),
        eval_executor=MagicMock(),
    )

    assert result.success is True
    assert captured["config"] is video_model_config
    assert captured["address"] == "localhost:10006"
    assert captured["skip"] is False
    assert captured["camera_catalog"] is not None
    assert (
        captured["unbound_create_kwargs"]["renderer_service"] is fake_renderer_service
    )
    assert captured["rollout_kwargs"]["renderer_service"] is fake_renderer_service
    assert captured["rollout_kwargs"]["data_source"] is data_source
    # Renderer-specific artifact access stays behind data_source.
    assert "artifact_path" not in captured["rollout_kwargs"]


@pytest.mark.asyncio
@pytest.mark.parametrize("log_level", [logging.INFO, logging.WARNING])
async def test_worker_logging_respects_configured_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    log_level: int,
) -> None:
    """Worker file and console logs retain warnings and honor the selected level."""
    from alpasim_runtime.worker import main as worker

    monkeypatch.setattr(
        worker,
        "typed_parse_config",
        lambda *_: SimpleNamespace(extra_cameras=[]),
    )
    monkeypatch.setattr(worker, "build_data_source_loader", lambda _: None)
    names = ["alpasim_runtime", "alpasim_utils", "alpasim_grpc"]

    async def emit_logs(**kwargs) -> None:
        for name in names:
            logging.getLogger(name).info("worker-test-info %s", name)
            logging.getLogger(name).warning("worker-test-warning %s", name)

    monkeypatch.setattr(worker, "run_worker_loop", emit_logs)
    args = SimpleNamespace(
        shared_rpc_tracking=None,
        user_config_path="unused.yaml",
        log_dir=str(tmp_path),
        worker_id=0,
        num_workers=2,
        num_consumers=1,
        telemetry_port=0,
        log_level=log_level,
        job_queue=SimpleNamespace(qsize=lambda: 0),
        result_queue=None,
        eval_config=None,
        version_ids=None,
        parent_pid=None,
        ego_mask_overrides=None,
        nav_routes=None,
    )
    loggers = [logging.getLogger(name) for name in names]
    saved = [(log.level, list(log.handlers), log.propagate) for log in loggers]
    try:
        await worker.worker_async_main(args)
        outputs = [
            capsys.readouterr().out,
            (tmp_path / "txt-logs/runtime_worker_0.log").read_text(),
        ]
        for output in outputs:
            for name in names:
                assert f"worker-test-warning {name}" in output
                assert (f"worker-test-info {name}" in output) == (
                    log_level == logging.INFO
                )
    finally:
        for log, (level, handlers, propagate) in zip(loggers, saved, strict=True):
            for handler in log.handlers:
                handler.close()
            log.handlers[:] = handlers
            log.setLevel(level)
            log.propagate = propagate
