# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Unit tests for worker main artifact loading helpers."""

from __future__ import annotations

from multiprocessing import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import ServiceAddress
from alpasim_runtime.config import RendererConfig, RendererKind, VideoModelConfig
from alpasim_runtime.worker.ipc import (
    SHUTDOWN_SENTINEL,
    AssignedRolloutJob,
    JobResult,
    ServiceEndpoints,
)
from alpasim_runtime.worker.main import run_single_rollout, run_worker_loop


@pytest.mark.asyncio
async def test_run_worker_loop_uses_parent_version_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_worker_loop should pass parent-provided version IDs into rollouts."""
    seen_version_ids = None

    async def _fake_run_single_rollout(
        job,
        user_config,
        data_source,
        camera_catalog,
        version_ids,
        rollouts_dir,
        eval_config,
        eval_executor=None,
    ) -> JobResult:
        del (
            user_config,
            data_source,
            camera_catalog,
            rollouts_dir,
            eval_config,
            eval_executor,
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
        scene_id="scene-1",
        rollout_spec_index=0,
        endpoints=endpoints,
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
    scene_loader = MagicMock()
    scene_loader.get_data_source.return_value = MagicMock()

    completed = await run_worker_loop(
        worker_id=0,
        job_queue=job_queue,
        result_queue=result_queue,
        num_consumers=1,
        user_config=user_config,
        scene_loader=scene_loader,
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
        return SimpleNamespace(rollout_uuid="rollout-uuid", scene_id="scene-1")

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
        scene_id="scene-1",
        rollout_spec_index=0,
        endpoints=ServiceEndpoints(
            driver=ServiceAddress("localhost:10001", skip=False),
            renderer=ServiceAddress("localhost:10006", skip=False),
            physics=ServiceAddress("localhost:10003", skip=False),
            trafficsim=ServiceAddress("localhost:10004", skip=False),
            controller=ServiceAddress("localhost:10005", skip=False),
        ),
    )
    video_model_config = VideoModelConfig(fps=24)
    user_config = SimpleNamespace(
        renderer=RendererConfig(
            kind=RendererKind.video_model,
            video_model_config=video_model_config,
        ),
        simulation_config=MagicMock(),
    )

    data_source = MagicMock()
    result = await run_single_rollout(
        job=job,
        user_config=user_config,
        data_source=data_source,
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
