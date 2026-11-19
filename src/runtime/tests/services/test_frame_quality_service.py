# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

import pytest
from alpasim_grpc.v0 import frame_quality_pb2
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.services.frame_quality_service import FrameQualityService
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.session_configs import FrameQualitySessionConfig
from alpasim_utils.types import ImageWithMetadata


@dataclass
class _Recorder:
    messages: list[LogEntry] = field(default_factory=list)

    async def on_message(self, message: LogEntry) -> None:
        self.messages.append(message)


class _FakeStub:
    def __init__(self) -> None:
        self.requests: list[frame_quality_pb2.FrameQualityRequest] = []
        self.release = asyncio.Event()
        self.block = False
        self.error: Exception | None = None
        self.checkpoint_id = "v1"
        self.checkpoint_sha256 = "abc123"

    async def score_frame(self, request, **kwargs):
        self.requests.append(request)
        if self.block:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return frame_quality_pb2.FrameQualityResponse(
            session_uuid=request.session_uuid,
            camera_logical_id=request.camera_logical_id,
            timestamp_us=request.timestamp_us,
            score=request.timestamp_us / 100.0,
            checkpoint_id=self.checkpoint_id,
            checkpoint_sha256=self.checkpoint_sha256,
        )


def _image(
    timestamp_us: int,
    *,
    logical_id: str = "front",
    frame_start_us: int = 1,
) -> ImageWithMetadata:
    return ImageWithMetadata(
        start_timestamp_us=frame_start_us,
        end_timestamp_us=timestamp_us,
        image_bytes=f"image-{timestamp_us}".encode(),
        camera_logical_id=logical_id,
    )


async def _make_service(
    monkeypatch,
    *,
    stub: _FakeStub | None = None,
    timeout_s: float = 12.0,
    skip: bool = False,
) -> tuple[FrameQualityService, _FakeStub, _Recorder, SessionInfo]:
    observed_timeouts: list[float] = []

    async def call_without_profiling(
        method_name, service_type, stub_call, *args, **kwargs
    ):
        observed_timeouts.append(kwargs["timeout"])
        return await stub_call(*args, **kwargs)

    monkeypatch.setattr(
        "alpasim_runtime.services.frame_quality_service.profiled_rpc_call",
        call_without_profiling,
    )
    recorder = _Recorder()
    broadcaster = MessageBroadcaster(handlers=[recorder])
    session_info = SessionInfo(
        uuid="rollout",
        broadcaster=broadcaster,
        session_config=FrameQualitySessionConfig(
            camera_logical_id="front",
            rpc_timeout_s=timeout_s,
            enabled=True,
        ),
    )
    service = FrameQualityService("quality:50051", skip=skip)
    fake_stub = stub or _FakeStub()
    service.stub = fake_stub
    service.session_info = session_info
    await service._initialize_session(session_info)
    service._observed_timeouts = observed_timeouts  # type: ignore[attr-defined]
    return service, fake_stub, recorder, session_info


@pytest.mark.asyncio
async def test_submit_frame_filters_camera_and_schedules_without_waiting(
    monkeypatch,
) -> None:
    service, stub, _, session_info = await _make_service(monkeypatch)
    stub.block = True

    await service.submit_frame(_image(10, logical_id="rear"))
    await asyncio.wait_for(
        service.submit_frame(_image(20, frame_start_us=10)), timeout=0.1
    )
    await asyncio.sleep(0)

    assert [request.timestamp_us for request in stub.requests] == [20]
    assert stub.requests[0].session_uuid == "rollout"
    assert stub.requests[0].camera_logical_id == "front"
    assert stub.requests[0].image_bytes == b"image-20"
    assert not service._tasks[20].done()

    stub.release.set()
    await service._cleanup_session(session_info)


@pytest.mark.asyncio
async def test_drain_broadcasts_complete_sorted_wire_responses(monkeypatch) -> None:
    service, _, recorder, session_info = await _make_service(monkeypatch)
    await service.submit_frame(_image(20))
    await service.submit_frame(_image(10))

    assert recorder.messages == []
    await service.drain()

    assert [message.WhichOneof("log_entry") for message in recorder.messages] == [
        "frame_quality_response",
        "frame_quality_response",
    ]
    responses = [message.frame_quality_response for message in recorder.messages]
    assert [response.timestamp_us for response in responses] == [10, 20]
    assert [response.score for response in responses] == pytest.approx([0.1, 0.2])
    assert {response.session_uuid for response in responses} == {"rollout"}
    assert {response.camera_logical_id for response in responses} == {"front"}
    assert {response.checkpoint_id for response in responses} == {"v1"}
    assert {response.checkpoint_sha256 for response in responses} == {"abc123"}
    assert service._observed_timeouts == [12.0, 12.0]  # type: ignore[attr-defined]
    await service._cleanup_session(session_info)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("rpc failed"), TimeoutError()])
async def test_rpc_failure_omits_metric_without_raising(monkeypatch, failure) -> None:
    stub = _FakeStub()
    stub.error = failure
    service, _, recorder, session_info = await _make_service(monkeypatch, stub=stub)

    await service.submit_frame(_image(10))
    await service.drain()

    assert recorder.messages == []
    await service._cleanup_session(session_info)


@pytest.mark.asyncio
async def test_checkpoint_change_omits_metric(monkeypatch) -> None:
    class _ChangingStub(_FakeStub):
        async def score_frame(self, request, **kwargs):
            response = await super().score_frame(request, **kwargs)
            response.checkpoint_id = f"v{request.timestamp_us}"
            return response

    service, _, recorder, session_info = await _make_service(
        monkeypatch, stub=_ChangingStub()
    )
    await service.submit_frame(_image(10))
    await service.submit_frame(_image(20))

    await service.drain()

    assert recorder.messages == []
    await service._cleanup_session(session_info)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "response_session",
        "response_camera",
        "response_timestamp",
        "score",
        "checkpoint_id",
    ),
    [
        ("wrong", "front", 10, 0.5, "v1"),
        ("rollout", "rear", 10, 0.5, "v1"),
        ("rollout", "front", 11, 0.5, "v1"),
        ("rollout", "front", 10, math.nan, "v1"),
        ("rollout", "front", 10, 1.1, "v1"),
        ("rollout", "front", 10, 0.5, ""),
    ],
)
async def test_invalid_response_omits_metric(
    monkeypatch,
    response_session,
    response_camera,
    response_timestamp,
    score,
    checkpoint_id,
) -> None:
    class _InvalidStub(_FakeStub):
        async def score_frame(self, request, **kwargs):
            return frame_quality_pb2.FrameQualityResponse(
                session_uuid=response_session,
                camera_logical_id=response_camera,
                timestamp_us=response_timestamp,
                score=score,
                checkpoint_id=checkpoint_id,
                checkpoint_sha256="abc123",
            )

    service, _, recorder, session_info = await _make_service(
        monkeypatch, stub=_InvalidStub()
    )
    await service.submit_frame(_image(10))

    await service.drain()

    assert recorder.messages == []
    await service._cleanup_session(session_info)


@pytest.mark.asyncio
async def test_backpressure_reaps_failure_without_raising(monkeypatch) -> None:
    monkeypatch.setattr(
        "alpasim_runtime.services.frame_quality_service._MAX_PENDING_REQUESTS", 1
    )
    stub = _FakeStub()
    stub.block = True
    stub.error = RuntimeError("rpc failed")
    service, _, recorder, session_info = await _make_service(monkeypatch, stub=stub)
    await service.submit_frame(_image(10))
    second_submission = asyncio.create_task(service.submit_frame(_image(20)))
    await asyncio.sleep(0)
    assert not second_submission.done()

    stub.release.set()
    await second_submission
    await service.drain()

    assert recorder.messages == []
    await service._cleanup_session(session_info)


@pytest.mark.asyncio
async def test_cleanup_cancels_and_reaps_outstanding_tasks(monkeypatch) -> None:
    class _CancellationStub(_FakeStub):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = asyncio.Event()

        async def score_frame(self, request, **kwargs):
            try:
                await asyncio.Future()
            finally:
                self.cancelled.set()

    stub = _CancellationStub()
    service, _, _, session_info = await _make_service(monkeypatch, stub=stub)
    await service.submit_frame(_image(10))
    await asyncio.sleep(0)

    await service._cleanup_session(session_info)

    assert stub.cancelled.is_set()
    assert service._tasks == {}


@pytest.mark.asyncio
async def test_skip_mode_does_not_submit(monkeypatch) -> None:
    service, stub, recorder, session_info = await _make_service(monkeypatch, skip=True)

    await service.submit_frame(_image(10))
    await service.drain()

    assert stub.requests == []
    assert recorder.messages == []
    await service._cleanup_session(session_info)
