# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the frame-quality scoring service."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable

import cv2
import numpy as np
import pytest
import torch
from alpasim_grpc.v0 import frame_quality_pb2, frame_quality_pb2_grpc

import grpc
from frame_quality.server import (
    BatchInferenceResult,
    FrameQualityBatcher,
    FrameQualityModel,
    FrameQualityServicer,
    _decode_frame,
)


class _FakeModel:
    def eval(self):
        return self

    def __call__(self, batch):
        assert batch.dtype == torch.uint8
        assert tuple(batch.shape) == (2, 252, 448, 3)
        return torch.tensor([0.25, 0.75]), torch.tensor([1, 3])


def _result(scores: list[float]) -> BatchInferenceResult:
    return BatchInferenceResult(
        scores=scores,
        decode_duration_s=0.001,
        model_duration_s=0.002,
    )


def _request(timestamp_us: int) -> frame_quality_pb2.FrameQualityRequest:
    return frame_quality_pb2.FrameQualityRequest(
        timestamp_us=timestamp_us,
        image_bytes=f"frame-{timestamp_us}".encode(),
        session_uuid="session-123",
        camera_logical_id="camera_front_wide_120fov",
    )


async def _score_frame_over_grpc(
    score_frames: Callable[[list[bytes]], BatchInferenceResult],
) -> frame_quality_pb2.FrameQualityResponse:
    server = grpc.aio.server()
    async with FrameQualityBatcher(
        score_frames,
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=2,
        max_batch_wait_ms=1,
        max_queued_frames=8,
    ) as batcher:
        frame_quality_pb2_grpc.add_FrameQualityServiceServicer_to_server(
            FrameQualityServicer(batcher), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                return await frame_quality_pb2_grpc.FrameQualityServiceStub(
                    channel
                ).score_frame(_request(7))
        finally:
            await server.stop(grace=None)


def test_model_decodes_resizes_and_records_checkpoint(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "quality.pt"
    checkpoint.write_bytes(b"checkpoint")

    def load_model(*args, _extra_files, **kwargs):
        _extra_files["metadata.json"] = json.dumps({"checkpoint_id": "v1"}).encode()
        return _FakeModel()

    monkeypatch.setattr("torch.jit.load", load_model)
    image = np.zeros((320, 512, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    model = FrameQualityModel(checkpoint, "cpu")

    result = model.score_frames([encoded.tobytes(), encoded.tobytes()])

    assert result.scores == [0.25, 0.75]
    assert result.decode_duration_s >= 0.0
    assert result.model_duration_s >= 0.0
    assert model.checkpoint_id == "v1"
    assert model.checkpoint_sha256 == hashlib.sha256(b"checkpoint").hexdigest()


@pytest.mark.parametrize(
    "metadata",
    [{}, {"checkpoint_id": ""}, {"checkpoint_id": 1}],
)
def test_model_requires_checkpoint_id_in_artifact_metadata(
    monkeypatch, tmp_path, metadata
) -> None:
    checkpoint = tmp_path / "quality.pt"
    checkpoint.write_bytes(b"checkpoint")

    def load_model(*args, _extra_files, **kwargs):
        _extra_files["metadata.json"] = json.dumps(metadata).encode()
        return _FakeModel()

    monkeypatch.setattr("torch.jit.load", load_model)

    with pytest.raises(
        ValueError,
        match="checkpoint metadata must contain a non-empty string checkpoint_id",
    ):
        FrameQualityModel(checkpoint, "cpu")


@pytest.mark.asyncio
async def test_batcher_combines_concurrent_requests_and_routes_scores() -> None:
    calls: list[list[bytes]] = []

    def score_frames(frames: list[bytes]) -> BatchInferenceResult:
        calls.append(frames)
        return _result([float(frame[-1:]) / 10 for frame in frames])

    async with FrameQualityBatcher(
        score_frames,
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=4,
        max_batch_wait_ms=20,
        max_queued_frames=8,
    ) as batcher:
        responses = await asyncio.gather(
            *(batcher.score_frame(_request(timestamp)) for timestamp in [1, 2, 3])
        )

    assert len(calls) == 1
    assert [response.timestamp_us for response in responses] == [1, 2, 3]
    assert [response.score for response in responses] == pytest.approx([0.1, 0.2, 0.3])
    assert {response.checkpoint_id for response in responses} == {"v1"}
    assert {response.session_uuid for response in responses} == {"session-123"}
    assert {response.camera_logical_id for response in responses} == {
        "camera_front_wide_120fov"
    }


@pytest.mark.asyncio
async def test_batcher_isolates_malformed_frame_from_valid_batch_request() -> None:
    calls: list[list[bytes]] = []

    def score_frames(frames: list[bytes]) -> BatchInferenceResult:
        calls.append(frames)
        if b"malformed" in frames:
            _decode_frame(b"malformed")
        return _result([float(frame[-1:]) / 10 for frame in frames])

    malformed_request = _request(2)
    malformed_request.image_bytes = b"malformed"
    async with FrameQualityBatcher(
        score_frames,
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=2,
        max_batch_wait_ms=20,
        max_queued_frames=8,
    ) as batcher:
        valid, malformed = await asyncio.gather(
            batcher.score_frame(_request(1)),
            batcher.score_frame(malformed_request),
            return_exceptions=True,
        )

    assert calls[0] == [b"frame-1", b"malformed"]
    assert isinstance(valid, frame_quality_pb2.FrameQualityResponse)
    assert valid.score == pytest.approx(0.1)
    assert isinstance(malformed, ValueError)
    assert str(malformed) == "Failed to decode an input frame"


@pytest.mark.asyncio
async def test_batcher_dispatches_lone_request_after_wait_bound() -> None:
    async with FrameQualityBatcher(
        lambda frames: _result([0.5] * len(frames)),
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=4,
        max_batch_wait_ms=10,
        max_queued_frames=8,
    ) as batcher:
        started_at = time.perf_counter()
        response = await batcher.score_frame(_request(1))
        elapsed_s = time.perf_counter() - started_at

    assert response.score == 0.5
    assert 0.005 <= elapsed_s < 0.2


@pytest.mark.asyncio
async def test_batcher_applies_queue_backpressure_without_dropping_requests() -> None:
    def blocking_score(frames: list[bytes]) -> BatchInferenceResult:
        time.sleep(0.05)
        return _result([0.5] * len(frames))

    async with FrameQualityBatcher(
        blocking_score,
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=1,
        max_batch_wait_ms=1,
        max_queued_frames=1,
    ) as batcher:
        first = asyncio.create_task(batcher.score_frame(_request(1)))
        await asyncio.sleep(0.005)
        second = asyncio.create_task(batcher.score_frame(_request(2)))
        await asyncio.sleep(0)
        third = asyncio.create_task(batcher.score_frame(_request(3)))
        await asyncio.sleep(0.01)

        assert not third.done()
        responses = await asyncio.gather(first, second, third)

    assert [response.timestamp_us for response in responses] == [1, 2, 3]


@pytest.mark.asyncio
async def test_batcher_skips_request_cancelled_before_inference() -> None:
    calls: list[list[bytes]] = []

    def score_frames(frames: list[bytes]) -> BatchInferenceResult:
        calls.append(frames)
        return _result([0.5] * len(frames))

    async with FrameQualityBatcher(
        score_frames,
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=4,
        max_batch_wait_ms=20,
        max_queued_frames=8,
    ) as batcher:
        first = asyncio.create_task(batcher.score_frame(_request(1)))
        cancelled = asyncio.create_task(batcher.score_frame(_request(2)))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await first

    assert calls == [[b"frame-1"]]


@pytest.mark.asyncio
async def test_batcher_continues_after_model_failure() -> None:
    call_count = 0

    def score_frames(frames: list[bytes]) -> BatchInferenceResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("model failed")
        return _result([0.5] * len(frames))

    async with FrameQualityBatcher(
        score_frames,
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=1,
        max_batch_wait_ms=1,
        max_queued_frames=8,
    ) as batcher:
        with pytest.raises(RuntimeError, match="model failed"):
            await batcher.score_frame(_request(1))
        response = await batcher.score_frame(_request(2))

    assert response.score == 0.5
    assert call_count == 2


@pytest.mark.asyncio
async def test_async_grpc_server_scores_single_frame() -> None:
    server = grpc.aio.server()
    async with FrameQualityBatcher(
        lambda frames: _result([0.5] * len(frames)),
        checkpoint_id="v1",
        checkpoint_sha256="abc",
        max_batch_size=2,
        max_batch_wait_ms=1,
        max_queued_frames=8,
    ) as batcher:
        frame_quality_pb2_grpc.add_FrameQualityServiceServicer_to_server(
            FrameQualityServicer(batcher), server
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                response = await frame_quality_pb2_grpc.FrameQualityServiceStub(
                    channel
                ).score_frame(_request(7))
        finally:
            await server.stop(grace=None)

    assert response.timestamp_us == 7
    assert response.score == 0.5
    assert response.checkpoint_id == "v1"
    assert response.session_uuid == "session-123"
    assert response.camera_logical_id == "camera_front_wide_120fov"


@pytest.mark.asyncio
async def test_grpc_server_reports_malformed_frame_as_invalid_argument() -> None:
    def score_frames(frames: list[bytes]) -> BatchInferenceResult:
        _decode_frame(frames[0])
        return _result([0.5])

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _score_frame_over_grpc(score_frames)

    assert exc_info.value.code() is grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_grpc_server_reports_unexpected_scorer_failure_as_internal() -> None:
    def score_frames(frames: list[bytes]) -> BatchInferenceResult:
        raise RuntimeError("model failed")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await _score_frame_over_grpc(score_frames)

    assert exc_info.value.code() is grpc.StatusCode.INTERNAL
