# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the frame-quality benchmark harness."""

from __future__ import annotations

import logging

import pytest
from alpasim_grpc.v0 import frame_quality_pb2

from frame_quality.server import BatchInferenceResult, FrameQualityBatcher

from .benchmark import BatchSizeCollector, benchmark_requests


@pytest.mark.asyncio
async def test_benchmark_reports_latency_throughput_and_batch_distribution() -> None:
    logger = logging.getLogger("frame_quality.server")
    requests: list[frame_quality_pb2.FrameQualityRequest] = []
    with BatchSizeCollector(logger) as batch_sizes:
        async with FrameQualityBatcher(
            lambda frames: BatchInferenceResult(
                scores=[0.5] * len(frames),
                decode_duration_s=0.001,
                model_duration_s=0.002,
            ),
            checkpoint_id="v1",
            checkpoint_sha256="abc",
            max_batch_size=2,
            max_batch_wait_ms=10,
            max_queued_frames=8,
        ) as batcher:

            async def score_frame(
                request: frame_quality_pb2.FrameQualityRequest,
            ) -> frame_quality_pb2.FrameQualityResponse:
                requests.append(request)
                return await batcher.score_frame(request)

            summary = await benchmark_requests(
                score_frame,
                frames=[b"a", b"b", b"c", b"d"],
                concurrency=2,
                session_uuid="benchmark-session",
                camera_logical_id="camera_front_wide_120fov",
            )

    assert summary["request_count"] == 4
    assert summary["concurrency"] == 2
    assert summary["throughput_fps"] > 0
    assert set(summary["latency_ms"]) == {
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
    }
    assert summary["latency_ms"]["max"] >= summary["latency_ms"]["p50"]
    assert sorted(request.timestamp_us for request in requests) == [0, 1, 2, 3]
    assert {request.session_uuid for request in requests} == {"benchmark-session"}
    assert {request.camera_logical_id for request in requests} == {
        "camera_front_wide_120fov"
    }
    assert batch_sizes.distribution() == {2: 2}


def test_batch_size_collector_ignores_unrelated_logs() -> None:
    logger = logging.getLogger("frame-quality-test")
    with BatchSizeCollector(logger) as batch_sizes:
        logger.info("unrelated")
        logger.info("batch", extra={"batch_size": 3})

    assert batch_sizes.distribution() == {3: 1}
