# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Reproducible microbenchmark for frame-quality dynamic batching."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import resource
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import torch
from alpasim_grpc.v0 import frame_quality_pb2

from frame_quality.server import FrameQualityBatcher, FrameQualityModel


class _BatchSizeHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def emit(self, record: logging.LogRecord) -> None:
        batch_size = getattr(record, "batch_size", None)
        if batch_size is not None:
            self.batch_sizes.append(int(batch_size))


class BatchSizeCollector:
    """Collect structured batch-size log fields without changing the server API."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._handler = _BatchSizeHandler()
        self._previous_level = logger.level

    def __enter__(self) -> Self:
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._previous_level)

    def distribution(self) -> dict[int, int]:
        return dict(sorted(Counter(self._handler.batch_sizes).items()))


async def benchmark_requests(
    score_frame: Callable[
        [frame_quality_pb2.FrameQualityRequest],
        Awaitable[frame_quality_pb2.FrameQualityResponse],
    ],
    *,
    frames: list[bytes],
    concurrency: int,
    session_uuid: str,
    camera_logical_id: str,
) -> dict[str, Any]:
    """Submit a fixed frame sequence with bounded client concurrency."""
    if not frames:
        raise ValueError("frames must not be empty")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
    for timestamp_us, image_bytes in enumerate(frames):
        queue.put_nowait((timestamp_us, image_bytes))

    latencies_ms: list[float] = []

    async def worker() -> None:
        while True:
            try:
                timestamp_us, image_bytes = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            request = frame_quality_pb2.FrameQualityRequest(
                timestamp_us=timestamp_us,
                image_bytes=image_bytes,
                session_uuid=session_uuid,
                camera_logical_id=camera_logical_id,
            )
            started_at = time.perf_counter()
            try:
                response = await score_frame(request)
                if response.timestamp_us != timestamp_us:
                    raise ValueError(
                        "frame quality response timestamp does not match request: "
                        f"{response.timestamp_us} != {timestamp_us}"
                    )
                if response.session_uuid != session_uuid:
                    raise ValueError(
                        "frame quality response session does not match request: "
                        f"{response.session_uuid!r} != {session_uuid!r}"
                    )
                if response.camera_logical_id != camera_logical_id:
                    raise ValueError(
                        "frame quality response camera does not match request: "
                        f"{response.camera_logical_id!r} != {camera_logical_id!r}"
                    )
                latencies_ms.append((time.perf_counter() - started_at) * 1000)
            finally:
                queue.task_done()

    benchmark_started_at = time.perf_counter()
    async with asyncio.TaskGroup() as task_group:
        for _ in range(min(concurrency, len(frames))):
            task_group.create_task(worker())
    duration_s = time.perf_counter() - benchmark_started_at

    percentiles = {
        name: float(np.percentile(latencies_ms, percentile))
        for name, percentile in [
            ("p50", 50),
            ("p90", 90),
            ("p95", 95),
            ("p99", 99),
        ]
    }
    percentiles["max"] = max(latencies_ms)
    return {
        "request_count": len(frames),
        "concurrency": concurrency,
        "duration_s": duration_s,
        "throughput_fps": len(frames) / duration_s,
        "latency_ms": percentiles,
    }


def _load_frames(frames_dir: Path) -> list[bytes]:
    paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise ValueError(f"No JPEG or PNG frames found in {frames_dir}")
    return [path.read_bytes() for path in paths]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    frames = _load_frames(args.frames_dir)
    model = FrameQualityModel(args.checkpoint, args.device)
    batcher = FrameQualityBatcher(
        model.score_frames,
        checkpoint_id=model.checkpoint_id,
        checkpoint_sha256=model.checkpoint_sha256,
        max_batch_size=args.max_batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        max_queued_frames=args.max_queued_frames,
    )

    async with batcher:
        await benchmark_requests(
            batcher.score_frame,
            frames=frames[: min(len(frames), args.max_batch_size)],
            concurrency=args.concurrency,
            session_uuid=args.session_uuid,
            camera_logical_id=args.camera_logical_id,
        )
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        with BatchSizeCollector(logging.getLogger("frame_quality.server")) as batches:
            summary = await benchmark_requests(
                batcher.score_frame,
                frames=frames,
                concurrency=args.concurrency,
                session_uuid=args.session_uuid,
                camera_logical_id=args.camera_logical_id,
            )

    summary.update(
        {
            "checkpoint_id": model.checkpoint_id,
            "checkpoint_sha256": model.checkpoint_sha256,
            "device": args.device,
            "max_batch_size": args.max_batch_size,
            "max_batch_wait_ms": args.max_batch_wait_ms,
            "max_queued_frames": args.max_queued_frames,
            "batch_size_distribution": {
                str(size): count for size, count in batches.distribution().items()
            },
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "gpu_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(args.device) / (1024 * 1024)
                if args.device.startswith("cuda")
                else None
            ),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--session-uuid", default="benchmark")
    parser.add_argument("--camera-logical-id", default="camera_front_wide_120fov")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-wait-ms", type=float, default=10)
    parser.add_argument("--max-queued-frames", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = asyncio.run(_run(args))
    output = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(f"{output}\n")


if __name__ == "__main__":
    main()
