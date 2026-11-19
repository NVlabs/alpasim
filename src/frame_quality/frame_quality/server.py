# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Asynchronous, dynamically batched frame-quality service."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, frame_quality_pb2, frame_quality_pb2_grpc

import grpc

logger = logging.getLogger(__name__)

_INPUT_WIDTH = 448
_INPUT_HEIGHT = 252
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class _InvalidRequestError(ValueError):
    """Raised when an RPC request contains an invalid frame."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_frame(image_bytes: bytes) -> np.ndarray:
    image_bgr = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image_bgr is None:
        raise _InvalidRequestError("Failed to decode an input frame")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(
        image_rgb,
        (_INPUT_WIDTH, _INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


@dataclass(frozen=True)
class BatchInferenceResult:
    scores: list[float]
    decode_duration_s: float
    model_duration_s: float


class FrameQualityModel:
    """Own and synchronously execute one exported frame quality model."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"frame quality checkpoint not found: {checkpoint}")

        self._device = torch.device(device)
        extra_files = {"metadata.json": ""}
        self._model = torch.jit.load(
            str(checkpoint),
            map_location=self._device,
            _extra_files=extra_files,
        ).eval()
        try:
            checkpoint_id = json.loads(extra_files["metadata.json"])["checkpoint_id"]
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            raise ValueError(
                "checkpoint metadata must contain a non-empty string checkpoint_id"
            ) from exc
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError(
                "checkpoint metadata must contain a non-empty string checkpoint_id"
            )
        self.checkpoint_id = checkpoint_id
        self.checkpoint_sha256 = _sha256(checkpoint)
        logger.info(
            "Loaded frame quality checkpoint %s (%s) on %s",
            checkpoint_id,
            self.checkpoint_sha256,
            self._device,
        )

    def score_frames(self, image_bytes: list[bytes]) -> BatchInferenceResult:
        decode_started_at = time.perf_counter()
        batch = np.stack([_decode_frame(frame) for frame in image_bytes])
        decode_duration_s = time.perf_counter() - decode_started_at

        model_started_at = time.perf_counter()
        tensor = torch.from_numpy(batch).to(self._device)
        with torch.inference_mode():
            scores, _ = self._model(tensor)
        score_values = [float(score) for score in scores.detach().cpu().reshape(-1)]
        model_duration_s = time.perf_counter() - model_started_at
        if len(score_values) != len(image_bytes):
            raise ValueError(
                "frame quality model returned "
                f"{len(score_values)} scores for {len(image_bytes)} frames"
            )
        return BatchInferenceResult(
            scores=score_values,
            decode_duration_s=decode_duration_s,
            model_duration_s=model_duration_s,
        )


@dataclass(frozen=True)
class _BatchItem:
    request: frame_quality_pb2.FrameQualityRequest
    response: asyncio.Future[frame_quality_pb2.FrameQualityResponse]
    enqueued_at: float


class FrameQualityBatcher:
    """Batch concurrent single-frame requests for one model executor."""

    def __init__(
        self,
        score_frames: Callable[[list[bytes]], BatchInferenceResult],
        *,
        checkpoint_id: str,
        checkpoint_sha256: str,
        max_batch_size: int,
        max_batch_wait_ms: float,
        max_queued_frames: int,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_batch_wait_ms <= 0:
            raise ValueError("max_batch_wait_ms must be positive")
        if max_queued_frames <= 0:
            raise ValueError("max_queued_frames must be positive")

        self._score_frames = score_frames
        self._checkpoint_id = checkpoint_id
        self._checkpoint_sha256 = checkpoint_sha256
        self._max_batch_size = max_batch_size
        self._max_batch_wait_s = max_batch_wait_ms / 1000
        self._queue: asyncio.Queue[_BatchItem | None] = asyncio.Queue(
            maxsize=max_queued_frames
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="frame-quality",
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._accepting = False

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    def start(self) -> None:
        if self._worker_task is not None:
            raise RuntimeError("frame quality batcher is already started")
        self._accepting = True
        self._worker_task = asyncio.create_task(
            self._run(), name="frame-quality-batcher"
        )

    async def close(self) -> None:
        if self._worker_task is None:
            self._executor.shutdown(wait=True)
            return
        self._accepting = False
        await self._queue.join()
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None
        self._executor.shutdown(wait=True)

    async def score_frame(
        self, request: frame_quality_pb2.FrameQualityRequest
    ) -> frame_quality_pb2.FrameQualityResponse:
        if not self._accepting:
            raise RuntimeError("frame quality batcher is not accepting requests")

        loop = asyncio.get_running_loop()
        response: asyncio.Future[frame_quality_pb2.FrameQualityResponse] = (
            loop.create_future()
        )
        item = _BatchItem(
            request=request,
            response=response,
            enqueued_at=loop.time(),
        )
        try:
            await self._queue.put(item)
            return await response
        except asyncio.CancelledError:
            response.cancel()
            raise

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return

            items = [first]
            deadline = first.enqueued_at + self._max_batch_wait_s
            while len(items) < self._max_batch_size:
                timeout_s = deadline - loop.time()
                if timeout_s <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout_s)
                except TimeoutError:
                    break
                assert item is not None
                items.append(item)

            live_items = [item for item in items if not item.response.cancelled()]
            if live_items:
                batch_started_at = time.perf_counter()
                queue_wait_s = loop.time() - min(
                    item.enqueued_at for item in live_items
                )
                try:
                    inference = await loop.run_in_executor(
                        self._executor,
                        self._score_frames,
                        [item.request.image_bytes for item in live_items],
                    )
                    self._set_results(live_items, inference)
                    logger.info(
                        "frame quality batch",
                        extra={
                            "batch_size": len(live_items),
                            "queue_wait_ms": queue_wait_s * 1000,
                            "decode_ms": inference.decode_duration_s * 1000,
                            "model_ms": inference.model_duration_s * 1000,
                            "total_ms": (time.perf_counter() - batch_started_at) * 1000,
                        },
                    )
                except _InvalidRequestError:
                    for item in live_items:
                        try:
                            inference = await loop.run_in_executor(
                                self._executor,
                                self._score_frames,
                                [item.request.image_bytes],
                            )
                            self._set_results([item], inference)
                        except Exception as exc:  # noqa: BLE001
                            if not item.response.cancelled():
                                item.response.set_exception(exc)
                except Exception as exc:  # noqa: BLE001
                    for item in live_items:
                        if not item.response.cancelled():
                            item.response.set_exception(exc)

            for _ in items:
                self._queue.task_done()

    def _set_results(
        self, items: list[_BatchItem], inference: BatchInferenceResult
    ) -> None:
        if len(inference.scores) != len(items):
            raise ValueError(
                "frame quality scorer returned "
                f"{len(inference.scores)} scores for {len(items)} frames"
            )
        for item, score in zip(items, inference.scores, strict=True):
            if not item.response.cancelled():
                item.response.set_result(
                    frame_quality_pb2.FrameQualityResponse(
                        timestamp_us=item.request.timestamp_us,
                        score=score,
                        checkpoint_id=self._checkpoint_id,
                        checkpoint_sha256=self._checkpoint_sha256,
                        session_uuid=item.request.session_uuid,
                        camera_logical_id=item.request.camera_logical_id,
                    )
                )


class FrameQualityServicer(frame_quality_pb2_grpc.FrameQualityServiceServicer):
    def __init__(self, batcher: FrameQualityBatcher) -> None:
        self._batcher = batcher

    async def get_version(
        self, request: common_pb2.Empty, context: grpc.aio.ServicerContext
    ) -> common_pb2.VersionId:
        return common_pb2.VersionId(
            version_id=importlib.metadata.version("alpasim-frame-quality"),
            git_hash="n/a",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    async def score_frame(
        self,
        request: frame_quality_pb2.FrameQualityRequest,
        context: grpc.aio.ServicerContext,
    ) -> frame_quality_pb2.FrameQualityResponse:
        try:
            return await self._batcher.score_frame(request)
        except asyncio.CancelledError:
            raise
        except _InvalidRequestError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise AssertionError("context.abort returned unexpectedly")
        except Exception as exc:  # noqa: BLE001
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            raise AssertionError("context.abort returned unexpectedly")


async def _serve(
    *,
    checkpoint: Path,
    device: str,
    host: str,
    port: int,
    max_batch_size: int,
    max_batch_wait_ms: float,
    max_queued_frames: int,
) -> None:
    cv2.setNumThreads(1)
    model = FrameQualityModel(checkpoint, device)
    batcher = FrameQualityBatcher(
        model.score_frames,
        checkpoint_id=model.checkpoint_id,
        checkpoint_sha256=model.checkpoint_sha256,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        max_queued_frames=max_queued_frames,
    )
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
        ]
    )
    frame_quality_pb2_grpc.add_FrameQualityServiceServicer_to_server(
        FrameQualityServicer(batcher), server
    )
    address = f"{host}:{port}"
    server.add_insecure_port(address)
    async with batcher:
        await server.start()
        logger.info("Serving frame quality scoring on %s", address)
        try:
            await server.wait_for_termination()
        finally:
            await server.stop(grace=5)


def serve(
    *,
    checkpoint: Path,
    device: str,
    host: str,
    port: int,
    max_batch_size: int = 32,
    max_batch_wait_ms: float = 10,
    max_queued_frames: int = 256,
) -> None:
    asyncio.run(
        _serve(
            checkpoint=checkpoint,
            device=device,
            host=host,
            port=port,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            max_queued_frames=max_queued_frames,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-wait-ms", type=float, default=10)
    parser.add_argument("--max-queued-frames", type=int, default=256)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
    )
    serve(
        checkpoint=args.checkpoint,
        device=args.device,
        host=args.host,
        port=args.port,
        max_batch_size=args.max_batch_size,
        max_batch_wait_ms=args.max_batch_wait_ms,
        max_queued_frames=args.max_queued_frames,
    )


if __name__ == "__main__":
    main()
