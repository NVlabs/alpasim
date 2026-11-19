# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Nonblocking runtime client for frame-quality scoring."""

from __future__ import annotations

import asyncio
import logging
import math
from time import perf_counter

from alpasim_grpc.v0 import frame_quality_pb2
from alpasim_grpc.v0.frame_quality_pb2_grpc import FrameQualityServiceStub
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo
from alpasim_runtime.services.session_configs import FrameQualitySessionConfig
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_runtime.telemetry.telemetry_context import try_get_context
from alpasim_utils.types import ImageWithMetadata

logger = logging.getLogger(__name__)

_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_MAX_PENDING_REQUESTS = 64


class FrameQualityService(ServiceBase[FrameQualityServiceStub]):
    """Accept rendered frames and asynchronously score the selected camera."""

    def __init__(self, address: str, skip: bool = False) -> None:
        super().__init__(address, skip)
        self._config: FrameQualitySessionConfig | None = None
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._results: dict[int, frame_quality_pb2.FrameQualityResponse] = {}
        self._submitted_count = 0
        self._failed = False

    @property
    def stub_class(self) -> type[FrameQualityServiceStub]:
        return FrameQualityServiceStub

    @property
    def channel_options(self) -> list[tuple[str, int]]:
        return [
            ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
        ]

    async def _initialize_session(self, session_info: SessionInfo) -> None:
        if not isinstance(session_info.session_config, FrameQualitySessionConfig):
            raise TypeError(
                "FrameQualityService requires FrameQualitySessionConfig, got "
                f"{type(session_info.session_config).__name__}"
            )
        self._config = session_info.session_config
        self._tasks = {}
        self._results = {}
        self._submitted_count = 0
        self._failed = False

    async def _cleanup_session(self, session_info: SessionInfo) -> None:
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._record_pending_requests(0)
        self._tasks = {}
        self._results = {}
        self._config = None

    async def submit_frame(self, image: ImageWithMetadata) -> None:
        """Schedule selected camera frames without propagating scoring errors."""
        try:
            await self._submit_frame(image)
        except Exception:
            self._failed = True
            logger.exception("Failed to submit a frame quality frame")

    async def _submit_frame(self, image: ImageWithMetadata) -> None:
        config = self._config
        if self.skip or config is None or not config.enabled:
            return

        if image.camera_logical_id != config.camera_logical_id:
            return

        timestamp_us = image.end_timestamp_us
        self._submitted_count += 1
        if timestamp_us in self._tasks:
            raise ValueError(f"Duplicate frame quality frame timestamp {timestamp_us}")

        while self._unfinished_task_count() >= _MAX_PENDING_REQUESTS:
            context = try_get_context()
            if context is not None:
                context.record_frame_quality_backpressure()
            await self._reap_oldest_unfinished_task()

        self._tasks[timestamp_us] = asyncio.create_task(
            self._score_frame(timestamp_us, image.image_bytes),
            name=f"frame-quality-{timestamp_us}",
        )
        self._record_pending_requests(self._unfinished_task_count())

    def _unfinished_task_count(self) -> int:
        return sum(not task.done() for task in self._tasks.values())

    async def _reap_oldest_unfinished_task(self) -> None:
        for task in self._tasks.values():
            if not task.done():
                await task
                return
        raise AssertionError("No unfinished frame quality request to reap")

    async def _score_frame(self, timestamp_us: int, image_bytes: bytes) -> None:
        config = self._config
        if config is None or self.stub is None:
            self._failed = True
            logger.error("frame quality service used outside an open session")
            return

        request = frame_quality_pb2.FrameQualityRequest(
            session_uuid=self._require_session_info().uuid,
            camera_logical_id=config.camera_logical_id,
            timestamp_us=timestamp_us,
            image_bytes=image_bytes,
        )
        try:
            response = await profiled_rpc_call(
                "score_frame",
                "frame_quality",
                self.stub.score_frame,
                request,
                timeout=config.rpc_timeout_s,
            )
            score = float(response.score)
            if response.session_uuid != request.session_uuid:
                raise ValueError(
                    "frame quality response session does not match request: "
                    f"{response.session_uuid!r} != {request.session_uuid!r}"
                )
            if response.camera_logical_id != request.camera_logical_id:
                raise ValueError(
                    "frame quality response camera does not match request: "
                    f"{response.camera_logical_id!r} != {request.camera_logical_id!r}"
                )
            if int(response.timestamp_us) != timestamp_us:
                raise ValueError(
                    "frame quality response timestamp does not match request: "
                    f"{response.timestamp_us} != {timestamp_us}"
                )
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"frame quality score must be finite and in [0, 1], got {score}"
                )
            if not response.checkpoint_id or not response.checkpoint_sha256:
                raise ValueError(
                    "frame quality response is missing checkpoint identity metadata"
                )
            self._results[timestamp_us] = response
        except Exception:
            self._failed = True
            logger.exception(
                "frame quality scoring failed for frame %d; the rollout metric "
                "will be omitted",
                timestamp_us,
            )
        finally:
            self._record_pending_requests(max(0, self._unfinished_task_count() - 1))

    async def drain(self) -> None:
        """Await accepted frames and broadcast their validated raw responses."""
        config = self._config
        if self.skip or config is None or not config.enabled or not self._tasks:
            return

        started_at = perf_counter()
        try:
            await self._drain_complete_metric()
        finally:
            context = try_get_context()
            if context is not None:
                context.record_frame_quality_drain(perf_counter() - started_at)

    async def _drain_complete_metric(self) -> None:
        """Broadcast responses only when every submitted frame is valid."""

        task_results = await asyncio.gather(
            *self._tasks.values(), return_exceptions=True
        )
        if any(isinstance(result, BaseException) for result in task_results):
            self._failed = True

        expected_timestamps = set(self._tasks)
        if (
            self._failed
            or self._submitted_count != len(expected_timestamps)
            or set(self._results) != expected_timestamps
        ):
            logger.error(
                "frame quality metric incomplete for session %s: "
                "submitted=%d unique=%d completed=%d",
                self._require_session_info().uuid,
                self._submitted_count,
                len(expected_timestamps),
                len(self._results),
            )
            return

        checkpoint_metadata = {
            (result.checkpoint_id, result.checkpoint_sha256)
            for result in self._results.values()
        }
        if len(checkpoint_metadata) != 1:
            self._failed = True
            logger.error(
                "frame quality checkpoint changed within session %s: %s",
                self._require_session_info().uuid,
                sorted(checkpoint_metadata),
            )
            return

        for response in sorted(
            self._results.values(), key=lambda result: result.timestamp_us
        ):
            await self._require_session_info().broadcaster.broadcast(
                LogEntry(frame_quality_response=response)
            )

    def _record_pending_requests(self, count: int) -> None:
        context = try_get_context()
        if context is not None:
            context.record_frame_quality_pending_requests(count)
