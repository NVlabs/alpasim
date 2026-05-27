# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Video model prefetch + per-frame events.

Two events live here together because they form a single logical pipeline:
``VideoModelPrefetchEvent`` requests a chunk of frames from the server and
schedules one ``VideoModelFrameEvent`` per camera-frame at the right
simulation time. The frame event is a thin per-frame dispatch helper -- it
only exists so the simulation loop can deliver pre-rendered frames to the
driver one at a time at their proper timestamps.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.events.base import Event, EventQueue
from alpasim_runtime.events.state import RolloutState
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.types import RuntimeCamera
from alpasim_utils.types import ImageWithMetadata
from alpasim_video_model.service import ChunkResult, VideoModelService
from alpasim_video_model.utils import build_trajectory_for_video_model

logger = logging.getLogger(__name__)


class VideoModelFrameEvent(Event):
    """Forwards a single pre-rendered video model frame to downstream consumers.

    Created by :class:`VideoModelPrefetchEvent`, not scheduled directly.
    """

    priority: int = 10

    def __init__(
        self,
        timestamp_us: int,
        frame: ImageWithMetadata,
        camera_logical_id: str,
        driver: DriverService,
        should_submit_to_driver: bool = True,
    ):
        """Initialize a pre-rendered frame event.

        Args:
            timestamp_us: Simulation timestamp when this frame should be
                delivered.
            frame: Image payload and metadata returned by the video model.
            camera_logical_id: Logical camera stream to update in rollout state.
            driver: Driver service that receives the image observation.
            should_submit_to_driver: Whether to forward this frame to the
                driver. Debug streams can update freshness without becoming
                policy inputs.
        """
        super().__init__(timestamp_us=timestamp_us)
        self.frame = frame
        self.camera_logical_id = camera_logical_id
        self.driver = driver
        self.should_submit_to_driver = should_submit_to_driver

    def description(self) -> str:
        """Return a short event-queue description.

        Returns:
            Human-readable event summary for logs and failure diagnostics.
        """
        return (
            f"VideoModelFrameEvent({self.camera_logical_id}, "
            f"submit={self.should_submit_to_driver})"
        )

    async def handle(self, state: RolloutState, queue: EventQueue) -> None:
        """Submit this frame to the driver and update camera freshness state.

        Args:
            state: Mutable rollout state to update with the latest camera time.
            queue: Event queue, unused because this event does not schedule
                follow-up work.
        """
        del queue
        if self.should_submit_to_driver:
            await self.driver.submit_image(self.frame)
        state.last_camera_frame_us[self.camera_logical_id] = self.timestamp_us


class VideoModelPrefetchEvent(Event):
    """Prefetches a chunk of frames from the video model and emits frame events.

    ``timestamp_us`` is when the prefetch runs.  For follow-up chunks it is the
    previous chunk's last generated frame timestamp, while ``chunk_start_us`` is
    the next frame timestamp requested from the server.

    Priority is intentionally above ``EventPriority.STEP`` (80) so this fires
    *after* :class:`StepEvent` has committed the current control step's
    propagated poses to ``state.ego_trajectory``. Otherwise the prefetch
    would read a trajectory that ends at the previous control step and
    every chunk frame would clamp to that one pose -- visible as ego
    snapping between two locations as chunks advance. With this ordering,
    ``state.ego_trajectory`` always extends through
    ``timestamp_us+control_timestep`` by the time we sample the requested
    generated frames.  For regular chunks the request starts one frame after
    the prefetch timestamp and ends one control step after it, so interpolation
    stays bounds-respecting.
    """

    priority: int = 90

    def __init__(
        self,
        timestamp_us: int,
        chunk_start_us: int,
        chunk_size: int,
        is_first_chunk: bool,
        video_model: VideoModelService,
        broadcaster: MessageBroadcaster,
        runtime_cameras: list[RuntimeCamera],
        driver: DriverService,
    ):
        """Initialize a chunk prefetch event.

        Args:
            timestamp_us: Simulation timestamp at which this prefetch runs.
            chunk_start_us: Timestamp of the first frame requested from the
                video model.
            chunk_size: Number of frames to request in this chunk.
            is_first_chunk: Whether this is the server's short initial chunk.
            video_model: Renderer service that owns the remote session.
            broadcaster: Rollout message broadcaster, reserved for future
                request/response logging.
            runtime_cameras: Runtime camera config for the active rollout.
            driver: Driver service that receives scheduled frame events.
        """
        super().__init__(timestamp_us=timestamp_us)
        self.chunk_start_us = chunk_start_us
        self.chunk_size = chunk_size
        self.is_first_chunk = is_first_chunk
        self.video_model = video_model
        self.broadcaster = broadcaster
        self.runtime_cameras = runtime_cameras
        self.driver = driver

    def description(self) -> str:
        """Return a short event-queue description.

        Returns:
            Human-readable event summary for logs and failure diagnostics.
        """
        return (
            f"VideoModelPrefetchEvent(chunk_start={self.chunk_start_us:_}, "
            f"size={self.chunk_size})"
        )

    async def handle(self, state: RolloutState, queue: EventQueue) -> None:
        """Render a chunk, enqueue its frames, and schedule the next prefetch.

        Args:
            state: Current rollout state, including the ego trajectory used to
                build the server request.
            queue: Event queue to receive frame events and the next prefetch.
        """
        frame_interval_us = self.video_model.frame_interval_us
        prefetch_t0 = time.perf_counter() if self.is_first_chunk else 0.0

        trajectory = build_trajectory_for_video_model(
            state.ego_trajectory,
            self.chunk_start_us,
            self.chunk_size,
            frame_interval_us,
        )
        chunk_result = await self.video_model.render_chunk(
            trajectory_local_to_rig=trajectory
        )
        self._emit_frames(chunk_result, queue)

        if self.is_first_chunk:
            logger.info(
                "SCENE_LOAD_TIMING scene=%s phase=first_prefetch_total elapsed=%.3fs",
                state.unbound.scene_id,
                time.perf_counter() - prefetch_t0,
            )

        self._schedule_next_prefetch(queue, frame_interval_us)

    def _schedule_next_prefetch(
        self, queue: EventQueue, frame_interval_us: int
    ) -> None:
        next_start = self.chunk_start_us + self.chunk_size * frame_interval_us
        next_prefetch_us = next_start - frame_interval_us
        queue.submit(
            VideoModelPrefetchEvent(
                timestamp_us=next_prefetch_us,
                chunk_start_us=next_start,
                chunk_size=self.video_model.chunk_size,
                is_first_chunk=False,
                video_model=self.video_model,
                broadcaster=self.broadcaster,
                runtime_cameras=self.runtime_cameras,
                driver=self.driver,
            )
        )

    def _emit_frames(self, chunk_result: ChunkResult, queue: EventQueue) -> None:
        for cam_id, frames in chunk_result.rgb_frames_per_camera.items():
            mask = self._build_forwarding_mask(frames)
            for frame, submit in zip(frames, mask, strict=True):
                queue.submit(
                    VideoModelFrameEvent(
                        timestamp_us=frame.start_timestamp_us,
                        frame=frame,
                        camera_logical_id=cam_id,
                        driver=self.driver,
                        should_submit_to_driver=submit,
                    )
                )

        if self.video_model.config.forward_hdmap_to_driver:
            # Each hdmap frame's camera_logical_id already carries the
            # ``hdmap_<camera>`` prefix (set in ``VideoModelService.render_chunk``),
            # so we don't need the dict key here.
            for frames in chunk_result.hdmap_frames_per_camera.values():
                mask = self._build_forwarding_mask(frames)
                for frame, submit in zip(frames, mask, strict=True):
                    queue.submit(
                        VideoModelFrameEvent(
                            timestamp_us=frame.start_timestamp_us,
                            frame=frame,
                            camera_logical_id=frame.camera_logical_id,
                            driver=self.driver,
                            should_submit_to_driver=submit,
                        )
                    )

        if self.video_model.config.forward_bev_to_driver:
            mask = self._build_forwarding_mask(chunk_result.bev_frames)
            for frame, submit in zip(chunk_result.bev_frames, mask, strict=True):
                queue.submit(
                    VideoModelFrameEvent(
                        timestamp_us=frame.start_timestamp_us,
                        frame=frame,
                        camera_logical_id=frame.camera_logical_id,
                        driver=self.driver,
                        should_submit_to_driver=submit,
                    )
                )

    def _build_forwarding_mask(self, frames: list[ImageWithMetadata]) -> list[bool]:
        if not frames:
            return []
        if self.video_model.config.frame_forwarding_mode != "subsample":
            return [True] * len(frames)

        timestamps = np.array([f.start_timestamp_us for f in frames], dtype=np.int64)
        sc = self.video_model.config.subsample_count
        si = self.video_model.config.subsample_interval_us
        if sc <= 0 or si <= 0:
            return [True] * len(frames)

        sorted_ts = np.sort(timestamps)
        targets = sorted_ts[-1] - np.arange(sc, dtype=np.int64) * si
        dists = np.abs(sorted_ts[:, None] - targets[None, :])
        min_dists = dists.min(axis=0, keepdims=True)
        nearest = dists == min_dists
        rev_idx = np.argmax(nearest[::-1, :], axis=0)
        nearest_idx = (len(sorted_ts) - 1) - rev_idx
        selected = sorted_ts[nearest_idx]
        return np.isin(timestamps, selected).tolist()


def make_initial_video_model_render_event(
    *,
    scene_start_us: int,
    render_start_timestamp_us: int | None = None,
    closed_loop_start_us: int | None = None,
    simulation_end_us: int | None = None,
    control_timestep_us: int,
    runtime_cameras: list[RuntimeCamera],
    renderer_service: Any,
    driver: DriverService,
    broadcaster: MessageBroadcaster,
    use_aggregated_render: bool = False,
) -> Event:
    """Plugin factory for the initial video-model prefetch event.

    Called by :class:`alpasim_video_model.service.VideoModelService` so core's
    ``EventBasedRollout`` can instantiate it uniformly alongside the built-in
    sensorsim renderer.

    ``render_start_timestamp_us`` is the centralized render anchor: the USDZ
    first frame supplied during session start corresponds to that timestamp.
    Generated frames begin one frame interval later, so the initial prefetch
    runs at the render anchor with
    ``chunk_start_us=render_start_timestamp_us + frame_interval_us``.

    Args:
        scene_start_us: Start of the rollout egomotion context.
        render_start_timestamp_us: Central render anchor for the rollout.
        control_timestep_us: Regular chunk/control step duration. For the
            video model this must match ``chunk_frames * frame_interval_us``.
        runtime_cameras: Runtime camera definitions for the active rollout.
        renderer_service: Active :class:`VideoModelService` instance.
        driver: Driver service receiving scheduled frame events.
        broadcaster: Rollout broadcaster, reserved for future logging.
        use_aggregated_render: Built-in sensorsim compatibility flag, unused
            for the chunked video model renderer.

    Returns:
        The first :class:`VideoModelPrefetchEvent`.
    """
    del simulation_end_us, closed_loop_start_us, use_aggregated_render
    render_anchor_us = (
        render_start_timestamp_us
        if render_start_timestamp_us is not None
        else scene_start_us
    )
    first_chunk_size = renderer_service.chunk_size
    frame_interval_us = renderer_service.frame_interval_us
    regular_chunk_duration_us = renderer_service.config.chunk_frames * frame_interval_us
    if control_timestep_us != regular_chunk_duration_us:
        raise ValueError(
            "For renderer_type='video_model', control_timestep_us "
            f"({control_timestep_us}) must equal chunk_frames * "
            f"frame_interval_us ({regular_chunk_duration_us})."
        )
    initial_chunk_start_us = render_anchor_us + frame_interval_us
    return VideoModelPrefetchEvent(
        timestamp_us=render_anchor_us,
        chunk_start_us=initial_chunk_start_us,
        chunk_size=first_chunk_size,
        is_first_chunk=True,
        video_model=renderer_service,
        broadcaster=broadcaster,
        runtime_cameras=runtime_cameras,
        driver=driver,
    )
