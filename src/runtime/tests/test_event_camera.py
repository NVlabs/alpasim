# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for camera frame render events."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_runtime.config import RenderBundling
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.camera import (
    CameraFrameEvent,
    CameraRenderFlushEvent,
    make_initial_sensorsim_render_event,
)
from alpasim_runtime.events.state import RolloutState
from alpasim_runtime.force_gt_frame_cache import ForceGtFrameCache, ForceGtFrameKey
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils.types import ImageWithMetadata


def _make_camera(logical_id: str) -> RuntimeCamera:
    return RuntimeCamera(
        logical_id=logical_id,
        render_resolution_hw=(720, 1280),
        clock=Clock(
            interval_us=100_000,
            duration_us=33_000,
            start_us=0,
        ),
    )


@pytest.mark.asyncio
async def test_non_aggregated_camera_event_renders_immediately(
    rollout_state: RolloutState,
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
):
    camera = _make_camera("cam_front")
    fake_image = MagicMock(spec=ImageWithMetadata)
    mock_sensorsim.render.return_value = fake_image

    event = CameraFrameEvent(
        camera=camera,
        trigger=camera.clock.ith_trigger(0),
        sensorsim=mock_sensorsim,
        driver=mock_driver,
    )

    queue = EventQueue()
    await event.handle(rollout_state, queue)

    mock_sensorsim.render.assert_awaited_once()
    mock_sensorsim.aggregated_render.assert_not_awaited()
    assert rollout_state.last_camera_frame_us["cam_front"] == 33_000
    assert len(rollout_state.step_context.outstanding_tasks) == 1

    await rollout_state.step_context.drain_outstanding_tasks()
    mock_driver.submit_image.assert_awaited_once_with(fake_image)
    assert len(queue) == 1


@pytest.mark.asyncio
async def test_aggregated_camera_events_flush_same_timestamp_together(
    rollout_state: RolloutState,
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
):
    cam_front = _make_camera("cam_front")
    cam_rear = _make_camera("cam_rear")
    trigger_front = Clock.Trigger(range(0, 33_000), sequential_idx=0)
    trigger_rear = Clock.Trigger(range(1_000, 33_000), sequential_idx=0)
    fake_image = MagicMock(spec=ImageWithMetadata)
    mock_sensorsim.aggregated_render.return_value = ([fake_image], b"driver-data")
    rollout_state.unbound.render_bundling = RenderBundling.RENDER_AGGREGATED

    queue = EventQueue()
    await CameraFrameEvent(
        cam_front, trigger_front, mock_sensorsim, mock_driver
    ).handle(rollout_state, queue)
    await CameraFrameEvent(cam_rear, trigger_rear, mock_sensorsim, mock_driver).handle(
        rollout_state, queue
    )

    assert len(rollout_state.pending_camera_triggers[33_000]) == 2
    assert len(queue) == 3  # one flush plus one next frame per camera

    flush = next(
        event for event in queue.queue if isinstance(event, CameraRenderFlushEvent)
    )
    await flush.handle(rollout_state, queue)

    mock_sensorsim.aggregated_render.assert_awaited_once()
    mock_sensorsim.render.assert_not_awaited()
    assert rollout_state.data_sensorsim_to_driver == b"driver-data"

    await rollout_state.step_context.drain_outstanding_tasks()
    mock_driver.submit_image.assert_awaited_once_with(fake_image)


@pytest.mark.asyncio
async def test_bundled_flush_uses_batch_render_when_flag_set(
    rollout_state: RolloutState,
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
):
    """render_bundling=BATCH_RENDER_RGB routes the flush to NRE batch_render."""
    rollout_state.unbound.render_bundling = RenderBundling.BATCH_RENDER_RGB
    camera = _make_camera("cam_front")
    trigger = Clock.Trigger(range(0, 33_000), sequential_idx=0)
    fake_image = MagicMock(spec=ImageWithMetadata)
    mock_sensorsim.batch_render.return_value = ([fake_image], None)

    queue = EventQueue()
    await CameraFrameEvent(camera, trigger, mock_sensorsim, mock_driver).handle(
        rollout_state, queue
    )
    flush = next(
        event for event in queue.queue if isinstance(event, CameraRenderFlushEvent)
    )
    await flush.handle(rollout_state, queue)

    mock_sensorsim.batch_render.assert_awaited_once()
    mock_sensorsim.aggregated_render.assert_not_awaited()
    await rollout_state.step_context.drain_outstanding_tasks()
    mock_driver.submit_image.assert_awaited_once_with(fake_image)


@pytest.mark.asyncio
async def test_camera_event_schedules_next_frame_ending_at_rollout_boundary(
    rollout_state: RolloutState,
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
):
    camera = _make_camera("cam_front")
    rollout_state.unbound.end_timestamp_us = 133_000
    mock_sensorsim.render.return_value = MagicMock(spec=ImageWithMetadata)

    queue = EventQueue()
    await CameraFrameEvent(
        camera=camera,
        trigger=camera.clock.ith_trigger(0),
        sensorsim=mock_sensorsim,
        driver=mock_driver,
    ).handle(rollout_state, queue)

    assert len(queue) == 1
    next_event = queue.queue[0]
    assert isinstance(next_event, CameraFrameEvent)
    assert next_event.trigger.time_range_us.stop == 133_000


def _image(camera: str, start: int, end: int, data: bytes) -> ImageWithMetadata:
    return ImageWithMetadata(
        start_timestamp_us=start,
        end_timestamp_us=end,
        image_bytes=data,
        camera_logical_id=camera,
    )


# Matches conftest ``unbound.image_format = 2`` (JPEG -> "jpeg" extension).
_SCENE_UUID = "scene-uuid-0001"
_RENDER_SIGNATURE = "jpeg-abc123"


def _frame_key(camera: str, start: int = 0, end: int = 33_000) -> ForceGtFrameKey:
    return ForceGtFrameKey(
        scene_uuid=_SCENE_UUID,
        render_signature=_RENDER_SIGNATURE,
        camera_logical_id=camera,
        frame_start_us=start,
        frame_end_us=end,
        extension="jpeg",
    )


def _enable_force_gt_cache(
    rollout_state: RolloutState,
    tmp_path,
    enabled: bool = True,
    period: range = range(0, 100_000),
    signature: str | None = _RENDER_SIGNATURE,
) -> None:
    rollout_state.unbound.force_gt_period = period
    rollout_state.unbound.use_cached_frames_during_force_gt = enabled
    rollout_state.force_gt_frame_cache = ForceGtFrameCache(tmp_path)
    rollout_state.force_gt_scene_uuid = _SCENE_UUID if signature else None
    rollout_state.force_gt_render_signature = signature


@pytest.mark.asyncio
async def test_force_gt_immediate_render_populates_then_replays_cache(
    rollout_state: RolloutState,
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
    tmp_path,
):
    """First force-GT render hits the renderer; a fresh cache replays from disk."""
    _enable_force_gt_cache(rollout_state, tmp_path)
    camera = _make_camera("cam_front")
    trigger = camera.clock.ith_trigger(0)  # ends at 33_000, within force-GT
    mock_sensorsim.render.return_value = _image("cam_front", 0, 33_000, b"px")

    await CameraFrameEvent(camera, trigger, mock_sensorsim, mock_driver).handle(
        rollout_state, EventQueue()
    )
    await rollout_state.step_context.drain_outstanding_tasks()

    mock_sensorsim.render.assert_awaited_once()
    assert ForceGtFrameCache(tmp_path).get(_frame_key("cam_front")) == b"px"

    # A separate cache instance (another worker/run on the same mount) replays.
    mock_driver.submit_image.reset_mock()
    rollout_state.force_gt_frame_cache = ForceGtFrameCache(tmp_path)
    await CameraFrameEvent(camera, trigger, mock_sensorsim, mock_driver).handle(
        rollout_state, EventQueue()
    )
    await rollout_state.step_context.drain_outstanding_tasks()

    mock_sensorsim.render.assert_awaited_once()  # still only the first call
    assert mock_driver.submit_image.await_args.args[0].image_bytes == b"px"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"period": range(0, 10_000)},  # frame ends at 33_000: closed-loop, not cached
        # Caching off resolves to a missing signature (set upstream in event_loop).
        {"signature": None},
    ],
)
@pytest.mark.asyncio
async def test_force_gt_render_not_cached(
    rollout_state: RolloutState,
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
    tmp_path,
    kwargs,
):
    _enable_force_gt_cache(rollout_state, tmp_path, **kwargs)
    camera = _make_camera("cam_front")
    trigger = camera.clock.ith_trigger(0)
    mock_sensorsim.render.return_value = _image("cam_front", 0, 33_000, b"px")

    await CameraFrameEvent(camera, trigger, mock_sensorsim, mock_driver).handle(
        rollout_state, EventQueue()
    )

    mock_sensorsim.render.assert_awaited_once()
    assert rollout_state.force_gt_frame_cache.get(_frame_key("cam_front")) is None


@pytest.mark.asyncio
async def test_force_gt_bundled_flush_populates_then_replays_cache(
    rollout_state: RolloutState,
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
    tmp_path,
):
    """Bundled force-GT flush renders once, then replays both cameras from cache."""
    _enable_force_gt_cache(rollout_state, tmp_path)
    rollout_state.unbound.render_bundling = RenderBundling.RENDER_AGGREGATED

    cam_front = _make_camera("cam_front")
    cam_rear = _make_camera("cam_rear")
    trigger = Clock.Trigger(range(0, 33_000), sequential_idx=0)
    mock_sensorsim.aggregated_render.return_value = (
        [_image("cam_front", 0, 33_000, b"f"), _image("cam_rear", 0, 33_000, b"r")],
        b"driver-data",
    )

    async def run_flush() -> None:
        rollout_state.pending_camera_triggers.clear()
        rollout_state.pending_camera_flush_timestamps.clear()
        queue = EventQueue()
        for cam in (cam_front, cam_rear):
            await CameraFrameEvent(cam, trigger, mock_sensorsim, mock_driver).handle(
                rollout_state, queue
            )
        flush = next(e for e in queue.queue if isinstance(e, CameraRenderFlushEvent))
        await flush.handle(rollout_state, queue)

    await run_flush()
    mock_sensorsim.aggregated_render.assert_awaited_once()
    # The render (cache-miss) forwards driver_data normally; caching no longer
    # special-cases it (caching requires skip_driver_during_force_gt, so
    # PolicyEvent drops it during force-GT).
    assert rollout_state.data_sensorsim_to_driver == b"driver-data"
    assert rollout_state.force_gt_frame_cache.get(_frame_key("cam_front")) == b"f"
    assert rollout_state.force_gt_frame_cache.get(_frame_key("cam_rear")) == b"r"

    # Replay: a fresh flush serves both frames from cache without a render RPC.
    await run_flush()
    mock_sensorsim.aggregated_render.assert_awaited_once()  # no new render
    await rollout_state.step_context.drain_outstanding_tasks()
    submitted = {
        call.args[0].camera_logical_id: call.args[0].image_bytes
        for call in mock_driver.submit_image.await_args_list
    }
    assert submitted == {"cam_front": b"f", "cam_rear": b"r"}


def test_initial_sensorsim_render_events_skip_first_triggers_outside_rollout(
    mock_sensorsim: AsyncMock,
    mock_driver: AsyncMock,
):
    inside = RuntimeCamera(
        logical_id="inside",
        render_resolution_hw=(720, 1280),
        clock=Clock(interval_us=100_000, duration_us=33_000, start_us=0),
    )
    exact_end = RuntimeCamera(
        logical_id="exact_end",
        render_resolution_hw=(720, 1280),
        clock=Clock(interval_us=100_000, duration_us=33_000, start_us=67_000),
    )
    outside = RuntimeCamera(
        logical_id="outside",
        render_resolution_hw=(720, 1280),
        clock=Clock(interval_us=100_000, duration_us=33_000, start_us=100_000),
    )

    events = make_initial_sensorsim_render_event(
        scene_start_us=0,
        render_start_timestamp_us=0,
        closed_loop_start_us=100_000,
        simulation_end_us=100_000,
        control_timestep_us=100_000,
        runtime_cameras=[inside, exact_end, outside],
        renderer_service=mock_sensorsim,
        driver=mock_driver,
        broadcaster=MagicMock(),
    )

    assert [event.camera.logical_id for event in events] == ["inside", "exact_end"]
