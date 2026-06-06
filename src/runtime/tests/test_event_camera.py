# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for camera frame render events."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.camera import (
    CameraFrameEvent,
    CameraRenderFlushEvent,
    make_initial_sensorsim_render_event,
)
from alpasim_runtime.events.state import RolloutState
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
        use_aggregated_render=False,
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

    queue = EventQueue()
    await CameraFrameEvent(
        cam_front, trigger_front, mock_sensorsim, mock_driver, True
    ).handle(rollout_state, queue)
    await CameraFrameEvent(
        cam_rear, trigger_rear, mock_sensorsim, mock_driver, True
    ).handle(rollout_state, queue)

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
        use_aggregated_render=False,
    ).handle(rollout_state, queue)

    assert len(queue) == 1
    next_event = queue.queue[0]
    assert isinstance(next_event, CameraFrameEvent)
    assert next_event.trigger.time_range_us.stop == 133_000


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
