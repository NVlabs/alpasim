# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for GroupedRenderEvent (both aggregated and parallel render modes)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.camera import GroupedRenderEvent
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


class TestGroupedRenderEventParallelMode:
    """Tests for GroupedRenderEvent with use_aggregated_render=False (parallel individual RPCs)."""

    @pytest.fixture
    def cameras(self) -> list[RuntimeCamera]:
        return [_make_camera("cam_front"), _make_camera("cam_rear")]

    @pytest.fixture
    def parallel_event(
        self,
        cameras: list[RuntimeCamera],
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ) -> GroupedRenderEvent:
        return GroupedRenderEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            cameras=cameras,
            sensorsim=mock_sensorsim,
            driver=mock_driver,
            scene_start_us=0,
            use_aggregated_render=False,
        )

    def test_priority(self, parallel_event: GroupedRenderEvent):
        assert parallel_event.priority == 10

    def test_description(self, parallel_event: GroupedRenderEvent):
        desc = parallel_event.description()
        assert "GroupedRenderEvent" in desc

    @pytest.mark.asyncio
    async def test_parallel_render_calls_individual_render_per_camera(
        self,
        parallel_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
    ):
        """Each camera trigger should result in an individual render() call."""
        fake_image = MagicMock(spec=ImageWithMetadata)
        mock_sensorsim.render.return_value = fake_image

        queue = EventQueue()
        await parallel_event.run(rollout_state, queue)

        # With 2 cameras that both have triggers in (0, 100_000], render
        # should be called once per camera trigger.
        assert mock_sensorsim.render.await_count >= 1
        mock_sensorsim.aggregated_render.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_parallel_render_submits_images_to_driver(
        self,
        parallel_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ):
        """All rendered images should be submitted to the driver."""
        fake_image = MagicMock(spec=ImageWithMetadata)
        mock_sensorsim.render.return_value = fake_image

        queue = EventQueue()
        await parallel_event.run(rollout_state, queue)

        render_count = mock_sensorsim.render.await_count
        # Drain tracked tasks to complete submissions
        await rollout_state.step_context.drain_outstanding_tasks()
        assert mock_driver.submit_image.await_count == render_count

    @pytest.mark.asyncio
    async def test_image_submissions_are_tracked_as_tasks(
        self,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ):
        """submit_image calls should be tracked as outstanding tasks on StepContext."""
        cameras = [_make_camera("cam_front")]
        fake_image = MagicMock(spec=ImageWithMetadata)
        mock_sensorsim.render.return_value = fake_image

        event = GroupedRenderEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            cameras=cameras,
            sensorsim=mock_sensorsim,
            driver=mock_driver,
            scene_start_us=0,
            use_aggregated_render=False,
        )
        queue = EventQueue()
        await event.run(rollout_state, queue)

        # Image submissions are tracked as tasks, not directly awaited
        assert len(rollout_state.step_context.outstanding_tasks) >= 1

        # Drain completes the submissions
        await rollout_state.step_context.drain_outstanding_tasks()
        mock_driver.submit_image.assert_awaited()

    @pytest.mark.asyncio
    async def test_parallel_render_does_not_set_driver_data(
        self,
        parallel_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
    ):
        """Parallel mode uses individual render(), which has no driver_data."""
        mock_sensorsim.render.return_value = MagicMock(spec=ImageWithMetadata)
        rollout_state.data_sensorsim_to_driver = None

        queue = EventQueue()
        await parallel_event.run(rollout_state, queue)

        assert rollout_state.data_sensorsim_to_driver is None

    @pytest.mark.asyncio
    async def test_parallel_render_tracks_last_camera_frame(
        self,
        parallel_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
    ):
        """Camera frame timestamps should be tracked for sync validation."""
        mock_sensorsim.render.return_value = MagicMock(spec=ImageWithMetadata)

        queue = EventQueue()
        await parallel_event.run(rollout_state, queue)

        # At least one camera should have its frame time tracked
        assert len(rollout_state.last_camera_frame_us) > 0

    @pytest.mark.asyncio
    async def test_no_triggers_skips_render(
        self,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ):
        """When no triggers fall in the window, no render calls are made."""
        camera = MagicMock()
        camera.clock = MagicMock()
        camera.clock.triggers_completed_in_range.return_value = []

        event = GroupedRenderEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            cameras=[camera],
            sensorsim=mock_sensorsim,
            driver=mock_driver,
            scene_start_us=0,
            use_aggregated_render=False,
        )

        queue = EventQueue()
        await event.run(rollout_state, queue)

        mock_sensorsim.render.assert_not_awaited()
        mock_driver.submit_image.assert_not_awaited()


class TestGroupedRenderEventAggregatedMode:
    """Tests for GroupedRenderEvent with use_aggregated_render=True (single batched RPC)."""

    @pytest.fixture
    def aggregated_event(
        self,
        runtime_camera: RuntimeCamera,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ) -> GroupedRenderEvent:
        return GroupedRenderEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            cameras=[runtime_camera],
            sensorsim=mock_sensorsim,
            driver=mock_driver,
            scene_start_us=0,
            use_aggregated_render=True,
        )

    @pytest.mark.asyncio
    async def test_aggregated_render_calls_aggregated_rpc(
        self,
        aggregated_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
    ):
        """Aggregated mode should call aggregated_render, not individual render."""
        mock_sensorsim.aggregated_render.return_value = ([], None)

        queue = EventQueue()
        await aggregated_event.run(rollout_state, queue)

        mock_sensorsim.aggregated_render.assert_awaited_once()
        mock_sensorsim.render.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_submissions_are_tracked_as_tasks_aggregated(
        self,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
        runtime_camera: RuntimeCamera,
    ):
        """submit_image calls should be tracked as outstanding tasks in aggregated mode."""
        fake_image = MagicMock(spec=ImageWithMetadata)
        mock_sensorsim.aggregated_render.return_value = ([fake_image], None)

        event = GroupedRenderEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            cameras=[runtime_camera],
            sensorsim=mock_sensorsim,
            driver=mock_driver,
            scene_start_us=0,
            use_aggregated_render=True,
        )
        queue = EventQueue()
        await event.run(rollout_state, queue)

        assert len(rollout_state.step_context.outstanding_tasks) >= 1

        await rollout_state.step_context.drain_outstanding_tasks()
        mock_driver.submit_image.assert_awaited()

    @pytest.mark.asyncio
    async def test_aggregated_render_stores_driver_data(
        self,
        aggregated_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
    ):
        """Aggregated mode stores driver_data on rollout_state."""
        driver_data_blob = b"aggregated_payload"
        mock_sensorsim.aggregated_render.return_value = ([], driver_data_blob)

        queue = EventQueue()
        await aggregated_event.run(rollout_state, queue)

        assert rollout_state.data_sensorsim_to_driver == driver_data_blob
