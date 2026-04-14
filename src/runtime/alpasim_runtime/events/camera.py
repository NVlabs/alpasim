# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Camera render event for event-based simulation loop.

GroupedRenderEvent fires once per control step, collects all camera triggers
whose shutters closed in the (past_us, now_us] window, and renders them.
Two render strategies are supported:
- Aggregated: single ``aggregated_render`` RPC bundling all cameras.
- Parallel: individual ``render`` RPCs dispatched concurrently via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging

from alpasim_runtime.events.base import EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils import geometry

logger = logging.getLogger(__name__)


def _remember_rendered_images(
    state: RolloutState,
    images: list,
) -> None:
    for image in images:
        state.last_rendered_images[image.camera_logical_id] = image


class GroupedRenderEvent(RecurringEvent):
    """Render all cameras whose shutters closed within a control step window.

    When ``use_aggregated_render`` is True, issues a single
    ``aggregated_render`` RPC that bundles all cameras.  When False,
    dispatches individual ``render`` RPCs concurrently via
    ``asyncio.gather``.
    """

    priority: int = EventPriority.CAMERA

    def __init__(
        self,
        timestamp_us: int,
        control_timestep_us: int,
        cameras: list[RuntimeCamera],
        sensorsim: SensorsimService,
        driver: DriverService,
        scene_start_us: int,
        use_aggregated_render: bool = False,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = control_timestep_us
        self.cameras = cameras
        self.sensorsim = sensorsim
        self.driver = driver
        self.use_aggregated_render = use_aggregated_render
        # Track the previous step boundary for trigger collection
        self._prev_timestamp_us = scene_start_us

    def description(self) -> str:
        return f"GroupedRenderEvent(now={self.timestamp_us:_}us)"

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        """Collect camera triggers, render images, forward to driver."""
        past_us = self._prev_timestamp_us
        now_us = self.timestamp_us

        # Determine which triggers completed in the (past_us, now_us] window
        skip_straddles = past_us == (
            state.unbound.start_timestamp_us + state.unbound.time_start_offset_us
        )
        camera_triggers: list[tuple[RuntimeCamera, Clock.Trigger]] = []
        for camera in self.cameras:
            triggers = camera.clock.triggers_completed_in_range(
                range(past_us, now_us), skip_straddles
            )
            camera_triggers.extend([(camera, trigger) for trigger in triggers])

        if not camera_triggers:
            # No triggers to render this step
            self._prev_timestamp_us = now_us
            return

        # Build traffic trajectories
        traffic_trajs: dict[str, geometry.Trajectory] = {
            track_id: obj.trajectory
            for track_id, obj in state.traffic_objs.items()
            if not obj.is_static
        }
        if state.unbound.hidden_traffic_objs:
            for hid, hobj in state.unbound.hidden_traffic_objs.items():
                traffic_trajs[hid] = hobj.trajectory

        if self.use_aggregated_render:
            await self._render_aggregated(state, camera_triggers, traffic_trajs)
        else:
            await self._render_parallel(state, camera_triggers, traffic_trajs)

        # Track camera frame times for sync validation
        for camera, trigger in camera_triggers:
            state.last_camera_frame_us[camera.logical_id] = trigger.time_range_us.stop

        self._prev_timestamp_us = now_us

    async def _render_aggregated(
        self,
        state: RolloutState,
        camera_triggers: list[tuple[RuntimeCamera, Clock.Trigger]],
        traffic_trajs: dict[str, geometry.Trajectory],
    ) -> None:
        """Single aggregated_render RPC for all cameras."""
        assert (
            state.step_context is not None
        ), "StepContext must exist before camera render"
        images_with_metadata, driver_data = await self.sensorsim.aggregated_render(
            camera_triggers,
            ego_trajectory=state.ego_trajectory,
            traffic_trajectories=traffic_trajs,
            scene_id=state.unbound.scene_id,
            image_format=state.unbound.image_format,
            ego_mask_rig_config_id=state.unbound.ego_mask_rig_config_id,
        )

        if state.rendered_images_handler is not None:
            await state.rendered_images_handler(images_with_metadata)

        _remember_rendered_images(state, images_with_metadata)

        for image in images_with_metadata:
            state.step_context.track_task(self.driver.submit_image(image))

        state.data_sensorsim_to_driver = driver_data

    async def _render_parallel(
        self,
        state: RolloutState,
        camera_triggers: list[tuple[RuntimeCamera, Clock.Trigger]],
        traffic_trajs: dict[str, geometry.Trajectory],
    ) -> None:
        """Individual render RPCs dispatched concurrently via asyncio.gather."""
        assert (
            state.step_context is not None
        ), "StepContext must exist before camera render"
        images = await asyncio.gather(
            *[
                self.sensorsim.render(
                    ego_trajectory=state.ego_trajectory,
                    traffic_trajectories=traffic_trajs,
                    trigger=trigger,
                    camera=camera,
                    scene_id=state.unbound.scene_id,
                    image_format=state.unbound.image_format,
                    ego_mask_rig_config_id=state.unbound.ego_mask_rig_config_id,
                )
                for camera, trigger in camera_triggers
            ]
        )

        if state.rendered_images_handler is not None:
            await state.rendered_images_handler(images)

        _remember_rendered_images(state, images)

        for image in images:
            state.step_context.track_task(self.driver.submit_image(image))
