# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Camera render events for the event-based simulation loop."""

from __future__ import annotations

import logging
from typing import Any

from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.events.base import Event, EventPriority, EventQueue
from alpasim_runtime.events.state import RolloutState
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils import geometry

logger = logging.getLogger(__name__)


def _traffic_trajectories(state: RolloutState) -> dict[str, geometry.Trajectory]:
    traffic_trajs: dict[str, geometry.Trajectory] = {
        track_id: obj.trajectory
        for track_id, obj in state.traffic_objs.items()
        if not obj.is_static
    }
    if state.unbound.hidden_traffic_objs:
        for hid, hobj in state.unbound.hidden_traffic_objs.items():
            traffic_trajs[hid] = hobj.trajectory
    return traffic_trajs


class CameraFrameEvent(Event):
    """Render or register one camera frame at its shutter-close timestamp."""

    priority: int = EventPriority.CAMERA

    def __init__(
        self,
        camera: RuntimeCamera,
        trigger: Clock.Trigger,
        sensorsim: SensorsimService,
        driver: DriverService,
        use_aggregated_render: bool = False,
    ):
        super().__init__(timestamp_us=trigger.time_range_us.stop)
        self.camera = camera
        self.trigger = trigger
        self.sensorsim = sensorsim
        self.driver = driver
        self.use_aggregated_render = use_aggregated_render

    def description(self) -> str:
        return (
            f"CameraFrameEvent({self.camera.logical_id}, "
            f"{self.trigger.time_range_us.start:_}->{self.trigger.time_range_us.stop:_}us)"
        )

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        if self.use_aggregated_render:
            self._register_for_aggregated_render(rollout_state, queue)
        else:
            await self._render_immediately(rollout_state)

        self._record_frame(rollout_state)
        self._schedule_next(rollout_state, queue)

    def _register_for_aggregated_render(
        self, state: RolloutState, queue: EventQueue
    ) -> None:
        state.pending_camera_triggers.setdefault(self.timestamp_us, []).append(
            (self.camera, self.trigger)
        )
        if self.timestamp_us not in state.pending_camera_flush_timestamps:
            state.pending_camera_flush_timestamps.add(self.timestamp_us)
            queue.submit(
                CameraRenderFlushEvent(
                    timestamp_us=self.timestamp_us,
                    sensorsim=self.sensorsim,
                    driver=self.driver,
                )
            )

    async def _render_immediately(self, state: RolloutState) -> None:
        assert state.step_context is not None, "StepContext must exist before render"
        render_coro = self.sensorsim.render(
            ego_trajectory=state.ego_trajectory,
            traffic_trajectories=_traffic_trajectories(state),
            trigger=self.trigger,
            camera=self.camera,
            scene_id=state.unbound.scene_id,
            image_format=state.unbound.image_format,
            ego_mask_rig_config_id=state.unbound.ego_mask_rig_config_id,
        )
        image = await render_coro

        state.step_context.track_task(self.driver.submit_image(image))

    def _record_frame(self, state: RolloutState) -> None:
        state.last_camera_frame_us[self.camera.logical_id] = (
            self.trigger.time_range_us.stop
        )
        state.last_camera_frame_start_us[self.camera.logical_id] = (
            self.trigger.time_range_us.start
        )

    def _schedule_next(self, state: RolloutState, queue: EventQueue) -> None:
        next_trigger = self.camera.clock.ith_trigger(self.trigger.sequential_idx + 1)
        if next_trigger.time_range_us.stop > state.unbound.end_timestamp_us:
            return
        queue.submit(
            CameraFrameEvent(
                camera=self.camera,
                trigger=next_trigger,
                sensorsim=self.sensorsim,
                driver=self.driver,
                use_aggregated_render=self.use_aggregated_render,
            )
        )


class CameraRenderFlushEvent(Event):
    """Render all registered camera frames sharing one frame-end timestamp."""

    priority: int = EventPriority.CAMERA_FLUSH

    def __init__(
        self,
        timestamp_us: int,
        sensorsim: SensorsimService,
        driver: DriverService,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.sensorsim = sensorsim
        self.driver = driver

    def description(self) -> str:
        return f"CameraRenderFlushEvent(now={self.timestamp_us:_}us)"

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        del queue
        assert (
            rollout_state.step_context is not None
        ), "StepContext must exist before render"
        camera_triggers = rollout_state.pending_camera_triggers.pop(
            self.timestamp_us, []
        )
        rollout_state.pending_camera_flush_timestamps.discard(self.timestamp_us)
        if not camera_triggers:
            return

        render_coro = self.sensorsim.aggregated_render(
            camera_triggers,
            ego_trajectory=rollout_state.ego_trajectory,
            traffic_trajectories=_traffic_trajectories(rollout_state),
            scene_id=rollout_state.unbound.scene_id,
            image_format=rollout_state.unbound.image_format,
            ego_mask_rig_config_id=rollout_state.unbound.ego_mask_rig_config_id,
        )
        images_with_metadata, driver_data = await render_coro

        for image in images_with_metadata:
            rollout_state.step_context.track_task(self.driver.submit_image(image))

        rollout_state.data_sensorsim_to_driver = driver_data


def make_initial_sensorsim_render_event(
    *,
    scene_start_us: int,
    render_start_timestamp_us: int,
    closed_loop_start_us: int,
    simulation_end_us: int,
    control_timestep_us: int,
    runtime_cameras: list[RuntimeCamera],
    renderer_service: Any,
    driver: DriverService,
    broadcaster: MessageBroadcaster,
    use_aggregated_render: bool = False,
) -> list[Event]:
    """Built-in factory for initial sensorsim camera frame events.

    Each ``RuntimeCamera.clock`` already carries its first shutter range,
    including any zero-decision-delay normalization, so sensorsim starts from
    those authoritative per-camera ranges.
    """
    del (
        render_start_timestamp_us,
        closed_loop_start_us,
        control_timestep_us,
        broadcaster,
    )
    return [
        CameraFrameEvent(
            camera=camera,
            trigger=trigger,
            sensorsim=renderer_service,
            driver=driver,
            use_aggregated_render=use_aggregated_render,
        )
        for camera in runtime_cameras
        for trigger in [camera.clock.ith_trigger(0)]
        if scene_start_us <= trigger.time_range_us.stop <= simulation_end_us
    ]
