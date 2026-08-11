# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Camera render events for the event-based simulation loop."""

from __future__ import annotations

import logging
from typing import Any

from alpasim_grpc.v0.sensorsim_pb2 import ImageFormat
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.config import RenderBundling
from alpasim_runtime.events.base import Event, EventPriority, EventQueue
from alpasim_runtime.events.state import RolloutState
from alpasim_runtime.force_gt_frame_cache import ForceGtFrameKey
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils import geometry
from alpasim_utils.types import ImageWithMetadata

logger = logging.getLogger(__name__)


def _force_gt_cache_key(
    state: RolloutState,
    camera_logical_id: str,
    trigger: Clock.Trigger,
) -> ForceGtFrameKey | None:
    """Return the force-GT cache key for this frame, or ``None`` if uncacheable.

    ``force_gt_render_signature`` is set exactly when force-GT caching is enabled
    (feature on and a cache attached), so it alone gates cacheability. A frame is
    cacheable when it is present and the shutter closes within the force-GT
    period (where the ego follows the recorded trajectory).
    """
    if state.force_gt_render_signature is None:
        return None
    if trigger.time_range_us.stop not in state.unbound.force_gt_period:
        return None
    assert state.force_gt_scene_uuid is not None
    return ForceGtFrameKey(
        scene_uuid=state.force_gt_scene_uuid,
        render_signature=state.force_gt_render_signature,
        camera_logical_id=camera_logical_id,
        frame_start_us=trigger.time_range_us.start,
        frame_end_us=trigger.time_range_us.stop,
        extension=ImageFormat.Name(int(state.unbound.image_format)).lower(),
    )


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


async def _render_frames(
    state: RolloutState,
    sensorsim: SensorsimService,
    camera_triggers: list[tuple[RuntimeCamera, Clock.Trigger]],
) -> tuple[list[ImageWithMetadata], bytes | None]:
    """Render a group of camera frames via the RPC selected by ``render_bundling``.

    Returns ``(images, driver_data)``; the per-camera ``render_rgb`` path carries
    no renderer->driver payload, so its ``driver_data`` is always ``None``.
    """
    bundling = state.unbound.render_bundling
    if bundling == RenderBundling.NONE:
        images = [
            await sensorsim.render(
                ego_trajectory=state.ego_trajectory,
                traffic_trajectories=_traffic_trajectories(state),
                trigger=trigger,
                camera=camera,
                scene_id=state.unbound.scene_id,
                image_format=state.unbound.image_format,
                ego_mask_rig_config_id=state.unbound.ego_mask_rig_config_id,
            )
            for camera, trigger in camera_triggers
        ]
        return images, None
    else:
        bundled_render = (
            sensorsim.batch_render
            if bundling == RenderBundling.BATCH_RENDER_RGB
            else sensorsim.aggregated_render
        )
        return await bundled_render(
            camera_triggers,
            ego_trajectory=state.ego_trajectory,
            traffic_trajectories=_traffic_trajectories(state),
            scene_id=state.unbound.scene_id,
            image_format=state.unbound.image_format,
            ego_mask_rig_config_id=state.unbound.ego_mask_rig_config_id,
        )


async def _render_or_replay(
    state: RolloutState,
    sensorsim: SensorsimService,
    driver: DriverService,
    camera_triggers: list[tuple[RuntimeCamera, Clock.Trigger]],
) -> None:
    """Submit each camera frame to the driver, serving it from the force-GT cache
    when possible and rendering (then caching) otherwise.

    Caching applies only when every frame in the group is a force-GT frame (has a
    key). A full cache replay carries no renderer payload; freshly rendered frames
    forward theirs via ``data_sensorsim_to_driver`` (caching requires
    ``skip_driver_during_force_gt``, so on a replay it is never consumed).
    """
    assert state.step_context is not None, "StepContext must exist before render"
    cache = state.force_gt_frame_cache
    keys = {
        camera.logical_id: _force_gt_cache_key(state, camera.logical_id, trigger)
        for camera, trigger in camera_triggers
    }
    is_cacheable = cache is not None and all(k is not None for k in keys.values())

    cached_frames = (
        {cam.logical_id: cache.get(keys[cam.logical_id]) for cam, _ in camera_triggers}
        if is_cacheable
        else {}
    )
    if cached_frames and all(frame is not None for frame in cached_frames.values()):
        driver_data: bytes | None = None
        images = [
            ImageWithMetadata(
                start_timestamp_us=trigger.time_range_us.start,
                end_timestamp_us=trigger.time_range_us.stop,
                image_bytes=cached_frames[camera.logical_id],
                camera_logical_id=camera.logical_id,
            )
            for camera, trigger in camera_triggers
        ]
    else:
        images, driver_data = await _render_frames(state, sensorsim, camera_triggers)
        if is_cacheable:
            for image in images:
                cache.put(keys[image.camera_logical_id], image.image_bytes)

    for image in images:
        state.step_context.track_task(driver.submit_image(image))
    state.data_sensorsim_to_driver = driver_data


class CameraFrameEvent(Event):
    """Render one camera frame at its shutter-close timestamp.

    When ``unbound.render_bundling`` is NONE, render this camera immediately with
    one ``render_rgb`` RPC. Otherwise register the frame so a single
    ``CameraRenderFlushEvent`` renders all same-timestamp cameras in one RPC.
    """

    priority: int = EventPriority.CAMERA

    def __init__(
        self,
        camera: RuntimeCamera,
        trigger: Clock.Trigger,
        sensorsim: SensorsimService,
        driver: DriverService,
    ):
        super().__init__(timestamp_us=trigger.time_range_us.stop)
        self.camera = camera
        self.trigger = trigger
        self.sensorsim = sensorsim
        self.driver = driver

    def description(self) -> str:
        return (
            f"CameraFrameEvent({self.camera.logical_id}, "
            f"{self.trigger.time_range_us.start:_}->{self.trigger.time_range_us.stop:_}us)"
        )

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        if rollout_state.unbound.render_bundling != RenderBundling.NONE:
            self._register_for_bundled_render(rollout_state, queue)
        else:
            await _render_or_replay(
                rollout_state,
                self.sensorsim,
                self.driver,
                [(self.camera, self.trigger)],
            )

        self._record_frame(rollout_state)
        self._schedule_next(rollout_state, queue)

    def _register_for_bundled_render(
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
        camera_triggers = rollout_state.pending_camera_triggers.pop(
            self.timestamp_us, []
        )
        rollout_state.pending_camera_flush_timestamps.discard(self.timestamp_us)
        if not camera_triggers:
            return

        await _render_or_replay(
            rollout_state, self.sensorsim, self.driver, camera_triggers
        )


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
        )
        for camera in runtime_cameras
        for trigger in [camera.clock.ith_trigger(0)]
        if scene_start_us <= trigger.time_range_us.stop <= simulation_end_us
    ]
