# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Sensorsim service implementation."""

from __future__ import annotations

import logging
from asyncio import Lock
from typing import Any, Dict, Type

from alpasim_grpc.v0.common_pb2 import Empty
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_grpc.v0.sensorsim_pb2 import (
    AggregatedRenderRequest,
    AggregatedRenderReturn,
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableEgoMasksReturn,
    DynamicObject,
    ImageFormat,
    PosePair,
    RGBRenderRequest,
    RGBRenderReturn,
)
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceStub
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import SimulationConfig
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo
from alpasim_runtime.services.session_configs import RendererSessionConfig
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils.geometry import Pose, Trajectory, pose_to_grpc
from alpasim_utils.types import ImageWithMetadata

logger = logging.getLogger(__name__)

WILDCARD_SCENE_ID = "*"
SENSORSIM_UNAVAILABLE_RETRY_DELAYS_S = (0.5, 2.0)


class SensorsimService(ServiceBase[SensorsimServiceStub]):
    """
    Sensorsim service implementation that handles both real and skip modes.

    Sensorsim is responsible for sensor simulation and image rendering.
    """

    def __init__(
        self,
        address: str,
        skip: bool,
        camera_catalog: CameraCatalog,
    ):
        super().__init__(address, skip)
        self._available_ego_masks: AvailableEgoMasksReturn | None = None
        self._available_ego_masks_lock = Lock()
        self._camera_catalog = camera_catalog
        self._available_cameras: Dict[
            str, list[AvailableCamerasReturn.AvailableCamera]
        ] = {}
        self._available_cameras_locks: Dict[str, Lock] = {}

    @property
    def stub_class(self) -> Type[SensorsimServiceStub]:
        return SensorsimServiceStub

    def make_initial_render_event(self, **kwargs: Any) -> Any:
        """Create the initial sensorsim render event for a rollout."""
        from alpasim_runtime.events.camera import make_initial_sensorsim_render_event

        return make_initial_sensorsim_render_event(
            renderer_service=self,
            **kwargs,
        )

    def validate_timing_alignment(self, simulation_config: SimulationConfig) -> None:
        """Validate rollout timing against sensorsim's control-step cadence."""
        force_gt_duration_us = simulation_config.force_gt_duration_us
        control_timestep_us = simulation_config.control_timestep_us

        if force_gt_duration_us < 0:
            raise ValueError(
                f"force_gt_duration_us must be >= 0, got {force_gt_duration_us}."
            )
        if force_gt_duration_us == 0:
            return
        if force_gt_duration_us % control_timestep_us != 0:
            raise ValueError(
                f"force_gt_duration_us ({force_gt_duration_us}) "
                f"must be a multiple of control_timestep_us "
                f"({control_timestep_us}). "
                f"Non-divisible durations cause ambiguous policy start timing."
            )

    def required_policy_start_timestmap_us(
        self,
        render_start_timestamp_us: int,
    ) -> int:
        """Sensorsim can start policy at the first rendered frame."""
        return render_start_timestamp_us

    @staticmethod
    def _copy_available_cameras(
        cameras: list[AvailableCamerasReturn.AvailableCamera],
    ) -> list[AvailableCamerasReturn.AvailableCamera]:
        """Return a deep copy of the available cameras list."""
        copied = []
        for camera in cameras:
            camera_copy = AvailableCamerasReturn.AvailableCamera()
            camera_copy.CopyFrom(camera)
            copied.append(camera_copy)
        return copied

    async def get_available_cameras(
        self, scene_id: str
    ) -> list[AvailableCamerasReturn.AvailableCamera]:
        """Fetch available cameras for `scene_id`, skipping RPC in skip mode."""
        if self.skip:
            return []

        session_info = self._require_session_info()

        if scene_id in self._available_cameras:
            return self._copy_available_cameras(self._available_cameras[scene_id])

        lock = self._available_cameras_locks.setdefault(scene_id, Lock())
        async with lock:
            if scene_id not in self._available_cameras:
                request = AvailableCamerasRequest(scene_id=scene_id)
                await session_info.broadcaster.broadcast(
                    LogEntry(available_cameras_request=request)
                )

                logger.info(f"Requesting available cameras for {scene_id=}")
                response: AvailableCamerasReturn = await profiled_rpc_call(
                    "get_available_cameras",
                    "sensorsim",
                    self.stub.get_available_cameras,
                    request,
                    unavailable_retry_delays_s=SENSORSIM_UNAVAILABLE_RETRY_DELAYS_S,
                )

                await session_info.broadcaster.broadcast(
                    LogEntry(available_cameras_return=response)
                )

                self._available_cameras[scene_id] = list(response.available_cameras)

        return self._copy_available_cameras(self._available_cameras[scene_id])

    async def get_available_ego_masks(self) -> AvailableEgoMasksReturn:
        """
        Get available ego masks.

        Returns an AvailableEgoMasksReturn containing the available ego masks.
        """
        if self.skip:
            return AvailableEgoMasksReturn()

        session_info = self._require_session_info()

        # Fast path: return cached value without acquiring lock
        if self._available_ego_masks is not None:
            return self._available_ego_masks

        async with self._available_ego_masks_lock:
            # Double-check after acquiring lock
            if self._available_ego_masks is not None:
                return self._available_ego_masks

            request = Empty()
            await session_info.broadcaster.broadcast(
                LogEntry(available_ego_masks_request=request)
            )

            self._available_ego_masks = await profiled_rpc_call(
                "get_available_ego_masks",
                "sensorsim",
                self.stub.get_available_ego_masks,
                request,
                unavailable_retry_delays_s=SENSORSIM_UNAVAILABLE_RETRY_DELAYS_S,
            )

            await session_info.broadcaster.broadcast(
                LogEntry(available_ego_masks_return=self._available_ego_masks)
            )

            logger.info(
                f"Available ego masks: {self._available_ego_masks} "
                f"(session={session_info.uuid}, service_addr={self.address})"
            )

        return self._available_ego_masks

    @staticmethod
    def determine_ego_mask_id(
        available_ego_masks: AvailableEgoMasksReturn,
        camera_logical_id: str,
        ego_mask_rig_config_id: str | None,
    ) -> str | None:
        """
        Determine the ego mask ID for a given camera and rig configuration.
        Returns the ego mask ID if found, otherwise None.
        """
        if ego_mask_rig_config_id is None:
            return None

        ego_mask_id = None
        for ego_mask_metadata in available_ego_masks.ego_mask_metadata:
            if (
                camera_logical_id == ego_mask_metadata.ego_mask_id.camera_logical_id
                and ego_mask_rig_config_id
                == ego_mask_metadata.ego_mask_id.rig_config_id
            ):
                ego_mask_id = ego_mask_metadata.ego_mask_id
                break

        return ego_mask_id

    def construct_rgb_render_request(
        self,
        ego_trajectory: Trajectory,
        traffic_trajectories: Dict[str, Trajectory],
        camera: RuntimeCamera,
        trigger: Clock.Trigger,
        scene_id: str,
        image_format: ImageFormat,
        ego_mask_id: str | None = None,
    ) -> RGBRenderRequest:
        """Construct an RGBRenderRequest for a single camera trigger.

        Interpolates ego and traffic poses at the trigger's time range,
        resolves the camera definition from the catalog, and assembles the
        full render request including dynamic objects and ego mask.

        Args:
            ego_trajectory: Ego vehicle trajectory in local frame.
            traffic_trajectories: Mapping of track_id to traffic trajectories.
            camera: Camera to render from.
            trigger: Clock trigger defining the time range for rendering.
            scene_id: Scene identifier for camera definition lookup.
            image_format: Desired output image format.
            ego_mask_id: Optional ego mask identifier to include in the render.
        """
        start_us = trigger.time_range_us.start
        end_us = trigger.time_range_us.stop

        def trajectory_to_pose_pair(
            trajectory: Trajectory, delta: Pose | None
        ) -> PosePair:
            """
            Interpolate pose between trigger start and end and package as PosePair.
            Optionally apply a delta transformation (such as rig_to_camera).
            """
            start_pose = trajectory.interpolate_pose(start_us)
            end_pose = trajectory.interpolate_pose(end_us)

            if delta is not None:
                start_pose = start_pose @ delta
                end_pose = end_pose @ delta

            return PosePair(
                start_pose=pose_to_grpc(start_pose),
                end_pose=pose_to_grpc(end_pose),
            )

        dynamic_objects = [
            DynamicObject(
                track_id=track_id,
                pose_pair=trajectory_to_pose_pair(track_traj, delta=None),
            )
            for track_id, track_traj in traffic_trajectories.items()
            if (
                start_us in track_traj.time_range_us
                and end_us in track_traj.time_range_us
            )
        ]

        definition = self._camera_catalog.get_camera_definition(
            scene_id, camera.logical_id
        )
        ego_pose = trajectory_to_pose_pair(ego_trajectory, delta=None)
        sensor_pose = trajectory_to_pose_pair(
            ego_trajectory,
            delta=definition.rig_to_camera,
        )

        return RGBRenderRequest(
            scene_id=scene_id,
            resolution_h=camera.render_resolution_hw[0],
            resolution_w=camera.render_resolution_hw[1],
            camera_intrinsics=definition.intrinsics,
            frame_start_us=start_us,
            frame_end_us=end_us,
            sensor_pose=sensor_pose,
            ego_pose=ego_pose,
            dynamic_objects=dynamic_objects,
            image_format=image_format,
            image_quality=95,
            insert_ego_mask=ego_mask_id is not None,
            ego_mask_id=ego_mask_id,
        )

    async def aggregated_render(
        self,
        camera_triggers: list[tuple[RuntimeCamera, Clock.Trigger]],
        ego_trajectory: Trajectory,
        traffic_trajectories: Dict[str, Trajectory],
        scene_id: str,
        image_format: ImageFormat,
        ego_mask_rig_config_id: str | None = None,
    ) -> (list[ImageWithMetadata], bytes | None):
        """
        Render multiple RGB images from the given scene and trajectories.
        Returns a tuple containing a list of ImageWithMetadata containing the rendered images
        and optional driver data bytes (forwarded without processing to the driver).
        """
        session_info = self._require_session_info()
        available_ego_masks = await self.get_available_ego_masks()

        request = AggregatedRenderRequest()

        for camera, trigger in camera_triggers:
            ego_mask_id = self.determine_ego_mask_id(
                available_ego_masks, camera.logical_id, ego_mask_rig_config_id
            )

            rgb_request = self.construct_rgb_render_request(
                ego_trajectory,
                traffic_trajectories,
                camera,
                trigger,
                scene_id,
                image_format,
                ego_mask_id,
            )
            request.rgb_requests.append(rgb_request)

        # TODO(mwatson): Add requests/handling for lidars
        await session_info.broadcaster.broadcast(
            LogEntry(aggregated_render_request=request)
        )

        response: AggregatedRenderReturn = await profiled_rpc_call(
            "render_aggregated",
            "sensorsim",
            self.stub.render_aggregated,
            request,
            unavailable_retry_delays_s=SENSORSIM_UNAVAILABLE_RETRY_DELAYS_S,
        )

        if len(response.rgb_returns) != len(camera_triggers):
            raise RuntimeError(
                f"sensorsim returned {len(response.rgb_returns)} rgb_returns "
                f"for {len(camera_triggers)} camera triggers"
            )

        images_with_metadata = []
        for rgb_return, (camera, trigger) in zip(response.rgb_returns, camera_triggers):
            images_with_metadata.append(
                ImageWithMetadata(
                    start_timestamp_us=trigger.time_range_us.start,
                    end_timestamp_us=trigger.time_range_us.stop,
                    image_bytes=rgb_return.image_bytes,
                    camera_logical_id=camera.logical_id,
                )
            )

        return (images_with_metadata, response.driver_data)

    async def render(
        self,
        ego_trajectory: Trajectory,
        traffic_trajectories: Dict[str, Trajectory],
        camera: RuntimeCamera,
        trigger: Clock.Trigger,
        scene_id: str,
        image_format: ImageFormat,
        ego_mask_rig_config_id: str | None = None,
    ) -> ImageWithMetadata:
        """
        Render an RGB image from the given scene and trajectories.

        Returns an ImageWithMetadata containing the rendered image.
        """
        if self.skip:
            logger.info("Skip mode: sensorsim returning empty image")
            # Return empty image for skip mode
            return ImageWithMetadata(
                start_timestamp_us=trigger.time_range_us.start,
                end_timestamp_us=trigger.time_range_us.stop,
                image_bytes=b"",  # TODO: fill in with a placeholder image
                camera_logical_id=camera.logical_id,
            )

        session_info = self._require_session_info()
        available_ego_masks = await self.get_available_ego_masks()
        ego_mask_id = self.determine_ego_mask_id(
            available_ego_masks, camera.logical_id, ego_mask_rig_config_id
        )

        request = self.construct_rgb_render_request(
            ego_trajectory,
            traffic_trajectories,
            camera,
            trigger,
            scene_id,
            image_format,
            ego_mask_id,
        )

        await session_info.broadcaster.broadcast(LogEntry(render_request=request))

        response: RGBRenderReturn = await profiled_rpc_call(
            "render_rgb",
            "sensorsim",
            self.stub.render_rgb,
            request,
            unavailable_retry_delays_s=SENSORSIM_UNAVAILABLE_RETRY_DELAYS_S,
        )

        return ImageWithMetadata(
            start_timestamp_us=trigger.time_range_us.start,
            end_timestamp_us=trigger.time_range_us.stop,
            image_bytes=response.image_bytes,
            camera_logical_id=camera.logical_id,
        )

    # -- Renderer session lifecycle hooks --------------------------------------

    async def _initialize_session(self, session_info: SessionInfo) -> None:
        """Initialize a sensorsim rollout session.

        Owns the sensorsim-specific camera discovery and camera catalog merge.
        Runs once per rollout, after the gRPC connection is open and before the
        event loop starts.

        Skip-mode handling matches pre-refactor behavior: callers (tests) that
        monkeypatch ``get_available_cameras`` can still populate camera
        definitions; callers that do not get the same empty-result failure
        mode as before at the first ``get_camera_definition`` lookup.
        """
        cfg = session_info.session_config
        if not isinstance(cfg, RendererSessionConfig):
            # Called without a RendererSessionConfig (direct-RPC tests, etc.).
            return

        scene_id = cfg.data_source.scene_id
        sensorsim_cameras = await self.get_available_cameras(scene_id)
        await self._camera_catalog.merge_local_and_sensorsim_cameras(
            scene_id, sensorsim_cameras
        )
