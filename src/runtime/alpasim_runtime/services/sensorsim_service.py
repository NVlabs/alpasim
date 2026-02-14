# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Sensorsim service implementation."""

from __future__ import annotations

import logging
from asyncio import Lock
from typing import Dict, Optional, Type

from alpasim_grpc.v0.common_pb2 import Empty
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
    ShutterType,
)
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceStub
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.logs import LogEntry
from alpasim_runtime.services.service_base import ServiceBase
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_runtime.types import Clock, ImageWithMetadata, RuntimeCamera
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory

logger = logging.getLogger(__name__)

WILDCARD_SCENE_ID = "*"


class SensorsimService(ServiceBase[SensorsimServiceStub]):
    """
    Sensorsim service implementation that handles both real and skip modes.

    Sensorsim is responsible for sensor simulation and image rendering.
    """

    def __init__(
        self,
        address: str,
        skip: bool,
        connection_timeout_s: int,
        id: int,
        camera_catalog: CameraCatalog,
    ):
        super().__init__(address, skip, connection_timeout_s, id)
        self._available_ego_masks: Optional[AvailableEgoMasksReturn] = None
        self._available_ego_masks_lock = Lock()
        self._camera_catalog = camera_catalog
        self._available_cameras: Dict[
            str, list[AvailableCamerasReturn.AvailableCamera]
        ] = {}
        self._available_cameras_locks: Dict[str, Lock] = {}

    @property
    def stub_class(self) -> Type[SensorsimServiceStub]:
        return SensorsimServiceStub

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
            logger.info("Skip mode: sensorsim returning fake CAM")
            # In skip mode, return fake cameras from extra_cameras config
            # so that merge_local_and_sensorsim_cameras can work properly
            local_overrides = self._camera_catalog.get_local_override_cameras()
            fake_cameras = []
            for cfg in local_overrides:
                # Create a minimal AvailableCamera from config
                # If config is complete, use it; otherwise create a minimal one
                # that will be filled in by merge_local_and_sensorsim_cameras
                camera = AvailableCamerasReturn.AvailableCamera(
                    logical_id=cfg.logical_id,
                )
                # If config has all required fields, create a complete camera
                if (
                    cfg.rig_to_camera is not None
                    and cfg.intrinsics is not None
                    and cfg.resolution_hw is not None
                    and cfg.shutter_type is not None
                ):
                    # Create camera definition from config and convert to proto
                    from alpasim_runtime.camera_catalog import CameraDefinition

                    camera_def = CameraDefinition.from_config(cfg)
                    camera = camera_def.as_proto()
                else:
                    # Create minimal camera with placeholder values
                    # These will be overridden by merge_local_and_sensorsim_cameras
                    camera.intrinsics.logical_id = cfg.logical_id
                    camera.intrinsics.resolution_h = cfg.resolution_hw[0] if cfg.resolution_hw else 1080
                    camera.intrinsics.resolution_w = cfg.resolution_hw[1] if cfg.resolution_hw else 1920
                    # Set shutter_type with default fallback
                    try:
                        camera.intrinsics.shutter_type = ShutterType.Value(
                            cfg.shutter_type or "GLOBAL"
                        )
                    except ValueError:
                        camera.intrinsics.shutter_type = ShutterType.GLOBAL
                    # Set minimal intrinsics (opencv_pinhole with default values)
                    # This is required for CameraDefinition.from_proto to work
                    if cfg.intrinsics is None:
                        pinhole = camera.intrinsics.opencv_pinhole_param
                        pinhole.focal_length_x = camera.intrinsics.resolution_w
                        pinhole.focal_length_y = camera.intrinsics.resolution_h
                        pinhole.principal_point_x = camera.intrinsics.resolution_w / 2.0
                        pinhole.principal_point_y = camera.intrinsics.resolution_h / 2.0
                    # Set default rig_to_camera (identity)
                    camera.rig_to_camera.quat.w = 1.0
                    if cfg.rig_to_camera is not None:
                        camera.rig_to_camera.vec.x = cfg.rig_to_camera.translation_m[0]
                        camera.rig_to_camera.vec.y = cfg.rig_to_camera.translation_m[1]
                        camera.rig_to_camera.vec.z = cfg.rig_to_camera.translation_m[2]
                        camera.rig_to_camera.quat.x = cfg.rig_to_camera.rotation_xyzw[0]
                        camera.rig_to_camera.quat.y = cfg.rig_to_camera.rotation_xyzw[1]
                        camera.rig_to_camera.quat.z = cfg.rig_to_camera.rotation_xyzw[2]
                        camera.rig_to_camera.quat.w = cfg.rig_to_camera.rotation_xyzw[3]
                fake_cameras.append(camera)
            logger.info(
                f"Skip mode: returning {len(fake_cameras)} fake cameras from extra_cameras config"
            )
            return fake_cameras

        if scene_id in self._available_cameras:
            return self._copy_available_cameras(self._available_cameras[scene_id])

        lock = self._available_cameras_locks.setdefault(scene_id, Lock())
        async with lock:
            if scene_id not in self._available_cameras:
                request = AvailableCamerasRequest(scene_id=scene_id)
                await self.session_info.log_writer.log_message(
                    LogEntry(available_cameras_request=request)
                )

                logger.info(f"Requesting available cameras for {scene_id=}")
                response: AvailableCamerasReturn = await profiled_rpc_call(
                    "get_available_cameras",
                    "sensorsim",
                    self.stub.get_available_cameras,
                    request,
                )

                await self.session_info.log_writer.log_message(
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

        # Fast path: return cached value without acquiring lock
        if self._available_ego_masks is not None:
            return self._available_ego_masks

        async with self._available_ego_masks_lock:
            # Double-check after acquiring lock
            if self._available_ego_masks is not None:
                return self._available_ego_masks

            self._available_ego_masks = await profiled_rpc_call(
                "get_available_ego_masks",
                "sensorsim",
                self.stub.get_available_ego_masks,
                Empty(),
            )
            logger.info(
                f"Available ego masks: {self._available_ego_masks} "
                f"(session={self.session_info.uuid}, service_addr={self.address})"
            )

        return self._available_ego_masks

    @staticmethod
    def determine_ego_mask_id(
        available_ego_masks: AvailableEgoMasksReturn,
        camera_logical_id: str,
        ego_mask_rig_config_id: Optional[str],
    ) -> Optional[str]:
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
        ego_mask_id: Optional[str] = None,
    ) -> RGBRenderRequest:
        start_us = trigger.time_range_us.start
        end_us = trigger.time_range_us.stop - 1

        def trajectory_to_pose_pair(
            trajectory: Trajectory, delta: Optional[QVec]
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
                start_pose=start_pose.as_grpc_pose(),
                end_pose=end_pose.as_grpc_pose(),
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
        sensor_pose = trajectory_to_pose_pair(
            ego_trajectory,
            delta=definition.rig_to_camera,
        )

        # Create ego_pose (ego vehicle pose without rig_to_camera transform)
        # This is used by MTGS renderer which needs the actual ego position
        ego_pose = trajectory_to_pose_pair(
            ego_trajectory,
            delta=None,  # No rig_to_camera transform for ego pose
        )

        # Get rig_to_camera as grpc Pose for MTGS renderer
        rig_to_camera_pose = (
            definition.rig_to_camera.as_grpc_pose()
            if definition.rig_to_camera is not None
            else None
        )

        return RGBRenderRequest(
            scene_id=scene_id,
            resolution_h=camera.render_resolution_hw[0],
            resolution_w=camera.render_resolution_hw[1],
            camera_intrinsics=definition.intrinsics,
            frame_start_us=start_us,
            frame_end_us=end_us,
            sensor_pose=sensor_pose,
            ego_pose=ego_pose,  # Add ego_pose for MTGS renderer
            rig_to_camera=rig_to_camera_pose,  # Add rig_to_camera for MTGS
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
        ego_mask_rig_config_id: Optional[str] = None,
    ) -> (list[ImageWithMetadata], Optional[bytes]):
        """
        Render multiple RGB images from the given scene and trajectories.
        Returns a tuple containing a list of ImageWithMetadata containing the rendered images
        and optional driver data bytes (forwarded without processing to the driver).
        """
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
        await self.session_info.log_writer.log_message(
            LogEntry(aggregated_render_request=request)
        )

        response: AggregatedRenderReturn = await profiled_rpc_call(
            "render_aggregated", "sensorsim", self.stub.render_aggregated, request
        )

        images_with_metadata = []
        # Match responses with requests to get metadata (timestamps, camera_id)
        # RGBRenderReturn only contains image_bytes, so we need to get metadata from the request
        for rgb_request, rgb_response in zip(request.rgb_requests, response.rgb_returns):
            images_with_metadata.append(
                ImageWithMetadata(
                    start_timestamp_us=rgb_request.frame_start_us,
                    end_timestamp_us=rgb_request.frame_end_us,
                    image_bytes=rgb_response.image_bytes,
                    camera_logical_id=rgb_request.camera_intrinsics.logical_id,
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
        ego_mask_rig_config_id: Optional[str] = None,
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

        await self.session_info.log_writer.log_message(LogEntry(render_request=request))

        response: RGBRenderReturn = await profiled_rpc_call(
            "render_rgb", "sensorsim", self.stub.render_rgb, request
        )

        return ImageWithMetadata(
            start_timestamp_us=trigger.time_range_us.start,
            end_timestamp_us=trigger.time_range_us.stop,
            image_bytes=response.image_bytes,
            camera_logical_id=camera.logical_id,
        )
