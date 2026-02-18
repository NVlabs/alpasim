# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Sensorsim service implementation."""

from __future__ import annotations

import logging
from asyncio import Lock
from typing import Dict, Optional, Type

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
from alpasim_runtime.services.service_base import ServiceBase
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory
from alpasim_utils.types import ImageWithMetadata

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
            return []

        if scene_id in self._available_cameras:
            return self._copy_available_cameras(self._available_cameras[scene_id])

        lock = self._available_cameras_locks.setdefault(scene_id, Lock())
        async with lock:
            if scene_id not in self._available_cameras:
                request = AvailableCamerasRequest(scene_id=scene_id)
                await self.session_info.broadcaster.broadcast(
                    LogEntry(available_cameras_request=request)
                )

                logger.info(f"Requesting available cameras for {scene_id=}")
                response: AvailableCamerasReturn = await profiled_rpc_call(
                    "get_available_cameras",
                    "sensorsim",
                    self.stub.get_available_cameras,
                    request,
                )

                await self.session_info.broadcaster.broadcast(
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

        return RGBRenderRequest(
            scene_id=scene_id,
            resolution_h=camera.render_resolution_hw[0],
            resolution_w=camera.render_resolution_hw[1],
            camera_intrinsics=definition.intrinsics,
            frame_start_us=start_us,
            frame_end_us=end_us,
            sensor_pose=sensor_pose,
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
        await self.session_info.broadcaster.broadcast(
            LogEntry(aggregated_render_request=request)
        )

        response: AggregatedRenderReturn = await profiled_rpc_call(
            "render_aggregated", "sensorsim", self.stub.render_aggregated, request
        )

        images_with_metadata = []
        for rgb_response in response.rgb_responses:
            images_with_metadata.append(
                ImageWithMetadata(
                    start_timestamp_us=rgb_response.start_timestamp_us,
                    end_timestamp_us=rgb_response.end_timestamp_us,
                    image_bytes=rgb_response.image_bytes,
                    camera_logical_id=rgb_response.camera_logical_id,
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

        await self.session_info.broadcaster.broadcast(LogEntry(render_request=request))

        response: RGBRenderReturn = await profiled_rpc_call(
            "render_rgb", "sensorsim", self.stub.render_rgb, request
        )

        return ImageWithMetadata(
            start_timestamp_us=trigger.time_range_us.start,
            end_timestamp_us=trigger.time_range_us.stop,
            image_bytes=response.image_bytes,
            camera_logical_id=camera.logical_id,
        )
