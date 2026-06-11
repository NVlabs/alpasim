# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""MTGS Sensorsim Service - gRPC server implementation.

Wraps the MTGS renderer and exposes it via gRPC as a SensorsimService,
allowing it to replace the default sensorsim in alpasim.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from alpasim_grpc.v0.common_pb2 import AvailableScenesReturn, Empty, VersionId
from alpasim_grpc.v0.sensorsim_pb2 import (
    AggregatedRenderRequest,
    AggregatedRenderReturn,
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableEgoMasksReturn,
    RGBRenderRequest,
    RGBRenderReturn,
)
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceServicer
from alpasim_mtgs.server.artifact_adapter import (
    get_available_cameras_from_data_source,
    rgb_render_request_to_render_state,
)
from alpasim_mtgs.server.engine.base_renderer import RenderState
from alpasim_mtgs.server.engine.mtgs import MTGS, _normalize_asset_id
from alpasim_mtgs.server.engine.utils.gaussian_utils import quat_to_rotmat
from pyquaternion import Quaternion

import grpc

logger = logging.getLogger(__name__)

VERSION_MESSAGE = VersionId(
    version_id="mtgs-sensorsim-1.0.0",
    git_hash="unknown",
)


class MTGSSensorsimService(SensorsimServiceServicer):
    """gRPC service that wraps MTGS renderer.

    Implements the SensorsimServiceServicer interface, allowing
    MTGS to be used as a drop-in replacement for the default sensorsim.
    """

    def __init__(
        self,
        server: grpc.Server,
        get_scene: callable,
        get_available_scene_ids: callable,
        cache_size: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.server = server
        self.get_scene = get_scene
        self.get_available_scene_ids = get_available_scene_ids
        self.device = device
        self.cache_size = cache_size

        logger.info(
            f"MTGS Sensorsim Service initialized (device={device}, cache_size={cache_size})"
        )

        self._renderer_cache: Dict[str, MTGS] = {}
        self._loaded_scene_ids: set[str] = set()

    def get_renderer(self, scene_id: str) -> MTGS:
        if scene_id in self._renderer_cache:
            return self._renderer_cache[scene_id]

        logger.info(f"Renderer cache miss, loading for scene {scene_id}")

        try:
            data_source = self.get_scene(scene_id)
        except Exception as e:
            logger.error(f"Failed to load scene {scene_id}: {e}")
            raise KeyError(f"Scene {scene_id} not available: {e}")

        asset_path = data_source.asset_path
        if asset_path is None:
            raise ValueError(
                f"Scene {scene_id} has no asset_path. "
                "Make sure asset_base_path is configured."
            )

        asset_folder_path = Path(asset_path).parent
        asset_id = Path(asset_path).name
        if not asset_id:
            asset_id = Path(asset_path).parent.name
        asset_id = _normalize_asset_id(asset_id)

        logger.info(f"Creating MTGS renderer for scene {scene_id}, asset_id={asset_id}")

        renderer = MTGS(device=self.device, asset_folder_path=asset_folder_path)
        renderer.reset(current_scene_id=scene_id, asset_id=asset_id)

        # Extract ego2globals from video_scene_dict
        ego2globals = None
        try:
            if (
                hasattr(renderer.asset_manager, "video_scene_dict")
                and renderer.asset_manager.video_scene_dict
            ):
                video_dict_raw = renderer.asset_manager.video_scene_dict
                video_dict = video_dict_raw
                if (
                    isinstance(video_dict_raw, dict)
                    and "ego2global" not in video_dict_raw
                    and "frame_infos" not in video_dict_raw
                ):
                    first_key = next(iter(video_dict_raw.keys()), None)
                    if first_key and isinstance(video_dict_raw[first_key], dict):
                        video_dict = video_dict_raw[first_key]

                if "frame_infos" in video_dict and len(video_dict["frame_infos"]) > 0:
                    frame_infos = video_dict["frame_infos"]
                    ego2globals_list = []
                    for fi in frame_infos:
                        if "ego2global" in fi:
                            ego2globals_list.append(np.array(fi["ego2global"]))
                        elif (
                            "ego2global_translation" in fi
                            and "ego2global_rotation" in fi
                        ):
                            transform = np.eye(4)
                            transform[:3, 3] = np.array(fi["ego2global_translation"])
                            rot = np.array(fi["ego2global_rotation"])
                            if rot.shape == (3, 3):
                                transform[:3, :3] = rot
                            elif rot.shape == (4,):
                                q = Quaternion(rot[0], rot[1], rot[2], rot[3])
                                transform[:3, :3] = q.rotation_matrix
                            ego2globals_list.append(transform)
                    if ego2globals_list:
                        ego2globals = np.stack(ego2globals_list)
                elif "ego2global" in video_dict:
                    ego2globals = np.array(video_dict["ego2global"])
                    if ego2globals.ndim == 2:
                        ego2globals = ego2globals[np.newaxis, ...]
        except Exception as e:
            logger.warning(f"Could not load ego2globals from video_scene_dict: {e}")

        # Fallback: compute from rig trajectory
        if ego2globals is None:
            try:
                rig = data_source.rig
                if rig.trajectory.poses is not None and len(rig.trajectory.poses) > 0:

                    poses = rig.trajectory.poses
                    ego2globals_list = []
                    for pose in poses:
                        if hasattr(pose.vec3, "__getitem__"):
                            trans = np.array([pose.vec3[0], pose.vec3[1], pose.vec3[2]])
                        else:
                            trans = np.array([pose.vec3.x, pose.vec3.y, pose.vec3.z])

                        if hasattr(pose.quat, "w"):
                            quat = np.array(
                                [pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z]
                            )
                        elif hasattr(pose.quat, "__getitem__"):
                            quat = np.array(pose.quat)
                        else:
                            raise ValueError(
                                f"Unsupported quaternion format: {type(pose.quat)}"
                            )

                        quat_tensor = torch.tensor(quat, dtype=torch.float64).unsqueeze(
                            0
                        )
                        rot_mat = quat_to_rotmat(quat_tensor).squeeze(0).numpy()

                        transform = np.eye(4)
                        transform[:3, :3] = rot_mat
                        transform[:3, 3] = trans
                        ego2globals_list.append(transform)

                    ego2globals = np.stack(ego2globals_list)
            except Exception as e:
                logger.warning(
                    f"Could not compute ego2globals from rig trajectory: {e}"
                )

        # Set world_to_nre
        if hasattr(data_source, "rig") and data_source.rig is not None:
            if (
                hasattr(data_source.rig, "world_to_nre")
                and data_source.rig.world_to_nre is not None
            ):
                renderer.set_world_to_nre(data_source.rig.world_to_nre)

        renderer.mtgs_agent2states = renderer.calibrate_agent_state(
            ego2globals=ego2globals
        )
        renderer.local_to_global_offset = None
        renderer._scene_id = scene_id
        renderer._asset_id = asset_id

        if len(self._renderer_cache) >= self.cache_size:
            oldest_key = next(iter(self._renderer_cache))
            del self._renderer_cache[oldest_key]
            torch.cuda.empty_cache()

        self._renderer_cache[scene_id] = renderer
        self._loaded_scene_ids.add(scene_id)
        return renderer

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        return VERSION_MESSAGE

    def get_available_scenes(
        self, _request: Empty, _context: grpc.ServicerContext
    ) -> AvailableScenesReturn:
        try:
            scene_ids = self.get_available_scene_ids()
            return AvailableScenesReturn(scene_ids=scene_ids)
        except Exception as e:
            logger.error(f"Failed to get available scenes: {e}")
            return AvailableScenesReturn(scene_ids=[])

    def get_available_cameras(
        self,
        request: AvailableCamerasRequest,
        context: grpc.ServicerContext,
    ) -> AvailableCamerasReturn:
        scene_id = request.scene_id
        try:
            renderer = self.get_renderer(scene_id)
            cameras = get_available_cameras_from_data_source(
                asset_manager=renderer.asset_manager
            )
            return AvailableCamerasReturn(available_cameras=cameras)
        except Exception as e:
            logger.error(f"Failed to get cameras for scene {scene_id}: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene {scene_id} not found: {e}")
            return AvailableCamerasReturn()

    def get_available_ego_masks(
        self, _request: Empty, _context: grpc.ServicerContext
    ) -> AvailableEgoMasksReturn:
        return AvailableEgoMasksReturn()

    def render_rgb(
        self, request: RGBRenderRequest, context: grpc.ServicerContext
    ) -> RGBRenderReturn:
        try:
            scene_id = request.scene_id
            renderer = self.get_renderer(scene_id)

            render_state_dict = rgb_render_request_to_render_state(
                request, asset_manager=renderer.asset_manager
            )

            render_state = RenderState()
            render_state[RenderState.TIMESTAMP] = render_state_dict["timestamp"]
            render_state[RenderState.AGENT_STATE] = render_state_dict["agent_state"]
            render_state[RenderState.CAMERAS] = render_state_dict["cameras"]
            render_state[RenderState.LIDAR] = render_state_dict.get("lidar", {})

            result = renderer.render(render_state)

            if "cameras" not in result:
                raise ValueError("Renderer did not return cameras in result")

            cameras_dict = result["cameras"]
            camera_name = request.camera_intrinsics.logical_id
            if camera_name in cameras_dict and "image" in cameras_dict[camera_name]:
                image = cameras_dict[camera_name]["image"]
            else:
                first_camera = next(iter(cameras_dict.values()))
                image = first_camera.get("image")

            if image is None:
                raise ValueError("No image found in render result")

            if isinstance(image, np.ndarray):
                if request.image_format == 1:
                    success, image_bytes = cv2.imencode(".png", image)
                elif request.image_format == 2:
                    success, image_bytes = cv2.imencode(
                        ".jpg",
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, int(request.image_quality)],
                    )
                else:
                    success, image_bytes = cv2.imencode(".jpg", image)

                if not success:
                    raise ValueError("Failed to encode image")

                return RGBRenderReturn(image_bytes=image_bytes.tobytes())
            else:
                raise ValueError(f"Unexpected image type: {type(image)}")

        except Exception as e:
            logger.exception(f"Error rendering RGB image: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return RGBRenderReturn()

    def render_aggregated(
        self,
        request: AggregatedRenderRequest,
        context: grpc.ServicerContext,
    ) -> AggregatedRenderReturn:
        if not request.rgb_requests:
            return AggregatedRenderReturn()

        first_request = request.rgb_requests[0]
        scene_id = first_request.scene_id
        frame_start_us = first_request.frame_start_us

        can_batch = all(
            req.scene_id == scene_id and req.frame_start_us == frame_start_us
            for req in request.rgb_requests
        )

        if not can_batch:
            rgb_returns = []
            for rgb_request in request.rgb_requests:
                rgb_return = self.render_rgb(rgb_request, context)
                rgb_returns.append(rgb_return)
                if not rgb_return.image_bytes:
                    # render_rgb set an error on context; stop here to avoid
                    # running further renders after the context is poisoned with
                    # a non-OK gRPC status code that cannot be reset to OK.
                    break
            return AggregatedRenderReturn(rgb_returns=rgb_returns)

        try:
            return self._render_aggregated_batch(request, context)
        except Exception as e:
            logger.exception(
                f"Batch rendering failed: {e}. Falling back to sequential."
            )
            rgb_returns = []
            for rgb_request in request.rgb_requests:
                try:
                    rgb_return = self.render_rgb(rgb_request, context)
                    rgb_returns.append(rgb_return)
                    if not rgb_return.image_bytes:
                        break
                except Exception:
                    rgb_returns.append(RGBRenderReturn())
                    break
            return AggregatedRenderReturn(rgb_returns=rgb_returns)

    def _render_aggregated_batch(
        self,
        request: AggregatedRenderRequest,
        context: grpc.ServicerContext,
    ) -> AggregatedRenderReturn:
        first_request = request.rgb_requests[0]
        scene_id = first_request.scene_id

        renderer = self.get_renderer(scene_id)
        render_state_dict = rgb_render_request_to_render_state(
            first_request, asset_manager=renderer.asset_manager
        )
        render_state = RenderState(**render_state_dict)

        result = renderer.render(render_state)

        if "cameras" not in result:
            raise ValueError("Renderer did not return cameras in result")

        cameras_dict = result["cameras"]
        rgb_returns = []
        for rgb_request in request.rgb_requests:
            camera_name = rgb_request.camera_intrinsics.logical_id

            if camera_name in cameras_dict and "image" in cameras_dict[camera_name]:
                image = cameras_dict[camera_name]["image"]
            else:
                first_camera = next(iter(cameras_dict.values()))
                image = first_camera.get("image")

            if image is None or not isinstance(image, np.ndarray):
                rgb_returns.append(RGBRenderReturn())
                continue

            if rgb_request.image_format == 1:
                success, image_bytes = cv2.imencode(".png", image)
            elif rgb_request.image_format == 2:
                success, image_bytes = cv2.imencode(
                    ".jpg",
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, int(rgb_request.image_quality)],
                )
            else:
                success, image_bytes = cv2.imencode(".jpg", image)

            if not success:
                rgb_returns.append(RGBRenderReturn())
                continue

            rgb_returns.append(RGBRenderReturn(image_bytes=image_bytes.tobytes()))

        return AggregatedRenderReturn(rgb_returns=rgb_returns)

    def shut_down(self, _request: Empty, context: grpc.ServicerContext) -> Empty:
        logger.info("shut_down")
        context.add_callback(self._shut_down)
        return Empty()

    def _shut_down(self) -> None:
        self.server.stop(0)
