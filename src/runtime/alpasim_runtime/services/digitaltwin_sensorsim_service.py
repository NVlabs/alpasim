# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
DigitalTwin Sensorsim Service implementation.

This service wraps the worldengine DigitalTwin renderer and exposes it via gRPC
as a SensorsimService, allowing it to replace the default sensorsim in alpasim.
"""

from __future__ import annotations

import functools
import io
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import grpc
import numpy as np
import torch

from alpasim_grpc.v0.common_pb2 import AvailableScenesReturn, Empty, VersionId
from alpasim_grpc.v0.sensorsim_pb2 import (
    AggregatedRenderRequest,
    AggregatedRenderReturn,
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableEgoMasksReturn,
    AvailableTrajectoriesRequest,
    AvailableTrajectoriesReturn,
    RGBRenderRequest,
    RGBRenderReturn,
)
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceServicer
from alpasim_utils.artifact import Artifact
from alpasim_utils.digitaltwin_artifact_adapter import (
    get_available_cameras_from_data_source,
    rgb_render_request_to_render_state,
)
from alpasim_utils.scene_data_source import SceneDataSource

logger = logging.getLogger(__name__)

# Import worldengine components
try:
    from worldengine.engine.engine_utils import get_global_config, set_global_config
    from worldengine.render.digitaltwin.digitaltwin import DigitalTwin
    from worldengine.render.base_renderer import RenderState
    WORLDENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"worldengine not available: {e}")
    WORLDENGINE_AVAILABLE = False
    DigitalTwin = None
    RenderState = None


VERSION_MESSAGE = VersionId(
    version_id="digitaltwin-sensorsim-1.0.0",
    git_hash="unknown",
)


class DigitalTwinSensorsimService(SensorsimServiceServicer):
    """
    gRPC service that wraps worldengine DigitalTwin renderer.
    
    This service implements the SensorsimServiceServicer interface, allowing
    DigitalTwin to be used as a drop-in replacement for the default sensorsim.
    """

    def __init__(
        self,
        server: grpc.Server,
        artifact_glob: str,
        asset_folder_path: str,
        scene_id_to_asset_id_mapping: Optional[Dict[str, str]] = None,
        cache_size: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the DigitalTwin sensorsim service.
        
        Args:
            server: gRPC server instance
            artifact_glob: Glob pattern to discover USDZ artifacts
            asset_folder_path: Path to DigitalTwin asset folder
            scene_id_to_asset_id_mapping: Optional mapping from scene_id to asset_id
            cache_size: Number of renderer instances to cache
            device: Device to use for rendering (cuda or cpu)
        """
        if not WORLDENGINE_AVAILABLE:
            raise ImportError("worldengine is required for DigitalTwinSensorsimService")
        
        self.server = server
        self.artifacts: Dict[str, Artifact] = Artifact.discover_from_glob(artifact_glob)
        self.asset_folder_path = Path(asset_folder_path)
        self.scene_id_to_asset_id_mapping = scene_id_to_asset_id_mapping or {}
        self.device = device
        
        logger.info(f"Available scenes: {list(self.artifacts.keys())}")
        logger.info(f"Asset folder path: {self.asset_folder_path}")
        
        # Cache renderer instances per scene
        # Note: We can't use lru_cache directly because it needs to access self.artifacts
        # Instead, we'll manually cache renderers
        self._renderer_cache: Dict[str, DigitalTwin] = {}
        
        def get_renderer(scene_id: str) -> DigitalTwin:
            if scene_id in self._renderer_cache:
                return self._renderer_cache[scene_id]
            
            if scene_id not in self.artifacts:
                raise KeyError(f"Scene {scene_id=} not available.")
            
            artifact = self.artifacts[scene_id]
            logger.info(f"Cache miss, loading renderer for {scene_id=}")
            
            # Get asset_id from mapping or use scene_id
            asset_id = self.scene_id_to_asset_id_mapping.get(scene_id, scene_id)
            
            # Initialize global config if needed
            try:
                get_global_config()
            except:
                # Create a minimal config
                from omegaconf import DictConfig
                config = DictConfig({
                    "asset_folder_path": str(self.asset_folder_path),
                })
                set_global_config(config)
            
            # Create renderer
            renderer = DigitalTwin(device=self.device)
            
            # Initialize renderer for this scene
            # We need to manually initialize since DigitalTwin.reset requires engine context
            # Instead, we'll directly use asset_manager to load assets
            asset_id_clean = asset_id
            if asset_id_clean[-4:].startswith('-'):  # Remove suffix like "-001"
                asset_id_clean = asset_id_clean[:-4]
            
            # Reset asset manager
            reset_asset = renderer.asset_manager.reset(asset_id_clean)
            if reset_asset:
                renderer._prepare_metas()
                renderer._init_gaussian_models()
                renderer.set_asset(renderer.asset_manager.background_asset)
                
                # Create a minimal digitaltwin_agent2states
                # Since we don't have engine context, we'll create a simplified version
                # that allows rendering to work. The actual agent states will come from
                # the render request.
                renderer.digitaltwin_agent2states = {}
                # Add ego if available from rig trajectory
                try:
                    rig = artifact.rig
                    if rig.trajectory.poses is not None and len(rig.trajectory.poses) > 0:
                        # Create a minimal ego state from first pose
                        first_pose = rig.trajectory.poses[0]
                        renderer.digitaltwin_agent2states['ego'] = {
                            'translation': torch.tensor(
                                [first_pose.vec3[0], first_pose.vec3[1], first_pose.vec3[2]],
                                device=self.device,
                                dtype=torch.float64
                            ).unsqueeze(0),
                            'rotation': torch.tensor(
                                [first_pose.quat.w, first_pose.quat.x, first_pose.quat.y, first_pose.quat.z],
                                device=self.device,
                                dtype=torch.float64
                            ).unsqueeze(0),
                        }
                except Exception as e:
                    logger.warning(f"Could not create ego state from rig: {e}")
                    # Create empty state - renderer will use original log data
                    renderer.digitaltwin_agent2states = {}
                
                renderer.sensor_caches = None
            
            renderer._scene_id = scene_id
            renderer._asset_id = asset_id
            renderer._initialized = True
            
            # Cache the renderer (simple LRU: keep only cache_size most recent)
            if len(self._renderer_cache) >= cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self._renderer_cache))
                del self._renderer_cache[oldest_key]
            
            self._renderer_cache[scene_id] = renderer
            return renderer
        
        self.get_renderer = get_renderer

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        """Return version information."""
        return VERSION_MESSAGE

    def get_available_scenes(
        self, request: Empty, context: grpc.ServicerContext
    ) -> AvailableScenesReturn:
        """Return list of available scene IDs."""
        return AvailableScenesReturn(scene_ids=list(self.artifacts.keys()))

    def get_available_cameras(
        self,
        request: AvailableCamerasRequest,
        context: grpc.ServicerContext,
    ) -> AvailableCamerasReturn:
        """Return available cameras for a scene."""
        scene_id = request.scene_id
        if scene_id not in self.artifacts:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene {scene_id} not found")
            return AvailableCamerasReturn()
        
        artifact = self.artifacts[scene_id]
        cameras = get_available_cameras_from_data_source(artifact)
        
        return AvailableCamerasReturn(available_cameras=cameras)

    def get_available_trajectories(
        self,
        request: AvailableTrajectoriesRequest,
        context: grpc.ServicerContext,
    ) -> AvailableTrajectoriesReturn:
        """Return available trajectories for a scene."""
        scene_id = request.scene_id
        if scene_id not in self.artifacts:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene {scene_id} not found")
            return AvailableTrajectoriesReturn()
        
        artifact = self.artifacts[scene_id]
        rig = artifact.rig
        
        return AvailableTrajectoriesReturn(
            available_trajectories=[
                AvailableTrajectoriesReturn.AvailableTrajectory(
                    trajectory=rig.trajectory.to_grpc()
                )
            ]
        )

    def get_available_ego_masks(
        self, request: Empty, context: grpc.ServicerContext
    ) -> AvailableEgoMasksReturn:
        """Return available ego masks (empty for now)."""
        # DigitalTwin doesn't support ego masks yet
        return AvailableEgoMasksReturn()

    def render_rgb(
        self, request: RGBRenderRequest, context: grpc.ServicerContext
    ) -> RGBRenderReturn:
        """Render an RGB image using DigitalTwin."""
        try:
            scene_id = request.scene_id
            
            if scene_id not in self.artifacts:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Scene {scene_id} not found")
                return RGBRenderReturn()
            
            artifact = self.artifacts[scene_id]
            
            # Get or create renderer for this scene
            renderer = self.get_renderer(scene_id)
            
            # Convert RGBRenderRequest to RenderState
            render_state_dict = rgb_render_request_to_render_state(request, artifact)
            
            # Create RenderState object
            # Note: RenderState is a dict subclass, so we can use dict assignment
            render_state = RenderState()
            render_state[RenderState.TIMESTAMP] = render_state_dict["timestamp"]
            render_state[RenderState.AGENT_STATE] = render_state_dict["agent_state"]
            render_state[RenderState.CAMERAS] = render_state_dict["cameras"]
            render_state[RenderState.LIDAR] = render_state_dict.get("lidar", {})
            
            # Set recon2global_translation if not set (needed for coordinate transformation)
            if not hasattr(renderer, 'recon2global_translation'):
                # Try to get from asset manager
                if hasattr(renderer.asset_manager, 'background_asset') and renderer.asset_manager.background_asset:
                    bg_asset = renderer.asset_manager.background_asset
                    if 'background' in bg_asset and 'config' in bg_asset['background']:
                        recon2global = bg_asset['background']['config'].get('recon2world_translation', [0.0, 0.0, 0.0])
                        renderer.recon2global_translation = torch.tensor(recon2global[:2], device=self.device)
                    else:
                        renderer.recon2global_translation = torch.zeros(2, device=self.device)
                else:
                    renderer.recon2global_translation = torch.zeros(2, device=self.device)
            
            # Ensure renderer is initialized for this scene
            # The renderer should already be initialized in get_renderer, but check anyway
            if not hasattr(renderer, '_initialized') or not renderer._initialized or renderer._scene_id != scene_id:
                # Re-initialize if needed
                asset_id = self.scene_id_to_asset_id_mapping.get(scene_id, scene_id)
                asset_id_clean = asset_id
                if asset_id_clean[-4:].startswith('-'):
                    asset_id_clean = asset_id_clean[:-4]
                
                reset_asset = renderer.asset_manager.reset(asset_id_clean)
                if reset_asset:
                    renderer._prepare_metas()
                    renderer._init_gaussian_models()
                    renderer.set_asset(renderer.asset_manager.background_asset)
                    
                    # Create minimal digitaltwin_agent2states
                    renderer.digitaltwin_agent2states = {}
                    try:
                        rig = artifact.rig
                        if rig.trajectory.poses is not None and len(rig.trajectory.poses) > 0:
                            first_pose = rig.trajectory.poses[0]
                            renderer.digitaltwin_agent2states['ego'] = {
                                'translation': torch.tensor(
                                    [first_pose.vec3[0], first_pose.vec3[1], first_pose.vec3[2]],
                                    device=self.device,
                                    dtype=torch.float64
                                ).unsqueeze(0),
                                'rotation': torch.tensor(
                                    [first_pose.quat.w, first_pose.quat.x, first_pose.quat.y, first_pose.quat.z],
                                    device=self.device,
                                    dtype=torch.float64
                                ).unsqueeze(0),
                            }
                    except Exception as e:
                        logger.warning(f"Could not create ego state: {e}")
                    
                    renderer.sensor_caches = None
                
                renderer._scene_id = scene_id
                renderer._asset_id = asset_id
                renderer._initialized = True
            
            # Render
            result = renderer.render(render_state)
            
            # Extract image from result
            # result should be a dict with 'cameras' key containing camera images
            if "cameras" not in result:
                raise ValueError("Renderer did not return cameras in result")
            
            cameras_dict = result["cameras"]
            if not cameras_dict:
                raise ValueError("No cameras in render result")
            
            # Get the first camera image (or the requested camera)
            camera_name = request.camera_intrinsics.logical_id
            if camera_name in cameras_dict and "image" in cameras_dict[camera_name]:
                image = cameras_dict[camera_name]["image"]
            else:
                # Fallback to first available camera
                first_camera = next(iter(cameras_dict.values()))
                image = first_camera.get("image")
            
            if image is None:
                raise ValueError("No image found in render result")
            
            # Convert image to bytes
            # image should be a numpy array (BGR format from OpenCV)
            if isinstance(image, np.ndarray):
                # Encode as JPEG or PNG based on request
                if request.image_format == 1:  # PNG
                    success, image_bytes = cv2.imencode(".png", image)
                elif request.image_format == 2:  # JPEG
                    success, image_bytes = cv2.imencode(
                        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(request.image_quality)]
                    )
                else:
                    # Default to JPEG
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
            raise

    def render_aggregated(
        self,
        request: AggregatedRenderRequest,
        context: grpc.ServicerContext,
    ) -> AggregatedRenderReturn:
        """Render multiple RGB images (not fully implemented yet)."""
        # For now, render each request individually
        rgb_returns = []
        for rgb_request in request.rgb_requests:
            rgb_return = self.render_rgb(rgb_request, context)
            rgb_returns.append(rgb_return)
        
        return AggregatedRenderReturn(rgb_returns=rgb_returns)

    def render_lidar(
        self, request, context: grpc.ServicerContext
    ):
        """Render LiDAR data (not implemented yet)."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("LiDAR rendering not implemented for DigitalTwin")
        raise NotImplementedError("LiDAR rendering not implemented")

    def shut_down(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        """Shut down the service."""
        logger.info("shut_down")
        context.add_callback(self._shut_down)
        return Empty()

    def _shut_down(self) -> None:
        """Internal shutdown callback."""
        self.server.stop(0)
