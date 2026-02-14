# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Adapter for converting alpasim data formats to MTGS renderer formats.

This module provides conversion utilities between:
- alpasim gRPC messages (RGBRenderRequest, PosePair, etc.)
- MTGS RenderState and camera configurations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from pyquaternion import Quaternion

from alpasim_grpc.v0.common_pb2 import Pose as GrpcPose
from alpasim_grpc.v0.sensorsim_pb2 import (
    AvailableCamerasReturn,
    CameraSpec,
    DynamicObject,
    RGBRenderRequest,
)
from alpasim_utils.qvec import QVec
from alpasim_utils.scene_data_source import SceneDataSource

logger = logging.getLogger(__name__)


# ==================================================================================
# Nuplan SDK Standard Camera/Lidar Extrinsics (Fixed for all Pacifica vehicles)
# ==================================================================================
# These values are used to override potentially incorrect values from video_scene_dict.pkl
# All sensor2ego transformations are relative to the rear axle (ego coordinate origin).

# LIDAR_TOP to ego (rear axle) transformation
NUPLAN_LIDAR2EGO = {
    'translation': np.array([1.5185133218765259, 0.0, 1.6308990716934204]),
    'rotation': np.array([-0.0016505558783280307, -0.00023289146777086609, 
                          0.003725490480134295, 0.9999916710390838]),  # [x, y, z, w]
}

# Flag to enable/disable override (set to True to use Nuplan standard values)
USE_NUPLAN_STANDARD_EXTRINSICS = True

logger.info(f"USE_NUPLAN_STANDARD_EXTRINSICS: {USE_NUPLAN_STANDARD_EXTRINSICS}")
if USE_NUPLAN_STANDARD_EXTRINSICS:
    logger.info("Will override sensor2ego from video_scene_dict with values from pkl (which are already correct)")
    logger.info("Will override lidar2ego with Nuplan standard values")


def grpc_pose_to_quaternion_and_translation(grpc_pose: GrpcPose) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert gRPC Pose to quaternion and translation.
    
    Args:
        grpc_pose: gRPC Pose message
        
    Returns:
        Tuple of (quaternion [w, x, y, z], translation [x, y, z])
    """
    quat = np.array([grpc_pose.quat.w, grpc_pose.quat.x, grpc_pose.quat.y, grpc_pose.quat.z])
    trans = np.array([grpc_pose.vec.x, grpc_pose.vec.y, grpc_pose.vec.z])
    return quat, trans


def camera_spec_to_worldengine_format(
    camera_spec: CameraSpec, rig_to_camera: Optional[GrpcPose] = None
) -> Dict:
    """
    Convert alpasim CameraSpec to MTGS camera format.
    
    Args:
        camera_spec: CameraSpec from gRPC message
        rig_to_camera: Optional rig_to_camera pose transformation
        
    Returns:
        Dictionary in MTGS camera format:
        {
            "channel": str,
            "sensor2ego_rotation": np.ndarray (quaternion),
            "sensor2ego_translation": np.ndarray ([x, y, z]),
            "intrinsic": np.ndarray (3x3),
            "distortion": np.ndarray,
            "height": int,
            "width": int
        }
    """
    camera_dict = {
        "channel": camera_spec.logical_id,
        "height": camera_spec.resolution_h,
        "width": camera_spec.resolution_w,
    }
    
    # Extract intrinsic parameters
    if camera_spec.HasField("opencv_pinhole_param"):
        param = camera_spec.opencv_pinhole_param
        intrinsic = np.array([
            [param.focal_length_x, 0, param.principal_point_x],
            [0, param.focal_length_y, param.principal_point_y],
            [0, 0, 1],
        ], dtype=np.float32)
        
        # Combine radial and tangential distortion
        radial = list(param.radial_coeffs) if param.radial_coeffs else []
        tangential = list(param.tangential_coeffs) if param.tangential_coeffs else []
        # Pad to at least 5 elements (k1, k2, p1, p2, k3)
        distortion = (radial + [0.0] * (5 - len(radial)))[:5]
        if len(tangential) >= 2:
            distortion[2:4] = tangential[:2]
        distortion = np.array(distortion, dtype=np.float32)
        
        camera_dict["intrinsic"] = intrinsic
        camera_dict["distortion"] = distortion
    elif camera_spec.HasField("opencv_fisheye_param"):
        param = camera_spec.opencv_fisheye_param
        intrinsic = np.array([
            [param.focal_length_x, 0, param.principal_point_x],
            [0, param.focal_length_y, param.principal_point_y],
            [0, 0, 1],
        ], dtype=np.float32)
        distortion = np.array(list(param.radial_coeffs) if param.radial_coeffs else [0.0] * 4, dtype=np.float32)
        camera_dict["intrinsic"] = intrinsic
        camera_dict["distortion"] = distortion
    else:
        logger.warning(f"Unsupported camera parameter type for {camera_spec.logical_id}")
        # Default intrinsic matrix
        camera_dict["intrinsic"] = np.eye(3, dtype=np.float32)
        camera_dict["distortion"] = np.zeros(5, dtype=np.float32)
    
    # Extract sensor2ego transformation
    if rig_to_camera is not None:
        # rig_to_camera is actually camera_to_rig in the proto (see comment in proto)
        # We need camera_to_ego, which is the inverse of rig_to_camera
        quat, trans = grpc_pose_to_quaternion_and_translation(rig_to_camera)
        # Invert the transformation
        q_inv = Quaternion(quat[0], quat[1], quat[2], quat[3]).inverse
        camera_dict["sensor2ego_rotation"] = np.array([q_inv.w, q_inv.x, q_inv.y, q_inv.z])
        # Translation inversion: t' = -R^T * t
        rot_matrix = q_inv.rotation_matrix
        camera_dict["sensor2ego_translation"] = -rot_matrix.T @ trans
    else:
        # Default: identity transformation
        camera_dict["sensor2ego_rotation"] = np.array([1.0, 0.0, 0.0, 0.0])
        camera_dict["sensor2ego_translation"] = np.array([0.0, 0.0, 0.0])
    
    return camera_dict


def pose_pair_to_agent_state(
    start_pose: GrpcPose, end_pose: GrpcPose, use_start: bool = True
) -> np.ndarray:
    """
    Convert PosePair to MTGS agent_state format [x, y, z, 0, 0, heading].
    
    Args:
        start_pose: Start pose from PosePair
        end_pose: End pose from PosePair
        use_start: If True, use start_pose; otherwise use end_pose
        
    Returns:
        numpy array [x, y, z, 0, 0, heading] where heading is yaw angle in radians
    """
    pose = start_pose if use_start else end_pose
    quat, trans = grpc_pose_to_quaternion_and_translation(pose)
    
    # Extract yaw from quaternion
    q = Quaternion(quat[0], quat[1], quat[2], quat[3])
    yaw = q.yaw_pitch_roll[0]
    
    # MTGS format: [x, y, z, 0, 0, heading]
    return np.array([trans[0], trans[1], trans[2], 0.0, 0.0, yaw], dtype=np.float64)


def rgb_render_request_to_render_state(
    request: RGBRenderRequest,
    asset_manager: Optional[Any] = None,
) -> Dict:
    """
    Convert RGBRenderRequest to MTGS RenderState format.
    
    Simplified version: Only extracts scene_id, AGENT_STATE, and TIMESTAMP from request.
    Camera information is loaded from asset_manager's video_scene_dict (pkl file).
    
    Args:
        request: RGBRenderRequest from gRPC (only scene_id, agent_state, timestamp are used)
        data_source: Optional SceneDataSource (used for backward compatibility with sensor_pose)
        asset_manager: MTGSAssetManager for accessing video_scene_dict (required for cameras)
        
    Returns:
        Dictionary compatible with MTGS RenderState:
        {
            "timestamp": int (microseconds),
            "agent_state": Dict[str, np.ndarray],  # {object_id: [x, y, z, 0, 0, heading]}
            "cameras": Dict[str, Dict],  # {camera_name: camera_config} - loaded from pkl
        }
    """
    # Extract timestamp
    timestamp = request.frame_start_us
    
    # Extract ego agent state
    # Priority: 1) ego_pose (new field), 2) sensor_pose (backward compatibility)
    agent_states = {}
    if request.HasField("ego_pose"):
        # Use ego_pose directly (preferred for MTGS renderer)
        ego_state = pose_pair_to_agent_state(
            request.ego_pose.start_pose,
            request.ego_pose.end_pose,
            use_start=True
        )
        agent_states["ego"] = ego_state
    elif request.HasField("sensor_pose"):
        # Fallback: use sensor_pose as ego_pose (may be inaccurate but allows testing)
        logger.warning(
            "ego_pose not found in request, using sensor_pose as fallback. "
            "Consider updating Runtime to send ego_pose field."
        )
        ego_state = pose_pair_to_agent_state(
            request.sensor_pose.start_pose,
            request.sensor_pose.end_pose,
            use_start=True
        )
        agent_states["ego"] = ego_state
    else:
        raise ValueError("RGBRenderRequest must contain either ego_pose or sensor_pose")

    # Extract traffic agent states from dynamic_objects
    for dyn_obj in request.dynamic_objects:
        track_id = dyn_obj.track_id
        agent_state = pose_pair_to_agent_state(
            dyn_obj.pose_pair.start_pose,
            dyn_obj.pose_pair.end_pose,
            use_start=True
        )
        agent_states[track_id] = agent_state
    
    # Load camera configuration from asset_manager's video_scene_dict (pkl file)
    cameras = {}
    if asset_manager is not None and hasattr(asset_manager, 'video_scene_dict') and asset_manager.video_scene_dict:
        
        video_info_raw = asset_manager.video_scene_dict
        
        # Handle nested structure: video_scene_dict may be {scene_id: {frame_infos, ...}}
        # or directly {frame_infos, ...}
        video_info = video_info_raw
        if isinstance(video_info_raw, dict) and 'frame_infos' not in video_info_raw:
            # Nested structure - get the first (and usually only) scene data
            first_key = next(iter(video_info_raw.keys()), None)
            if first_key and isinstance(video_info_raw[first_key], dict):
                video_info = video_info_raw[first_key]
        
        if 'frame_infos' not in video_info or len(video_info['frame_infos']) == 0:
            logger.warning("No frame_infos found in video_scene_dict")
        else:
            frame_info = video_info['frame_infos'][0]
            if 'cams' not in frame_info:
                logger.warning("No 'cams' found in frame_infos[0]")
            else:
                for cam_name, mtgs_cam_info in frame_info['cams'].items():
                    # Extract intrinsic and distortion (prefer colmap_param if available)
                    # These are from colmap optimization and should be used as-is
                    if 'colmap_param' in mtgs_cam_info:
                        intrinsic = np.array(mtgs_cam_info['colmap_param']['cam_intrinsic'])
                        distortion = np.array(mtgs_cam_info['colmap_param']['distortion'])
                    else:
                        intrinsic = np.array(mtgs_cam_info['cam_intrinsic'])
                        distortion = np.array(mtgs_cam_info['distortion'])
                    
                    # Extract sensor2ego transformation from pkl
                    # NOTE: Our analysis shows that sensor2ego in video_scene_dict is ALREADY correct
                    # (relative to rear axle, matching Nuplan SDK standard)
                    # So we use it directly without override
                    sensor2ego_rotation = mtgs_cam_info['sensor2ego_rotation']
                    if isinstance(sensor2ego_rotation, Quaternion):
                        sensor2ego_rotation = sensor2ego_rotation.elements
                    else:
                        sensor2ego_rotation = np.array(sensor2ego_rotation)
                    
                    sensor2ego_translation = np.array(mtgs_cam_info['sensor2ego_translation'])
                    
                    # Get resolution (default to 1080x1920)
                    height = mtgs_cam_info.get('height', 1080)
                    width = mtgs_cam_info.get('width', 1920)
                    
                    cameras[cam_name] = {
                        'channel': cam_name,
                        'sensor2ego_rotation': sensor2ego_rotation,
                        'sensor2ego_translation': sensor2ego_translation,
                        'intrinsic': intrinsic,
                        'distortion': distortion,
                        'height': height,
                        'width': width,
                    }
    else:
        logger.warning("asset_manager or video_scene_dict not available, cameras will be empty")
    
    # Setup lidar parameters with Nuplan standard values
    # The key issue: video_scene_dict may have lidar2ego=[0,0,0], which is incorrect
    # We override it with the standard Nuplan value here
    lidar_params = {}
    if USE_NUPLAN_STANDARD_EXTRINSICS:
        lidar_params = {
            'channel': 'LIDAR_TOP',
            'sensor2ego_translation': NUPLAN_LIDAR2EGO['translation'],
            'sensor2ego_rotation': NUPLAN_LIDAR2EGO['rotation'],
        }
    
    render_state = {
        "timestamp": timestamp,
        "agent_state": agent_states,
        "cameras": cameras,
        "lidar": lidar_params,
    }
    
    return render_state


def get_available_cameras_from_data_source(
    asset_manager: Optional[Any] = None,
) -> list[AvailableCamerasReturn.AvailableCamera]:
    """
    Extract available cameras from SceneDataSource or video_info.

    If asset_manager.video_scene_dict is available, extract cameras from pkl file.
    Otherwise, create default NuPlan cameras.

    Args:
        data_source: SceneDataSource instance
        asset_manager: Optional asset manager with video_scene_dict

    Returns:
        List of AvailableCamera messages
    """
    import numpy as np
    from pyquaternion import Quaternion
    
    cameras = []
    
    # Try to get cameras from video_info (pkl file)
    video_info = None
    if asset_manager is not None and hasattr(asset_manager, 'video_scene_dict') and asset_manager.video_scene_dict:
        try:
            # video_scene_dict is a dict, get the first video_info
            video_scene_dict = asset_manager.video_scene_dict
            if isinstance(video_scene_dict, dict):
                video_info = list(video_scene_dict.values())[0]
            else:
                video_info = video_scene_dict
        except Exception as e:
            logger.warning(f"Failed to get video_info from asset_manager: {e}")
    
    if video_info and 'frame_infos' in video_info and len(video_info['frame_infos']) > 0:
        # Extract cameras from video_info (pkl file)
        logger.info("Extracting cameras from video_info (pkl file)")
        frame_info = video_info['frame_infos'][0]
        
        for cam_name, cam_info in frame_info['cams'].items():
            # 1. Extract intrinsic parameters
            if 'colmap_param' in cam_info:
                intrinsic = np.array(cam_info['colmap_param']['cam_intrinsic'])
                distortion = np.array(cam_info['colmap_param']['distortion'])
            else:
                intrinsic = np.array(cam_info['cam_intrinsic'])
                distortion = np.array(cam_info.get('distortion', np.zeros(5)))
            
            # Extract focal length and principal point from intrinsic matrix
            # Intrinsic matrix format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            focal_length_x = float(intrinsic[0, 0])
            focal_length_y = float(intrinsic[1, 1])
            principal_point_x = float(intrinsic[0, 2])
            principal_point_y = float(intrinsic[1, 2])
            
            # 2. Extract extrinsic parameters (sensor2ego)
            sensor2ego_rotation = cam_info['sensor2ego_rotation']
            if isinstance(sensor2ego_rotation, Quaternion):
                quat_elements = sensor2ego_rotation.elements  # [w, x, y, z]
            else:
                quat_elements = np.array(sensor2ego_rotation)
            
            sensor2ego_translation = np.array(cam_info['sensor2ego_translation'])
            
            # 3. Get resolution
            height = cam_info.get('height', 1080)
            width = cam_info.get('width', 1920)
            
            # 4. Create CameraSpec
            camera_spec = CameraSpec(
                logical_id=cam_name,
                resolution_h=int(height),
                resolution_w=int(width),
            )
            
            # Set pinhole intrinsics from matrix
            pinhole = camera_spec.opencv_pinhole_param
            pinhole.focal_length_x = focal_length_x
            pinhole.focal_length_y = focal_length_y
            pinhole.principal_point_x = principal_point_x
            pinhole.principal_point_y = principal_point_y
            
            # 5. Create rig_to_camera pose
            # Note: sensor2ego means ego->sensor transformation
            # rig_to_camera means rig->camera transformation
            # In nuPlan, ego and rig are typically the same, so we can use sensor2ego directly
            # But rig_to_camera might need to be the inverse (camera->rig)
            # Check your coordinate system definition!
            # For now, assuming rig=ego and rig_to_camera = sensor2ego (ego->sensor = rig->camera)
            
            rig_to_camera = GrpcPose()
            rig_to_camera.vec.x = float(sensor2ego_translation[0])
            rig_to_camera.vec.y = float(sensor2ego_translation[1])
            rig_to_camera.vec.z = float(sensor2ego_translation[2])
            rig_to_camera.quat.w = float(quat_elements[0])  # w is first element
            rig_to_camera.quat.x = float(quat_elements[1])
            rig_to_camera.quat.y = float(quat_elements[2])
            rig_to_camera.quat.z = float(quat_elements[3])
            
            available_camera = AvailableCamerasReturn.AvailableCamera(
                intrinsics=camera_spec,
                rig_to_camera=rig_to_camera,
                logical_id=cam_name,
            )
            cameras.append(available_camera)
        
        logger.info(f"Extracted {len(cameras)} cameras from video_info: {[c.logical_id for c in cameras]}")
        return cameras
    
    # Fallback: Create default cameras if video_info is not available
    logger.info("No cameras found in video_info, creating default NuPlan cameras")
    default_cameras = [
        "CAM_F0",
        "CAM_L0", "CAM_L1", "CAM_L2",
        "CAM_R0", "CAM_R1", "CAM_R2",
        "CAM_B0"
    ]

    for camera_name in default_cameras:
        camera_spec = CameraSpec(
            logical_id=camera_name,
            resolution_h=1080,
            resolution_w=1920,
        )

        # Set default pinhole intrinsics
        pinhole = camera_spec.opencv_pinhole_param
        pinhole.focal_length_x = 1920.0
        pinhole.focal_length_y = 1080.0
        pinhole.principal_point_x = 960.0
        pinhole.principal_point_y = 560.0

        # Create identity pose (camera at rig origin, no rotation)
        rig_to_camera = GrpcPose()
        rig_to_camera.vec.x = 0.0
        rig_to_camera.vec.y = 0.0
        rig_to_camera.vec.z = 0.0
        rig_to_camera.quat.w = 1.0
        rig_to_camera.quat.x = 0.0
        rig_to_camera.quat.y = 0.0
        rig_to_camera.quat.z = 0.0

        available_camera = AvailableCamerasReturn.AvailableCamera(
            intrinsics=camera_spec,
            rig_to_camera=rig_to_camera,
            logical_id=camera_name,
        )
        cameras.append(available_camera)

    logger.info(f"Created {len(cameras)} default cameras: {[c.logical_id for c in cameras]}")
    return cameras
