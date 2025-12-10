# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Adapter for converting alpasim data formats to worldengine DigitalTwin renderer formats.

This module provides conversion utilities between:
- alpasim gRPC messages (RGBRenderRequest, PosePair, etc.)
- worldengine RenderState and camera configurations
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

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
    Convert alpasim CameraSpec to worldengine camera format.
    
    Args:
        camera_spec: CameraSpec from gRPC message
        rig_to_camera: Optional rig_to_camera pose transformation
        
    Returns:
        Dictionary in worldengine camera format:
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
    Convert PosePair to worldengine agent_state format [x, y, z, 0, 0, heading].
    
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
    
    # worldengine format: [x, y, z, 0, 0, heading]
    return np.array([trans[0], trans[1], trans[2], 0.0, 0.0, yaw], dtype=np.float64)


def rgb_render_request_to_render_state(
    request: RGBRenderRequest,
    data_source: Optional[SceneDataSource] = None,
) -> Dict:
    """
    Convert RGBRenderRequest to worldengine RenderState format.
    
    Args:
        request: RGBRenderRequest from gRPC
        data_source: Optional SceneDataSource for additional context
        
    Returns:
        Dictionary compatible with worldengine RenderState:
        {
            "timestamp": int (microseconds),
            "agent_state": Dict[str, np.ndarray],  # {object_id: [x, y, z, 0, 0, heading]}
            "cameras": Dict[str, Dict],  # {camera_name: camera_config}
        }
    """
    # Extract timestamp
    timestamp = request.frame_start_us
    
    # Extract ego agent state from sensor_pose
    agent_states = {}
    if request.HasField("sensor_pose"):
        # Note: sensor_pose is in camera frame, we need to convert to ego frame
        # For now, we'll use the start pose as ego pose
        # In a full implementation, we'd need to apply camera_to_rig transformation
        ego_state = pose_pair_to_agent_state(
            request.sensor_pose.start_pose,
            request.sensor_pose.end_pose,
            use_start=True
        )
        agent_states["ego"] = ego_state
    
    # Extract traffic agent states from dynamic_objects
    for dyn_obj in request.dynamic_objects:
        track_id = dyn_obj.track_id
        agent_state = pose_pair_to_agent_state(
            dyn_obj.pose_pair.start_pose,
            dyn_obj.pose_pair.end_pose,
            use_start=True
        )
        agent_states[track_id] = agent_state
    
    # Extract camera configuration
    cameras = {}
    if request.HasField("camera_intrinsics"):
        # For single camera request, create camera dict
        camera_dict = camera_spec_to_worldengine_format(request.camera_intrinsics)
        cameras[camera_dict["channel"]] = camera_dict
    
    render_state = {
        "timestamp": timestamp,
        "agent_state": agent_states,
        "cameras": cameras,
        "lidar": {},  # Empty for now, can be extended
    }
    
    return render_state


def get_available_cameras_from_data_source(
    data_source: SceneDataSource,
) -> list[AvailableCamerasReturn.AvailableCamera]:
    """
    Extract available cameras from SceneDataSource.
    
    Args:
        data_source: SceneDataSource instance
        
    Returns:
        List of AvailableCamera messages
    """
    cameras = []
    
    # Try to get cameras from rig
    try:
        rig = data_source.rig
        for camera_id in rig.camera_ids:
            # Create a basic CameraSpec (would need full calibration data in real implementation)
            camera_spec = CameraSpec(
                logical_id=camera_id.logical_name,
                resolution_h=1080,  # Default, should be from metadata
                resolution_w=1920,  # Default, should be from metadata
            )
            
            # Create rig_to_camera pose (would need actual calibration data)
            rig_to_camera = GrpcPose(
                vec=GrpcPose.Vec3(x=0.0, y=0.0, z=0.0),
                quat=GrpcPose.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
            )
            
            available_camera = AvailableCamerasReturn.AvailableCamera(
                intrinsics=camera_spec,
                rig_to_camera=rig_to_camera,
                logical_id=camera_id.logical_name,
            )
            cameras.append(available_camera)
    except Exception as e:
        logger.warning(f"Could not extract cameras from data source: {e}")
    
    return cameras
