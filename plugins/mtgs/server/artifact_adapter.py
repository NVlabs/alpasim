# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Adapter for converting alpasim gRPC messages to MTGS renderer formats."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from alpasim_grpc.v0.common_pb2 import Pose as GrpcPose
from alpasim_grpc.v0.sensorsim_pb2 import (
    AvailableCamerasReturn,
    CameraSpec,
    RGBRenderRequest,
)
from pyquaternion import Quaternion

logger = logging.getLogger(__name__)

try:
    from trajdata.dataset_specific.nuplan.nuplan_utils import (
        NUPLAN_REAL_LIDAR2EGO_ROTATION,
        NUPLAN_REAL_LIDAR2EGO_TRANSLATION,
    )

    NUPLAN_LIDAR2EGO = {
        "translation": np.array(NUPLAN_REAL_LIDAR2EGO_TRANSLATION),
        "rotation": np.array(NUPLAN_REAL_LIDAR2EGO_ROTATION),
    }
except ImportError:
    NUPLAN_LIDAR2EGO = {
        "translation": np.array([1.5185133218765259, 0.0, 1.6308990716934204]),
        "rotation": np.array(
            [
                -0.0016505558783280307,
                -0.00023289146777086609,
                0.003725490480134295,
                0.9999916710390838,
            ]
        ),
    }

USE_NUPLAN_STANDARD_EXTRINSICS = True


def grpc_pose_to_quaternion_and_translation(
    grpc_pose: GrpcPose,
) -> tuple[np.ndarray, np.ndarray]:
    quat = np.array(
        [grpc_pose.quat.w, grpc_pose.quat.x, grpc_pose.quat.y, grpc_pose.quat.z]
    )
    trans = np.array([grpc_pose.vec.x, grpc_pose.vec.y, grpc_pose.vec.z])
    return quat, trans


def camera_spec_to_worldengine_format(
    camera_spec: CameraSpec, rig_to_camera: Optional[GrpcPose] = None
) -> Dict:
    camera_dict = {
        "channel": camera_spec.logical_id,
        "height": camera_spec.resolution_h,
        "width": camera_spec.resolution_w,
    }

    if camera_spec.HasField("opencv_pinhole_param"):
        param = camera_spec.opencv_pinhole_param
        intrinsic = np.array(
            [
                [param.focal_length_x, 0, param.principal_point_x],
                [0, param.focal_length_y, param.principal_point_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        radial = list(param.radial_coeffs) if param.radial_coeffs else []
        tangential = list(param.tangential_coeffs) if param.tangential_coeffs else []
        distortion = (radial + [0.0] * (5 - len(radial)))[:5]
        if len(tangential) >= 2:
            distortion[2:4] = tangential[:2]
        distortion = np.array(distortion, dtype=np.float32)

        camera_dict["intrinsic"] = intrinsic
        camera_dict["distortion"] = distortion
    elif camera_spec.HasField("opencv_fisheye_param"):
        param = camera_spec.opencv_fisheye_param
        intrinsic = np.array(
            [
                [param.focal_length_x, 0, param.principal_point_x],
                [0, param.focal_length_y, param.principal_point_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        distortion = np.array(
            list(param.radial_coeffs) if param.radial_coeffs else [0.0] * 4,
            dtype=np.float32,
        )
        camera_dict["intrinsic"] = intrinsic
        camera_dict["distortion"] = distortion
    else:
        logger.warning(
            f"Unsupported camera parameter type for {camera_spec.logical_id}"
        )
        camera_dict["intrinsic"] = np.eye(3, dtype=np.float32)
        camera_dict["distortion"] = np.zeros(5, dtype=np.float32)

    if rig_to_camera is not None:
        quat, trans = grpc_pose_to_quaternion_and_translation(rig_to_camera)
        q_inv = Quaternion(quat[0], quat[1], quat[2], quat[3]).inverse
        camera_dict["sensor2ego_rotation"] = np.array(
            [q_inv.w, q_inv.x, q_inv.y, q_inv.z]
        )
        rot_matrix = q_inv.rotation_matrix
        camera_dict["sensor2ego_translation"] = -rot_matrix.T @ trans
    else:
        camera_dict["sensor2ego_rotation"] = np.array([1.0, 0.0, 0.0, 0.0])
        camera_dict["sensor2ego_translation"] = np.array([0.0, 0.0, 0.0])

    return camera_dict


def pose_pair_to_agent_state(
    start_pose: GrpcPose, end_pose: GrpcPose, use_start: bool = True
) -> np.ndarray:
    pose = start_pose if use_start else end_pose
    quat, trans = grpc_pose_to_quaternion_and_translation(pose)
    q = Quaternion(quat[0], quat[1], quat[2], quat[3])
    yaw = q.yaw_pitch_roll[0]
    return np.array([trans[0], trans[1], trans[2], 0.0, 0.0, yaw], dtype=np.float64)


def rgb_render_request_to_render_state(
    request: RGBRenderRequest,
    asset_manager: Optional[Any] = None,
) -> Dict:
    timestamp = request.frame_start_us

    agent_states = {}
    if request.HasField("ego_pose"):
        ego_state = pose_pair_to_agent_state(
            request.ego_pose.start_pose, request.ego_pose.end_pose, use_start=True
        )
        agent_states["ego"] = ego_state
    elif request.HasField("sensor_pose"):
        logger.warning("ego_pose not found in request, using sensor_pose as fallback.")
        ego_state = pose_pair_to_agent_state(
            request.sensor_pose.start_pose, request.sensor_pose.end_pose, use_start=True
        )
        agent_states["ego"] = ego_state
    else:
        raise ValueError("RGBRenderRequest must contain either ego_pose or sensor_pose")

    for dyn_obj in request.dynamic_objects:
        track_id = dyn_obj.track_id
        agent_state = pose_pair_to_agent_state(
            dyn_obj.pose_pair.start_pose, dyn_obj.pose_pair.end_pose, use_start=True
        )
        agent_states[track_id] = agent_state

    cameras = {}
    if (
        asset_manager is not None
        and hasattr(asset_manager, "video_scene_dict")
        and asset_manager.video_scene_dict
    ):
        video_info_raw = asset_manager.video_scene_dict
        video_info = video_info_raw
        if isinstance(video_info_raw, dict) and "frame_infos" not in video_info_raw:
            first_key = next(iter(video_info_raw.keys()), None)
            if first_key and isinstance(video_info_raw[first_key], dict):
                video_info = video_info_raw[first_key]

        if "frame_infos" not in video_info or len(video_info["frame_infos"]) == 0:
            logger.warning("No frame_infos found in video_scene_dict")
        else:
            frame_info = video_info["frame_infos"][0]
            if "cams" not in frame_info:
                logger.warning("No 'cams' found in frame_infos[0]")
            else:
                for cam_name, mtgs_cam_info in frame_info["cams"].items():
                    if "colmap_param" in mtgs_cam_info:
                        intrinsic = np.array(
                            mtgs_cam_info["colmap_param"]["cam_intrinsic"]
                        )
                        distortion = np.array(
                            mtgs_cam_info["colmap_param"]["distortion"]
                        )
                    else:
                        intrinsic = np.array(mtgs_cam_info["cam_intrinsic"])
                        distortion = np.array(mtgs_cam_info["distortion"])

                    sensor2ego_rotation = mtgs_cam_info["sensor2ego_rotation"]
                    if isinstance(sensor2ego_rotation, Quaternion):
                        sensor2ego_rotation = sensor2ego_rotation.elements
                    else:
                        sensor2ego_rotation = np.array(sensor2ego_rotation)

                    sensor2ego_translation = np.array(
                        mtgs_cam_info["sensor2ego_translation"]
                    )

                    height = mtgs_cam_info.get("height", 1080)
                    width = mtgs_cam_info.get("width", 1920)

                    cameras[cam_name] = {
                        "channel": cam_name,
                        "sensor2ego_rotation": sensor2ego_rotation,
                        "sensor2ego_translation": sensor2ego_translation,
                        "intrinsic": intrinsic,
                        "distortion": distortion,
                        "height": height,
                        "width": width,
                    }

    lidar_params = {}
    if USE_NUPLAN_STANDARD_EXTRINSICS:
        lidar_params = {
            "channel": "LIDAR_TOP",
            "sensor2ego_translation": NUPLAN_LIDAR2EGO["translation"],
            "sensor2ego_rotation": NUPLAN_LIDAR2EGO["rotation"],
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
    cameras = []

    video_info = None
    if (
        asset_manager is not None
        and hasattr(asset_manager, "video_scene_dict")
        and asset_manager.video_scene_dict
    ):
        try:
            video_scene_dict = asset_manager.video_scene_dict
            if isinstance(video_scene_dict, dict):
                video_info = list(video_scene_dict.values())[0]
            else:
                video_info = video_scene_dict
        except Exception as e:
            logger.warning(f"Failed to get video_info from asset_manager: {e}")

    if (
        video_info
        and "frame_infos" in video_info
        and len(video_info["frame_infos"]) > 0
    ):
        logger.info("Extracting cameras from video_info (pkl file)")
        frame_info = video_info["frame_infos"][0]

        for cam_name, cam_info in frame_info["cams"].items():
            if "colmap_param" in cam_info:
                intrinsic = np.array(cam_info["colmap_param"]["cam_intrinsic"])
            else:
                intrinsic = np.array(cam_info["cam_intrinsic"])

            focal_length_x = float(intrinsic[0, 0])
            focal_length_y = float(intrinsic[1, 1])
            principal_point_x = float(intrinsic[0, 2])
            principal_point_y = float(intrinsic[1, 2])

            sensor2ego_rotation = cam_info["sensor2ego_rotation"]
            if isinstance(sensor2ego_rotation, Quaternion):
                quat_elements = sensor2ego_rotation.elements
            else:
                quat_elements = np.array(sensor2ego_rotation)

            sensor2ego_translation = np.array(cam_info["sensor2ego_translation"])

            height = cam_info.get("height", 1080)
            width = cam_info.get("width", 1920)

            camera_spec = CameraSpec(
                logical_id=cam_name,
                resolution_h=int(height),
                resolution_w=int(width),
            )

            pinhole = camera_spec.opencv_pinhole_param
            pinhole.focal_length_x = focal_length_x
            pinhole.focal_length_y = focal_length_y
            pinhole.principal_point_x = principal_point_x
            pinhole.principal_point_y = principal_point_y

            rig_to_camera = GrpcPose()
            rig_to_camera.vec.x = float(sensor2ego_translation[0])
            rig_to_camera.vec.y = float(sensor2ego_translation[1])
            rig_to_camera.vec.z = float(sensor2ego_translation[2])
            rig_to_camera.quat.w = float(quat_elements[0])
            rig_to_camera.quat.x = float(quat_elements[1])
            rig_to_camera.quat.y = float(quat_elements[2])
            rig_to_camera.quat.z = float(quat_elements[3])

            available_camera = AvailableCamerasReturn.AvailableCamera(
                intrinsics=camera_spec,
                rig_to_camera=rig_to_camera,
                logical_id=cam_name,
            )
            cameras.append(available_camera)

        logger.info(
            f"Extracted {len(cameras)} cameras from video_info: {[c.logical_id for c in cameras]}"
        )
        return cameras

    logger.info("No cameras found in video_info, creating default NuPlan cameras")
    default_cameras = [
        "CAM_F0",
        "CAM_L0",
        "CAM_L1",
        "CAM_L2",
        "CAM_R0",
        "CAM_R1",
        "CAM_R2",
        "CAM_B0",
    ]

    for camera_name in default_cameras:
        camera_spec = CameraSpec(
            logical_id=camera_name,
            resolution_h=1080,
            resolution_w=1920,
        )
        pinhole = camera_spec.opencv_pinhole_param
        pinhole.focal_length_x = 1920.0
        pinhole.focal_length_y = 1080.0
        pinhole.principal_point_x = 960.0
        pinhole.principal_point_y = 560.0

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

    logger.info(
        f"Created {len(cameras)} default cameras: {[c.logical_id for c in cameras]}"
    )
    return cameras
