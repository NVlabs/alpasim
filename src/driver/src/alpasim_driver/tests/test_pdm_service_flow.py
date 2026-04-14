# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
import base64
import json
import pickle
from io import BytesIO
from pathlib import Path

import grpc.aio
import numpy as np
import pytest
from alpasim_grpc.v0.common_pb2 import DynamicState, Pose, PoseAtTime, Quat, Vec3
from alpasim_grpc.v0.common_pb2 import Trajectory as TrajectoryMsg
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
)
from omegaconf import OmegaConf
from PIL import Image

from ..main import EgoDriverService, _DRIVER_CONTROL_PREFIX
from ..schema import DriverConfig


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _make_png_bytes() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _encode_planner_context(planner_context: dict) -> bytes:
    payload = {
        "planner_context": planner_context,
        "renderer_data_b64": base64.b64encode(b"").decode("ascii"),
    }
    return _DRIVER_CONTROL_PREFIX + json.dumps(payload, separators=(",", ":")).encode(
        "utf-8"
    )


@pytest.mark.asyncio
async def test_pdm_drive_response_contains_debug_metadata(tmp_path: Path) -> None:
    repo_root = _get_repo_root()
    cfg_path = repo_root / "src" / "wizard" / "configs" / "driver" / "pdm.yaml"
    raw_cfg = OmegaConf.load(cfg_path)
    if "defaults" in raw_cfg:
        del raw_cfg["defaults"]
    raw_cfg.output_dir = str(tmp_path)
    raw_cfg.port = 0

    schema = OmegaConf.structured(DriverConfig)
    cfg = OmegaConf.merge(schema, raw_cfg)

    loop = asyncio.get_running_loop()
    server = grpc.aio.server()
    service = EgoDriverService(cfg=cfg, loop=loop, grpc_server=server)

    session_uuid = "pdm-session"
    camera_id = cfg.inference.use_cameras[0]
    rollout_spec = DriveSessionRequest.RolloutSpec()
    camera_def = rollout_spec.vehicle.available_cameras.add()
    camera_def.logical_id = camera_id
    camera_def.intrinsics.resolution_h = 32
    camera_def.intrinsics.resolution_w = 32
    ftheta = camera_def.intrinsics.ftheta_param
    ftheta.principal_point_x = 16.0
    ftheta.principal_point_y = 16.0
    ftheta.angle_to_pixeldist_poly.extend([0.0, 1.0])
    ftheta.pixeldist_to_angle_poly.extend([0.0, 1.0])

    start_request = DriveSessionRequest(
        session_uuid=session_uuid,
        random_seed=0,
        rollout_spec=rollout_spec,
    )

    try:
        await service.start_session(start_request, None)

        pose_ts = 1_000_000
        traj_msg = TrajectoryMsg()
        traj_msg.poses.append(
            PoseAtTime(
                pose=Pose(
                    vec=Vec3(x=0.0, y=0.0, z=0.0),
                    quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                timestamp_us=pose_ts,
            )
        )
        dynamic_state = DynamicState(
            linear_velocity=Vec3(x=6.0, y=0.0, z=0.0),
            angular_velocity=Vec3(x=0.0, y=0.0, z=0.0),
            linear_acceleration=Vec3(x=0.0, y=0.0, z=0.0),
            angular_acceleration=Vec3(x=0.0, y=0.0, z=0.0),
        )
        await service.submit_egomotion_observation(
            RolloutEgoTrajectory(
                session_uuid=session_uuid,
                trajectory=traj_msg,
                dynamic_states=[dynamic_state],
            ),
            None,
        )
        await service.submit_image_observation(
            RolloutCameraImage(
                session_uuid=session_uuid,
                camera_image=RolloutCameraImage.CameraImage(
                    logical_id=camera_id,
                    frame_start_us=pose_ts,
                    frame_end_us=pose_ts + 50_000,
                    image_bytes=_make_png_bytes(),
                ),
            ),
            None,
        )

        planner_context = {
            "ego": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
            "route_waypoints_in_rig": [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [24.0, 0.5, 0.0]],
            "nearby_lanes": [
                {
                    "id": "lane_main",
                    "centerline_in_rig": [[0.0, 0.0], [12.0, 0.0], [24.0, 0.5]],
                }
            ],
            "actors": [{"id": "lead", "position_in_rig": [18.0, 0.1], "yaw": 0.0}],
            "traffic_rules": {
                "wait_lines_in_rig": [{"type": "Stop", "points": [[8.0, -1.0], [8.0, 1.0]]}],
                "crosswalks_in_rig": [],
            },
        }
        response: DriveResponse = await service.drive(
            DriveRequest(
                session_uuid=session_uuid,
                time_now_us=pose_ts,
                time_query_us=pose_ts + 100_000,
                renderer_data=_encode_planner_context(planner_context),
            ),
            None,
        )

        assert len(response.trajectory.poses) > 1
        debug_info = pickle.loads(response.debug_info.unstructured_debug_info)
        assert debug_info["selected_model_type"] == "pdm"
        assert debug_info["planner_backend"] == "pdm_closed"
        assert debug_info["proposal_count"] >= 1
        assert debug_info["route_available"] is True
        assert debug_info["actor_count"] == 1
        assert debug_info["wait_line_count"] == 1
    finally:
        await service.stop_worker()
        if session_uuid in service._sessions:
            del service._sessions[session_uuid]
