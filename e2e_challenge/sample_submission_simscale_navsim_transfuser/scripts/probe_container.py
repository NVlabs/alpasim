# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import argparse
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc, sensorsim_pb2
from PIL import Image

import grpc

CAMERAS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)
NOW_US = 1_000_000
QUERY_US = 1_500_000
VERSION_ID = "simscale-ltf-navtest-e2e"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and greater than 0")
    return timeout


def jpeg_bytes(value: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1920, 1080), (value, value, value)).save(
        buffer,
        "JPEG",
        quality=90,
    )
    return buffer.getvalue()


def start_request(
    session_uuid: str,
    seed: int,
) -> egodriver_pb2.DriveSessionRequest:
    cameras = []
    for logical_id in CAMERAS:
        camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
            logical_id=logical_id
        )
        camera.intrinsics.logical_id = logical_id
        camera.intrinsics.resolution_h = 1080
        camera.intrinsics.resolution_w = 1920
        cameras.append(camera)
    return egodriver_pb2.DriveSessionRequest(
        session_uuid=session_uuid,
        random_seed=seed,
        rollout_spec=egodriver_pb2.DriveSessionRequest.RolloutSpec(
            vehicle=egodriver_pb2.DriveSessionRequest.RolloutSpec.VehicleDefinition(
                available_cameras=cameras
            )
        ),
    )


def seed_session(
    stub: egodriver_pb2_grpc.EgodriverServiceStub,
    session_uuid: str,
    x: float,
    pixel: int,
    route_y: float,
    timeout: float,
) -> None:
    stub.start_session(start_request(session_uuid, pixel), timeout=timeout)
    pose = common_pb2.PoseAtTime(
        timestamp_us=NOW_US,
        pose=common_pb2.Pose(vec=common_pb2.Vec3(x=x)),
    )
    pose.pose.quat.w = 1.0
    stub.submit_egomotion_observation(
        egodriver_pb2.RolloutEgoTrajectory(
            session_uuid=session_uuid,
            trajectory=common_pb2.Trajectory(poses=[pose]),
            dynamic_states=[
                common_pb2.DynamicState(linear_velocity=common_pb2.Vec3(x=5.0))
            ],
        ),
        timeout=timeout,
    )
    stub.submit_route(
        egodriver_pb2.RouteRequest(
            session_uuid=session_uuid,
            route=egodriver_pb2.Route(
                timestamp_us=NOW_US,
                waypoints=[common_pb2.Vec3(x=20.0, y=route_y)],
            ),
        ),
        timeout=timeout,
    )
    payload = jpeg_bytes(pixel)
    for logical_id in CAMERAS:
        stub.submit_image_observation(
            egodriver_pb2.RolloutCameraImage(
                session_uuid=session_uuid,
                camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
                    frame_start_us=NOW_US - 30_000,
                    frame_end_us=NOW_US,
                    image_bytes=payload,
                    logical_id=logical_id,
                ),
            ),
            timeout=timeout,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a live SimScale LTF driver")
    parser.add_argument("--address", default="127.0.0.1:6789")
    parser.add_argument("--timeout", type=_positive_timeout, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sessions = (
        ("probe-a", 0.0, 32, 5.0),
        ("probe-b", 100.0, 224, -5.0),
    )
    attempted_sessions: list[str] = []
    channel = grpc.insecure_channel(args.address)
    stub: egodriver_pb2_grpc.EgodriverServiceStub | None = None
    try:
        stub = egodriver_pb2_grpc.EgodriverServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=args.timeout)
        version = stub.get_version(
            common_pb2.Empty(),
            wait_for_ready=True,
            timeout=args.timeout,
        )
        _require(
            version.version_id == VERSION_ID,
            "driver version ID does not match the probe contract",
        )
        _require(
            version.grpc_api_version == API_VERSION_MESSAGE,
            "driver gRPC API version does not match the probe contract",
        )

        for session in sessions:
            attempted_sessions.append(session[0])
            seed_session(stub, *session, args.timeout)

        barrier = threading.Barrier(2)

        def drive(session: tuple[str, float, int, float]) -> dict[str, int | str]:
            session_uuid, expected_x, _, _ = session
            barrier.wait(timeout=5.0)
            response = stub.drive(
                egodriver_pb2.DriveRequest(
                    session_uuid=session_uuid,
                    time_now_us=NOW_US,
                    time_query_us=QUERY_US,
                ),
                timeout=args.timeout,
            )
            timestamps = [pose.timestamp_us for pose in response.trajectory.poses]
            _require(
                bool(timestamps),
                "drive response trajectory is empty",
            )
            _require(
                timestamps[0] == NOW_US,
                "trajectory does not start at the requested current time",
            )
            _require(
                timestamps[-1] >= QUERY_US,
                "trajectory does not reach the requested query time",
            )
            _require(
                timestamps == sorted(set(timestamps)),
                "trajectory timestamps are not sorted and unique",
            )
            _require(
                abs(response.trajectory.poses[0].pose.vec.x - expected_x) < 1e-4,
                "trajectory anchor does not match the session pose",
            )
            return {
                "session": session_uuid,
                "points": len(timestamps),
                "last_us": timestamps[-1],
            }

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(drive, sessions))
        print(
            json.dumps(
                {"version": version.version_id, "sessions": results},
                sort_keys=True,
            )
        )
    finally:
        if stub is not None:
            for session_uuid in attempted_sessions:
                try:
                    stub.close_session(
                        egodriver_pb2.DriveSessionCloseRequest(
                            session_uuid=session_uuid
                        ),
                        timeout=10.0,
                    )
                except grpc.RpcError:
                    continue
        channel.close()


if __name__ == "__main__":
    main()
