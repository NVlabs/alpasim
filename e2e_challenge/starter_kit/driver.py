#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Minimal CPU-only contestant driver for the AlpaSim e2e challenge path."""

from __future__ import annotations

import logging
import math
import os
import signal
import threading
import time
from concurrent import futures
from dataclasses import dataclass

from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc

import grpc

LOGGER = logging.getLogger("alpasim_e2e_starter_driver")


@dataclass
class SessionState:
    latest_pose: common_pb2.PoseAtTime | None = None


def _yaw_from_quaternion(quat: common_pb2.Quat) -> float:
    siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
    cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
    return math.atan2(siny_cosp, cosy_cosp)


def build_straight_line_trajectory(
    start_pose: common_pb2.PoseAtTime | None,
    time_now_us: int,
    *,
    speed_mps: float = 5.0,
    horizon_s: float = 5.0,
    dt_us: int = 100_000,
) -> common_pb2.Trajectory:
    """Build a constant-speed straight-line trajectory from the latest pose."""
    if start_pose is None:
        start_pose = common_pb2.PoseAtTime(
            timestamp_us=time_now_us,
            pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=0.0, y=0.0, z=0.0),
                quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        )

    yaw = _yaw_from_quaternion(start_pose.pose.quat)
    dx = math.cos(yaw)
    dy = math.sin(yaw)
    n_points = max(2, math.ceil(horizon_s * 1_000_000 / dt_us) + 1)

    trajectory = common_pb2.Trajectory()
    for i in range(n_points):
        distance_m = speed_mps * (i * dt_us / 1_000_000.0)
        trajectory.poses.append(
            common_pb2.PoseAtTime(
                timestamp_us=time_now_us + i * dt_us,
                pose=common_pb2.Pose(
                    vec=common_pb2.Vec3(
                        x=start_pose.pose.vec.x + dx * distance_m,
                        y=start_pose.pose.vec.y + dy * distance_m,
                        z=start_pose.pose.vec.z,
                    ),
                    quat=start_pose.pose.quat,
                ),
            )
        )
    return trajectory


class StarterDriver(egodriver_pb2_grpc.EgodriverServiceServicer):
    """Simple gRPC driver service that contestants can copy and modify."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.RLock()
        self._server: grpc.Server | None = None

    def attach_server(self, server: grpc.Server) -> None:
        self._server = server

    def start_session(
        self,
        request: egodriver_pb2.DriveSessionRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.SessionRequestStatus:
        with self._lock:
            self._sessions[request.session_uuid] = SessionState()
        LOGGER.info("started session %s", request.session_uuid)
        return common_pb2.SessionRequestStatus()

    def close_session(
        self,
        request: egodriver_pb2.DriveSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        with self._lock:
            self._sessions.pop(request.session_uuid, None)
        LOGGER.info("closed session %s", request.session_uuid)
        return common_pb2.Empty()

    def submit_image_observation(
        self,
        request: egodriver_pb2.RolloutCameraImage,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        return common_pb2.Empty()

    def submit_egomotion_observation(
        self,
        request: egodriver_pb2.RolloutEgoTrajectory,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        if request.trajectory.poses:
            session = self._get_session(request.session_uuid, context)
            with self._lock:
                session.latest_pose = request.trajectory.poses[-1]
        return common_pb2.Empty()

    def submit_route(
        self,
        request: egodriver_pb2.RouteRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        return common_pb2.Empty()

    def submit_recording_ground_truth(
        self,
        request: egodriver_pb2.GroundTruthRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        return common_pb2.Empty()

    def drive(
        self,
        request: egodriver_pb2.DriveRequest,
        context: grpc.ServicerContext,
    ) -> egodriver_pb2.DriveResponse:
        session = self._get_session(request.session_uuid, context)
        with self._lock:
            latest_pose = session.latest_pose
        return egodriver_pb2.DriveResponse(
            trajectory=build_straight_line_trajectory(
                latest_pose,
                request.time_now_us,
            )
        )

    def get_version(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.VersionId:
        return common_pb2.VersionId(
            version_id="e2e-starter-driver",
            git_hash="local",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def shut_down(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        if self._server is not None:
            threading.Thread(target=self._stop_server, daemon=True).start()
        return common_pb2.Empty()

    def _stop_server(self) -> None:
        time.sleep(0.05)
        if self._server is not None:
            self._server.stop(grace=0.0)

    def _get_session(
        self,
        session_uuid: str,
        context: grpc.ServicerContext,
    ) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_uuid)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {session_uuid}")
            raise AssertionError("unreachable")
        return session


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("ALPASIM_DRIVER_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    host = os.environ.get("ALPASIM_DRIVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ALPASIM_DRIVER_PORT", "6789"))

    service = StarterDriver()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(service, server)
    service.attach_server(server)

    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"failed to bind {host}:{port}")

    def request_stop(signum: int, frame: object) -> None:
        LOGGER.info("received signal %s, stopping", signum)
        server.stop(grace=0.0)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    server.start()
    LOGGER.info("starter driver listening on %s:%d", host, bound_port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
