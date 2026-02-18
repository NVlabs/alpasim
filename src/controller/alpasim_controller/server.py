# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
This file implements a gRPC server for the alpasim controller service: a service
that provides a vehicle model and controller simulation environment.
"""

import argparse
import importlib.metadata
import logging
from concurrent import futures
from threading import Lock

from alpasim_controller.mpc_controller import MPCImplementation
from alpasim_controller.system_manager import SystemManager
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2, controller_pb2, controller_pb2_grpc

import grpc

logger = logging.getLogger(__name__)


def construct_version() -> common_pb2.VersionId:
    response = common_pb2.VersionId(
        version_id=importlib.metadata.version("alpasim_controller"),
        grpc_api_version=API_VERSION_MESSAGE,
        git_hash="n/a",
    )
    return response


class VDCSimService(controller_pb2_grpc.VDCServiceServicer):
    """
    VDCSimService (Vehicle Dynamics and Control) is a gRPC service that interacts with
    a SystemManager backend.
    """

    def __init__(
        self, server: grpc.Server, log_dir: str, mpc_implementation: str | None = None
    ):
        logger.info(f"VDCServicer initialized logging to: {log_dir}")
        self._server = server
        self._backend = SystemManager(log_dir, mpc_implementation=mpc_implementation)
        self._lock = Lock()

    def get_version(self, request: common_pb2.Empty, context: grpc.ServicerContext):
        return construct_version()

    def start_session(
        self, request: common_pb2.SessionRequestStatus, context: grpc.ServicerContext
    ):
        logger.info(f"start_session for session_uuid: {request.session_uuid}")
        with self._lock:
            self._backend.start_session(request.session_uuid)
        return common_pb2.SessionRequestStatus()

    def close_session(
        self,
        request: controller_pb2.VDCSessionCloseRequest,
        context: grpc.ServicerContext,
    ):
        logger.info(f"close_session for session_uuid: {request.session_uuid}")
        with self._lock:
            self._backend.close_session(request)
        return common_pb2.Empty()

    def run_controller_and_vehicle(
        self,
        request: controller_pb2.RunControllerAndVehicleModelRequest,
        context: grpc.ServicerContext,
    ):
        logger.debug(
            f"run_controller_and_vehicle called for session_uuid: {request.session_uuid}"
        )
        with self._lock:
            response = self._backend.run_controller_and_vehicle_model(request)
        return response

    def shut_down(self, request: common_pb2.Empty, context: grpc.ServicerContext):
        logger.info("shut_down")
        context.add_callback(self._shut_down)
        return common_pb2.Empty()

    def _shut_down(self) -> None:
        self._server.stop(0)


def serve(host, port: int, log_dir: str, mpc_implementation: str | None = None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    controller_pb2_grpc.add_VDCServiceServicer_to_server(
        VDCSimService(server, log_dir, mpc_implementation=mpc_implementation), server
    )
    address = f"{host}:{port}"
    logger.info(f"Starting server on {address}")
    server.add_insecure_port(address)
    server.start()
    logger.info("Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, help="Port to listen on", default=50051)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--mpc-implementation",
        type=MPCImplementation,
        choices=list(MPCImplementation),
        default=MPCImplementation.LINEAR,
        help="MPC implementation: linear (OSQP, default) or nonlinear (CasADi)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    serve(args.host, args.port, args.log_dir, args.mpc_implementation)
