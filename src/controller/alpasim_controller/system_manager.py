# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
SystemManager - manages multiple systems, each with vehicle dynamics and controller
"""

import logging

from alpasim_controller.mpc_controller import MPCImplementation
from alpasim_controller.system import System, create_system
from alpasim_grpc.v0 import common_pb2, controller_pb2


class SystemManager:
    """Manages multiple vehicle dynamics and control systems.

    Args:
        log_dir: Directory for controller log files.
        mpc_implementation: MPC implementation to use (default: LINEAR)
    """

    def __init__(
        self, log_dir: str, mpc_implementation: MPCImplementation | None = None
    ):
        self._log_dir = log_dir
        # Registered sessions (session_uuid -> System or None if not yet initialized)
        self._sessions: dict[str, System | None] = {}
        self._mpc_implementation = mpc_implementation or MPCImplementation.LINEAR

        logging.info(f"SystemManager using {self._mpc_implementation} MPC")

    def start_session(self, session_uuid: str) -> None:
        """Register a new session.

        The actual System is created lazily on the first run_controller_and_vehicle
        call, since that provides the initial state needed for initialization.
        """
        if session_uuid in self._sessions:
            raise KeyError(f"Session {session_uuid} already exists")
        logging.info(f"Registering session: {session_uuid}")
        self._sessions[session_uuid] = None  # Placeholder until first run

    def close_session(
        self, request: controller_pb2.VDCSessionCloseRequest
    ) -> common_pb2.Empty:
        """Close a session and remove its System."""
        session_uuid = request.session_uuid
        if session_uuid not in self._sessions:
            raise KeyError(f"Session {session_uuid} does not exist")

        system = self._sessions[session_uuid]
        if system is not None:
            logging.info(f"Closing session: {session_uuid}")
        else:
            logging.warning(
                f"Closing session: {session_uuid} (was registered but never used)"
            )
        del self._sessions[session_uuid]
        return common_pb2.Empty()

    def _create_system(
        self, session_uuid: str, state: common_pb2.StateAtTime
    ) -> System:
        """Create a new System for a session."""
        logging.info(
            f"Creating system for session_uuid: {session_uuid} "
            f"(mpc: {self._mpc_implementation})"
        )
        system = create_system(
            log_file=f"{self._log_dir}/alpasim_controller_{session_uuid}.csv",
            initial_state=state,
            mpc_implementation=self._mpc_implementation,
        )
        self._sessions[session_uuid] = system
        return system

    def run_controller_and_vehicle_model(
        self, request: controller_pb2.RunControllerAndVehicleModelRequest
    ) -> controller_pb2.RunControllerAndVehicleModelResponse:
        """Run the controller and vehicle model for a request."""
        session_uuid = request.session_uuid
        logging.debug(
            f"run_controller_and_vehicle called for session_uuid: {session_uuid}"
        )

        if session_uuid not in self._sessions:
            raise KeyError(
                f"Session {session_uuid} does not exist. "
                "Call start_session before run_controller_and_vehicle."
            )

        system = self._sessions[session_uuid]
        if system is None:
            # First call for this session - create the system now
            system = self._create_system(session_uuid, request.state)

        return system.run_controller_and_vehicle_model(request)
