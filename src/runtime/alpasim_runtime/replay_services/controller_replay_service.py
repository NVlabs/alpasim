# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Controller (VDC) replay service implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from alpasim_grpc.v0 import controller_pb2_grpc
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

from .base_replay_servicer import BaseReplayServicer

logger = logging.getLogger(__name__)


class ControllerReplayService(
    BaseReplayServicer, controller_pb2_grpc.VDCServiceServicer
):
    """Replay service for the controller/VDC service.

    Note: The controller's start_session/close_session RPCs are not recorded
    in the ASL log (only run_controller_and_vehicle is), so we override the
    base class to skip request validation and just track session state.
    """

    def __init__(self, asl_reader: ASLReader):
        super().__init__(asl_reader, "controller")

    def start_session(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Track session open without ASL validation (not recorded in ASL)."""
        return self.track_session(request.session_uuid, "open", context)

    def run_controller_and_vehicle(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Return recorded propagated poses"""
        return self.validate_request("run_controller_and_vehicle", request, context)
