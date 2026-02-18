# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Driver (Egodriver) replay service implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from alpasim_grpc.v0 import egodriver_pb2_grpc
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

from .base_replay_servicer import BaseReplayServicer

logger = logging.getLogger(__name__)


class DriverReplayService(
    BaseReplayServicer, egodriver_pb2_grpc.EgodriverServiceServicer
):
    """Replay service for the driver/policy service"""

    def __init__(self, asl_reader: ASLReader):
        super().__init__(asl_reader, "driver")

    def drive(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Return recorded trajectory"""
        return self.validate_request("drive", request, context)

    def submit_image_observation(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Validate image submission"""
        return self.validate_request("submit_image_observation", request, context)

    def submit_egomotion_observation(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Validate egomotion data"""
        return self.validate_request("submit_egomotion_observation", request, context)

    def submit_route(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Validate route request"""
        return self.validate_request("submit_route", request, context)

    def submit_recording_ground_truth(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Validate ground truth"""
        return self.validate_request("submit_recording_ground_truth", request, context)
