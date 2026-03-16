# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Sensorsim replay service implementation with image correlation.
"""

from __future__ import annotations

import logging
from typing import Any

from alpasim_grpc.v0 import sensorsim_pb2, sensorsim_pb2_grpc
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

from .base_replay_servicer import BaseReplayServicer

logger = logging.getLogger(__name__)


class SensorsimReplayService(
    BaseReplayServicer, sensorsim_pb2_grpc.SensorsimServiceServicer
):
    """Replay service for the sensorsim service"""

    def __init__(self, asl_reader: ASLReader):
        super().__init__(asl_reader, "sensorsim")

    def get_available_ego_masks(
        self, request: Any, context: grpc.ServicerContext
    ) -> sensorsim_pb2.AvailableEgoMasksReturn:
        """Return available ego masks from ASL log."""
        return self.validate_request("get_available_ego_masks", request, context)

    def render_rgb(
        self, request: sensorsim_pb2.RGBRenderRequest, context: grpc.ServicerContext
    ) -> sensorsim_pb2.RGBRenderReturn:
        """Validate render request and return appropriate image"""
        # Validate the render request
        self.validate_request("render_rgb", request, context)

        # Extract camera ID from the request
        camera_id: str = request.camera_intrinsics.logical_id

        # Find corresponding image data from driver_camera_image entries
        # Pass the timestamp from the render request for accurate matching
        timestamp_us = request.frame_start_us
        image_data = self.asl_reader.get_driver_image_for_camera(
            camera_id, timestamp_us
        )

        if image_data is None:
            raise ValueError(f"No image data found for camera {camera_id}")

        # Create and return the response
        return sensorsim_pb2.RGBRenderReturn(image_bytes=image_data)

    def get_available_cameras(
        self,
        request: sensorsim_pb2.AvailableCamerasRequest,
        context: grpc.ServicerContext,
    ) -> sensorsim_pb2.AvailableCamerasReturn:
        """Return camera configuration from ASL log"""
        # Validate the request against the ASL log and get the recorded response
        return self.validate_request("get_available_cameras", request, context)

    def get_available_trajectories(
        self,
        request: sensorsim_pb2.AvailableTrajectoriesRequest,
        context: grpc.ServicerContext,
    ) -> sensorsim_pb2.AvailableTrajectoriesReturn:
        """Return available trajectories (if any)"""
        # Not typically used in standard simulations
        return sensorsim_pb2.AvailableTrajectoriesReturn()
