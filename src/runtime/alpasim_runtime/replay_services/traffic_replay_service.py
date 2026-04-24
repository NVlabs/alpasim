# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Traffic replay service implementation.
"""

from __future__ import annotations

import logging

from alpasim_grpc.v0 import traffic_pb2, traffic_pb2_grpc
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

from .base_replay_servicer import BaseReplayServicer

logger = logging.getLogger(__name__)


class TrafficReplayService(BaseReplayServicer, traffic_pb2_grpc.TrafficServiceServicer):
    """Replay service for the traffic service"""

    def __init__(self, asl_reader: ASLReader):
        super().__init__(asl_reader, "trafficsim")

    def get_metadata(
        self, request: traffic_pb2.MetadataRequest, context: grpc.ServicerContext
    ) -> traffic_pb2.MetadataResponse:
        """Return metadata from ASL"""
        metadata = traffic_pb2.TrafficModuleMetadata()
        version_id = self.get_version(request, context)
        metadata.version_id.CopyFrom(version_id)
        metadata.minimum_history_length_us = 1000000  # 1 second default

        map_id = self.asl_reader.get_map_id()
        metadata.supported_map_ids.append(map_id)
        return metadata

    def simulate(
        self, request: traffic_pb2.SimulateRequest, context: grpc.ServicerContext
    ) -> traffic_pb2.SimulateResponse:
        """Return recorded traffic poses"""
        return self.validate_request("simulate", request, context)
