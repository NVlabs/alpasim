# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Driver (Egodriver) replay service implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from alpasim_grpc.v0 import common_pb2, egodriver_pb2_grpc
from alpasim_grpc.v0.egodriver_pb2 import RolloutEgoTrajectory
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

from .base_replay_servicer import BaseReplayServicer

logger = logging.getLogger(__name__)


def _split_ego_trajectory(request: RolloutEgoTrajectory) -> list[RolloutEgoTrajectory]:
    """Split a multi-pose egomotion request into single-pose requests.

    The runtime may batch multiple poses into a single
    ``submit_egomotion_observation`` call (e.g. the first PolicyEvent sends
    both t0 and t1 together).  The ASL recording stores one entry per pose,
    so we decompose here to match at the individual-pose level.

    If the request already contains a single pose, returns it unchanged
    (wrapped in a list).
    """
    poses = request.trajectory.poses
    dynamics = request.dynamic_states
    if len(poses) <= 1:
        return [request]

    parts: list[RolloutEgoTrajectory] = []
    for i, pose in enumerate(poses):
        part = RolloutEgoTrajectory(
            session_uuid=request.session_uuid,
            trajectory=common_pb2.Trajectory(poses=[pose]),
        )
        if i < len(dynamics):
            part.dynamic_states.append(dynamics[i])
        parts.append(part)
    return parts


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
        """Validate egomotion data, decomposing multi-pose batches.

        The runtime may send multiple poses in a single call.  We split the
        request into per-pose pieces and validate each against the recording
        individually so that message grouping doesn't matter.
        """
        result: Any = None
        for part in _split_ego_trajectory(request):
            result = self.validate_request(
                "submit_egomotion_observation", part, context
            )
        return result

    def submit_route(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Validate route request"""
        return self.validate_request("submit_route", request, context)

    def submit_recording_ground_truth(
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        """Validate ground truth"""
        return self.validate_request("submit_recording_ground_truth", request, context)
