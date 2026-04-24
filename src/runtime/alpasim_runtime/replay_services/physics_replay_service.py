# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Physics replay service implementation.

Handles N-to-1 decomposition: when the event-based loop sends a single
multi-pose ``ground_intersection`` request where the old loop sent N
per-pose requests, this service decomposes the incoming request into
per-pose sub-requests, matches each against the recorded ASL exchanges,
and merges the responses.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from alpasim_grpc.v0 import common_pb2, physics_pb2, physics_pb2_grpc
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

from .base_replay_servicer import BaseReplayServicer

logger = logging.getLogger(__name__)


def _decompose_multi_pose_request(
    request: physics_pb2.PhysicsGroundIntersectionRequest,
) -> list[physics_pb2.PhysicsGroundIntersectionRequest] | None:
    """Decompose a multi-pose request into per-pose sub-requests.

    Mirrors the old loop's ``apply_physics_to_trajectory`` pattern where
    each pose was sent as a separate single-pose request with::

        sub_now_us  = pose_timestamp - dt
        sub_future_us = pose_timestamp

    where ``dt = first_pose_timestamp - request.now_us``.

    Returns None if the request has fewer than 2 poses (no decomposition
    needed).
    """
    poses = request.ego_data.ego_trajectory_aabb.poses
    if len(poses) < 2:
        return None

    first_ts = poses[0].timestamp_us
    dt = first_ts - request.now_us

    sub_requests = []
    for pose_at_time in poses:
        ts = pose_at_time.timestamp_us

        sub_trajectory = common_pb2.Trajectory(
            poses=[common_pb2.PoseAtTime(timestamp_us=ts, pose=pose_at_time.pose)]
        )

        sub_request = physics_pb2.PhysicsGroundIntersectionRequest(
            scene_id=request.scene_id,
            now_us=ts - dt,
            future_us=ts,
            ego_data=physics_pb2.PhysicsGroundIntersectionRequest.EgoData(
                aabb=request.ego_data.aabb,
                ego_trajectory_aabb=sub_trajectory,
            ),
            other_objects=list(request.other_objects),
        )
        sub_requests.append(sub_request)

    return sub_requests


def _merge_physics_responses(
    responses: list[physics_pb2.PhysicsGroundIntersectionReturn],
) -> physics_pb2.PhysicsGroundIntersectionReturn:
    """Merge per-pose responses into a single multi-pose response."""
    merged_poses = []
    merged_statuses = []
    merged_other_poses = []

    for resp in responses:
        merged_poses.extend(resp.ego_trajectory_aabb.poses)
        merged_statuses.extend(resp.ego_status)
        # other_poses are empty for the initial trajectory case, but
        # include the last response's set for completeness.
        merged_other_poses = list(resp.other_poses)

    return physics_pb2.PhysicsGroundIntersectionReturn(
        ego_trajectory_aabb=common_pb2.Trajectory(poses=merged_poses),
        ego_status=merged_statuses,
        other_poses=merged_other_poses,
    )


class PhysicsReplayService(BaseReplayServicer, physics_pb2_grpc.PhysicsServiceServicer):
    """Replay service for the physics service."""

    def __init__(self, asl_reader: ASLReader):
        super().__init__(asl_reader, "physics")

    def ground_intersection(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Return recorded constrained poses.

        If the incoming request has multiple poses and doesn't match any
        single recorded exchange, decompose it into per-pose sub-requests
        (matching the old loop's pattern), match each individually, and
        merge the responses.
        """
        # Fast path: try exact match first.
        match = self.asl_reader.find_and_consume_matching_request(
            request, self.service_name, "ground_intersection"
        )
        if match is not None:
            _index, response = match
            return response

        # Slow path: try N-to-1 decomposition for multi-pose requests.
        sub_requests = _decompose_multi_pose_request(request)
        if sub_requests is not None:
            result = self._try_decomposed_match(sub_requests, context)
            if result is not None:
                return result

        # Neither path matched — generate the standard error.
        error_msg = self.asl_reader.generate_no_match_error(
            request, self.service_name, "ground_intersection"
        )
        self.logger.error(error_msg)
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
        return None  # unreachable

    def _try_decomposed_match(
        self,
        sub_requests: list[physics_pb2.PhysicsGroundIntersectionRequest],
        context: grpc.ServicerContext,
    ) -> physics_pb2.PhysicsGroundIntersectionReturn | None:
        """Match each per-pose sub-request and merge responses.

        Returns the merged response on success, or None if any
        sub-request fails to match.
        """
        results: list[Tuple[int, physics_pb2.PhysicsGroundIntersectionReturn]] = []

        for sub_req in sub_requests:
            match = self.asl_reader.find_and_consume_matching_request(
                sub_req, self.service_name, "ground_intersection"
            )
            if match is None:
                # Partial match — consumed indices from earlier sub-requests
                # are not rolled back.  This is acceptable because a partial
                # match indicates a real divergence worth investigating.
                self.logger.warning(
                    "Decomposed physics match failed on sub-request %d/%d",
                    len(results) + 1,
                    len(sub_requests),
                )
                return None
            results.append(match)

        responses = [resp for _idx, resp in results]
        merged = _merge_physics_responses(responses)
        self.logger.info(
            "Matched multi-pose physics request via %d-way decomposition",
            len(sub_requests),
        )
        return merged
