# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Base class for replay servicers that validates requests against ASL recordings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Set

from alpasim_grpc.v0 import common_pb2
from alpasim_runtime.replay_services.asl_reader import ASLReader

import grpc

logger = logging.getLogger(__name__)


class BaseReplayServicer:
    """Base class for all replay servicers with common validation logic"""

    def __init__(self, asl_reader: ASLReader, service_name: str):
        self.asl_reader = asl_reader
        self.service_name = service_name
        self.open_sessions: Set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.{service_name}")

    def validate_request(
        self,
        method_name: str,
        request: Any,
        context: grpc.ServicerContext,
    ) -> Any:
        """
        Validate request against ASL recording and return expected response.

        Args:
            method_name: Name of the gRPC method being called
            request: The incoming request message
            context: gRPC context for error handling

        Returns:
            The recorded response from ASL, or False if not found
        """
        # Try to find and consume a matching request
        recorded_response = self.asl_reader.find_and_consume_matching_request(
            request, self.service_name, method_name
        )

        if recorded_response:
            unused_match_index, recorded_response = recorded_response
            return recorded_response

        # No match found - generate detailed error
        error_msg = self.asl_reader.generate_no_match_error(
            request, self.service_name, method_name
        )

        self.logger.error(error_msg)
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)

        # Unreachable code, but needed for type checking.
        return None

    def track_session(
        self,
        session_uuid: str,
        action: Literal["open", "close"],
        context: grpc.ServicerContext,
    ) -> common_pb2.SessionRequestStatus | common_pb2.Empty:
        """Track session state for services that maintain sessions"""
        if action == "open":
            self.open_sessions.add(session_uuid)
            self.logger.info(
                "Session opened: %s (total: %d)", session_uuid, len(self.open_sessions)
            )
            return common_pb2.SessionRequestStatus()

        if session_uuid not in self.open_sessions:
            error_msg = f"Session not found: {session_uuid}"
            self.logger.error(error_msg)
            # Aborts the call, below code is not executed.
            context.abort(grpc.StatusCode.NOT_FOUND, error_msg)
            return common_pb2.Empty()

        self.open_sessions.remove(session_uuid)
        self.logger.info(
            "Session closed: %s (remaining: %d)",
            session_uuid,
            len(self.open_sessions),
        )
        return common_pb2.Empty()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about replay progress"""
        summary = self.asl_reader.get_exchange_summary()

        # Filter summary for all methods of this service
        service_stats = {}
        for key, stats in summary.items():
            if key.startswith(f"{self.service_name}."):
                service_stats[key] = stats

        return {
            "service": self.service_name,
            "open_sessions": len(self.open_sessions),
            "exchanges": service_stats,
        }

    def get_version(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Return version from ASL metadata"""

        asl_version = self.asl_reader.get_service_version(self.service_name)
        version_id = common_pb2.VersionId()
        version_id.CopyFrom(asl_version)

        # Prepend indicator that this is a replay service using ASL version
        version_id.version_id = (
            f"{self.service_name}-replay-asl-{asl_version.version_id}"
        )

        return version_id

    def start_session(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Validate session start request"""
        self.validate_request("start_session", request, context)
        return self.track_session(request.session_uuid, "open", context)

    def close_session(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Validate session close request"""
        return self.track_session(request.session_uuid, "close", context)

    def shut_down(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Handle shutdown request"""
        self.logger.info("Shutdown requested")
        return common_pb2.Empty()

    def get_available_scenes(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Return available scenes from ASL metadata"""
        available_scenes = common_pb2.AvailableScenesReturn()
        map_id = self.asl_reader.get_map_id()
        scene_id = "clipgt-" + map_id
        available_scenes.scene_ids.append(scene_id)
        self.logger.info(f"Added available scene_id: {scene_id}")

        return available_scenes
