# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Traffic service implementation."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Type

import numpy as np
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import common_pb2
from alpasim_grpc.v0.common_pb2 import Empty, PoseAtTime, VersionId
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_grpc.v0.traffic_pb2 import (
    ObjectTrajectory,
    ObjectTrajectoryUpdate,
    TrafficModuleMetadata,
    TrafficRequest,
    TrafficReturn,
    TrafficSessionCloseRequest,
    TrafficSessionRequest,
)
from alpasim_grpc.v0.traffic_pb2_grpc import TrafficServiceStub
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.services.service_base import (
    WILDCARD_SCENE_ID,
    ServiceBase,
    SessionInfo,
)
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_utils.qvec import QVec
from alpasim_utils.scenario import AABB, TrafficObjects
from alpasim_utils.trajectory import Trajectory

logger = logging.getLogger(__name__)


def _extract_map_id(scene_id: str) -> str:
    """Extract map ID from scene ID."""
    # Assuming scene_id clipgt-<sequence_id>
    pattern = r"^clipgt-([0-9a-fA-F-]{36})$"
    match = re.match(pattern, scene_id)
    if match is None:
        # Fallback to more generic pattern
        match = re.search(r"([a-fA-F0-9-]+)$", scene_id)
        if match is None:
            raise RuntimeError("scene_id does not contain a valid map ID")
    return match.group(1)


class TrafficService(ServiceBase[TrafficServiceStub]):
    """
    Traffic service implementation that handles both real and skip modes.

    Traffic manages sessions and simulates traffic agents in the scene.
    """

    def __init__(
        self,
        address: str,
        skip: bool = False,
        connection_timeout_s: int = 30,
        id: int = 0,
    ):
        super().__init__(address, skip, connection_timeout_s, id)
        self._metadata: Optional[TrafficModuleMetadata] = None

    @property
    def stub_class(self) -> Type[TrafficServiceStub]:
        return TrafficServiceStub

    # Override the session method to add typed parameters
    def session(  # type: ignore[override]
        self,
        uuid: str,
        broadcaster: MessageBroadcaster,
        traffic_objs: TrafficObjects,
        scene_id: str,
        ego_aabb: AABB,
        gt_ego_aabb_trajectory: Trajectory,
        start_timestamp_us: int,
        random_seed: int,
    ) -> "ServiceBase[TrafficServiceStub]":
        """Create a traffic session with typed parameters.

        These are used in `_initialize_session()`.
        """
        return super().session(
            uuid=uuid,
            broadcaster=broadcaster,
            traffic_objs=traffic_objs,
            scene_id=scene_id,
            ego_aabb=ego_aabb,
            gt_ego_aabb_trajectory=gt_ego_aabb_trajectory,
            start_timestamp_us=start_timestamp_us,
            random_seed=random_seed,
        )

    async def get_metadata(self) -> TrafficModuleMetadata:
        """Get metadata from the traffic service."""
        if self.skip:
            # Return minimal metadata for skip mode
            return TrafficModuleMetadata(
                version_id=VersionId(
                    version_id="<skip>",
                    git_hash="<skip>",
                    grpc_api_version=API_VERSION_MESSAGE,
                ),
                supported_map_ids=[],
            )

        if self._metadata is None:
            async with self:
                self._metadata = await profiled_rpc_call(
                    "get_metadata",
                    "traffic",
                    self.stub.get_metadata,
                    Empty(),
                    wait_for_ready=True,
                    timeout=self.connection_timeout_s,
                )

        return self._metadata

    async def get_version(self) -> VersionId:
        """Override to get version from metadata."""
        return (await self.get_metadata()).version_id

    async def get_available_scenes(self) -> List[str]:
        """Get available scenes from the traffic service."""
        if self.skip:
            return [WILDCARD_SCENE_ID]

        map_ids = (await self.get_metadata()).supported_map_ids
        return [f"clipgt-{map_id}" for map_id in map_ids]

    # Note: The pass-through methods have been removed and their logic merged into the public methods

    async def _initialize_session(
        self, session_info: SessionInfo, **kwargs: Any
    ) -> None:
        """Initialize traffic session after gRPC connection is established."""
        # First call parent to set up session_info
        await super()._initialize_session(session_info=session_info)

        # Extract session parameters
        self._traffic_objs = session_info.additional_args["traffic_objs"]
        scene_id = session_info.additional_args["scene_id"]
        ego_aabb = session_info.additional_args["ego_aabb"]
        gt_ego_aabb_trajectory = session_info.additional_args["gt_ego_aabb_trajectory"]
        start_timestamp_us = session_info.additional_args["start_timestamp_us"]
        random_seed = session_info.additional_args["random_seed"]

        # Extract map ID from scene ID
        map_id = _extract_map_id(scene_id)

        # Convert traffic objects to ObjectTrajectory format
        logged_object_trajectories = []

        # First add the ego trajectory
        logged_object_trajectories.append(
            ObjectTrajectory(
                object_id="EGO",
                trajectory=gt_ego_aabb_trajectory.to_grpc(),
                aabb=ego_aabb.to_grpc(),
                is_static=False,
            )
        )

        # Then add all traffic objects
        for obj_id, obj in self._traffic_objs.items():
            logged_object_trajectories.append(
                ObjectTrajectory(
                    object_id=obj_id,
                    trajectory=obj.trajectory.to_grpc(),
                    aabb=obj.aabb.to_grpc(),
                    is_static=obj.is_static,
                )
            )

        # Create session request
        session_request = TrafficSessionRequest(
            session_uuid=self.session_info.uuid,
            map_id=map_id,
            random_seed=random_seed,
            logged_object_trajectories=logged_object_trajectories,
            handover_time_us=start_timestamp_us + int(1e6),  # Add 1 second for warm-up
        )

        # Log and start session
        await self.session_info.broadcaster.broadcast(
            LogEntry(traffic_session_request=session_request)
        )

        if self.skip:
            logger.debug("Skip mode: traffic start_session no-op")
            return

        await profiled_rpc_call(
            "start_session", "traffic", self.stub.start_session, session_request
        )

    async def _cleanup_session(self, **kwargs: Any) -> None:
        """Clean up traffic session."""
        if self.skip:
            logger.debug("Skip mode: traffic close_session no-op")
            # Clean up traffic-specific attributes
            if hasattr(self, "_traffic_objs"):
                delattr(self, "_traffic_objs")
            return

        close_request = TrafficSessionCloseRequest(session_uuid=self.session_info.uuid)
        await profiled_rpc_call(
            "close_session", "traffic", self.stub.close_session, close_request
        )

        # Clean up traffic-specific attributes
        if hasattr(self, "_traffic_objs"):
            delattr(self, "_traffic_objs")

    async def simulate_traffic(
        self,
        ego_aabb_pose_future: QVec,
        future_us: int,
    ) -> TrafficReturn:
        """Simulate traffic for a given ego pose update."""
        # Skip expensive gRPC request construction when in skip mode
        if self.skip:
            logger.debug("Skip mode: replaying traffic from recorded data")

            # In skip mode, return traffic positions from recorded trajectories
            # without constructing the expensive TrafficRequest
            object_trajectory_updates = []
            for obj_id, obj in self._traffic_objs.items():
                if obj_id == "EGO":
                    continue  # Skip EGO in replay

                # Get the trajectory at the requested timestamp
                if future_us in obj.trajectory.time_range_us:
                    traj = obj.trajectory.interpolate_to_timestamps(
                        np.array([future_us], dtype=np.uint64)
                    )
                    object_trajectory_updates.append(
                        ObjectTrajectoryUpdate(
                            object_id=obj_id,
                            trajectory=traj.to_grpc(),
                        )
                    )

            return TrafficReturn(object_trajectory_updates=object_trajectory_updates)

        # Create ego trajectory update with the provided pose
        object_trajectory_updates = [
            ObjectTrajectoryUpdate(
                object_id="EGO",
                trajectory=common_pb2.Trajectory(
                    poses=[
                        PoseAtTime(
                            pose=ego_aabb_pose_future.as_grpc_pose(),
                            timestamp_us=future_us,
                        )
                    ]
                ),
            )
        ]

        traffic_request = TrafficRequest(
            session_uuid=self.session_info.uuid,
            time_query_us=future_us,
            object_trajectory_updates=object_trajectory_updates,
        )

        await self.session_info.broadcaster.broadcast(
            LogEntry(traffic_request=traffic_request)
        )

        traffic_return = await profiled_rpc_call(
            "simulate", "traffic", self.stub.simulate, traffic_request
        )

        # Log response
        await self.session_info.broadcaster.broadcast(
            LogEntry(traffic_return=traffic_return)
        )

        return traffic_return
