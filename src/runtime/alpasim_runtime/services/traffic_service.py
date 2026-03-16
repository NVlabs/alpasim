# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Traffic service implementation."""

from __future__ import annotations

import logging
import random
import re
from typing import Type

import numpy as np
from alpasim_grpc.v0 import common_pb2
from alpasim_grpc.v0.common_pb2 import PoseAtTime
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_grpc.v0.traffic_pb2 import (
    ObjectTrajectory,
    ObjectTrajectoryUpdate,
    TrafficRequest,
    TrafficReturn,
    TrafficSessionCloseRequest,
    TrafficSessionRequest,
)
from alpasim_grpc.v0.traffic_pb2_grpc import TrafficServiceStub
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo
from alpasim_runtime.services.session_configs import TrafficSessionConfig
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_utils.geometry import Pose, pose_to_grpc, trajectory_to_grpc

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

    @property
    def stub_class(self) -> Type[TrafficServiceStub]:
        return TrafficServiceStub

    async def _initialize_session(self, session_info: SessionInfo) -> None:
        """Initialize traffic session after gRPC connection is established."""
        cfg = session_info.session_config
        if not isinstance(cfg, TrafficSessionConfig):
            raise TypeError(
                "TrafficService._initialize_session requires a TrafficSessionConfig "
                f"via session_config, got {type(cfg).__name__}."
            )

        self._traffic_objs = cfg.traffic_objs
        scene_id = cfg.scene_id
        ego_aabb = cfg.ego_aabb
        gt_ego_aabb_trajectory = cfg.gt_ego_aabb_trajectory
        start_timestamp_us = cfg.start_timestamp_us

        # Extract map ID from scene ID
        map_id = _extract_map_id(scene_id)

        # Convert traffic objects to ObjectTrajectory format
        logged_object_trajectories = []

        # First add the ego trajectory
        logged_object_trajectories.append(
            ObjectTrajectory(
                object_id="EGO",
                trajectory=trajectory_to_grpc(gt_ego_aabb_trajectory),
                aabb=ego_aabb.to_grpc(),
                is_static=False,
            )
        )

        # Then add all traffic objects
        for obj_id, obj in self._traffic_objs.items():
            logged_object_trajectories.append(
                ObjectTrajectory(
                    object_id=obj_id,
                    trajectory=trajectory_to_grpc(obj.trajectory),
                    aabb=obj.aabb.to_grpc(),
                    is_static=obj.is_static,
                )
            )

        # Create session request
        session_request = TrafficSessionRequest(
            session_uuid=session_info.uuid,
            map_id=map_id,
            random_seed=random.randint(0, 2**32 - 1),
            logged_object_trajectories=logged_object_trajectories,
            handover_time_us=start_timestamp_us + int(1e6),  # Add 1 second for warm-up
        )

        # Log and start session
        await session_info.broadcaster.broadcast(
            LogEntry(traffic_session_request=session_request)
        )

        if self.skip:
            logger.debug("Skip mode: traffic start_session no-op")
            return

        await profiled_rpc_call(
            "start_session", "traffic", self.stub.start_session, session_request
        )

    async def _cleanup_session(self, session_info: SessionInfo) -> None:
        """Clean up traffic session."""
        if self.skip:
            logger.debug("Skip mode: traffic close_session no-op")
            # Clean up traffic-specific attributes
            if hasattr(self, "_traffic_objs"):
                delattr(self, "_traffic_objs")
            return

        close_request = TrafficSessionCloseRequest(session_uuid=session_info.uuid)
        await profiled_rpc_call(
            "close_session", "traffic", self.stub.close_session, close_request
        )

        # Clean up traffic-specific attributes
        if hasattr(self, "_traffic_objs"):
            delattr(self, "_traffic_objs")

    async def simulate_traffic(
        self,
        ego_aabb_pose_future: Pose,
        future_us: int,
    ) -> TrafficReturn:
        """Simulate traffic for a given ego pose update."""
        session_info = self._require_session_info()
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
                    traj = obj.trajectory.interpolate(
                        np.array([future_us], dtype=np.uint64)
                    )
                    object_trajectory_updates.append(
                        ObjectTrajectoryUpdate(
                            object_id=obj_id,
                            trajectory=trajectory_to_grpc(traj),
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
                            pose=pose_to_grpc(ego_aabb_pose_future),
                            timestamp_us=future_us,
                        )
                    ]
                ),
            )
        ]

        traffic_request = TrafficRequest(
            session_uuid=session_info.uuid,
            time_query_us=future_us,
            object_trajectory_updates=object_trajectory_updates,
        )

        await session_info.broadcaster.broadcast(
            LogEntry(traffic_request=traffic_request)
        )

        traffic_return = await profiled_rpc_call(
            "simulate", "traffic", self.stub.simulate, traffic_request
        )

        # Log response
        await session_info.broadcaster.broadcast(
            LogEntry(traffic_return=traffic_return)
        )

        return traffic_return
