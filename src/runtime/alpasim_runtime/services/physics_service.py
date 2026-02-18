# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Physics service implementation."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

from alpasim_grpc.v0.common_pb2 import Pose
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_grpc.v0.physics_pb2 import PhysicsGroundIntersectionRequest
from alpasim_grpc.v0.physics_pb2_grpc import PhysicsServiceStub
from alpasim_runtime.config import PhysicsUpdateMode, ScenarioConfig
from alpasim_runtime.services.service_base import ServiceBase
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_utils.qvec import QVec
from alpasim_utils.scenario import AABB

logger = logging.getLogger(__name__)


class PhysicsService(ServiceBase[PhysicsServiceStub]):
    """
    Physics service implementation that handles both real and skip modes.

    Physics is responsible for ground intersection calculations,
    determining how objects interact with the ground plane.
    """

    @property
    def stub_class(self) -> Type[PhysicsServiceStub]:
        return PhysicsServiceStub

    async def ground_intersection(
        self,
        scene_id: str,
        delta_start_us: int,
        delta_end_us: int,
        pose_now: QVec,
        pose_future: QVec,
        traffic_poses: Dict[str, QVec],
        ego_aabb: AABB,
        skip: bool = False,
    ) -> Tuple[QVec, Dict[str, QVec]]:
        """
        Calculate ground intersection for ego and traffic vehicles.

        Args:
            skip: If True, return traffic poses unchanged without
                making a gRPC call. Use this when objects are following
                trajectories that already have correct physics applied (e.g.,
                recorded ground truth or when traffic sim is skipped).

        Returns:
            Tuple of (ego_pose, traffic_poses) after ground intersection
        """

        if self.skip or skip:
            # Return the future poses unchanged
            return pose_future, traffic_poses

        assert traffic_poses is not None or (
            pose_now is not None and pose_future is not None
        ), "Either traffic_poses or pose_now and pose_future must be provided."

        traffic_poses = traffic_poses or {}

        request = self._prepare_request(
            scene_id,
            delta_start_us,
            delta_end_us,
            pose_now.as_grpc_pose(),
            pose_future.as_grpc_pose(),
            [p.as_grpc_pose() for p in traffic_poses.values()],
            ego_aabb=ego_aabb,
        )

        await self.session_info.broadcaster.broadcast(LogEntry(physics_request=request))

        response = await profiled_rpc_call(
            "ground_intersection", "physics", self.stub.ground_intersection, request
        )

        await self.session_info.broadcaster.broadcast(LogEntry(physics_return=response))

        ego_response = QVec.from_grpc_pose(response.ego_pose.pose)
        traffic_responses = {
            k: QVec.from_grpc_pose(v.pose)
            for k, v in zip(traffic_poses.keys(), response.other_poses)
        }

        return ego_response, traffic_responses

    def _prepare_request(
        self,
        scene_id: str,
        delta_start_us: int,
        delta_end_us: int,
        ego_pose_now: Pose,
        ego_pose_future: Pose,
        other_poses: List[Pose],
        ego_aabb: AABB,
    ) -> PhysicsGroundIntersectionRequest:
        """Prepare the physics ground intersection request."""
        return PhysicsGroundIntersectionRequest(
            scene_id=scene_id,
            now_us=delta_start_us,
            future_us=delta_end_us,
            ego_data=PhysicsGroundIntersectionRequest.EgoData(
                aabb=ego_aabb.to_grpc(),
                pose_pair=PhysicsGroundIntersectionRequest.PosePair(
                    now_pose=ego_pose_now, future_pose=ego_pose_future
                ),
            ),
            other_objects=[
                PhysicsGroundIntersectionRequest.OtherObject(
                    # TODO[RDL] extract AABB from NRE reconstruction, this
                    # is placeholder assuming all cars are equally sized
                    aabb=ego_aabb.to_grpc(),
                    pose_pair=PhysicsGroundIntersectionRequest.PosePair(
                        now_pose=other_pose, future_pose=other_pose
                    ),
                )
                for other_pose in other_poses
            ],
        )

    async def find_scenario_incompatibilities(
        self, scenario: ScenarioConfig
    ) -> List[str]:
        """Check if physics service can handle the given scenario."""

        incompatibilities = await super().find_scenario_incompatibilities(scenario)

        # If physics is in skip mode, it can handle any scenario
        if not self.skip and scenario.physics_update_mode == PhysicsUpdateMode.NONE:
            incompatibilities.append("Physics is disabled for this scenario.")

        return incompatibilities
