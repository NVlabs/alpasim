# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
EvalDataAccumulator - unified message accumulation for evaluation.

This module provides the EvalDataAccumulator class that accumulates all
eval-relevant data from LogEntry messages, providing a unified code path
for both:
- ASL file loading (post-eval)
- Runtime evaluation (in-runtime)

The accumulator handles all message types and builds ScenarioEvalInput,
eliminating code duplication between asl_loader.py and runtime_evaluator.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from alpasim_grpc.v0.common_pb2 import AABB
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata
from alpasim_utils.geometry import (
    Pose,
    Trajectory,
    pose_from_grpc,
    trajectory_from_grpc,
)
from trajdata.maps import VectorMap

from eval.data import (
    RAABB,
    Cameras,
    DriverResponses,
    RenderableTrajectory,
    Routes,
    ScenarioEvalInput,
)
from eval.schema import EvalConfig

logger = logging.getLogger("alpasim_eval.accumulator")


@dataclass
class EvalDataAccumulator:
    """Accumulates all eval-relevant data from LogEntry messages.

    This class provides a unified code path for both:
    - ASL file loading (post-eval)
    - Runtime evaluation (in-runtime)

    Usage:
        accumulator = EvalDataAccumulator(cfg=eval_config)

        # Feed messages (from ASL file or runtime broadcaster)
        for message in messages:
            accumulator.handle_message(message)

        # Build final output
        scenario_input = accumulator.build_scenario_eval_input(
            run_uuid=..., run_name=..., batch_id=..., vec_map=...
        )
    """

    cfg: EvalConfig

    # Internal state - populated from rollout_metadata message
    _session_metadata: Optional[RolloutMetadata.SessionMetadata] = field(
        default=None, init=False
    )
    _ego_coords_rig_to_aabb_center: Optional[Pose] = field(default=None, init=False)
    _ego_aabb_dims: Optional[tuple[float, float, float]] = field(
        default=None, init=False
    )
    _gt_ego_trajectory: Optional[Trajectory] = field(default=None, init=False)

    # Actor data from rollout_metadata + actor_poses
    _actor_aabb_dims: dict[str, tuple[float, float, float]] = field(
        default_factory=dict, init=False
    )
    _actor_trajectory_data: dict[str, list[tuple[int, Pose]]] = field(
        default_factory=dict, init=False
    )

    # Driver request/response pairing
    _pending_request: Optional[tuple[int, int]] = field(default=None, init=False)
    _driver_responses: list[tuple[int, int, DriveResponse]] = field(
        default_factory=list, init=False
    )

    # Camera and route data
    _cameras: Cameras = field(default_factory=Cameras, init=False)
    _routes: Routes = field(default_factory=Routes, init=False)

    @property
    def session_metadata(self) -> Optional[RolloutMetadata.SessionMetadata]:
        """Access session metadata after accumulation."""
        return self._session_metadata

    def handle_message(self, message: LogEntry) -> None:
        """Handle any LogEntry message type.

        This method processes all eval-relevant message types:
        - rollout_metadata: Extract session metadata, AABB dims, transforms, gt trajectory
        - actor_poses: Accumulate poses to build trajectories
        - driver_camera_image: Add to cameras
        - route_request: Add to routes
        - driver_request: Store timestamps for pairing with driver_return
        - driver_return: Pair with pending request and accumulate
        - available_cameras_return: Add camera calibrations

        Args:
            message: The LogEntry protobuf message to process.
        """
        msg_type = message.WhichOneof("log_entry")

        if msg_type == "rollout_metadata":
            self._handle_rollout_metadata(message.rollout_metadata)
        elif msg_type == "actor_poses":
            self._handle_actor_poses(message.actor_poses)
        elif msg_type == "driver_camera_image":
            self._cameras.add_camera_image(message.driver_camera_image.camera_image)
        elif msg_type == "route_request":
            self._routes.add_route(message.route_request.route)
        elif msg_type == "driver_request":
            self._pending_request = (
                message.driver_request.time_now_us,
                message.driver_request.time_query_us,
            )
        elif msg_type == "driver_return":
            if self._pending_request is not None:
                self._driver_responses.append(
                    (*self._pending_request, message.driver_return)
                )
                self._pending_request = None
        elif msg_type == "available_cameras_return":
            for available_camera in message.available_cameras_return.available_cameras:
                self._cameras.add_calibration(available_camera)

    def _handle_rollout_metadata(self, metadata: RolloutMetadata) -> None:
        """Extract data from rollout_metadata message.

        Args:
            metadata: The RolloutMetadata protobuf message.
        """
        self._session_metadata = metadata.session_metadata

        # Extract actor AABB dimensions and initialize trajectory data
        for actor_aabb in metadata.actor_definitions.actor_aabb:
            actor_id = actor_aabb.actor_id
            aabb = actor_aabb.aabb
            self._actor_aabb_dims[actor_id] = (
                aabb.size_x,
                aabb.size_y,
                aabb.size_z,
            )
            self._actor_trajectory_data[actor_id] = []

        # Extract ego coordinate transformation
        self._ego_coords_rig_to_aabb_center = pose_from_grpc(
            metadata.transform_ego_coords_rig_to_aabb
        )

        # Get ego AABB dims (EGO is always first in actor_definitions)
        ego_aabb = metadata.actor_definitions.actor_aabb[0].aabb
        self._ego_aabb_dims = (
            ego_aabb.size_x,
            ego_aabb.size_y,
            ego_aabb.size_z,
        )

        # Parse and transform ground truth trajectory to AABB frame
        self._gt_ego_trajectory = trajectory_from_grpc(
            metadata.ego_rig_recorded_ground_truth_trajectory
        ).transform(self._ego_coords_rig_to_aabb_center, is_relative=True)

    def _handle_actor_poses(self, poses_message: ActorPoses) -> None:
        """Accumulate actor poses for trajectory building.

        Args:
            poses_message: The ActorPoses protobuf message.
        """
        timestamp_us = poses_message.timestamp_us
        for pose in poses_message.actor_poses:
            actor_id = pose.actor_id
            if actor_id in self._actor_trajectory_data:
                self._actor_trajectory_data[actor_id].append(
                    (timestamp_us, pose_from_grpc(pose.actor_pose))
                )

    def _build_actor_trajectories(
        self,
    ) -> dict[str, tuple[Trajectory, tuple[float, float, float]]]:
        """Build actor trajectories from accumulated pose data.

        Returns:
            Dictionary mapping actor_id to (Trajectory, (aabb_x, aabb_y, aabb_z))
        """
        actor_trajectories: dict[str, tuple[Trajectory, tuple[float, float, float]]] = (
            {}
        )

        for actor_id, pose_data in self._actor_trajectory_data.items():
            if not pose_data:
                continue

            # Sort by timestamp and build trajectory
            pose_data.sort(key=lambda x: x[0])
            timestamps = np.array([p[0] for p in pose_data], dtype=np.uint64)
            poses = [p[1] for p in pose_data]
            trajectory = Trajectory.from_poses(timestamps=timestamps, poses=poses)

            # Get AABB dims for this actor
            aabb_dims = self._actor_aabb_dims.get(
                actor_id, (4.5, 2.0, 1.5)
            )  # Default dims

            actor_trajectories[actor_id] = (trajectory, aabb_dims)

        return actor_trajectories

    def _build_ego_renderable_trajectory(
        self,
        actor_trajectories: dict[str, tuple[Trajectory, tuple[float, float, float]]],
    ) -> Optional[RenderableTrajectory]:
        """Build ego RenderableTrajectory from actor trajectories.

        Args:
            actor_trajectories: The built actor trajectories dict.

        Returns:
            RenderableTrajectory for EGO, or None if EGO not found.
        """
        if "EGO" not in actor_trajectories or self._ego_aabb_dims is None:
            return None

        ego_traj, _ = actor_trajectories["EGO"]
        ego_raabb = RAABB.from_grpc(
            AABB(
                size_x=self._ego_aabb_dims[0],
                size_y=self._ego_aabb_dims[1],
                size_z=self._ego_aabb_dims[2],
            ),
            self.cfg.vehicle,
        )
        return RenderableTrajectory.from_trajectory(ego_traj, ego_raabb)

    def _build_driver_responses(
        self, ego_renderable: Optional[RenderableTrajectory]
    ) -> DriverResponses:
        """Build DriverResponses from accumulated data.

        Args:
            ego_renderable: The ego RenderableTrajectory (can be placeholder).

        Returns:
            DriverResponses with all accumulated responses.
        """
        if self._ego_coords_rig_to_aabb_center is None or self._ego_aabb_dims is None:
            raise ValueError(
                "Cannot build DriverResponses without rollout_metadata. "
                "Ensure rollout_metadata message was processed first."
            )

        # Create placeholder if no ego trajectory available
        if ego_renderable is None:
            ego_raabb = RAABB.from_grpc(
                AABB(
                    size_x=self._ego_aabb_dims[0],
                    size_y=self._ego_aabb_dims[1],
                    size_z=self._ego_aabb_dims[2],
                ),
                self.cfg.vehicle,
            )
            ego_renderable = RenderableTrajectory.create_empty_with_bbox(ego_raabb)

        driver_responses = DriverResponses(
            ego_coords_rig_to_aabb_center=self._ego_coords_rig_to_aabb_center,
            ego_trajectory_local=ego_renderable,
        )

        # Add all accumulated driver responses
        for now_us, query_us, drive_response in self._driver_responses:
            driver_responses.add_drive_response(drive_response, now_us, query_us)

        return driver_responses

    def build_scenario_eval_input(
        self,
        run_uuid: str,
        run_name: str,
        batch_id: str,
        vec_map: Optional[VectorMap] = None,
    ) -> ScenarioEvalInput:
        """Build ScenarioEvalInput from all accumulated data.

        This method should be called after all messages have been processed.
        It builds actor trajectories from accumulated poses, constructs
        DriverResponses, and assembles the final ScenarioEvalInput.

        Args:
            run_uuid: Unique identifier for the evaluation run.
            run_name: Human-readable name for the evaluation run.
            batch_id: Batch identifier (typically "0" for single-batch runs).
            vec_map: Optional vector map for offroad detection.

        Returns:
            ScenarioEvalInput ready for evaluation via ScenarioEvaluator.

        Raises:
            ValueError: If rollout_metadata was not processed.
        """
        # Validate required data from rollout_metadata
        if self._session_metadata is None:
            raise ValueError(
                "Cannot build ScenarioEvalInput without session_metadata. "
                "Ensure rollout_metadata message was processed."
            )
        if self._ego_coords_rig_to_aabb_center is None:
            raise ValueError(
                "Cannot build ScenarioEvalInput without ego_coords_rig_to_aabb_center. "
                "Ensure rollout_metadata message was processed."
            )
        if self._ego_aabb_dims is None:
            raise ValueError(
                "Cannot build ScenarioEvalInput without ego_aabb_dims. "
                "Ensure rollout_metadata message was processed."
            )
        if self._gt_ego_trajectory is None:
            raise ValueError(
                "Cannot build ScenarioEvalInput without gt_ego_trajectory. "
                "Ensure rollout_metadata message was processed."
            )

        # Build actor trajectories from accumulated pose data
        actor_trajectories = self._build_actor_trajectories()

        # Build ego renderable trajectory
        ego_renderable = self._build_ego_renderable_trajectory(actor_trajectories)

        # Build DriverResponses
        driver_responses = self._build_driver_responses(ego_renderable)

        # Update driver_responses with actual EGO trajectory if available
        if ego_renderable is not None:
            driver_responses.ego_trajectory_local = ego_renderable

        # Convert routes to global frame using EGO trajectory
        if ego_renderable is not None and self._routes.routes_in_rig_frame:
            self._routes.convert_routes_to_global_frame(
                ego_trajectory=ego_renderable,
                ego_coords_rig_to_aabb_center=self._ego_coords_rig_to_aabb_center,
            )

        return ScenarioEvalInput(
            session_metadata=self._session_metadata,
            ego_coords_rig_to_aabb_center=self._ego_coords_rig_to_aabb_center,
            ego_aabb_x_m=self._ego_aabb_dims[0],
            ego_aabb_y_m=self._ego_aabb_dims[1],
            ego_aabb_z_m=self._ego_aabb_dims[2],
            actor_trajectories=actor_trajectories,
            ego_recorded_ground_truth_trajectory=self._gt_ego_trajectory,
            driver_responses=driver_responses,
            vec_map=vec_map,
            cameras=self._cameras if self._cameras.camera_by_logical_id else None,
            routes=self._routes if self._routes.routes_in_rig_frame else None,
            run_uuid=run_uuid,
            run_name=run_name,
            batch_id=batch_id,
        )
