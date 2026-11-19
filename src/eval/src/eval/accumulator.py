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

import json
import logging
import math
from dataclasses import dataclass, field

import numpy as np
from alpasim_grpc.v0.common_pb2 import AABB
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse
from alpasim_grpc.v0.frame_quality_pb2 import FrameQualityResponse
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata
from alpasim_grpc.v0.traffic_pb2 import TrafficReturn, TrafficSessionRequest
from alpasim_utils.geometry import (
    Pose,
    Trajectory,
    pose_from_grpc,
    trajectory_from_grpc,
)
from trajdata.maps import VectorMap

from eval.data import (
    RAABB,
    AggregationType,
    Cameras,
    DriverResponses,
    MetricReturn,
    NavInstructionAtTime,
    NavInstructions,
    RenderableTrajectory,
    Routes,
    ScenarioEvalInput,
    TrafficPredictions,
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
            run_uuid=..., run_name=..., vec_map=...
        )
    """

    cfg: EvalConfig

    # Internal state - populated from rollout_metadata message
    _session_metadata: RolloutMetadata.SessionMetadata | None = field(
        default=None, init=False
    )
    _ego_coords_rig_to_aabb_center: Pose | None = field(default=None, init=False)
    _ego_aabb_dims: tuple[float, float, float] | None = field(default=None, init=False)
    _gt_ego_trajectory: Trajectory | None = field(default=None, init=False)
    _force_gt_duration_us: int | None = field(default=None, init=False)

    # Actor data from rollout_metadata + actor_poses
    _actor_aabb_dims: dict[str, tuple[float, float, float]] = field(
        default_factory=dict, init=False
    )
    _actor_trajectory_data: dict[str, list[tuple[int, Pose]]] = field(
        default_factory=dict, init=False
    )

    # Driver request/response pairing
    _pending_request: tuple[int, int] | None = field(default=None, init=False)
    _driver_responses: list[tuple[int, int, DriveResponse]] = field(
        default_factory=list, init=False
    )

    # Traffic request/response pairing and metadata
    _pending_traffic_query_us: int | None = field(default=None, init=False)
    _static_actor_ids: set[str] = field(default_factory=set, init=False)
    _traffic_predictions: TrafficPredictions = field(
        default_factory=TrafficPredictions, init=False
    )

    # Camera and route data
    _cameras: Cameras = field(default_factory=Cameras, init=False)
    _routes: Routes = field(default_factory=Routes, init=False)
    _nav_instructions: NavInstructions = field(
        default_factory=NavInstructions, init=False
    )
    _frame_quality_responses: dict[int, FrameQualityResponse] = field(
        default_factory=dict, init=False
    )

    @property
    def session_metadata(self) -> RolloutMetadata.SessionMetadata | None:
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
        elif msg_type == "navigation_instruction":
            nav = message.navigation_instruction
            self._nav_instructions.add(
                NavInstructionAtTime(
                    timestamp_us=int(nav.timestamp_us),
                    instruction_text=nav.instruction_text,
                    remaining_distance_m=float(nav.debug_info.remaining_distance_m),
                    status=nav.DebugInfo.Status.Name(nav.debug_info.status),
                    route_offset_m=float(nav.debug_info.route_offset_m),
                )
            )
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
        elif msg_type == "traffic_session_request":
            self._handle_traffic_session_request(message.traffic_session_request)
        elif msg_type == "traffic_request":
            self._pending_traffic_query_us = int(message.traffic_request.time_query_us)
        elif msg_type == "traffic_return":
            self._handle_traffic_return(message.traffic_return)
        elif msg_type == "available_cameras_return":
            for available_camera in message.available_cameras_return.available_cameras:
                self._cameras.add_calibration(available_camera)
        elif msg_type == "frame_quality_response":
            self._handle_frame_quality_response(message.frame_quality_response)

    def _handle_frame_quality_response(self, response: FrameQualityResponse) -> None:
        if (
            not response.session_uuid
            or not response.camera_logical_id
            or not response.checkpoint_id
            or not response.checkpoint_sha256
        ):
            raise ValueError("frame quality response has incomplete provenance")
        score = float(response.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"frame quality score must be finite and in [0, 1], got {score}"
            )

        timestamp_us = int(response.timestamp_us)
        if timestamp_us in self._frame_quality_responses:
            raise ValueError(
                f"frame quality response has duplicate timestamp {timestamp_us}"
            )
        if self._frame_quality_responses:
            previous = next(iter(self._frame_quality_responses.values()))
            if (
                response.session_uuid != previous.session_uuid
                or response.camera_logical_id != previous.camera_logical_id
                or response.checkpoint_id != previous.checkpoint_id
                or response.checkpoint_sha256 != previous.checkpoint_sha256
            ):
                raise ValueError("frame quality responses have conflicting provenance")

        stored_response = FrameQualityResponse()
        stored_response.CopyFrom(response)
        self._frame_quality_responses[timestamp_us] = stored_response

    def _build_frame_quality_metrics(self) -> list[MetricReturn]:
        if not self._frame_quality_responses:
            return []
        if self._session_metadata is None:
            raise ValueError(
                "Cannot validate frame quality response without session metadata"
            )

        responses = sorted(
            self._frame_quality_responses.values(),
            key=lambda response: response.timestamp_us,
        )
        first = responses[0]
        if first.session_uuid != self._session_metadata.session_uuid:
            raise ValueError(
                "frame quality response session does not match rollout metadata: "
                f"{first.session_uuid!r} != {self._session_metadata.session_uuid!r}"
            )
        camera = self._cameras.camera_by_logical_id.get(first.camera_logical_id)
        response_timestamps = {int(response.timestamp_us) for response in responses}
        camera_timestamps = set(camera.timestamps_us) if camera is not None else set()
        if response_timestamps != camera_timestamps:
            raise ValueError(
                "frame quality response stream is incomplete for camera "
                f"{first.camera_logical_id!r}: scored={len(response_timestamps)} "
                f"frames={len(camera_timestamps)}"
            )

        return [
            MetricReturn(
                name="frame_quality_score",
                timestamps_us=[int(response.timestamp_us) for response in responses],
                values=[float(response.score) for response in responses],
                valid=[True] * len(responses),
                time_aggregation=AggregationType.P90,
                info=json.dumps(
                    {
                        "metadata": {
                            "camera_logical_id": first.camera_logical_id,
                            "checkpoint_id": first.checkpoint_id,
                            "checkpoint_sha256": first.checkpoint_sha256,
                        },
                        "source": "frame_quality",
                    },
                    sort_keys=True,
                ),
            )
        ]

    def _handle_rollout_metadata(self, metadata: RolloutMetadata) -> None:
        """Extract data from rollout_metadata message.

        Args:
            metadata: The RolloutMetadata protobuf message.
        """
        self._session_metadata = metadata.session_metadata
        if metadata.force_gt_duration is not None:
            force_gt = int(metadata.force_gt_duration)
            if force_gt < 0:
                raise ValueError(f"force_gt_duration must be >= 0, got {force_gt}")
            self._force_gt_duration_us = force_gt

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

    def _handle_traffic_session_request(self, request: TrafficSessionRequest) -> None:
        self._static_actor_ids = {
            str(obj.object_id)
            for obj in request.logged_object_trajectories
            if obj.is_static
        }

    def _handle_traffic_return(self, traffic_return: TrafficReturn) -> None:
        if self._pending_traffic_query_us is None:
            return

        query_time_us = self._pending_traffic_query_us
        self._pending_traffic_query_us = None
        object_trajectories: dict[str, Trajectory] = {}
        for update in traffic_return.object_trajectory_updates:
            object_id = str(update.object_id)
            if object_id == "EGO" or object_id in self._static_actor_ids:
                continue
            trajectory = trajectory_from_grpc(update.trajectory)
            if trajectory.is_empty() or trajectory.time_range_us.stop <= query_time_us:
                continue
            object_trajectories[object_id] = trajectory

        self._traffic_predictions.add_prediction(query_time_us, object_trajectories)

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
    ) -> RenderableTrajectory | None:
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
        self, ego_renderable: RenderableTrajectory | None
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
            parse_unstructured_debug_info=self.cfg.parse_unstructured_debug_info,
        )

        # Add all accumulated driver responses
        for now_us, query_us, drive_response in self._driver_responses:
            driver_responses.add_drive_response(drive_response, now_us, query_us)

        return driver_responses

    def build_scenario_eval_input(
        self,
        run_uuid: str,
        run_name: str,
        vec_map: VectorMap | None = None,
    ) -> ScenarioEvalInput:
        """Build ScenarioEvalInput from all accumulated data.

        This method should be called after all messages have been processed.
        It builds actor trajectories from accumulated poses, constructs
        DriverResponses, and assembles the final ScenarioEvalInput.

        Args:
            run_uuid: Unique identifier for the evaluation run.
            run_name: Human-readable name for the evaluation run.
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
            traffic_predictions=self._traffic_predictions,
            vec_map=vec_map,
            cameras=self._cameras if self._cameras.camera_by_logical_id else None,
            routes=self._routes if self._routes.routes_in_rig_frame else None,
            nav_instructions=(
                self._nav_instructions if self._nav_instructions.timestamps_us else None
            ),
            run_uuid=run_uuid,
            run_name=run_name,
            force_gt_duration_us=self._force_gt_duration_us,
            recorded_metrics=self._build_frame_quality_metrics(),
        )
