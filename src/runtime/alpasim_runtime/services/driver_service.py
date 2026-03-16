# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Driver service implementation."""

from __future__ import annotations

import logging
import random
from typing import Optional, Type

import numpy as np
from alpasim_grpc.v0.common_pb2 import DynamicState
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    GroundTruth,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    RouteRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import EgodriverServiceStub
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo
from alpasim_runtime.services.session_configs import DriverSessionConfig
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_utils.geometry import (
    Polyline,
    Pose,
    Trajectory,
    polyline_to_grpc_route,
    trajectory_from_grpc,
    trajectory_to_grpc,
)
from alpasim_utils.types import ImageWithMetadata

logger = logging.getLogger(__name__)


class DriverService(ServiceBase[EgodriverServiceStub]):
    """
    Service for interacting with the autonomous driving policy.

    This service handles communication with the driver model, including
    submitting sensor observations and receiving driving decisions.
    """

    @property
    def stub_class(self) -> Type[EgodriverServiceStub]:
        return EgodriverServiceStub

    async def _initialize_session(self, session_info: SessionInfo) -> None:
        """Initialize driver session after gRPC connection is established."""
        cfg = session_info.session_config
        if not isinstance(cfg, DriverSessionConfig):
            raise TypeError(
                "DriverService._initialize_session requires a DriverSessionConfig "
                f"via session_config, got {type(cfg).__name__}."
            )

        scene_id = cfg.scene_id
        sensorsim_cameras = cfg.sensorsim_cameras

        rollout_spec = DriveSessionRequest.RolloutSpec(
            vehicle=DriveSessionRequest.RolloutSpec.VehicleDefinition(
                available_cameras=sensorsim_cameras,
            ),
        )

        request = DriveSessionRequest(
            session_uuid=session_info.uuid,
            random_seed=random.randint(0, 2**32 - 1),
            debug_info=DriveSessionRequest.DebugInfo(scene_id=scene_id),
            rollout_spec=rollout_spec,
        )

        await session_info.broadcaster.broadcast(
            LogEntry(driver_session_request=request)
        )

        if self.skip:
            return

        await profiled_rpc_call(
            "start_session", "driver", self.stub.start_session, request
        )

    async def _cleanup_session(self, session_info: SessionInfo) -> None:
        """Clean up driver session."""
        if self.skip:
            return

        close_request = DriveSessionCloseRequest(session_uuid=session_info.uuid)
        await profiled_rpc_call(
            "close_session", "driver", self.stub.close_session, close_request
        )

    async def submit_image(self, image: ImageWithMetadata) -> None:
        """Submit an image observation for the current session."""
        session_info = self._require_session_info()
        request = RolloutCameraImage(
            session_uuid=session_info.uuid,
            camera_image=RolloutCameraImage.CameraImage(
                frame_start_us=image.start_timestamp_us,
                frame_end_us=image.end_timestamp_us,
                image_bytes=image.image_bytes,
                logical_id=image.camera_logical_id,
            ),
        )

        await session_info.broadcaster.broadcast(LogEntry(driver_camera_image=request))

        if self.skip:
            return

        await profiled_rpc_call(
            "submit_image_observation",
            "driver",
            self.stub.submit_image_observation,
            request,
        )

    async def submit_trajectory(
        self,
        trajectory: Trajectory,
        dynamic_states_in_rig: list[DynamicState],
    ) -> None:
        """Submit an egomotion trajectory for the current session.

        Args:
            trajectory: The estimated ego trajectory.
            dynamic_states_in_rig: Estimated dynamic states (velocities, accelerations)
                resolved in the rig frame, one per pose in the trajectory.
        """
        session_info = self._require_session_info()
        request = RolloutEgoTrajectory(
            session_uuid=session_info.uuid,
            trajectory=trajectory_to_grpc(trajectory),
            dynamic_states=dynamic_states_in_rig,
        )

        await session_info.broadcaster.broadcast(
            LogEntry(driver_ego_trajectory=request)
        )

        if self.skip:
            return

        await profiled_rpc_call(
            "submit_egomotion_observation",
            "driver",
            self.stub.submit_egomotion_observation,
            request,
        )

    async def submit_route(
        self, timestamp_us: int, route_polyline_in_rig: Polyline
    ) -> None:
        """Submit a route for the current session."""
        session_info = self._require_session_info()
        # Convert the route polyline to gRPC Route format
        grpc_route = polyline_to_grpc_route(route_polyline_in_rig, timestamp_us)

        request = RouteRequest(
            session_uuid=session_info.uuid,
            route=grpc_route,
        )

        await session_info.broadcaster.broadcast(LogEntry(route_request=request))

        if self.skip:
            return

        await profiled_rpc_call(
            "submit_route", "driver", self.stub.submit_route, request
        )

    async def submit_recording_ground_truth(
        self, timestamp_us: int, trajectory: Trajectory
    ) -> None:
        """Submit ground truth from recording for the current session."""
        session_info = self._require_session_info()
        request = GroundTruthRequest(
            session_uuid=session_info.uuid,
            ground_truth=GroundTruth(
                timestamp_us=timestamp_us,
                trajectory=trajectory_to_grpc(trajectory),
            ),
        )

        await session_info.broadcaster.broadcast(LogEntry(ground_truth_request=request))

        if self.skip:
            return

        await profiled_rpc_call(
            "submit_recording_ground_truth",
            "driver",
            self.stub.submit_recording_ground_truth,
            request,
        )

    async def drive(
        self, time_now_us: int, time_query_us: int, renderer_data: Optional[bytes]
    ) -> Trajectory:
        """Request a drive decision for the current session.

        Returns:
            Trajectory containing the selected trajectory for the ego vehicle.
        """
        session_info = self._require_session_info()
        # Create request with both old and new fields for backward compatibility
        request = DriveRequest(
            session_uuid=session_info.uuid,
            time_now_us=time_now_us,
            time_query_us=time_query_us,
            renderer_data=renderer_data or b"",
        )

        await session_info.broadcaster.broadcast(LogEntry(driver_request=request))

        if self.skip:
            # Create a trajectory response with multiple future timestamps.
            # This enables plan_deviation scorer to compute metrics by comparing
            # overlapping timestamps between consecutive drive calls.

            # Generate timestamps extending 5 seconds into the future at 100ms intervals
            num_points = 50
            interval_us = 100_000  # 100ms
            timestamps = np.array(
                [time_now_us + i * interval_us for i in range(num_points)],
                dtype=np.uint64,
            )

            # Create poses - simple straight-line trajectory moving forward
            poses = [
                Pose(
                    position=np.array(
                        [i * 0.5, 0.0, 0.0], dtype=np.float32
                    ),  # 0.5m per step = 5m/s
                    quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                )
                for i in range(num_points)
            ]

            trajectory = Trajectory.from_poses(
                timestamps=timestamps,
                poses=poses,
            )
            response = DriveResponse(
                trajectory=trajectory_to_grpc(trajectory),
            )
        else:
            response = await profiled_rpc_call(
                "drive", "driver", self.stub.drive, request
            )

        await session_info.broadcaster.broadcast(LogEntry(driver_return=response))

        return trajectory_from_grpc(response.trajectory)
