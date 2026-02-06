# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Controller service implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Type

import numpy as np
from alpasim_grpc.v0.common_pb2 import DynamicState, Vec3
from alpasim_grpc.v0.controller_pb2 import (
    RunControllerAndVehicleModelRequest,
    VDCSessionCloseRequest,
    VDCSessionRequest,
)
from alpasim_grpc.v0.controller_pb2_grpc import VDCServiceStub
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_runtime.services.service_base import (
    WILDCARD_SCENE_ID,
    ServiceBase,
    SessionInfo,
)
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_utils.qvec import QVec
from alpasim_utils.trajectory import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class PropagatedPoses:
    """
    A pair of poses and dynamic states, one representing the true predicted output from
    the vehicle model, and the other representing the estimate used by the controller in the loop.
    """

    pose_local_to_rig: QVec  # The pose of the vehicle in the local frame
    pose_local_to_rig_estimate: (
        QVec  # The "software" estimated pose of the vehicle in the local frame
    )
    dynamic_state: DynamicState  # The true dynamic state (velocities, accelerations)
    dynamic_state_estimated: DynamicState  # The estimated dynamic state


class ControllerService(ServiceBase[VDCServiceStub]):
    """
    Controller service implementation that handles both real and skip modes.
    """

    @property
    def stub_class(self) -> Type[VDCServiceStub]:
        return VDCServiceStub

    @staticmethod
    def create_run_controller_and_vehicle_request(
        session_uuid: str,
        now_us: int,
        pose_local_to_rig: QVec,
        rig_linear_velocity_in_rig: np.ndarray,
        rig_angular_velocity_in_rig: np.ndarray,
        rig_reference_trajectory_in_rig: Trajectory,
        future_us: int,
        force_gt: bool,
    ) -> RunControllerAndVehicleModelRequest:
        """
        Helper method to generate a RunControllerAndVehicleModelRequest.
        """
        request = RunControllerAndVehicleModelRequest()
        request.session_uuid = session_uuid

        request.state.pose.CopyFrom(pose_local_to_rig.as_grpc_pose())
        request.state.timestamp_us = now_us
        request.state.state.linear_velocity.CopyFrom(
            Vec3(
                x=rig_linear_velocity_in_rig[0],
                y=rig_linear_velocity_in_rig[1],
                z=rig_linear_velocity_in_rig[2],
            )
        )
        request.state.state.angular_velocity.CopyFrom(
            Vec3(
                x=rig_angular_velocity_in_rig[0],
                y=rig_angular_velocity_in_rig[1],
                z=rig_angular_velocity_in_rig[2],
            )
        )

        request.planned_trajectory_in_rig.CopyFrom(
            rig_reference_trajectory_in_rig.to_grpc()
        )

        request.future_time_us = future_us

        request.coerce_dynamic_state = force_gt
        return request

    async def _initialize_session(
        self, session_info: SessionInfo, **kwargs: Any
    ) -> None:
        """Initialize a controller service session."""
        if self.stub:
            request = VDCSessionRequest(session_uuid=session_info.uuid)
            await profiled_rpc_call(
                "start_session", "controller", self.stub.start_session, request
            )
        else:
            if self.skip:
                logger.info("Skip mode: no stub, session cannot be initialized")
            else:
                raise RuntimeError(
                    "ControllerService stub is not initialized, cannot start session"
                )

    async def _cleanup_session(self, session_info: SessionInfo, **kwargs: Any) -> None:
        """Cleanup resources associated with the session"""
        if self.stub:
            await profiled_rpc_call(
                "close_session",
                "controller",
                self.stub.close_session,
                VDCSessionCloseRequest(session_uuid=session_info.uuid),
            )
        else:
            if self.skip:
                logger.info("Skip mode: no stub, session cannot be cleaned up")
            else:
                raise RuntimeError(
                    "ControllerService stub is not initialized, cannot clean up session"
                )

    async def run_controller_and_vehicle(
        self,
        now_us: int,
        pose_local_to_rig: QVec,
        rig_linear_velocity_in_rig: np.ndarray,
        rig_angular_velocity_in_rig: np.ndarray,
        rig_reference_trajectory_in_rig: Trajectory,
        future_us: int,
        force_gt: bool,
        fallback_pose_local_to_rig_future: QVec,
    ) -> PropagatedPoses:
        """
        Run controller and vehicle model to get future pose.
        """

        # Skip expensive gRPC request construction when in skip mode
        if self.skip:
            logger.debug("Skip mode: controller returning fallback pose")
            return PropagatedPoses(
                pose_local_to_rig=fallback_pose_local_to_rig_future,
                pose_local_to_rig_estimate=fallback_pose_local_to_rig_future,
                dynamic_state=DynamicState(),
                dynamic_state_estimated=DynamicState(),
            )

        request = self.create_run_controller_and_vehicle_request(
            session_uuid=self.session_info.uuid,
            now_us=now_us,
            pose_local_to_rig=pose_local_to_rig,
            rig_linear_velocity_in_rig=rig_linear_velocity_in_rig,
            rig_angular_velocity_in_rig=rig_angular_velocity_in_rig,
            rig_reference_trajectory_in_rig=rig_reference_trajectory_in_rig,
            future_us=future_us,
            force_gt=force_gt,
        )

        await self.session_info.broadcaster.broadcast(
            LogEntry(controller_request=request)
        )

        response = await profiled_rpc_call(
            "run_controller_and_vehicle",
            "controller",
            self.stub.run_controller_and_vehicle,
            request,
        )

        await self.session_info.broadcaster.broadcast(
            LogEntry(controller_return=response)
        )

        return PropagatedPoses(
            pose_local_to_rig=QVec.from_grpc_pose(response.pose_local_to_rig.pose),
            pose_local_to_rig_estimate=QVec.from_grpc_pose(
                response.pose_local_to_rig_estimated.pose
            ),
            dynamic_state=response.dynamic_state,
            dynamic_state_estimated=response.dynamic_state_estimated,
        )

    async def get_available_scenes(self) -> List[str]:
        """Get list of available scenes from the controller service.

        The controller supports all scenes.
        """
        return [WILDCARD_SCENE_ID]
