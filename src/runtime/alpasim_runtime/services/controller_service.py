# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Controller service implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Type

import numpy as np
from alpasim_grpc.v0.common_pb2 import DynamicState, Vec3
from alpasim_grpc.v0.controller_pb2 import (
    RunControllerAndVehicleModelRequest,
    VDCSessionCloseRequest,
    VDCSessionRequest,
)
from alpasim_grpc.v0.controller_pb2_grpc import VDCServiceStub
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_utils.geometry import (
    Pose,
    Trajectory,
    pose_from_grpc,
    pose_to_grpc,
    trajectory_to_grpc,
)

logger = logging.getLogger(__name__)


@dataclass
class PropagatedPosesAtTime:
    """Single pose + dynamic state at a specific timestamp."""

    timestamp_us: int
    pose_local_to_rig: Pose  # The pose of the vehicle in the local frame
    pose_local_to_rig_estimate: Pose  # The "software" estimated pose in local frame
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
        pose_local_to_rig: Pose,
        rig_linear_velocity_in_rig: np.ndarray,
        rig_angular_velocity_in_rig: np.ndarray,
        rig_reference_trajectory_in_rig: Trajectory,
        future_us: int,
        force_gt: bool,
        pose_reporting_interval_us: int = 0,
    ) -> RunControllerAndVehicleModelRequest:
        """
        Helper method to generate a RunControllerAndVehicleModelRequest.
        """
        request = RunControllerAndVehicleModelRequest()
        request.session_uuid = session_uuid

        request.state.pose.CopyFrom(pose_to_grpc(pose_local_to_rig))
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
            trajectory_to_grpc(rig_reference_trajectory_in_rig)
        )

        request.future_time_us = future_us

        request.coerce_dynamic_state = force_gt
        request.pose_reporting_interval_us = pose_reporting_interval_us
        return request

    async def _initialize_session(self, session_info: SessionInfo) -> None:
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

    async def _cleanup_session(self, session_info: SessionInfo) -> None:
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

    # TODO(mwatson): Simplify this once deprecated fields are removed
    @staticmethod
    def _ensure_intermediates(
        propagated_states: list[PropagatedPosesAtTime],
        fallback_trajectory_local_to_rig: Trajectory,
        now_us: int,
        future_us: int,
        pose_reporting_interval_us: int,
    ) -> list[PropagatedPosesAtTime]:
        """Backfill intermediate states if the result only contains the final state.

        When the controller (or skip mode) returns only the final pose but the
        caller expects intermediate poses at ``pose_reporting_interval_us``
        spacing, this method generates them by interpolating
        ``fallback_trajectory_local_to_rig``.
        """
        expected_intermediate_timestamps = (
            list(
                range(
                    now_us + pose_reporting_interval_us,
                    future_us,
                    pose_reporting_interval_us,
                )
            )
            if pose_reporting_interval_us > 0
            else []
        )

        has_intermediates = len(propagated_states) > 1
        if not has_intermediates and expected_intermediate_timestamps:
            logger.debug(
                "Generating %d intermediate states by interpolation",
                len(expected_intermediate_timestamps),
            )
            ts_array = np.array(expected_intermediate_timestamps, dtype=np.uint64)
            poses = fallback_trajectory_local_to_rig.interpolate_poses_list(ts_array)
            intermediates = [
                PropagatedPosesAtTime(
                    timestamp_us=t,
                    pose_local_to_rig=pose,
                    pose_local_to_rig_estimate=pose,
                    dynamic_state=DynamicState(),
                    dynamic_state_estimated=DynamicState(),
                )
                for t, pose in zip(expected_intermediate_timestamps, poses, strict=True)
            ]
            return intermediates + propagated_states

        logger.debug(
            "Controller generated %d intermediate states",
            len(propagated_states) - 1,
        )
        return propagated_states

    async def run_controller_and_vehicle(
        self,
        now_us: int,
        pose_local_to_rig: Pose,
        rig_linear_velocity_in_rig: np.ndarray,
        rig_angular_velocity_in_rig: np.ndarray,
        rig_reference_trajectory_in_rig: Trajectory,
        future_us: int,
        force_gt: bool,
        fallback_trajectory_local_to_rig: Trajectory,
        pose_reporting_interval_us: int = 0,
    ) -> list[PropagatedPosesAtTime]:
        """Run controller and vehicle model to propagate the ego pose to *future_us*.

        Args:
            now_us: Current simulation timestamp in microseconds.
            pose_local_to_rig: Current ego pose in local frame.
            rig_linear_velocity_in_rig: Linear velocity vector in rig frame.
            rig_angular_velocity_in_rig: Angular velocity vector in rig frame.
            rig_reference_trajectory_in_rig: Planned reference trajectory in rig frame.
            future_us: Target timestamp to propagate to.
            force_gt: If True, coerce the vehicle model to use ground-truth state.
            fallback_trajectory_local_to_rig: Trajectory used in skip mode or
                force_gt mode; interpolated at future_us to produce the fallback pose.
            pose_reporting_interval_us: Interval for intermediate state reporting.
                When > 0, intermediate states are generated between now_us and future_us.

        Returns:
            List of PropagatedPosesAtTime in chronological order. The last element
            is the final state at future_us; preceding elements are intermediates.
        """
        session_info = self._require_session_info()

        # Skip expensive gRPC request construction when in skip mode
        if self.skip:
            logger.debug("Skip mode: controller returning fallback pose")
            fallback_pose_local_to_rig = (
                fallback_trajectory_local_to_rig.interpolate_pose(future_us)
            )
            result = [
                PropagatedPosesAtTime(
                    timestamp_us=future_us,
                    pose_local_to_rig=fallback_pose_local_to_rig,
                    pose_local_to_rig_estimate=fallback_pose_local_to_rig,
                    dynamic_state=DynamicState(),
                    dynamic_state_estimated=DynamicState(),
                )
            ]
            return self._ensure_intermediates(
                result,
                fallback_trajectory_local_to_rig,
                now_us,
                future_us,
                pose_reporting_interval_us,
            )

        request = self.create_run_controller_and_vehicle_request(
            session_uuid=session_info.uuid,
            now_us=now_us,
            pose_local_to_rig=pose_local_to_rig,
            rig_linear_velocity_in_rig=rig_linear_velocity_in_rig,
            rig_angular_velocity_in_rig=rig_angular_velocity_in_rig,
            rig_reference_trajectory_in_rig=rig_reference_trajectory_in_rig,
            future_us=future_us,
            force_gt=force_gt,
            pose_reporting_interval_us=pose_reporting_interval_us,
        )

        await session_info.broadcaster.broadcast(LogEntry(controller_request=request))

        response = await profiled_rpc_call(
            "run_controller_and_vehicle",
            "controller",
            self.stub.run_controller_and_vehicle,
            request,
        )

        await session_info.broadcaster.broadcast(LogEntry(controller_return=response))

        # When force_gt, ignore the controller response and populate from the
        # fallback (ground-truth) trajectory so downstream always sees GT poses.
        if force_gt:
            fallback_pose_local_to_rig = (
                fallback_trajectory_local_to_rig.interpolate_pose(future_us)
            )
            result = [
                PropagatedPosesAtTime(
                    timestamp_us=future_us,
                    pose_local_to_rig=fallback_pose_local_to_rig,
                    pose_local_to_rig_estimate=fallback_pose_local_to_rig,
                    dynamic_state=DynamicState(),
                    dynamic_state_estimated=DynamicState(),
                )
            ]
        elif response.states:
            # Prefer the new `states` field
            result = [
                PropagatedPosesAtTime(
                    timestamp_us=s.timestamp_us,
                    pose_local_to_rig=pose_from_grpc(s.pose_local_to_rig),
                    pose_local_to_rig_estimate=pose_from_grpc(
                        s.pose_local_to_rig_estimated
                    ),
                    dynamic_state=s.dynamic_state,
                    dynamic_state_estimated=s.dynamic_state_estimated,
                )
                for s in response.states
            ]
        else:  # Deprecated path: read from deprecated fields
            result = [
                PropagatedPosesAtTime(
                    timestamp_us=future_us,
                    pose_local_to_rig=pose_from_grpc(response.pose_local_to_rig.pose),
                    pose_local_to_rig_estimate=pose_from_grpc(
                        response.pose_local_to_rig_estimated.pose
                    ),
                    dynamic_state=response.dynamic_state,
                    dynamic_state_estimated=response.dynamic_state_estimated,
                )
            ]

        return self._ensure_intermediates(
            result,
            fallback_trajectory_local_to_rig,
            now_us,
            future_us,
            pose_reporting_interval_us,
        )
