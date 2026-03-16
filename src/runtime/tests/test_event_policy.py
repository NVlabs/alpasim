# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for PolicyEvent and its helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.events.base import EventQueue
from alpasim_runtime.events.physics import PhysicsEvent, PhysicsTarget
from alpasim_runtime.events.policy import (
    PolicyEvent,
    assert_sensors_up_to_date,
    transform_trajectory_from_noisy_to_true_local_frame,
)
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_utils.geometry import Polyline, Pose, Trajectory

# ---------------------------------------------------------------------------
# PolicyEvent tests
# ---------------------------------------------------------------------------


class TestPolicyEvent:
    @pytest.fixture
    def policy_event(self, service_bundle: ServiceBundle) -> PolicyEvent:
        return PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
        )

    def test_priority(self, policy_event: PolicyEvent):
        assert policy_event.priority == 20

    def test_description(self, policy_event: PolicyEvent):
        desc = policy_event.description()
        assert "PolicyEvent" in desc
        assert "200_000" in desc

    @pytest.mark.asyncio
    async def test_run_calls_drive(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """PolicyEvent.run() should call driver.drive with correct timestamps."""
        # Setup driver to return a trajectory
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        mock_driver.drive.assert_awaited_once()
        call_kwargs = mock_driver.drive.call_args.kwargs
        assert call_kwargs["time_now_us"] == 200_000
        assert call_kwargs["time_query_us"] == 300_000

    @pytest.mark.asyncio
    async def test_handle_reschedules_same_event(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """RecurringEvent.handle() runs then reschedules with advanced timestamp."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        queue = EventQueue()
        await policy_event.handle(rollout_state, queue)

        # RecurringEvent resubmits itself with advanced timestamp
        assert len(queue) == 1
        next_event = queue.pop()
        assert next_event is policy_event  # Same object, mutated
        assert next_event.timestamp_us == 300_000  # 200_000 + 100_000

    @pytest.mark.asyncio
    async def test_run_clears_sensorsim_data(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """data_sensorsim_to_driver should be cleared after policy consumes it."""
        rollout_state.data_sensorsim_to_driver = b"some_data"

        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        assert rollout_state.data_sensorsim_to_driver is None

    @pytest.mark.asyncio
    async def test_run_fills_step_context_timing(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """PolicyEvent.run() fills timing fields on the existing StepContext."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        assert rollout_state.step_context is not None
        assert rollout_state.step_context.step_start_us == 200_000
        assert rollout_state.step_context.target_time_us == 300_000
        assert rollout_state.step_context.driver_trajectory is not None

    @pytest.mark.asyncio
    async def test_drains_outstanding_tasks_before_drive(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ) -> None:
        """PolicyEvent must drain all outstanding tasks before calling driver.drive()."""
        import asyncio

        gate = asyncio.Event()

        async def blocked_send() -> None:
            await gate.wait()

        # Simulate a pending camera submission from GroupedRenderEvent
        rollout_state.step_context.track_task(blocked_send())

        mock_driver.drive.return_value = simple_trajectory.clip(200_000, 300_001)

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
        )
        queue = EventQueue()
        policy_task = asyncio.create_task(event.run(rollout_state, queue))

        await asyncio.sleep(0)
        mock_driver.drive.assert_not_awaited()  # blocked on gate

        gate.set()
        await policy_task
        mock_driver.drive.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_raises_when_route_timestamp_mismatches_ego_state(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
    ):
        route = Polyline(
            points=np.array(
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        )
        event = PolicyEvent(
            timestamp_us=100_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=cast(
                Any,
                SimpleNamespace(generate_route=lambda *_args, **_kwargs: route),
            ),
            send_recording_ground_truth=False,
        )

        with pytest.raises(ValueError, match="Timestamp mismatch"):
            await event.run(rollout_state, EventQueue())

        mock_driver.submit_route.assert_not_awaited()
        mock_driver.drive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_raises_when_ground_truth_timestamp_mismatches_ego_state(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        mock_driver.drive.return_value = simple_trajectory.clip(100_000, 200_001)

        event = PolicyEvent(
            timestamp_us=100_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=True,
        )

        with pytest.raises(ValueError, match="Timestamp mismatch"):
            await event.run(rollout_state, EventQueue())

        mock_driver.submit_recording_ground_truth.assert_not_awaited()
        mock_driver.drive.assert_not_awaited()


# ---------------------------------------------------------------------------
# assert_sensors_up_to_date tests
# ---------------------------------------------------------------------------


class TestAssertSensorsUpToDate:
    def test_passes_when_synchronized(self, rollout_state: RolloutState):
        # ego_trajectory_estimate covers [0, 200_000], so step_start_us must
        # match the latest timestamp for the egopose freshness check.
        step_start_us = 200_000
        rollout_state.last_camera_frame_us = {"cam_front": step_start_us}

        # Should not raise
        assert_sensors_up_to_date(rollout_state, step_start_us, ["cam_front"])

    def test_raises_on_camera_mismatch(self, rollout_state: RolloutState):
        # ego_trajectory_estimate latest is 200_000 — use that as step_start_us
        # so the egopose check passes and the camera check is exercised.
        rollout_state.last_camera_frame_us = {
            "cam_front": 50_000,
            "cam_rear": 200_000,
        }

        with pytest.raises(ValueError, match="Cameras not up to date"):
            assert_sensors_up_to_date(rollout_state, 200_000, ["cam_front", "cam_rear"])

    def test_skips_camera_check_on_first_policy_call(self, rollout_state: RolloutState):
        """First policy call has no camera frames yet — should not raise."""
        rollout_state.last_camera_frame_us = {}  # No frames yet

        # Should not raise even though cameras have no frames
        assert_sensors_up_to_date(rollout_state, 200_000, ["cam_front"])

    def test_empty_camera_ids(self, rollout_state: RolloutState):
        rollout_state.last_camera_frame_us = {}

        # No cameras to check — should pass (egopose check still runs)
        assert_sensors_up_to_date(rollout_state, 200_000, [])

    def test_raises_on_stale_egopose(self, rollout_state: RolloutState):
        """Egopose freshness check fires when trajectory estimate is behind."""
        # ego_trajectory_estimate covers [0, 200_000], but step_start_us is
        # ahead of that — the latest committed egopose is stale.
        rollout_state.last_camera_frame_us = {"cam_front": 300_000}

        with pytest.raises(ValueError, match="Egopose not up to date"):
            assert_sensors_up_to_date(rollout_state, 300_000, ["cam_front"])


# ---------------------------------------------------------------------------
# transform_trajectory tests
# ---------------------------------------------------------------------------


class TestTransformTrajectory:
    def test_identity_transform(self, rollout_state: RolloutState):
        """With identity ego poses, transform should be ~identity."""
        # Create a small trajectory to transform
        input_traj = Trajectory.from_poses(
            timestamps=np.array([200_000, 300_000], dtype=np.uint64),
            poses=[
                Pose(
                    np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([2.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
            ],
        )
        result = transform_trajectory_from_noisy_to_true_local_frame(
            rollout_state, input_traj
        )

        # Should return a trajectory (the exact values depend on ego_trajectory.last_pose)
        assert len(result.timestamps_us) == 2


# ---------------------------------------------------------------------------
# apply_traffic_physics tests
# ---------------------------------------------------------------------------


class TestPhysicsEventTraffic:
    @pytest.mark.asyncio
    async def test_returns_poses_from_traffic_response(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_physics: AsyncMock,
    ):
        """Traffic poses should be extracted from TrafficReturn."""
        from alpasim_grpc.v0.traffic_pb2 import ObjectTrajectoryUpdate, TrafficReturn
        from alpasim_utils import geometry

        # Make physics_update_mode != ALL_ACTORS to skip physics call
        rollout_state.unbound.physics_update_mode = PhysicsUpdateMode.NONE

        # Build an ego trajectory covering the target time
        ego_traj = Trajectory.from_poses(
            timestamps=np.array([200_000, 300_000], dtype=np.uint64),
            poses=[
                Pose(
                    np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([2.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
            ],
        )

        # Create a traffic response with one object using real protobuf types
        obj_traj = Trajectory.from_poses(
            timestamps=np.array([200_000, 300_000], dtype=np.uint64),
            poses=[
                Pose(
                    np.array([5.0, 1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([6.0, 1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
            ],
        )

        traffic_update = ObjectTrajectoryUpdate(
            object_id="car_1",
            trajectory=geometry.trajectory_to_grpc(obj_traj),
        )

        traffic_response = TrafficReturn(object_trajectory_updates=[traffic_update])

        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=300_000,
            force_gt=False,
            corrected_ego_trajectory=ego_traj,
            traffic_response=traffic_response,
        )
        rollout_state.step_context = ctx

        event = PhysicsEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            services=service_bundle,
            target=PhysicsTarget.TRAFFIC,
        )
        await event.run(rollout_state, EventQueue())

        assert "car_1" in ctx.traffic_trajectories
        mock_physics.ground_intersection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_traffic_response(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_physics: AsyncMock,
    ):
        from alpasim_grpc.v0.traffic_pb2 import TrafficReturn

        rollout_state.unbound.physics_update_mode = PhysicsUpdateMode.NONE

        ego_traj = Trajectory.from_poses(
            timestamps=np.array([200_000, 300_000], dtype=np.uint64),
            poses=[
                Pose(
                    np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([2.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
            ],
        )

        traffic_response = TrafficReturn(object_trajectory_updates=[])

        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=300_000,
            force_gt=False,
            corrected_ego_trajectory=ego_traj,
            traffic_response=traffic_response,
        )
        rollout_state.step_context = ctx

        event = PhysicsEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            services=service_bundle,
            target=PhysicsTarget.TRAFFIC,
        )
        await event.run(rollout_state, EventQueue())

        assert ctx.traffic_trajectories == {}
