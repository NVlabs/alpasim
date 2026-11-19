# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for PolicyEvent and its helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from alpasim_grpc.v0.egodriver_pb2 import VectorizedObservation
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.events.base import EndSimulationException, EventQueue
from alpasim_runtime.events.force_gt_utils import controller_reference_trajectory
from alpasim_runtime.events.physics import PhysicsEvent, PhysicsTarget
from alpasim_runtime.events.policy import (
    PolicyEvent,
    assert_sensors_up_to_date,
    transform_trajectory_from_noisy_to_true_local_frame,
)
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_utils.geometry import DynamicTrajectory, Polyline, Pose, Trajectory

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
            send_vectorized_observation=False,
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
        monkeypatch: pytest.MonkeyPatch,
    ):
        """PolicyEvent.run() should call driver.drive with correct timestamps."""
        # Setup driver to return a trajectory
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = (drive_traj, False)
        estimate_timestamps_us = np.array([0, 100_000, 300_000], dtype=np.uint64)
        estimate_context = Trajectory.from_poses(
            timestamps=estimate_timestamps_us,
            poses=[
                Pose(
                    np.array([10.0 + float(ts) / 1e6, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                )
                for ts in estimate_timestamps_us
            ],
        )
        zero_dynamics = np.zeros(
            (len(estimate_context.timestamps_us), 12), dtype=np.float64
        )
        rollout_state.ego_trajectory_estimate = (
            DynamicTrajectory.from_trajectory_and_dynamics(
                estimate_context,
                zero_dynamics,
            )
        )
        captured_vector_kwargs: dict[str, Any] = {}

        def fake_build_vectorized_observation(
            **kwargs: Any,
        ) -> VectorizedObservation:
            captured_vector_kwargs.update(kwargs)
            return VectorizedObservation(timestamp_us=kwargs["timestamp_us"])

        monkeypatch.setattr(
            "alpasim_runtime.events.policy.build_vectorized_observation",
            fake_build_vectorized_observation,
        )

        policy_event.send_vectorized_observation = True
        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        mock_driver.drive.assert_awaited_once()
        mock_driver.submit_vectorized_observation.assert_awaited_once()
        call_kwargs = mock_driver.drive.call_args.kwargs
        assert call_kwargs["time_now_us"] == 200_000
        assert call_kwargs["time_query_us"] == 300_000
        assert captured_vector_kwargs["timestamp_us"] == 200_000
        np.testing.assert_allclose(
            captured_vector_kwargs["pose_local_to_physical_ego"].vec3[0],
            0.2,
        )
        np.testing.assert_array_equal(
            rollout_state.step_context.driver_trajectory.timestamps_us,
            drive_traj.timestamps_us,
        )

    @pytest.mark.asyncio
    async def test_run_skips_vector_observation_when_disabled(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Disabled vector observations should not pay vector build cost."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = (drive_traj, False)

        def fail_build_vectorized_observation(**kwargs: Any) -> VectorizedObservation:
            raise AssertionError("vector observation should not be built")

        monkeypatch.setattr(
            "alpasim_runtime.events.policy.build_vectorized_observation",
            fail_build_vectorized_observation,
        )

        await policy_event.run(rollout_state, EventQueue())

        mock_driver.drive.assert_awaited_once()
        mock_driver.submit_vectorized_observation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_first_run_submits_full_initial_egomotion_context(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        timestamps_us = np.array([0, 100_000, 200_000], dtype=np.uint64)
        ego_context = simple_trajectory.interpolate(timestamps_us)
        zero_dynamics = np.zeros((len(ego_context), 12), dtype=np.float64)
        rollout_state.ego_trajectory_estimate = (
            DynamicTrajectory.from_trajectory_and_dynamics(
                ego_context,
                zero_dynamics,
            )
        )
        rollout_state.last_egopose_update_us = None
        mock_driver.drive.return_value = (
            simple_trajectory.clip(200_000, 300_001),
            False,
        )

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=[],
            route_generator=None,
            send_recording_ground_truth=False,
            send_vectorized_observation=False,
        )
        await event.run(rollout_state, EventQueue())

        mock_driver.submit_trajectory.assert_awaited_once()
        mock_driver.submit_route.assert_not_awaited()
        trajectory, dynamic_states = mock_driver.submit_trajectory.call_args.args
        assert trajectory.timestamps_us.tolist() == [0, 100_000, 200_000]
        assert len(dynamic_states) == 3
        assert rollout_state.last_egopose_update_us == 200_000

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
        mock_driver.drive.return_value = (drive_traj, False)

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
        mock_driver.drive.return_value = (drive_traj, False)

        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        assert rollout_state.data_sensorsim_to_driver is None

    @pytest.mark.asyncio
    async def test_run_raises_end_simulation_on_terminate(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """When driver returns terminate_session=True, PolicyEvent raises EndSimulationException."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = (drive_traj, True)

        with pytest.raises(EndSimulationException):
            await policy_event.run(rollout_state, EventQueue())

    @pytest.mark.asyncio
    async def test_handle_does_not_reschedule_on_terminate(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """When driver requests termination, the event is not rescheduled."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = (drive_traj, True)

        queue = EventQueue()
        with pytest.raises(EndSimulationException):
            await policy_event.handle(rollout_state, queue)

        assert len(queue) == 0  # Not rescheduled

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
        mock_driver.drive.return_value = (drive_traj, False)

        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        assert rollout_state.step_context is not None
        assert rollout_state.step_context.step_start_us == 200_000
        assert rollout_state.step_context.target_time_us == 300_000
        assert rollout_state.step_context.driver_trajectory is not None

    @pytest.mark.asyncio
    async def test_force_gt_calls_drive_by_default_and_stores_driver_trajectory(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
    ) -> None:
        """Force-GT should query policy by default and leave controller override downstream."""
        route = Polyline(
            points=np.array(
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        )
        rollout_state.unbound.force_gt_period = range(0, 300_001)
        rollout_state.data_sensorsim_to_driver = b"renderer-data"
        expected_driver_trajectory = Trajectory.from_poses(
            np.array([200_000, 300_000], dtype=np.uint64),
            [
                Pose(
                    np.array([99.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([100.0, 0.0, 0.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
            ],
        )
        mock_driver.drive.return_value = (expected_driver_trajectory, False)

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=cast(
                Any,
                SimpleNamespace(generate_route=lambda *_args, **_kwargs: route),
            ),
            send_recording_ground_truth=True,
            send_vectorized_observation=False,
        )

        await event.run(rollout_state, EventQueue())

        mock_driver.submit_trajectory.assert_awaited_once()
        mock_driver.submit_route.assert_awaited_once()
        mock_driver.submit_recording_ground_truth.assert_awaited_once()
        mock_driver.drive.assert_awaited_once()
        assert mock_driver.drive.call_args.kwargs["renderer_data"] == b"renderer-data"
        assert rollout_state.data_sensorsim_to_driver is None

        assert rollout_state.step_context is not None
        driver_trajectory = rollout_state.step_context.driver_trajectory
        assert driver_trajectory is not None
        np.testing.assert_array_equal(
            driver_trajectory.timestamps_us, expected_driver_trajectory.timestamps_us
        )
        np.testing.assert_allclose(
            driver_trajectory.positions,
            np.array([[99.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32),
        )

    @pytest.mark.asyncio
    async def test_force_gt_skip_driver_flag_skips_drive(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
    ) -> None:
        """Feature flag should preserve skip-driver behavior during force-GT."""
        rollout_state.unbound.force_gt_period = range(0, 300_001)
        rollout_state.unbound.skip_driver_during_force_gt = True
        rollout_state.data_sensorsim_to_driver = b"renderer-data"
        force_gt_trajectory = Trajectory.from_poses(
            np.array([0, 100_000, 200_000, 300_000], dtype=np.uint64),
            [
                Pose(
                    np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([1.0, 0.0, 2.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([2.0, 0.0, 3.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
                Pose(
                    np.array([3.0, 0.0, 4.0], dtype=np.float32),
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                ),
            ],
        )
        rollout_state.force_gt_ego_trajectory = force_gt_trajectory

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
            send_vectorized_observation=False,
        )

        await event.run(rollout_state, EventQueue())

        mock_driver.drive.assert_not_awaited()
        assert rollout_state.data_sensorsim_to_driver is None

        assert rollout_state.step_context is not None
        driver_trajectory = rollout_state.step_context.driver_trajectory
        assert driver_trajectory is not None
        expected_trajectory = controller_reference_trajectory(
            force_gt_trajectory, 200_000, rollout_state.unbound.control_timestep_us
        )
        np.testing.assert_array_equal(
            driver_trajectory.timestamps_us, expected_trajectory.timestamps_us
        )
        np.testing.assert_allclose(
            driver_trajectory.positions,
            expected_trajectory.positions,
        )

    @pytest.mark.asyncio
    async def test_run_treats_empty_force_gt_period_as_noop(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """Zero force-GT duration should never mark a policy step as force-GT."""
        rollout_state.unbound.force_gt_period = range(100_000, 100_000)
        mock_driver.drive.return_value = (
            simple_trajectory.clip(100_000, 200_001),
            False,
        )
        event = PolicyEvent(
            timestamp_us=100_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
            send_vectorized_observation=False,
        )

        await event.run(rollout_state, EventQueue())

        assert rollout_state.step_context is not None
        assert rollout_state.step_context.force_gt is False

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

        # Simulate a pending camera submission from a camera render event.
        rollout_state.step_context.track_task(blocked_send())

        mock_driver.drive.return_value = (
            simple_trajectory.clip(200_000, 300_001),
            False,
        )

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
            send_vectorized_observation=False,
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
            send_vectorized_observation=False,
        )

        with pytest.raises(ValueError, match="Timestamp mismatch"):
            await event.run(rollout_state, EventQueue())

        mock_driver.submit_route.assert_not_awaited()
        mock_driver.drive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_raises_when_the_estimate_has_no_pose_for_this_step(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
    ):
        """The route is built from the estimate, so a stale estimate is fatal."""
        rollout_state.ego_trajectory_estimate = (
            DynamicTrajectory.from_trajectory_and_dynamics(
                rollout_state.ego_trajectory.trajectory().clip(0, 100_001),
                np.zeros((2, 12), dtype=np.float64),
            )
        )

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=cast(
                Any,
                SimpleNamespace(
                    generate_route=lambda *_args, **_kwargs: Polyline(
                        points=np.zeros((3, 3), dtype=np.float32)
                    )
                ),
            ),
            send_recording_ground_truth=False,
            send_vectorized_observation=False,
        )

        with pytest.raises(
            ValueError,
            match=r"Timestamp mismatch: ego_trajectory_estimate at \d+ != 200000",
        ):
            await event.run(rollout_state, EventQueue())

        mock_driver.submit_route.assert_not_awaited()
        mock_driver.drive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_route_is_placed_on_the_map_but_sent_in_the_estimated_frame(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """Egomotion noise splits the two frames the route submission needs.

        Generating the route places the ego on the map, which only the true pose
        does correctly, while the driver reads the waypoints in the rig frame it
        reconstructs from the estimate it was sent.
        """
        mock_driver.drive.return_value = (
            simple_trajectory.clip(200_000, 300_001),
            False,
        )
        # The fixture drives along x with no rotation, so the true pose for this
        # step is at x = 0.2 m and the estimate is 1 m ahead and 3 m to the left.
        true_position = np.array([0.2, 0.0, 0.0], dtype=np.float32)
        estimated_position = np.array([1.2, 3.0, 0.0], dtype=np.float32)
        estimated_pose = Pose(
            estimated_position,
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        rollout_state.ego_trajectory_estimate = (
            DynamicTrajectory.from_trajectory_and_dynamics(
                Trajectory.from_poses(
                    np.array([0, 200_000], dtype=np.uint64),
                    [rollout_state.ego_trajectory.first_pose, estimated_pose],
                ),
                np.zeros((2, 12), dtype=np.float64),
            )
        )
        route_in_true_rig = np.array(
            [[10.0, 0.0, 0.0], [14.0, 0.0, 0.0], [18.0, 0.0, 0.0]], dtype=np.float32
        )
        poses_used = []

        def _generate_route(_timestamp_us: int, pose_local_to_rig: Pose) -> Polyline:
            poses_used.append(pose_local_to_rig)
            return Polyline(points=route_in_true_rig.copy())

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=cast(Any, SimpleNamespace(generate_route=_generate_route)),
            send_recording_ground_truth=False,
            send_vectorized_observation=False,
        )

        await event.run(rollout_state, EventQueue())

        np.testing.assert_allclose(poses_used[0].vec3, true_position)
        submitted = mock_driver.submit_route.call_args.args[1]
        np.testing.assert_allclose(
            submitted.waypoints[: len(route_in_true_rig)],
            route_in_true_rig + (true_position - estimated_position),
            atol=1e-5,
        )

    @pytest.mark.asyncio
    async def test_run_raises_when_ground_truth_timestamp_mismatches_ego_state(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        mock_driver.drive.return_value = (
            simple_trajectory.clip(100_000, 200_001),
            False,
        )

        event = PolicyEvent(
            timestamp_us=100_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=True,
            send_vectorized_observation=False,
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

    def test_allows_older_camera_frame(self, rollout_state: RolloutState):
        rollout_state.last_camera_frame_us = {
            "cam_front": 50_000,
            "cam_rear": 200_000,
        }

        assert_sensors_up_to_date(rollout_state, 200_000, ["cam_front", "cam_rear"])

    def test_raises_when_required_camera_has_no_frame(
        self, rollout_state: RolloutState
    ):
        rollout_state.last_camera_frame_us = {}  # No frames yet

        with pytest.raises(ValueError, match="Cameras not up to date"):
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


@pytest.mark.asyncio
async def test_navigation_instruction_is_looked_up_from_the_true_pose(
    rollout_state: RolloutState,
    service_bundle: ServiceBundle,
    mock_driver: AsyncMock,
    simple_trajectory: Trajectory,
):
    """The instruction reflects where the ego is along the recorded route."""
    from alpasim_runtime.nav_route import (
        NavStatus,
        RouteInstruction,
        RouteNavigator,
        SessionRouteIndex,
    )

    # The recording covers 0..1 m of a session; the turn sits 0.6 m ahead of the
    # ego, which is at x=0.2 m at step 200_000.
    route_index = SessionRouteIndex(
        timestamps=np.array([0, 500_000, 1_000_000], dtype=np.int64),
        cumulative_distances=np.array([100.0, 100.5, 101.0]),
        instructions=[RouteInstruction(100.8, "TURN_LEFT", "Turn left")],
    )
    mock_driver.drive.return_value = (simple_trajectory.clip(200_000, 300_001), False)
    event = PolicyEvent(
        timestamp_us=200_000,
        policy_timestep_us=100_000,
        services=service_bundle,
        camera_ids=[],
        route_generator=None,
        send_recording_ground_truth=False,
        send_vectorized_observation=False,
        navigation=RouteNavigator(route_index, rollout_state.unbound.gt_ego_trajectory),
    )

    await event.run(rollout_state, EventQueue())

    mock_driver.submit_navigation.assert_awaited_once()
    timestamp_us, instruction = mock_driver.submit_navigation.call_args.args
    assert timestamp_us == 200_000
    assert instruction.status is NavStatus.UPCOMING
    assert instruction.instruction_text == "Turn left in 1m"
    assert instruction.remaining_distance_m == pytest.approx(0.6, abs=1e-5)


@pytest.mark.asyncio
async def test_geometry_navigation_sends_text_classified_from_the_route(
    rollout_state: RolloutState,
    service_bundle: ServiceBundle,
    mock_driver: AsyncMock,
    simple_trajectory: Trajectory,
):
    """With source ``geometry`` the runtime, not the driver, reads the route."""
    from alpasim_runtime.nav_route import GeometryNavigator, NavStatus
    from alpasim_utils.geometry import Polyline

    ahead = [(x, 0.0, 0.0) for x in np.arange(0.0, 20.0, 4.0)]
    left = [(16.0, y, 0.0) for y in np.arange(4.0, 24.0, 4.0)]
    route_generator = MagicMock()
    route_generator.generate_route.return_value = Polyline(
        np.array(ahead + left, dtype=np.float64)
    )
    mock_driver.drive.return_value = (simple_trajectory.clip(200_000, 300_001), False)
    event = PolicyEvent(
        timestamp_us=200_000,
        policy_timestep_us=100_000,
        services=service_bundle,
        camera_ids=[],
        route_generator=route_generator,
        send_recording_ground_truth=False,
        send_vectorized_observation=False,
        navigation=GeometryNavigator(),
    )

    await event.run(rollout_state, EventQueue())

    mock_driver.submit_route.assert_awaited_once()
    mock_driver.submit_navigation.assert_awaited_once()
    timestamp_us, instruction = mock_driver.submit_navigation.call_args.args
    assert timestamp_us == 200_000
    assert instruction.instruction_text == "Turn left in 16m"
    assert instruction.status is NavStatus.UPCOMING


@pytest.mark.asyncio
async def test_geometry_navigation_without_usable_route_sends_the_empty_instruction(
    rollout_state: RolloutState,
    service_bundle: ServiceBundle,
    mock_driver: AsyncMock,
    simple_trajectory: Trajectory,
):
    from alpasim_runtime.nav_route import GeometryNavigator, NavStatus
    from alpasim_utils.geometry import Polyline

    route_generator = MagicMock()
    route_generator.generate_route.return_value = Polyline(np.zeros((0, 3)))
    mock_driver.drive.return_value = (simple_trajectory.clip(200_000, 300_001), False)
    event = PolicyEvent(
        timestamp_us=200_000,
        policy_timestep_us=100_000,
        services=service_bundle,
        camera_ids=[],
        route_generator=route_generator,
        send_recording_ground_truth=False,
        send_vectorized_observation=False,
        navigation=GeometryNavigator(),
    )

    await event.run(rollout_state, EventQueue())

    # The driver keeps the last instruction, so "nothing to say" is sent explicitly.
    mock_driver.submit_navigation.assert_awaited_once()
    timestamp_us, instruction = mock_driver.submit_navigation.call_args.args
    assert timestamp_us == 200_000
    assert instruction.instruction_text == ""
    assert instruction.status is NavStatus.UNAVAILABLE
