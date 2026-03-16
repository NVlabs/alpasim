# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Parity tests for event-based loop regression fixes.

Each test class targets a specific regression fix and verifies that the
event-based loop reproduces the behavior of the original sequential loop
for that fix area.

Phase 1A: Dynamic state preservation (via StepContext/StepEvent)
Phase 1B: Controller + ego physics pipeline
Phase 2E: Grouped render data flow
Phase 2F: Noisy/true transform
Phase 3G: End-of-rollout flush ordering
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from alpasim_grpc.v0.traffic_pb2 import ObjectTrajectoryUpdate, TrafficReturn
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.events.base import (
    EndSimulationException,
    Event,
    EventQueue,
    SimulationEndEvent,
)
from alpasim_runtime.events.camera import GroupedRenderEvent
from alpasim_runtime.events.controller import ControllerEvent
from alpasim_runtime.events.physics import PhysicsEvent, PhysicsTarget
from alpasim_runtime.events.policy import (
    PolicyEvent,
    transform_trajectory_from_noisy_to_true_local_frame,
)
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_runtime.events.step import StepEvent
from alpasim_runtime.events.traffic import TrafficEvent
from alpasim_utils import geometry
from alpasim_utils.geometry import DynamicTrajectory, Pose, Trajectory
from alpasim_utils.scenario import AABB, TrafficObject

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pose(x: float, y: float, z: float) -> Pose:
    """Create a pose at (x, y, z) with identity rotation."""
    return Pose(
        np.array([x, y, z], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def _make_trajectory_from_range(
    start_us: int, stop_us: int, step_us: int
) -> Trajectory:
    """Create a straight-line trajectory with evenly spaced timestamps."""
    timestamps = list(range(start_us, stop_us + 1, step_us))
    poses = [
        Pose(
            np.array([float(t) / 1e6, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        for t in timestamps
    ]
    return Trajectory.from_poses(
        timestamps=np.array(timestamps, dtype=np.uint64),
        poses=poses,
    )


# ---------------------------------------------------------------------------
# Phase 1A: Dynamic state preservation
# ---------------------------------------------------------------------------


class TestDynamicStatePreservation:
    """Phase 1A: dynamics are preserved through DynamicTrajectory operations.

    In the new architecture, dynamics live inside DynamicTrajectory instances.
    StepEvent extends state trajectories via concat. These tests verify
    that dynamics data survives concatenation and can be queried.
    """

    def test_ego_trajectory_has_dynamics(
        self,
        rollout_state: RolloutState,
    ):
        """ego_trajectory_estimate carries a dynamics array."""
        dynamics = rollout_state.ego_trajectory_estimate.dynamics
        assert dynamics.shape == (2, 12)

    def test_dynamics_preserved_through_concat(
        self,
        rollout_state: RolloutState,
    ):
        """Concatenating new entries preserves existing dynamics."""
        initial_len = len(rollout_state.ego_trajectory_estimate.timestamps_us)

        # Build a new DynamicTrajectory with non-zero dynamics
        new_ts = np.array([300_000], dtype=np.uint64)
        new_pose = _make_pose(0.3, 0.0, 0.0)
        new_traj = Trajectory.from_poses(new_ts, [new_pose])
        new_dynamics = np.array([[1.0] * 12], dtype=np.float64)
        new_dt = DynamicTrajectory.from_trajectory_and_dynamics(new_traj, new_dynamics)

        result = rollout_state.ego_trajectory_estimate.concat(new_dt)
        assert len(result.timestamps_us) == initial_len + 1
        # Original entries have zero dynamics
        np.testing.assert_array_equal(result.dynamics[0], np.zeros(12))
        # New entry has ones
        np.testing.assert_array_equal(result.dynamics[-1], np.ones(12))

    def test_step_context_carries_dynamic_trajectory(
        self,
        rollout_state: RolloutState,
    ):
        """StepContext ego_true and ego_estimated are DynamicTrajectory."""
        ts = np.array([300_000], dtype=np.uint64)
        pose = _make_pose(0.3, 0.0, 0.0)
        traj = Trajectory.from_poses(ts, [pose])
        dyn = np.array([[2.0] * 12], dtype=np.float64)
        dt = DynamicTrajectory.from_trajectory_and_dynamics(traj, dyn)

        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=300_000,
            force_gt=False,
            ego_true=dt,
            ego_estimated=dt,
        )

        assert ctx.ego_true is not None
        np.testing.assert_array_equal(ctx.ego_true.dynamics[0], [2.0] * 12)

    def test_step_context_has_traffic_trajectories(self):
        """StepContext initializes with empty traffic_trajectories dict."""
        ctx = StepContext(
            step_start_us=100_000,
            target_time_us=200_000,
            force_gt=False,
        )
        assert ctx.traffic_trajectories == {}
        assert isinstance(ctx.traffic_trajectories, dict)

    def test_initial_ego_trajectory_has_zero_dynamics(
        self,
        rollout_state: RolloutState,
    ):
        """The fixture-initialized ego trajectories start with zero dynamics."""
        for row in rollout_state.ego_trajectory.dynamics:
            np.testing.assert_array_equal(row, np.zeros(12))
        for row in rollout_state.ego_trajectory_estimate.dynamics:
            np.testing.assert_array_equal(row, np.zeros(12))


# ---------------------------------------------------------------------------
# Phase 1B: Controller + ego physics pipeline
# ---------------------------------------------------------------------------


class TestControllerAndEgoPhysicsPipeline:
    """Phase 1B: Controller and ego physics are now separate pipeline events.

    In the new architecture, PolicyEvent only queries the driver.
    ControllerEvent and PhysicsEvent handle controller and physics separately.
    These tests verify the reusable helper functions.
    """

    @pytest.mark.asyncio
    async def test_ego_physics_skipped_when_mode_none(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_physics: AsyncMock,
    ):
        """When physics mode is NONE, ego physics returns the trajectory unchanged."""
        rollout_state.unbound.physics_update_mode = PhysicsUpdateMode.NONE

        input_pose = _make_pose(1.0, 2.0, 3.0)
        input_traj = Trajectory.from_poses(
            np.array([200_000], dtype=np.uint64), [input_pose]
        )
        ego_true = DynamicTrajectory.from_trajectory_and_dynamics(
            input_traj, np.zeros((1, 12), dtype=np.float64)
        )

        ctx = StepContext(
            step_start_us=100_000,
            target_time_us=200_000,
            force_gt=False,
            ego_true=ego_true,
        )
        rollout_state.step_context = ctx

        event = PhysicsEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            services=service_bundle,
            target=PhysicsTarget.EGO,
        )
        await event.run(rollout_state, EventQueue())

        result = ctx.corrected_ego_trajectory
        assert result is not None
        np.testing.assert_array_almost_equal(result.get_pose(0).vec3, input_pose.vec3)
        mock_physics.ground_intersection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ego_physics_called_when_mode_ego_only(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_physics: AsyncMock,
    ):
        """When physics mode is EGO_ONLY, ground_intersection is called."""
        rollout_state.unbound.physics_update_mode = PhysicsUpdateMode.EGO_ONLY

        constrained_pose = _make_pose(1.0, 2.0, 0.0)
        constrained_traj = Trajectory.from_poses(
            np.array([200_000], dtype=np.uint64), [constrained_pose]
        )
        mock_physics.ground_intersection.return_value = (constrained_traj, {})

        input_pose = _make_pose(1.0, 2.0, 3.0)
        input_traj = Trajectory.from_poses(
            np.array([200_000], dtype=np.uint64), [input_pose]
        )
        ego_true = DynamicTrajectory.from_trajectory_and_dynamics(
            input_traj, np.zeros((1, 12), dtype=np.float64)
        )

        ctx = StepContext(
            step_start_us=100_000,
            target_time_us=200_000,
            force_gt=False,
            ego_true=ego_true,
        )
        rollout_state.step_context = ctx

        event = PhysicsEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            services=service_bundle,
            target=PhysicsTarget.EGO,
        )
        await event.run(rollout_state, EventQueue())

        mock_physics.ground_intersection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ego_physics_called_when_mode_all_actors(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_physics: AsyncMock,
    ):
        """When physics mode is ALL_ACTORS, ground_intersection is called for ego."""
        rollout_state.unbound.physics_update_mode = PhysicsUpdateMode.ALL_ACTORS

        constrained_pose = _make_pose(1.0, 2.0, 0.0)
        constrained_traj = Trajectory.from_poses(
            np.array([200_000], dtype=np.uint64), [constrained_pose]
        )
        mock_physics.ground_intersection.return_value = (constrained_traj, {})

        input_pose = _make_pose(1.0, 2.0, 3.0)
        input_traj = Trajectory.from_poses(
            np.array([200_000], dtype=np.uint64), [input_pose]
        )
        ego_true = DynamicTrajectory.from_trajectory_and_dynamics(
            input_traj, np.zeros((1, 12), dtype=np.float64)
        )

        ctx = StepContext(
            step_start_us=100_000,
            target_time_us=200_000,
            force_gt=False,
            ego_true=ego_true,
        )
        rollout_state.step_context = ctx

        event = PhysicsEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            services=service_bundle,
            target=PhysicsTarget.EGO,
        )
        await event.run(rollout_state, EventQueue())

        mock_physics.ground_intersection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_controller_event_passes_delay_buffer_trajectory(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_controller: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """ControllerEvent feeds the planner delay buffer output to the controller."""
        ref_trajectory = simple_trajectory.clip(200_000, 400_001)

        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=300_000,
            force_gt=False,
            driver_trajectory=ref_trajectory,
        )
        rollout_state.step_context = ctx

        event = ControllerEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            services=service_bundle,
        )
        await event.run(rollout_state, EventQueue())

        mock_controller.run_controller_and_vehicle.assert_awaited_once()
        call_kwargs = mock_controller.run_controller_and_vehicle.call_args.kwargs
        assert call_kwargs["rig_reference_trajectory_in_rig"] is not None

    @pytest.mark.asyncio
    async def test_controller_event_uses_full_gt_fallback_during_force_gt(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_controller: AsyncMock,
    ):
        """Force-GT should pass the full GT trajectory as the fallback input."""
        ctx = StepContext(
            step_start_us=100_000,
            target_time_us=200_000,
            force_gt=True,
            driver_trajectory=MagicMock(timestamps_us=[]),
        )
        rollout_state.step_context = ctx

        event = ControllerEvent(
            timestamp_us=100_000,
            control_timestep_us=100_000,
            services=service_bundle,
        )
        await event.run(rollout_state, EventQueue())

        call_kwargs = mock_controller.run_controller_and_vehicle.call_args.kwargs
        assert (
            call_kwargs["fallback_trajectory_local_to_rig"]
            is rollout_state.unbound.gt_ego_trajectory
        )

    @pytest.mark.asyncio
    async def test_policy_event_creates_step_context_with_driver_trajectory(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """PolicyEvent creates a StepContext and writes driver_trajectory to it."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
        )

        queue = EventQueue()
        await event.run(rollout_state, queue)

        assert rollout_state.step_context is not None
        assert rollout_state.step_context.driver_trajectory is not None
        assert rollout_state.step_context.step_start_us == 200_000
        assert rollout_state.step_context.target_time_us == 300_000


# ---------------------------------------------------------------------------
# Phase 2E: Grouped render data flow
# ---------------------------------------------------------------------------


class TestGroupedRenderDataFlow:
    """Phase 2E: When group_render_requests=True, policy receives renderer data."""

    @pytest.fixture
    def grouped_render_event(
        self,
        runtime_camera: MagicMock,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ) -> GroupedRenderEvent:
        return GroupedRenderEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            cameras=[runtime_camera],
            sensorsim=mock_sensorsim,
            driver=mock_driver,
            scene_start_us=0,
            use_aggregated_render=True,
        )

    @pytest.mark.asyncio
    async def test_grouped_render_populates_driver_data(
        self,
        grouped_render_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ):
        """GroupedRenderEvent stores driver_data on rollout_state."""
        driver_data_blob = b"aggregated_render_payload"
        mock_sensorsim.aggregated_render.return_value = ([], driver_data_blob)

        queue = EventQueue()
        await grouped_render_event.run(rollout_state, queue)

        assert rollout_state.data_sensorsim_to_driver == driver_data_blob

    @pytest.mark.asyncio
    async def test_grouped_render_no_triggers_leaves_data_none(
        self,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ):
        """When no camera triggers fall in the window, driver_data is not set."""
        camera = MagicMock()
        camera.clock = MagicMock()
        camera.clock.triggers_completed_in_range.return_value = []

        event = GroupedRenderEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            cameras=[camera],
            sensorsim=mock_sensorsim,
            driver=mock_driver,
            scene_start_us=0,
            use_aggregated_render=True,
        )

        rollout_state.data_sensorsim_to_driver = None

        queue = EventQueue()
        await event.run(rollout_state, queue)

        mock_sensorsim.aggregated_render.assert_not_awaited()
        assert rollout_state.data_sensorsim_to_driver is None

    @pytest.mark.asyncio
    async def test_policy_clears_renderer_data_after_consumption(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """PolicyEvent sets data_sensorsim_to_driver = None after consuming it."""
        rollout_state.data_sensorsim_to_driver = b"some_renderer_payload"

        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
        )

        queue = EventQueue()
        await event.run(rollout_state, queue)

        assert rollout_state.data_sensorsim_to_driver is None

    @pytest.mark.asyncio
    async def test_policy_passes_renderer_data_to_driver(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """PolicyEvent forwards data_sensorsim_to_driver to driver.drive."""
        payload = b"renderer_payload_for_driver"
        rollout_state.data_sensorsim_to_driver = payload

        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        event = PolicyEvent(
            timestamp_us=200_000,
            policy_timestep_us=100_000,
            services=service_bundle,
            camera_ids=["cam_front"],
            route_generator=None,
            send_recording_ground_truth=False,
        )

        queue = EventQueue()
        await event.run(rollout_state, queue)

        call_kwargs = mock_driver.drive.call_args.kwargs
        assert call_kwargs["renderer_data"] == payload

    @pytest.mark.asyncio
    async def test_grouped_render_forwards_images_to_driver(
        self,
        grouped_render_event: GroupedRenderEvent,
        rollout_state: RolloutState,
        mock_sensorsim: AsyncMock,
        mock_driver: AsyncMock,
    ):
        """GroupedRenderEvent forwards all images from aggregated_render to driver."""
        fake_images = [MagicMock(), MagicMock(), MagicMock()]
        mock_sensorsim.aggregated_render.return_value = (fake_images, b"data")

        queue = EventQueue()
        await grouped_render_event.run(rollout_state, queue)

        # Drain tracked tasks to complete submissions
        await rollout_state.step_context.drain_outstanding_tasks()

        assert mock_driver.submit_image.await_count == 3
        for i, call in enumerate(mock_driver.submit_image.call_args_list):
            assert call[0][0] is fake_images[i]


# ---------------------------------------------------------------------------
# Phase 2F: Noisy/true transform
# ---------------------------------------------------------------------------


class TestNoisyTrueTransform:
    """Phase 2F: transform_trajectory corrects for egomotion noise."""

    _zero_dyn = np.zeros(12, dtype=np.float64)

    def test_identity_when_estimate_equals_true(
        self,
        rollout_state: RolloutState,
    ):
        """Transform is identity when estimate and true trajectories match."""
        input_traj = Trajectory.from_poses(
            timestamps=np.array([200_000, 300_000], dtype=np.uint64),
            poses=[_make_pose(1.0, 0.0, 0.0), _make_pose(2.0, 0.0, 0.0)],
        )

        result = transform_trajectory_from_noisy_to_true_local_frame(
            rollout_state, input_traj
        )

        for i in range(len(result.timestamps_us)):
            original_pose = input_traj.get_pose(i)
            transformed_pose = result.get_pose(i)
            np.testing.assert_array_almost_equal(
                transformed_pose.vec3, original_pose.vec3, decimal=4
            )

    def test_non_identity_when_estimate_and_true_differ(
        self,
        rollout_state: RolloutState,
    ):
        """Transform is non-identity when estimate and true last poses diverge."""
        shifted_pose = _make_pose(0.5, 0.0, 0.0)
        rollout_state.ego_trajectory_estimate.update_absolute(
            300_000, shifted_pose, self._zero_dyn
        )

        true_pose = _make_pose(0.0, 0.0, 0.0)
        rollout_state.ego_trajectory.update_absolute(300_000, true_pose, self._zero_dyn)

        input_traj = Trajectory.from_poses(
            timestamps=np.array([400_000, 500_000], dtype=np.uint64),
            poses=[_make_pose(1.0, 0.0, 0.0), _make_pose(2.0, 0.0, 0.0)],
        )

        result = transform_trajectory_from_noisy_to_true_local_frame(
            rollout_state, input_traj
        )

        result_pose_0 = result.get_pose(0)
        input_pose_0 = input_traj.get_pose(0)

        diff = result_pose_0.vec3[0] - input_pose_0.vec3[0]
        assert abs(diff) > 0.1, f"Expected non-identity transform but got diff={diff}"

    def test_transform_preserves_timestamp_count(
        self,
        rollout_state: RolloutState,
    ):
        """The transformed trajectory has the same number of timestamps."""
        input_traj = Trajectory.from_poses(
            timestamps=np.array([200_000, 250_000, 300_000], dtype=np.uint64),
            poses=[
                _make_pose(1.0, 0.0, 0.0),
                _make_pose(1.5, 0.0, 0.0),
                _make_pose(2.0, 0.0, 0.0),
            ],
        )

        result = transform_trajectory_from_noisy_to_true_local_frame(
            rollout_state, input_traj
        )
        assert len(result.timestamps_us) == 3

    def test_transform_preserves_timestamps(
        self,
        rollout_state: RolloutState,
    ):
        """Timestamps are preserved across the transform."""
        timestamps = np.array([200_000, 250_000, 300_000], dtype=np.uint64)
        input_traj = Trajectory.from_poses(
            timestamps=timestamps,
            poses=[
                _make_pose(1.0, 0.0, 0.0),
                _make_pose(1.5, 0.0, 0.0),
                _make_pose(2.0, 0.0, 0.0),
            ],
        )

        result = transform_trajectory_from_noisy_to_true_local_frame(
            rollout_state, input_traj
        )
        np.testing.assert_array_equal(result.timestamps_us, timestamps)

    def test_roundtrip_noise_correction(
        self,
        rollout_state: RolloutState,
    ):
        """The correction undoes the estimate frame and applies the true frame."""
        rollout_state.ego_trajectory_estimate.update_absolute(
            300_000, _make_pose(1.0, 0.0, 0.0), self._zero_dyn
        )
        rollout_state.ego_trajectory.update_absolute(
            300_000, _make_pose(0.0, 0.0, 0.0), self._zero_dyn
        )

        input_traj = Trajectory.from_poses(
            timestamps=np.array([400_000], dtype=np.uint64),
            poses=[_make_pose(3.0, 0.0, 0.0)],
        )

        result = transform_trajectory_from_noisy_to_true_local_frame(
            rollout_state, input_traj
        )

        # Expected: (3 - 1) = 2.0 in the true frame
        result_x = result.get_pose(0).vec3[0]
        np.testing.assert_almost_equal(result_x, 2.0, decimal=3)


# ---------------------------------------------------------------------------
# Phase 3G: End-of-rollout flush ordering
# ---------------------------------------------------------------------------


class TestEndOfRolloutFlush:
    """Phase 3G: SimulationEndEvent fires after policy but before controller."""

    def test_simulation_end_priority_is_30(self):
        """SimulationEndEvent must have priority 30."""
        event = SimulationEndEvent(timestamp_us=1_000_000)
        assert event.priority == 30

    def test_end_fires_after_policy_but_before_controller(self):
        """At the same timestamp, camera (10) → policy (20) → end (30) → controller (40)."""
        q = EventQueue()
        end = SimulationEndEvent(timestamp_us=1000)
        camera = _DummyEvent(timestamp_us=1000, priority=10)
        policy = _DummyEvent(timestamp_us=1000, priority=20)
        controller = _DummyEvent(timestamp_us=1000, priority=40)

        q.submit(end)
        q.submit(camera)
        q.submit(policy)
        q.submit(controller)

        assert q.pop().priority == 10
        assert q.pop().priority == 20
        popped_end = q.pop()
        assert isinstance(popped_end, SimulationEndEvent)
        assert popped_end.priority == 30
        assert q.pop().priority == 40

    def test_end_preempts_controller_at_same_timestamp(self):
        """SimulationEndEvent (30) fires before ControllerEvent (40)."""
        q = EventQueue()
        end = SimulationEndEvent(timestamp_us=1000)
        controller = _DummyEvent(timestamp_us=1000, priority=40)

        q.submit(controller)
        q.submit(end)

        first = q.pop()
        assert isinstance(first, SimulationEndEvent)
        second = q.pop()
        assert second.priority == 40

    @pytest.mark.asyncio
    async def test_simulation_end_raises_exception(
        self,
        rollout_state: RolloutState,
    ):
        """SimulationEndEvent raises EndSimulationException on handle."""
        event = SimulationEndEvent(timestamp_us=1_000_000)
        queue = EventQueue()

        with pytest.raises(EndSimulationException):
            await event.handle(rollout_state, queue)

    def test_earlier_timestamp_always_wins_over_priority(self):
        """Events at earlier timestamps always precede later ones."""
        q = EventQueue()
        late_high_prio = SimulationEndEvent(timestamp_us=2000)
        early_low_prio = _DummyEvent(timestamp_us=1000, priority=99)

        q.submit(late_high_prio)
        q.submit(early_low_prio)

        first = q.pop()
        assert first.timestamp_us == 1000
        second = q.pop()
        assert second.timestamp_us == 2000

    def test_full_end_of_rollout_event_ordering(self):
        """Verify the complete end-of-rollout ordering at final timestamp.

        SimulationEndEvent (30) fires after camera (10) and policy (20),
        allowing the policy event to submit final observations and call
        drive() at the boundary, but before controller (40) and the rest
        of the pipeline.
        """
        final_ts = 1_000_000
        q = EventQueue()

        prior_camera = _DummyEvent(timestamp_us=final_ts - 100_000, priority=10)
        q.submit(prior_camera)

        camera = _DummyEvent(timestamp_us=final_ts, priority=10)
        policy = _DummyEvent(timestamp_us=final_ts, priority=20)
        end = SimulationEndEvent(timestamp_us=final_ts)
        controller = _DummyEvent(timestamp_us=final_ts, priority=40)
        ego_physics = _DummyEvent(timestamp_us=final_ts, priority=50)
        traffic = _DummyEvent(timestamp_us=final_ts, priority=60)
        commit = _DummyEvent(timestamp_us=final_ts, priority=80)

        for e in [commit, traffic, ego_physics, controller, end, policy, camera]:
            q.submit(e)

        assert q.pop().timestamp_us == final_ts - 100_000

        priorities_seen = []
        for _ in range(7):
            event = q.pop()
            priorities_seen.append(event.priority)

        assert priorities_seen == [10, 20, 30, 40, 50, 60, 80]


# ---------------------------------------------------------------------------
# Cross-cutting: PolicyEvent pipeline integration
# ---------------------------------------------------------------------------


class TestPolicyEventPipelineIntegration:
    """Cross-cutting integration tests for the PolicyEvent stage."""

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

    @pytest.mark.asyncio
    async def test_policy_event_creates_step_context(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """PolicyEvent creates StepContext with driver_trajectory."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        mock_driver.drive.assert_awaited_once()

        ctx = rollout_state.step_context
        assert ctx is not None
        assert ctx.step_start_us == 200_000
        assert ctx.target_time_us == 300_000
        assert ctx.driver_trajectory is not None

    @pytest.mark.asyncio
    async def test_handle_reschedules_with_advanced_timestamp(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """RecurringEvent.handle() reschedules with timestamp += interval_us."""
        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        queue = EventQueue()
        await policy_event.handle(rollout_state, queue)

        assert len(queue) == 1
        next_event = queue.pop()
        assert next_event is policy_event  # Same object, mutated
        assert next_event.timestamp_us == 300_000

    @pytest.mark.asyncio
    async def test_pipeline_with_none_renderer_data(
        self,
        policy_event: PolicyEvent,
        rollout_state: RolloutState,
        mock_driver: AsyncMock,
        simple_trajectory: Trajectory,
    ):
        """When no grouped render data exists, driver receives None for renderer_data."""
        rollout_state.data_sensorsim_to_driver = None

        drive_traj = simple_trajectory.clip(200_000, 300_001)
        mock_driver.drive.return_value = drive_traj

        queue = EventQueue()
        await policy_event.run(rollout_state, queue)

        call_kwargs = mock_driver.drive.call_args.kwargs
        assert call_kwargs["renderer_data"] is None


# ---------------------------------------------------------------------------
# Traffic physics accumulation
# ---------------------------------------------------------------------------


class TestTrafficPhysicsAccumulation:
    """PhysicsEvent:TRAFFIC accumulates into ctx.traffic_trajectories."""

    @pytest.mark.asyncio
    async def test_traffic_physics_accumulates_trajectories(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ):
        """Two traffic physics rounds accumulate poses into traffic_trajectories."""
        ego_traj = _make_trajectory_from_range(200_000, 500_000, 100_000)
        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=500_000,
            force_gt=False,
            corrected_ego_trajectory=ego_traj,
            traffic_response=TrafficReturn(),  # empty response
        )
        rollout_state.step_context = ctx

        # First round at T=200_000 (target=300_000)
        event1 = PhysicsEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            services=service_bundle,
            target=PhysicsTarget.TRAFFIC,
        )
        await event1.run(rollout_state, EventQueue())

        # Second round at T=300_000 (target=400_000)
        ctx.traffic_response = TrafficReturn()
        event2 = PhysicsEvent(
            timestamp_us=300_000,
            control_timestep_us=100_000,
            services=service_bundle,
            target=PhysicsTarget.TRAFFIC,
        )
        await event2.run(rollout_state, EventQueue())

        # traffic_trajectories should exist (possibly empty since TrafficReturn is empty)
        assert isinstance(ctx.traffic_trajectories, dict)


# ---------------------------------------------------------------------------
# Traffic event timing
# ---------------------------------------------------------------------------


class TestStepEventTrafficTrajectories:
    """StepEvent commits accumulated traffic trajectories."""

    @pytest.mark.asyncio
    async def test_commit_applies_accumulated_traffic_poses(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ):
        """StepEvent writes all accumulated traffic poses to state."""
        # Set up a traffic object with trajectory up to 200_000
        initial_traj = _make_trajectory_from_range(0, 200_000, 100_000)
        rollout_state.traffic_objs["car_1"] = TrafficObject(
            track_id="car_1",
            aabb=AABB(x=4.0, y=2.0, z=1.5),
            trajectory=initial_traj,
            is_static=False,
            label_class="VEHICLE",
        )

        # Build ego data for the step
        ego_traj = Trajectory.from_poses(
            np.array([300_000], dtype=np.uint64),
            [_make_pose(0.3, 0.0, 0.0)],
        )
        ego_dyn = DynamicTrajectory.from_trajectory_and_dynamics(
            ego_traj, np.zeros((1, 12), dtype=np.float64)
        )

        # Build accumulated traffic trajectory with two poses
        pose_300 = _make_pose(1.0, 0.0, 0.0)
        pose_400 = _make_pose(2.0, 0.0, 0.0)
        traffic_traj = Trajectory.from_poses(
            np.array([300_000], dtype=np.uint64), [pose_300]
        )
        traffic_traj.update_absolute(400_000, pose_400)

        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=400_000,
            force_gt=False,
            ego_true=ego_dyn,
            ego_estimated=ego_dyn,
            corrected_ego_trajectory=ego_traj,
            traffic_trajectories={"car_1": traffic_traj},
        )
        rollout_state.step_context = ctx

        event = StepEvent(
            timestamp_us=200_000,
            control_timestep_us=200_000,
            services=service_bundle,
        )

        queue = EventQueue()
        await event.run(rollout_state, queue)

        # car_1 should now have poses at 300_000 and 400_000
        car_traj = rollout_state.traffic_objs["car_1"].trajectory
        assert 300_000 in car_traj.time_range_us
        assert 400_000 in car_traj.time_range_us


class TestTrafficEventTiming:
    """Traffic events derive their own target time, not from StepContext."""

    @pytest.mark.asyncio
    async def test_traffic_event_uses_own_target_time(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ):
        """TrafficEvent queries trafficsim with its own target_time, not ctx.target_time_us."""
        # Driver cycle covers 200_000 → 500_000 (3 traffic steps of 100_000)
        ego_traj = _make_trajectory_from_range(200_000, 500_000, 100_000)
        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=500_000,  # driver's target — NOT what traffic should use
            force_gt=False,
            corrected_ego_trajectory=ego_traj,
        )
        rollout_state.step_context = ctx

        # Traffic event at 200_000 with interval 100_000 → target should be 300_000
        event = TrafficEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            services=service_bundle,
        )

        queue = EventQueue()
        await event.run(rollout_state, queue)

        call_kwargs = service_bundle.trafficsim.simulate_traffic.call_args.kwargs
        assert call_kwargs["future_us"] == 300_000  # NOT 500_000

    @pytest.mark.asyncio
    async def test_traffic_event_interpolates_ego_pose(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ):
        """TrafficEvent interpolates ego pose at its own target, not last_pose."""
        ego_traj = _make_trajectory_from_range(200_000, 500_000, 100_000)
        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=500_000,
            force_gt=False,
            corrected_ego_trajectory=ego_traj,
        )
        rollout_state.step_context = ctx

        event = TrafficEvent(
            timestamp_us=200_000,
            control_timestep_us=100_000,
            services=service_bundle,
        )

        queue = EventQueue()
        await event.run(rollout_state, queue)

        call_kwargs = service_bundle.trafficsim.simulate_traffic.call_args.kwargs
        ego_pose_used = call_kwargs["ego_aabb_pose_future"]
        # With identity ds_to_aabb transform, ego pose x should be 0.3 (at t=300_000)
        # not 0.5 (at t=500_000, the last_pose).
        # The trajectory helper _make_trajectory_from_range creates poses where x = t/1e6
        np.testing.assert_almost_equal(ego_pose_used.vec3[0], 0.3, decimal=3)


# ---------------------------------------------------------------------------
# Multi-cadence traffic integration
# ---------------------------------------------------------------------------


class TestMultiCadenceTraffic:
    """Integration: driver cycle spans multiple traffic steps."""

    @pytest.mark.asyncio
    async def test_three_traffic_steps_per_driver_cycle(
        self,
        rollout_state: RolloutState,
        service_bundle: ServiceBundle,
    ):
        """Run 3 traffic rounds within one driver cycle, verify all poses committed."""
        # Setup: traffic object and ego trajectory covering full range
        # corrected_ego_trajectory starts after the existing state trajectory
        # (state ends at 200_000), so it covers 300_000..500_000.
        ego_traj = _make_trajectory_from_range(300_000, 500_000, 100_000)
        ego_dyn = DynamicTrajectory.from_trajectory_and_dynamics(
            ego_traj, np.zeros((len(ego_traj), 12), dtype=np.float64)
        )

        initial_traffic_traj = _make_trajectory_from_range(0, 200_000, 100_000)
        rollout_state.traffic_objs["car_1"] = TrafficObject(
            track_id="car_1",
            aabb=AABB(x=4.0, y=2.0, z=1.5),
            trajectory=initial_traffic_traj,
            is_static=False,
            label_class="VEHICLE",
        )

        # Create StepContext as PolicyEvent would
        ctx = StepContext(
            step_start_us=200_000,
            target_time_us=500_000,
            force_gt=False,
            ego_true=ego_dyn,
            ego_estimated=ego_dyn,
            corrected_ego_trajectory=ego_traj,
        )
        rollout_state.step_context = ctx

        # Configure trafficsim mock to return a car_1 trajectory update each round
        def make_traffic_return(future_us: int) -> TrafficReturn:
            car_pose = _make_pose(float(future_us) / 1e6, 1.0, 0.0)
            traj_proto = geometry.trajectory_to_grpc(
                Trajectory.from_poses(
                    np.array([future_us], dtype=np.uint64), [car_pose]
                )
            )
            return TrafficReturn(
                object_trajectory_updates=[
                    ObjectTrajectoryUpdate(object_id="car_1", trajectory=traj_proto)
                ]
            )

        service_bundle.trafficsim.simulate_traffic.side_effect = lambda **kwargs: (
            make_traffic_return(kwargs["future_us"])
        )

        # Run 3 traffic + traffic-physics rounds
        traffic_dt = 100_000
        for step_offset in range(3):
            traffic_ts = 200_000 + step_offset * traffic_dt

            traffic_event = TrafficEvent(
                timestamp_us=traffic_ts,
                control_timestep_us=traffic_dt,
                services=service_bundle,
            )
            await traffic_event.run(rollout_state, EventQueue())

            physics_event = PhysicsEvent(
                timestamp_us=traffic_ts,
                control_timestep_us=traffic_dt,
                services=service_bundle,
                target=PhysicsTarget.TRAFFIC,
            )
            await physics_event.run(rollout_state, EventQueue())

        # Verify 3 poses accumulated
        assert "car_1" in ctx.traffic_trajectories
        assert len(ctx.traffic_trajectories["car_1"]) == 3

        # Run commit (driver cycle timestep = 3 * traffic_dt)
        step_event = StepEvent(
            timestamp_us=200_000,
            control_timestep_us=300_000,
            services=service_bundle,
        )
        await step_event.run(rollout_state, EventQueue())

        # Verify all 3 timestamps committed to state
        car_traj = rollout_state.traffic_objs["car_1"].trajectory
        assert 300_000 in car_traj.time_range_us
        assert 400_000 in car_traj.time_range_us
        assert 500_000 in car_traj.time_range_us

        # Verify step context was replaced with a fresh one
        assert rollout_state.step_context is not None
        assert rollout_state.step_context.ego_true is None  # fresh, not the old ctx


# ---------------------------------------------------------------------------
# Dummy event for ordering tests
# ---------------------------------------------------------------------------


class _DummyEvent(Event):
    """Minimal concrete event for priority/ordering tests."""

    def __init__(self, timestamp_us: int, priority: int = 50):
        super().__init__(timestamp_us)
        self.priority = priority

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        pass
