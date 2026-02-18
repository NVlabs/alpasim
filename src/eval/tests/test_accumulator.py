# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for EvalDataAccumulator."""

import pytest
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata
from conftest import create_test_eval_config

from eval.accumulator import EvalDataAccumulator
from eval.data import ScenarioEvalInput
from eval.schema import EvalConfig


def _create_rollout_metadata() -> RolloutMetadata:
    """Create a minimal RolloutMetadata for testing."""
    metadata = RolloutMetadata()

    # Session metadata
    metadata.session_metadata.session_uuid = "test-uuid-123"
    metadata.session_metadata.scene_id = "test-scene"
    metadata.session_metadata.batch_size = 1
    metadata.session_metadata.n_sim_steps = 10
    metadata.session_metadata.start_timestamp_us = 0
    metadata.session_metadata.control_timestep_us = 100_000  # 100ms

    # Actor definitions - EGO vehicle AABB
    ego_aabb = metadata.actor_definitions.actor_aabb.add()
    ego_aabb.actor_id = "EGO"
    ego_aabb.aabb.size_x = 4.5
    ego_aabb.aabb.size_y = 2.0
    ego_aabb.aabb.size_z = 1.5
    ego_aabb.actor_label = "EGO"

    # Add a traffic actor
    traffic_aabb = metadata.actor_definitions.actor_aabb.add()
    traffic_aabb.actor_id = "TRAFFIC_1"
    traffic_aabb.aabb.size_x = 4.0
    traffic_aabb.aabb.size_y = 1.8
    traffic_aabb.aabb.size_z = 1.4
    traffic_aabb.actor_label = "vehicle"

    # Identity transform for rig to aabb (no transformation)
    metadata.transform_ego_coords_rig_to_aabb.vec.x = 0.0
    metadata.transform_ego_coords_rig_to_aabb.vec.y = 0.0
    metadata.transform_ego_coords_rig_to_aabb.vec.z = 0.0
    metadata.transform_ego_coords_rig_to_aabb.quat.x = 0.0
    metadata.transform_ego_coords_rig_to_aabb.quat.y = 0.0
    metadata.transform_ego_coords_rig_to_aabb.quat.z = 0.0
    metadata.transform_ego_coords_rig_to_aabb.quat.w = 1.0

    # Ground truth trajectory - simple straight line
    for i in range(10):
        pose_at_time = metadata.ego_rig_recorded_ground_truth_trajectory.poses.add()
        pose_at_time.timestamp_us = i * 100_000
        pose_at_time.pose.vec.x = float(i)
        pose_at_time.pose.vec.y = 0.0
        pose_at_time.pose.vec.z = 0.0
        pose_at_time.pose.quat.x = 0.0
        pose_at_time.pose.quat.y = 0.0
        pose_at_time.pose.quat.z = 0.0
        pose_at_time.pose.quat.w = 1.0

    return metadata


def _create_actor_poses(
    timestamp_us: int, ego_x: float, traffic_x: float | None = None
) -> ActorPoses:
    """Create ActorPoses message for EGO and optionally traffic."""
    actor_poses = ActorPoses()
    actor_poses.timestamp_us = timestamp_us

    ego_pose = actor_poses.actor_poses.add()
    ego_pose.actor_id = "EGO"
    ego_pose.actor_pose.vec.x = ego_x
    ego_pose.actor_pose.vec.y = 0.0
    ego_pose.actor_pose.vec.z = 0.0
    ego_pose.actor_pose.quat.x = 0.0
    ego_pose.actor_pose.quat.y = 0.0
    ego_pose.actor_pose.quat.z = 0.0
    ego_pose.actor_pose.quat.w = 1.0

    if traffic_x is not None:
        traffic_pose = actor_poses.actor_poses.add()
        traffic_pose.actor_id = "TRAFFIC_1"
        traffic_pose.actor_pose.vec.x = traffic_x
        traffic_pose.actor_pose.vec.y = 5.0  # Offset in y
        traffic_pose.actor_pose.vec.z = 0.0
        traffic_pose.actor_pose.quat.x = 0.0
        traffic_pose.actor_pose.quat.y = 0.0
        traffic_pose.actor_pose.quat.z = 0.0
        traffic_pose.actor_pose.quat.w = 1.0

    return actor_poses


def _create_driver_request(time_now_us: int, time_query_us: int) -> LogEntry:
    """Create a driver_request LogEntry."""
    entry = LogEntry()
    entry.driver_request.time_now_us = time_now_us
    entry.driver_request.time_query_us = time_query_us
    return entry


def _create_driver_return() -> LogEntry:
    """Create a driver_return LogEntry with a simple trajectory."""
    entry = LogEntry()

    # Add a simple trajectory with 2 poses
    for i in range(2):
        pose_at_time = entry.driver_return.trajectory.poses.add()
        pose_at_time.timestamp_us = i * 100_000
        pose_at_time.pose.vec.x = float(i)
        pose_at_time.pose.vec.y = 0.0
        pose_at_time.pose.vec.z = 0.0
        pose_at_time.pose.quat.x = 0.0
        pose_at_time.pose.quat.y = 0.0
        pose_at_time.pose.quat.z = 0.0
        pose_at_time.pose.quat.w = 1.0

    return entry


@pytest.fixture
def default_eval_config() -> EvalConfig:
    """Create a default EvalConfig for testing."""
    return create_test_eval_config()


class TestEvalDataAccumulatorHandleMessage:
    """Tests for EvalDataAccumulator.handle_message()."""

    def test_handles_rollout_metadata(self, default_eval_config: EvalConfig) -> None:
        """Test that rollout_metadata message is processed correctly."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)
        metadata = _create_rollout_metadata()

        accumulator.handle_message(LogEntry(rollout_metadata=metadata))

        # Check session metadata was extracted
        assert accumulator.session_metadata is not None
        assert accumulator.session_metadata.session_uuid == "test-uuid-123"
        assert accumulator.session_metadata.scene_id == "test-scene"

        # Check internal state was populated
        assert accumulator._ego_aabb_dims == (4.5, 2.0, 1.5)
        assert accumulator._ego_coords_rig_to_aabb_center is not None
        assert accumulator._gt_ego_trajectory is not None

    def test_handles_actor_poses(self, default_eval_config: EvalConfig) -> None:
        """Test that actor_poses messages are accumulated correctly."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # First send metadata to initialize actor tracking
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Send actor poses
        for i in range(5):
            poses = _create_actor_poses(
                timestamp_us=i * 100_000, ego_x=float(i), traffic_x=float(i) + 10.0
            )
            accumulator.handle_message(LogEntry(actor_poses=poses))

        # Check poses were accumulated
        assert "EGO" in accumulator._actor_trajectory_data
        assert len(accumulator._actor_trajectory_data["EGO"]) == 5

        assert "TRAFFIC_1" in accumulator._actor_trajectory_data
        assert len(accumulator._actor_trajectory_data["TRAFFIC_1"]) == 5

    def test_handles_driver_request_return_pairing(
        self, default_eval_config: EvalConfig
    ) -> None:
        """Test that driver_request and driver_return are paired correctly."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send metadata first
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Send driver_request
        accumulator.handle_message(_create_driver_request(100_000, 200_000))
        assert accumulator._pending_request == (100_000, 200_000)

        # Send driver_return
        accumulator.handle_message(_create_driver_return())
        assert accumulator._pending_request is None

        # Check response was accumulated with correct timestamps
        assert len(accumulator._driver_responses) == 1
        now_us, query_us, response = accumulator._driver_responses[0]
        assert now_us == 100_000
        assert query_us == 200_000
        assert isinstance(response, DriveResponse)

    def test_ignores_orphan_driver_return(
        self, default_eval_config: EvalConfig
    ) -> None:
        """Test that driver_return without prior driver_request is ignored."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send driver_return without prior driver_request
        accumulator.handle_message(_create_driver_return())

        # Should not accumulate
        assert len(accumulator._driver_responses) == 0

    def test_handles_unknown_message_types(
        self, default_eval_config: EvalConfig
    ) -> None:
        """Test that unknown message types are silently ignored."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Create a message with an unhandled type
        entry = LogEntry()
        entry.egomotion_estimate_error.timestamp_us = 100_000

        # Should not raise
        accumulator.handle_message(entry)


class TestEvalDataAccumulatorBuildScenarioEvalInput:
    """Tests for EvalDataAccumulator.build_scenario_eval_input()."""

    def test_builds_scenario_eval_input_successfully(
        self, default_eval_config: EvalConfig
    ) -> None:
        """Test that ScenarioEvalInput is built correctly from accumulated data."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send metadata
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Send actor poses
        for i in range(10):
            poses = _create_actor_poses(timestamp_us=i * 100_000, ego_x=float(i))
            accumulator.handle_message(LogEntry(actor_poses=poses))

        # Build ScenarioEvalInput
        result = accumulator.build_scenario_eval_input(
            run_uuid="test-run",
            run_name="test",
            batch_id="0",
            vec_map=None,
        )

        assert isinstance(result, ScenarioEvalInput)
        assert result.session_metadata.session_uuid == "test-uuid-123"
        assert result.run_uuid == "test-run"
        assert result.run_name == "test"
        assert result.batch_id == "0"

    def test_builds_actor_trajectories(self, default_eval_config: EvalConfig) -> None:
        """Test that actor trajectories are built from accumulated poses."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send metadata
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Send actor poses with traffic
        for i in range(10):
            poses = _create_actor_poses(
                timestamp_us=i * 100_000, ego_x=float(i), traffic_x=float(i) + 10.0
            )
            accumulator.handle_message(LogEntry(actor_poses=poses))

        # Build ScenarioEvalInput
        result = accumulator.build_scenario_eval_input(
            run_uuid="test-run",
            run_name="test",
            batch_id="0",
        )

        # Check EGO trajectory
        assert "EGO" in result.actor_trajectories
        ego_traj, ego_dims = result.actor_trajectories["EGO"]
        assert len(ego_traj.timestamps_us) == 10
        assert ego_dims == (4.5, 2.0, 1.5)

        # Check TRAFFIC_1 trajectory
        assert "TRAFFIC_1" in result.actor_trajectories
        traffic_traj, traffic_dims = result.actor_trajectories["TRAFFIC_1"]
        assert len(traffic_traj.timestamps_us) == 10
        # Use approximate comparison due to float32 precision in protobuf
        assert traffic_dims[0] == pytest.approx(4.0)
        assert traffic_dims[1] == pytest.approx(1.8)
        assert traffic_dims[2] == pytest.approx(1.4)

    def test_builds_driver_responses(self, default_eval_config: EvalConfig) -> None:
        """Test that DriverResponses are built from accumulated data."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send metadata
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Send actor poses (needed for ego trajectory)
        for i in range(10):
            poses = _create_actor_poses(timestamp_us=i * 100_000, ego_x=float(i))
            accumulator.handle_message(LogEntry(actor_poses=poses))

        # Send driver request/return pairs
        accumulator.handle_message(_create_driver_request(100_000, 200_000))
        accumulator.handle_message(_create_driver_return())

        # Build ScenarioEvalInput
        result = accumulator.build_scenario_eval_input(
            run_uuid="test-run",
            run_name="test",
            batch_id="0",
        )

        assert result.driver_responses is not None
        # Check that driver responses were added (they're processed in add_drive_response)
        assert len(result.driver_responses.per_timestep_driver_responses) == 1

    def test_ego_aabb_dimensions_populated(
        self, default_eval_config: EvalConfig
    ) -> None:
        """Test that ego AABB dimensions are correctly populated."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send metadata
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Send minimal poses
        for i in range(2):
            poses = _create_actor_poses(timestamp_us=i * 100_000, ego_x=float(i))
            accumulator.handle_message(LogEntry(actor_poses=poses))

        # Build ScenarioEvalInput
        result = accumulator.build_scenario_eval_input(
            run_uuid="test-run",
            run_name="test",
            batch_id="0",
        )

        assert result.ego_aabb_x_m == 4.5
        assert result.ego_aabb_y_m == 2.0
        assert result.ego_aabb_z_m == 1.5

    def test_ground_truth_trajectory_populated(
        self, default_eval_config: EvalConfig
    ) -> None:
        """Test that ground truth trajectory is correctly populated."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send metadata
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Send minimal poses
        for i in range(2):
            poses = _create_actor_poses(timestamp_us=i * 100_000, ego_x=float(i))
            accumulator.handle_message(LogEntry(actor_poses=poses))

        # Build ScenarioEvalInput
        result = accumulator.build_scenario_eval_input(
            run_uuid="test-run",
            run_name="test",
            batch_id="0",
        )

        assert result.ego_recorded_ground_truth_trajectory is not None
        assert len(result.ego_recorded_ground_truth_trajectory.timestamps_us) == 10

    def test_raises_without_rollout_metadata(
        self, default_eval_config: EvalConfig
    ) -> None:
        """Test that building without rollout_metadata raises ValueError."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        with pytest.raises(ValueError, match="session_metadata"):
            accumulator.build_scenario_eval_input(
                run_uuid="test-run",
                run_name="test",
                batch_id="0",
            )

    def test_handles_empty_actor_poses(self, default_eval_config: EvalConfig) -> None:
        """Test that building with only metadata (no actor_poses) works."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # Send only metadata
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # Build ScenarioEvalInput - should work but have empty trajectories
        result = accumulator.build_scenario_eval_input(
            run_uuid="test-run",
            run_name="test",
            batch_id="0",
        )

        assert isinstance(result, ScenarioEvalInput)
        # Actor trajectories should be empty since no poses were sent
        assert len(result.actor_trajectories) == 0


class TestEvalDataAccumulatorIntegration:
    """Integration tests for EvalDataAccumulator."""

    def test_full_message_sequence(self, default_eval_config: EvalConfig) -> None:
        """Test a complete sequence of messages like in a real simulation."""
        accumulator = EvalDataAccumulator(cfg=default_eval_config)

        # 1. Send rollout metadata (first message)
        accumulator.handle_message(
            LogEntry(rollout_metadata=_create_rollout_metadata())
        )

        # 2. Send actor poses for 10 timesteps
        for i in range(10):
            poses = _create_actor_poses(
                timestamp_us=i * 100_000, ego_x=float(i), traffic_x=float(i) + 10.0
            )
            accumulator.handle_message(LogEntry(actor_poses=poses))

        # 3. Send driver request/return pairs
        for i in range(5):
            now_us = i * 100_000
            query_us = (i + 1) * 100_000
            accumulator.handle_message(_create_driver_request(now_us, query_us))
            accumulator.handle_message(_create_driver_return())

        # 4. Build final ScenarioEvalInput
        result = accumulator.build_scenario_eval_input(
            run_uuid="integration-test",
            run_name="integration",
            batch_id="0",
        )

        # Verify result
        assert result.session_metadata.session_uuid == "test-uuid-123"
        assert "EGO" in result.actor_trajectories
        assert "TRAFFIC_1" in result.actor_trajectories
        assert len(result.actor_trajectories["EGO"][0].timestamps_us) == 10
        assert result.driver_responses is not None
        assert len(result.driver_responses.per_timestep_driver_responses) == 5
