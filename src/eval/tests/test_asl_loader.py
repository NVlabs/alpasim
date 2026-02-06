# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for ASL loader and ASL-to-evaluation pipeline."""

from pathlib import Path

import pytest
import pytest_asyncio
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata
from alpasim_utils import logs
from conftest import SimpleScenarioEvaluator, create_test_eval_config

from eval.asl_loader import load_scenario_eval_input_from_asl
from eval.data import ScenarioEvalInput
from eval.scenario_evaluator import ScenarioEvalResult
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


def _create_actor_poses(timestamp_us: int, x_position: float) -> ActorPoses:
    """Create ActorPoses message for EGO at given timestamp and position."""
    actor_poses = ActorPoses()
    actor_poses.timestamp_us = timestamp_us

    ego_pose = actor_poses.actor_poses.add()
    ego_pose.actor_id = "EGO"
    ego_pose.actor_pose.vec.x = x_position
    ego_pose.actor_pose.vec.y = 0.0
    ego_pose.actor_pose.vec.z = 0.0
    ego_pose.actor_pose.quat.x = 0.0
    ego_pose.actor_pose.quat.y = 0.0
    ego_pose.actor_pose.quat.z = 0.0
    ego_pose.actor_pose.quat.w = 1.0

    return actor_poses


@pytest_asyncio.fixture
async def minimal_asl_file(tmp_path: Path) -> Path:
    """Create a minimal valid ASL file for testing.

    The ASL file contains:
    - RolloutMetadata with session info, EGO AABB, and ground truth trajectory
    - ActorPoses messages for 10 timesteps with EGO moving in a straight line
    """
    asl_path = tmp_path / "rollouts" / "test-clipgt" / "0" / "rollout.asl"
    asl_path.parent.mkdir(parents=True, exist_ok=True)

    log_writer = logs.LogWriter(asl_path)
    async with log_writer:
        # Write rollout metadata (must be first message)
        metadata = _create_rollout_metadata()
        await log_writer.on_message(LogEntry(rollout_metadata=metadata))

        # Write actor poses for each timestep
        for i in range(10):
            timestamp_us = i * 100_000
            # EGO moves along x-axis, matching ground truth
            actor_poses = _create_actor_poses(timestamp_us, x_position=float(i))
            await log_writer.on_message(LogEntry(actor_poses=actor_poses))

    return asl_path


@pytest.fixture
def default_eval_config() -> EvalConfig:
    """Create a default EvalConfig for testing."""
    return create_test_eval_config()


class TestLoadScenarioEvalInputFromAsl:
    """Tests for load_scenario_eval_input_from_asl function."""

    @pytest.mark.asyncio
    async def test_loads_asl_file_successfully(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that ASL file loads successfully into ScenarioEvalInput."""
        result = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        assert isinstance(result, ScenarioEvalInput)

    @pytest.mark.asyncio
    async def test_session_metadata_populated(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that session metadata is correctly populated from ASL."""
        result = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        assert result.session_metadata.session_uuid == "test-uuid-123"
        assert result.session_metadata.scene_id == "test-scene"
        assert result.session_metadata.n_sim_steps == 10
        assert result.session_metadata.control_timestep_us == 100_000

    @pytest.mark.asyncio
    async def test_ego_aabb_populated(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that EGO AABB dimensions are correctly extracted."""
        result = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        assert result.ego_aabb_x_m == 4.5
        assert result.ego_aabb_y_m == 2.0
        assert result.ego_aabb_z_m == 1.5

    @pytest.mark.asyncio
    async def test_actor_trajectories_populated(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that actor trajectories are built from ActorPoses messages."""
        result = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        assert "EGO" in result.actor_trajectories
        ego_trajectory, ego_aabb_dims = result.actor_trajectories["EGO"]

        # Check trajectory has 10 poses
        assert len(ego_trajectory.timestamps_us) == 10

        # Check AABB dims (should match what's in metadata)
        assert ego_aabb_dims == (4.5, 2.0, 1.5)

    @pytest.mark.asyncio
    async def test_ground_truth_trajectory_populated(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that ground truth trajectory is extracted from metadata."""
        result = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        assert result.ego_recorded_ground_truth_trajectory is not None
        assert len(result.ego_recorded_ground_truth_trajectory.timestamps_us) == 10

    @pytest.mark.asyncio
    async def test_ids_extracted_from_path(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that clipgt_id, batch_id are extracted from file path."""
        result = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        # Path is .../rollouts/test-clipgt/0/rollout.asl
        assert result.batch_id == "0"


class TestEvaluateFromAslFile:
    """End-to-end tests for loading ASL and running evaluation.

    Note: These tests use SimpleScenarioEvaluator from conftest which excludes
    OffRoadScorer and other scorers that require complex fixtures (VectorMap,
    camera data, driver responses). Creating a proper VectorMap requires
    road geometry with RoadLane objects, center lines, edges, and spatial
    indices from the trajdata library - which is complex for unit tests.
    """

    @pytest.mark.asyncio
    async def test_evaluation_completes_successfully(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that evaluation runs successfully on ASL-loaded input."""
        # Load ASL file
        scenario_input = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        # Run evaluation
        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(scenario_input)

        assert isinstance(result, ScenarioEvalResult)
        assert result.timestep_metrics is not None
        assert result.aggregated_metrics is not None

    @pytest.mark.asyncio
    async def test_collision_metrics_computed(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that collision metrics are computed from ASL input."""
        scenario_input = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(scenario_input)

        # Check collision metrics are present
        metric_names = [m.name for m in result.timestep_metrics]
        assert "collision_any" in metric_names

        # With only EGO and no other actors, should be no collisions
        assert result.aggregated_metrics.get("collision_any", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_metrics_dataframe_generated(
        self, minimal_asl_file: Path, default_eval_config: EvalConfig
    ) -> None:
        """Test that metrics DataFrame is generated from evaluation."""
        scenario_input = await load_scenario_eval_input_from_asl(
            asl_file_path=str(minimal_asl_file),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={"run_uuid": "test-run", "run_name": "test"},
        )

        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(scenario_input)

        assert result.metrics_df is not None
        assert len(result.metrics_df) > 0

        # Check required columns exist
        assert "name" in result.metrics_df.columns
        assert "timestamps_us" in result.metrics_df.columns
        assert "values" in result.metrics_df.columns
