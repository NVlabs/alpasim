# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for the ScenarioEvaluator class."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata
from alpasim_utils import logs
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import AABB
from conftest import SimpleScenarioEvaluator, create_test_eval_config

from eval.accumulator import EvalDataAccumulator
from eval.asl_loader import load_scenario_eval_input_from_asl
from eval.data import AggregationType, MetricReturn, ScenarioEvalInput
from eval.scenario_evaluator import ScenarioEvalResult
from eval.schema import EvalConfig


@pytest.fixture
def simple_trajectory() -> Trajectory:
    """Create a simple straight-line trajectory."""
    n_points = 10
    timestamps = np.arange(0, n_points * 100_000, 100_000, dtype=np.uint64)

    poses = [
        Pose(
            position=np.array([i * 1.0, 0.0, 0.0], dtype=np.float32),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        for i in range(n_points)
    ]

    return Trajectory.from_poses(timestamps=timestamps, poses=poses)


@pytest.fixture
def session_metadata() -> RolloutMetadata.SessionMetadata:
    """Create sample session metadata."""
    return RolloutMetadata.SessionMetadata(
        session_uuid="test-uuid-123",
        scene_id="test-scene",
        batch_size=1,
        n_sim_steps=10,
        start_timestamp_us=0,
        control_timestep_us=100_000,
    )


@pytest.fixture
def minimal_eval_input(
    session_metadata: RolloutMetadata.SessionMetadata,
    simple_trajectory: Trajectory,
) -> ScenarioEvalInput:
    """Create a minimal ScenarioEvalInput for testing."""
    return ScenarioEvalInput(
        session_metadata=session_metadata,
        run_uuid="test-run-uuid",
        run_name="test-run",
        batch_id="0",
        ego_coords_rig_to_aabb_center=Pose.identity(),
        ego_aabb_x_m=4.5,
        ego_aabb_y_m=2.0,
        ego_aabb_z_m=1.5,
        actor_trajectories={
            "EGO": (simple_trajectory, (4.5, 2.0, 1.5)),
        },
        ego_recorded_ground_truth_trajectory=simple_trajectory,
        vec_map=None,  # No map, offroad metrics will be skipped
    )


class SimpleScenarioEvaluatorClass:
    """Test the ScenarioEvaluator class.

    Note: These tests use SimpleScenarioEvaluator from conftest which excludes
    OffRoadScorer and other scorers that require complex fixtures (VectorMap,
    camera data, driver responses). Creating a proper VectorMap requires
    road geometry with RoadLane objects, center lines, edges, and spatial
    indices from the trajdata library - which is complex for unit tests.
    """

    def test_create_with_config(self, default_eval_config: EvalConfig) -> None:
        """Test creating an evaluator with configuration."""
        evaluator = SimpleScenarioEvaluator(default_eval_config)

        assert evaluator is not None
        assert evaluator.cfg is not None
        assert evaluator.cfg.vehicle.vehicle_shrink_factor == 0.0
        assert evaluator.cfg.vehicle.vehicle_corner_roundness == 0.0

    def test_create_with_custom_params(self) -> None:
        """Test creating an evaluator with custom parameters."""
        config = create_test_eval_config(
            vehicle_shrink_factor=0.1,
            vehicle_corner_roundness=0.2,
        )
        evaluator = SimpleScenarioEvaluator(config)

        assert evaluator.cfg.vehicle.vehicle_shrink_factor == 0.1
        assert evaluator.cfg.vehicle.vehicle_corner_roundness == 0.2

    def test_evaluate_basic(
        self, minimal_eval_input: ScenarioEvalInput, default_eval_config: EvalConfig
    ) -> None:
        """Test basic evaluation with minimal input."""
        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(minimal_eval_input)

        assert isinstance(result, ScenarioEvalResult)
        assert isinstance(result.timestep_metrics, list)
        assert isinstance(result.aggregated_metrics, dict)
        assert result.metrics_df is not None

    def test_evaluate_returns_collision_metrics(
        self, minimal_eval_input: ScenarioEvalInput, default_eval_config: EvalConfig
    ) -> None:
        """Test that collision metrics are returned."""
        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(minimal_eval_input)

        # Check that collision metrics are present
        metric_names = [m.name for m in result.timestep_metrics]
        assert "collision_any" in metric_names
        assert "collision_front" in metric_names
        assert "collision_rear" in metric_names
        assert "collision_lateral" in metric_names

    def test_no_collision_scenario(
        self, minimal_eval_input: ScenarioEvalInput, default_eval_config: EvalConfig
    ) -> None:
        """Test that no collisions are detected for a simple straight trajectory."""
        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(minimal_eval_input)

        # With only EGO and no other actors, there should be no collisions
        assert result.aggregated_metrics.get("collision_any", 0.0) == 0.0


class TestScenarioEvalResult:
    """Test the ScenarioEvalResult dataclass."""

    def test_result_structure(
        self, minimal_eval_input: ScenarioEvalInput, default_eval_config: EvalConfig
    ) -> None:
        """Test the structure of the evaluation result."""
        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(minimal_eval_input)

        # Check timestep_metrics structure
        assert len(result.timestep_metrics) > 0
        for metric in result.timestep_metrics:
            assert isinstance(metric, MetricReturn)
            assert isinstance(metric.name, str)
            assert isinstance(metric.timestamps_us, list)
            assert isinstance(metric.values, (list, np.ndarray))
            assert isinstance(metric.valid, list)

        # Check aggregated_metrics structure
        for name, value in result.aggregated_metrics.items():
            assert isinstance(name, str)
            assert isinstance(value, (int, float))

    def test_metrics_df_contains_all_metrics(
        self, minimal_eval_input: ScenarioEvalInput, default_eval_config: EvalConfig
    ) -> None:
        """Test that metrics_df contains all timestep metrics."""
        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(minimal_eval_input)

        if result.metrics_df is not None and len(result.metrics_df) > 0:
            # Check required columns
            assert "name" in result.metrics_df.columns
            assert "timestamps_us" in result.metrics_df.columns
            assert "values" in result.metrics_df.columns
            assert "valid" in result.metrics_df.columns


class TestAggregation:
    """Test metric aggregation logic via MetricReturn.aggregate()."""

    def test_aggregate_max(self) -> None:
        """Test MAX aggregation type."""
        metric = MetricReturn(
            name="test_max",
            timestamps_us=[1000, 2000, 3000],
            values=[0.1, 0.5, 0.3],
            valid=[True, True, True],
            time_aggregation=AggregationType.MAX,
        )

        assert metric.aggregate() == 0.5

    def test_aggregate_mean(self) -> None:
        """Test MEAN aggregation type."""
        metric = MetricReturn(
            name="test_mean",
            timestamps_us=[1000, 2000, 3000],
            values=[1.0, 2.0, 3.0],
            valid=[True, True, True],
            time_aggregation=AggregationType.MEAN,
        )

        assert metric.aggregate() == 2.0

    def test_aggregate_last(self) -> None:
        """Test LAST aggregation type."""
        metric = MetricReturn(
            name="test_last",
            timestamps_us=[1000, 2000, 3000],
            values=[0.1, 0.5, 0.9],
            valid=[True, True, True],
            time_aggregation=AggregationType.LAST,
        )

        assert metric.aggregate() == 0.9

    def test_aggregate_with_invalid_values(self) -> None:
        """Test that invalid values are excluded from aggregation."""
        metric = MetricReturn(
            name="test_invalid",
            timestamps_us=[1000, 2000, 3000],
            values=[1.0, 2.0, 3.0],
            valid=[True, False, True],  # Middle value is invalid
            time_aggregation=AggregationType.MEAN,
        )

        # Should average only valid values: (1.0 + 3.0) / 2 = 2.0
        assert metric.aggregate() == 2.0


class TestCollisionDetection:
    """Test collision detection with multiple actors."""

    @pytest.fixture
    def colliding_trajectory(self, simple_trajectory: Trajectory) -> Trajectory:
        """Create a trajectory that would collide with simple_trajectory."""
        # Derive from simple_trajectory: same timestamps, same x positions (will collide)
        n_points = len(simple_trajectory)
        simple_positions = simple_trajectory.positions
        poses = [
            Pose(
                position=np.array([simple_positions[i, 0], 0.0, 0.0], dtype=np.float32),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
            for i in range(n_points)
        ]
        return Trajectory.from_poses(
            timestamps=simple_trajectory.timestamps_us.copy(),
            poses=poses,
        )

    def test_collision_detection(
        self,
        session_metadata: RolloutMetadata.SessionMetadata,
        simple_trajectory: Trajectory,
        colliding_trajectory: Trajectory,
        default_eval_config: EvalConfig,
    ) -> None:
        """Test that collisions are detected when actors overlap."""
        eval_input = ScenarioEvalInput(
            session_metadata=session_metadata,
            run_uuid="test-run-uuid",
            run_name="test-run",
            batch_id="0",
            ego_coords_rig_to_aabb_center=Pose.identity(),
            ego_aabb_x_m=4.5,
            ego_aabb_y_m=2.0,
            ego_aabb_z_m=1.5,
            actor_trajectories={
                "EGO": (simple_trajectory, (4.5, 2.0, 1.5)),
                "OTHER": (colliding_trajectory, (4.5, 2.0, 1.5)),
            },
            ego_recorded_ground_truth_trajectory=simple_trajectory,
            vec_map=None,
        )

        evaluator = SimpleScenarioEvaluator(default_eval_config)
        result = evaluator.evaluate(eval_input)

        # Should detect collision since trajectories overlap
        assert result.aggregated_metrics.get("collision_any", 0.0) == 1.0


def _build_rollout_metadata(simulation_data: dict[str, Any]) -> RolloutMetadata:
    """Build RolloutMetadata protobuf from simulation data.

    This helper creates a complete RolloutMetadata message including session
    metadata, EGO AABB, coordinate transform, and ground truth trajectory.
    """
    metadata = RolloutMetadata()

    # Copy session metadata
    sm = simulation_data["session_metadata"]
    metadata.session_metadata.session_uuid = sm.session_uuid
    metadata.session_metadata.scene_id = sm.scene_id
    metadata.session_metadata.batch_size = sm.batch_size
    metadata.session_metadata.n_sim_steps = sm.n_sim_steps
    metadata.session_metadata.start_timestamp_us = sm.start_timestamp_us
    metadata.session_metadata.control_timestep_us = sm.control_timestep_us

    # EGO AABB
    ego_aabb_proto = metadata.actor_definitions.actor_aabb.add()
    ego_aabb_proto.actor_id = "EGO"
    ego_aabb_proto.aabb.size_x = simulation_data["ego_aabb"].x
    ego_aabb_proto.aabb.size_y = simulation_data["ego_aabb"].y
    ego_aabb_proto.aabb.size_z = simulation_data["ego_aabb"].z
    ego_aabb_proto.actor_label = "EGO"

    # Transform
    transform = simulation_data["transform_ego_coords_ds_to_aabb"]
    metadata.transform_ego_coords_rig_to_aabb.vec.x = transform.vec3[0]
    metadata.transform_ego_coords_rig_to_aabb.vec.y = transform.vec3[1]
    metadata.transform_ego_coords_rig_to_aabb.vec.z = transform.vec3[2]
    metadata.transform_ego_coords_rig_to_aabb.quat.x = transform.quat[0]
    metadata.transform_ego_coords_rig_to_aabb.quat.y = transform.quat[1]
    metadata.transform_ego_coords_rig_to_aabb.quat.z = transform.quat[2]
    metadata.transform_ego_coords_rig_to_aabb.quat.w = transform.quat[3]

    # Ground truth trajectory
    gt_traj: Trajectory = simulation_data["gt_trajectory"]
    gt_positions = gt_traj.positions
    gt_quaternions = gt_traj.quaternions
    for i in range(len(gt_traj)):
        pose_at_time = metadata.ego_rig_recorded_ground_truth_trajectory.poses.add()
        pose_at_time.timestamp_us = int(gt_traj.timestamps_us[i])
        pose_at_time.pose.vec.x = float(gt_positions[i, 0])
        pose_at_time.pose.vec.y = float(gt_positions[i, 1])
        pose_at_time.pose.vec.z = float(gt_positions[i, 2])
        pose_at_time.pose.quat.x = float(gt_quaternions[i, 0])
        pose_at_time.pose.quat.y = float(gt_quaternions[i, 1])
        pose_at_time.pose.quat.z = float(gt_quaternions[i, 2])
        pose_at_time.pose.quat.w = float(gt_quaternions[i, 3])

    return metadata


def _build_actor_poses_messages(simulation_data: dict[str, Any]) -> list[ActorPoses]:
    """Build ActorPoses messages from simulation data.

    Returns a list of ActorPoses messages, one per timestep, containing
    the EGO trajectory poses.
    """
    messages = []
    ego_traj: Trajectory = simulation_data["ego_trajectory"]
    ego_positions = ego_traj.positions
    ego_quaternions = ego_traj.quaternions

    for i in range(len(ego_traj)):
        actor_poses = ActorPoses()
        actor_poses.timestamp_us = int(ego_traj.timestamps_us[i])
        ego_pose = actor_poses.actor_poses.add()
        ego_pose.actor_id = "EGO"
        ego_pose.actor_pose.vec.x = float(ego_positions[i, 0])
        ego_pose.actor_pose.vec.y = float(ego_positions[i, 1])
        ego_pose.actor_pose.vec.z = float(ego_positions[i, 2])
        ego_pose.actor_pose.quat.x = float(ego_quaternions[i, 0])
        ego_pose.actor_pose.quat.y = float(ego_quaternions[i, 1])
        ego_pose.actor_pose.quat.z = float(ego_quaternions[i, 2])
        ego_pose.actor_pose.quat.w = float(ego_quaternions[i, 3])

        messages.append(actor_poses)

    return messages


def _populate_accumulator(
    accumulator: EvalDataAccumulator, simulation_data: dict[str, Any]
) -> None:
    """Feed simulation data messages to an accumulator.

    This helper sends rollout_metadata and all actor_poses messages to
    the accumulator, simulating the runtime message flow.
    """
    # Feed rollout_metadata
    metadata = _build_rollout_metadata(simulation_data)
    accumulator.handle_message(LogEntry(rollout_metadata=metadata))

    # Feed actor_poses
    for actor_poses in _build_actor_poses_messages(simulation_data):
        accumulator.handle_message(LogEntry(actor_poses=actor_poses))


class TestRuntimeVsPostEvalEquivalence:
    """Tests verifying runtime and post-eval paths produce identical results.

    These tests ensure that evaluating simulation data via the RuntimeEvaluator
    (in-runtime path) produces the same metrics as evaluating the same data
    after loading it from an ASL file (post-eval path).
    """

    @pytest.fixture
    def simulation_data(
        self,
        simple_trajectory: Trajectory,
    ) -> dict[str, Any]:
        """Create simulation data usable by both runtime and post-eval paths.

        Returns a dictionary containing all data needed to evaluate a scenario
        through both paths, with consistent values.
        """
        # Use a non-trivial trajectory for more meaningful metrics comparison
        n_points = 10
        timestamps = np.arange(0, n_points * 100_000, 100_000, dtype=np.uint64)

        # Ego trajectory: moves along x-axis
        ego_poses = [
            Pose(
                position=np.array([i * 1.0, 0.0, 0.0], dtype=np.float32),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
            for i in range(n_points)
        ]
        ego_trajectory = Trajectory.from_poses(timestamps=timestamps, poses=ego_poses)

        # Ground truth trajectory: same as ego for this test
        gt_trajectory = ego_trajectory.clone()

        # Session metadata
        session_metadata = RolloutMetadata.SessionMetadata(
            session_uuid="test-uuid-runtime-vs-posteval",
            scene_id="test-scene-comparison",
            batch_size=1,
            n_sim_steps=n_points,
            start_timestamp_us=0,
            control_timestep_us=100_000,
        )

        # Ego AABB dimensions
        ego_aabb = AABB(x=4.5, y=2.0, z=1.5)

        return {
            "ego_trajectory": ego_trajectory,
            "gt_trajectory": gt_trajectory,
            "session_metadata": session_metadata,
            "transform_ego_coords_ds_to_aabb": Pose.identity(),
            "ego_aabb": ego_aabb,
            "rollout_uuid": "test-rollout-uuid-123",
            "scene_id": "test-scene-comparison",
            "run_uuid": "test-run-uuid",
            "run_name": "test-run",
            "timestamps": timestamps,
        }

    @pytest_asyncio.fixture
    async def asl_file_from_simulation(
        self,
        simulation_data: dict[str, Any],
        tmp_path: Path,
    ) -> Path:
        """Write an ASL file containing the simulation data.

        Creates an ASL file that, when loaded, should produce the same
        ScenarioEvalInput as building it directly via EvalDataAccumulator.
        """
        # Create directory structure expected by asl_loader
        asl_dir = tmp_path / "rollouts" / simulation_data["scene_id"] / "0"
        asl_dir.mkdir(parents=True, exist_ok=True)
        asl_path = asl_dir / "rollout.asl"

        log_writer = logs.LogWriter(asl_path)
        async with log_writer:
            # Write rollout metadata (must be first message)
            metadata = _build_rollout_metadata(simulation_data)
            await log_writer.on_message(LogEntry(rollout_metadata=metadata))

            # Write actor poses for each timestep
            for actor_poses in _build_actor_poses_messages(simulation_data):
                await log_writer.on_message(LogEntry(actor_poses=actor_poses))

        return asl_path

    @pytest.mark.asyncio
    async def test_runtime_vs_posteval_produce_same_metrics(
        self,
        simulation_data: dict[str, Any],
        asl_file_from_simulation: Path,
        default_eval_config: EvalConfig,
    ) -> None:
        """Verify both evaluation paths produce identical aggregated metrics.

        This test creates identical simulation data, evaluates it through both:
        1. Accumulator path: EvalDataAccumulator (simulating runtime message flow)
        2. Post-eval path: load_scenario_eval_input_from_asl() (which also uses
           EvalDataAccumulator internally)

        Since both paths now use EvalDataAccumulator, this test verifies that
        feeding messages directly vs reading from ASL produces identical results.
        """
        # ========== Accumulator Path (simulates runtime message flow) ==========
        accumulator = EvalDataAccumulator(cfg=default_eval_config)
        _populate_accumulator(accumulator, simulation_data)

        runtime_input = accumulator.build_scenario_eval_input(
            run_uuid=simulation_data["run_uuid"],
            run_name=simulation_data["run_name"],
            batch_id="0",
            vec_map=None,
        )

        # ========== Post-Eval Path ==========
        posteval_input = await load_scenario_eval_input_from_asl(
            asl_file_path=str(asl_file_from_simulation),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={
                "run_uuid": simulation_data["run_uuid"],
                "run_name": simulation_data["run_name"],
            },
        )

        # ========== Evaluate Both ==========
        evaluator = SimpleScenarioEvaluator(default_eval_config)
        runtime_result = evaluator.evaluate(runtime_input)
        posteval_result = evaluator.evaluate(posteval_input)

        # ========== Compare Results ==========
        # Both should have metrics
        assert runtime_result.metrics_df is not None
        assert posteval_result.metrics_df is not None
        assert len(runtime_result.metrics_df) > 0
        assert len(posteval_result.metrics_df) > 0

        # Compare aggregated metrics (main verification)
        runtime_metrics = runtime_result.aggregated_metrics
        posteval_metrics = posteval_result.aggregated_metrics

        # Same metric names should be present
        assert set(runtime_metrics.keys()) == set(posteval_metrics.keys()), (
            f"Metric names differ:\n"
            f"Runtime only: {set(runtime_metrics.keys()) - set(posteval_metrics.keys())}\n"
            f"Posteval only: {set(posteval_metrics.keys()) - set(runtime_metrics.keys())}"
        )

        # Compare metric values with tolerance for floating point
        for metric_name in runtime_metrics:
            runtime_val = runtime_metrics[metric_name]
            posteval_val = posteval_metrics[metric_name]
            assert runtime_val == pytest.approx(posteval_val, rel=1e-6, abs=1e-9), (
                f"Metric '{metric_name}' differs: "
                f"runtime={runtime_val}, posteval={posteval_val}"
            )

        # Verify DataFrame row counts match
        assert len(runtime_result.metrics_df) == len(posteval_result.metrics_df), (
            f"DataFrame row counts differ: "
            f"runtime={len(runtime_result.metrics_df)}, "
            f"posteval={len(posteval_result.metrics_df)}"
        )

    @pytest.mark.asyncio
    async def test_both_paths_detect_no_collision_for_single_ego(
        self,
        simulation_data: dict[str, Any],
        asl_file_from_simulation: Path,
        default_eval_config: EvalConfig,
    ) -> None:
        """Verify both paths correctly detect no collision with single actor."""
        # ========== Accumulator Path ==========
        accumulator = EvalDataAccumulator(cfg=default_eval_config)
        _populate_accumulator(accumulator, simulation_data)

        runtime_input = accumulator.build_scenario_eval_input(
            run_uuid=simulation_data["run_uuid"],
            run_name=simulation_data["run_name"],
            batch_id="0",
        )

        # ========== Post-eval path ==========
        posteval_input = await load_scenario_eval_input_from_asl(
            asl_file_path=str(asl_file_from_simulation),
            cfg=default_eval_config,
            artifacts={},
            run_metadata={
                "run_uuid": simulation_data["run_uuid"],
                "run_name": simulation_data["run_name"],
            },
        )

        evaluator = SimpleScenarioEvaluator(default_eval_config)
        runtime_result = evaluator.evaluate(runtime_input)
        posteval_result = evaluator.evaluate(posteval_input)

        # Both should report no collision
        assert runtime_result.aggregated_metrics.get("collision_any", 0.0) == 0.0
        assert posteval_result.aggregated_metrics.get("collision_any", 0.0) == 0.0
