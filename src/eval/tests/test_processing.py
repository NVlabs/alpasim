# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from eval.aggregation.modifiers import AddCombinedEvent, RemoveTimestepsAfterEvent
from eval.aggregation.processing import (
    ProcessedMetricDFs,
    UnprocessedMetricsDFs,
    add_rollout_and_trajectory_uids,
    aggregate_and_write_metrics_results_txt,
    aggregate_over_clips,
    get_avg_dist_between_incidents,
)
from eval.schema import SceneScoreConfig


# Fixtures for commonly used dataframes
@pytest.fixture
def unified_metrics_df() -> pl.DataFrame:
    """Comprehensive metrics dataframe that supports all test scenarios.

    Uses different values per run, clip, and rollout to verify aggregation correctness.

    Data Structure:
    ===============
    - 2 runs: test_run_1 (uuid1), test_run_2 (uuid2)
    - 2 clips per run: clip1, clip2
    - 2 rollouts per clip: rollout1, rollout2
    - 3 timestamps per trajectory: 1000, 2000, 3000 microseconds
    - 9 metrics per trajectory (see metrics list below)

    Total: 2×2×2×3×9 = 216 data points

    Multiplier System:
    ==================
    Values are generated using: base_value × run_multiplier × clip_multiplier × rollout_multiplier

    - run_multipliers: test_run_1=1, test_run_2=100
    - clip_multipliers: clip1=1, clip2=10
    - rollout_multipliers: rollout1=1, rollout2=2

    Expected Relationships:
    =======================
    For continuous metrics (dist_traveled_m, metric_a, metric_b):
    - rollout2 = 2× rollout1 (within same run/clip)
    - clip2 = 10× clip1 (averaged across rollouts, within same run)
    - run2 = 100× run1 (averaged across clips and rollouts)

    For binary metrics (collision_*, offroad, etc.):
    - Values remain unscaled to preserve their semantic meaning

    Metrics Included:
    =================
    Required by DEFAULT_MODIFIERS:
    - collision_any, collision_front, collision_lateral (max aggregation)
    - offroad, img_is_black (max aggregation)
    - dist_traveled_m (max aggregation) - SCALED
    - progress (last aggregation)
    - progress_rel_to_total (last aggregation)

    Test metrics:
    - metric_a (max aggregation) - SCALED
    - metric_b (mean aggregation) - SCALED

    Usage:
    ======
    This fixture enables testing that:
    1. Data doesn't get mixed up between different runs/clips/rollouts
    2. Aggregation logic (time, clip, rollout) works correctly
    3. Mathematical relationships are preserved through the pipeline
    4. Binary vs continuous metrics are handled appropriately
    """
    base_data = []

    # Two runs with different UUIDs and names
    runs = [
        {"run_uuid": "uuid1", "run_name": "test_run_1", "run_multiplier": 1},
        {"run_uuid": "uuid2", "run_name": "test_run_2", "run_multiplier": 100},
    ]

    # Two clips per run
    clips = [
        {"clip_id": "clip1", "clip_multiplier": 1},
        {"clip_id": "clip2", "clip_multiplier": 10},
    ]

    # Two rollouts per clip
    rollouts = [
        {"rollout_id": "rollout1", "rollout_multiplier": 1},
        {"rollout_id": "rollout2", "rollout_multiplier": 2},
    ]

    # Timestamps for each trajectory
    timestamps = [1000, 2000, 3000]

    # All metrics required by DEFAULT_MODIFIERS plus some test metrics
    # Base values that will be modified per run/clip/rollout
    metrics_config = [
        ("eval_relevant", "max", [1.0, 1.0, 1.0]),  # all relevant (no prerun filtering)
        ("collision_any", "max", [0.0, 1.0, 0.0]),
        ("collision_front", "max", [0.0, 0.0, 1.0]),
        ("collision_lateral", "max", [0.0, 0.0, 0.0]),
        ("offroad", "max", [0.0, 0.0, 1.0]),
        ("img_is_black", "max", [0.0, 0.0, 0.0]),
        ("dist_traveled_m", "max", [1.0, 2.0, 3.0]),  # Will be scaled by multipliers
        ("gt_dist_traveled_m", "last", [10.0, 10.0, 10.0]),
        ("dist_to_gt_trajectory", "max", [0.0, 0.0, 0.0]),
        ("progress", "last", [0.1, 0.5, 0.8]),
        ("progress_rel_to_total", "last", [0.1, 0.5, 0.8]),
        ("progress_rel", "min", [0.9, 0.8, 0.8]),
        ("metric_a", "max", [1.0, 2.0, 3.0]),  # Will be scaled by multipliers
        ("metric_b", "mean", [0.5, 1.5, 2.5]),  # Will be scaled by multipliers
    ]

    # Generate data for each combination with unique values per run/clip/rollout
    for run in runs:
        for clip in clips:
            for rollout in rollouts:
                for metric_name, time_agg, base_values in metrics_config:
                    for ts, base_val in zip(timestamps, base_values):
                        # Create unique values by combining run, clip, and rollout multipliers
                        if metric_name in ["dist_traveled_m", "metric_a", "metric_b"]:
                            # Scale continuous metrics by run, clip, and rollout multipliers
                            value = (
                                base_val
                                * run["run_multiplier"]
                                * clip["clip_multiplier"]
                                * rollout["rollout_multiplier"]
                            )
                        else:
                            # Keep binary/categorical metrics as-is to preserve their meaning
                            value = base_val

                        base_data.append(
                            {
                                "timestamps_us": ts,
                                "values": value,
                                "valid": True,
                                "name": metric_name,
                                "time_aggregation": time_agg,
                                "clipgt_id": clip["clip_id"],
                                "rollout_id": rollout["rollout_id"],
                                "run_uuid": run["run_uuid"],
                                "run_name": run["run_name"],
                            }
                        )

    return pl.DataFrame(base_data)


@pytest.fixture
def trajectory_uid_df() -> pl.DataFrame:
    """Sample trajectory UID mapping dataframe."""
    return pl.DataFrame(
        {
            "trajectory_uid": [1, 2, 3, 4],
            "rollout_uid": [1, 1, 2, 2],
            "run_name": ["test_run_1", "test_run_1", "test_run_2", "test_run_2"],
            "run_uuid": ["uuid1", "uuid1", "uuid2", "uuid2"],
            "clipgt_id": ["clip1", "clip2", "clip1", "clip2"],
            "rollout_id": ["rollout1", "rollout1", "rollout1", "rollout1"],
        }
    )


@pytest.fixture
def temp_directory() -> Generator[pathlib.Path, None, None]:
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


@pytest.fixture
def modified_metrics_df_with_img_black(
    unified_metrics_df: pl.DataFrame,
) -> pl.DataFrame:
    """Modified metrics dataframe where one specific rollout has img_is_black = 1.0.

    Specifically sets img_is_black = 1.0 for test_run_1 + clip1 + rollout1.
    This will trigger RemoveTrajectoryWithEvent(img_is_black > 0) to remove that complete trajectory.

    Expected impact:
    - Original: 8 trajectories (2 runs × 2 clips × 2 rollouts)
    - Removed: 1 trajectory (test_run_1 + clip1 + rollout1)
    - Remaining: 7 trajectories

    Rollouts per clip after filtering:
    - test_run_1 + clip1: 1 rollout (rollout2 remains, rollout1 removed)
    - test_run_1 + clip2: 2 rollouts (both rollout1 and rollout2 remain)
    - test_run_2 + clip1: 2 rollouts (both rollout1 and rollout2 remain)
    - test_run_2 + clip2: 2 rollouts (both rollout1 and rollout2 remain)
    """
    return unified_metrics_df.with_columns(
        # Set img_is_black = 1.0 for test_run_1 + clip1 + rollout1 only
        pl.when(
            (pl.col("run_name") == "test_run_1")
            & (pl.col("clipgt_id") == "clip1")
            & (pl.col("rollout_id") == "rollout1")
            & (pl.col("name") == "img_is_black")
        )
        .then(1.0)
        .otherwise(pl.col("values"))
        .alias("values")
    )


class TestUnprocessedMetricsDFs:
    """Test the UnprocessedMetricsDFs class."""

    def test_initialization(self, unified_metrics_df: pl.DataFrame) -> None:
        """Test basic initialization."""
        udf = UnprocessedMetricsDFs(unified_metrics_df)
        assert udf.unprocessed_df.equals(unified_metrics_df)

    def test_save_and_load(
        self, unified_metrics_df: pl.DataFrame, temp_directory: pathlib.Path
    ) -> None:
        """Test saving and loading dataframes."""
        udf = UnprocessedMetricsDFs(unified_metrics_df)
        udf.save_to(temp_directory)

        # Check file was created
        assert (temp_directory / "metrics_unprocessed.parquet").exists()

        # Load and verify
        loaded_udf = UnprocessedMetricsDFs.load_from(temp_directory)
        assert loaded_udf.unprocessed_df.equals(unified_metrics_df)

    def test_concat_basic(self, unified_metrics_df: pl.DataFrame) -> None:
        """Test concatenating multiple UnprocessedMetricsDFs."""
        udf1 = UnprocessedMetricsDFs(unified_metrics_df)
        udf2 = UnprocessedMetricsDFs(
            unified_metrics_df.with_columns(
                pl.col("run_uuid").str.replace("uuid1", "uuid2"),
                pl.col("run_name").str.replace("test_run_1", "test_run_2"),
            )
        )

        concatenated = UnprocessedMetricsDFs.concat([udf1, udf2])

        assert concatenated.unprocessed_df.height == unified_metrics_df.height * 2
        unique_runs = concatenated.unprocessed_df["run_name"].unique().sort()
        assert unique_runs.to_list() == ["test_run_1", "test_run_2"]

    def test_concat_with_rename(self, unified_metrics_df: pl.DataFrame) -> None:
        """Test concatenating with run name renaming."""
        # Use only test_run_1 data for this test to keep it simple
        df1 = unified_metrics_df.filter(pl.col("run_name") == "test_run_1")
        udf1 = UnprocessedMetricsDFs(df1)
        udf2 = UnprocessedMetricsDFs(
            df1.with_columns(pl.col("run_name").str.replace("test_run_1", "old_name"))
        )

        rename_map = {"old_name": "new_name"}
        concatenated = UnprocessedMetricsDFs.concat(
            [udf1, udf2], rename_run_names=rename_map
        )

        unique_runs = concatenated.unprocessed_df["run_name"].unique().sort()
        assert unique_runs.to_list() == ["new_name", "test_run_1"]

    def test_n_rollouts_simulated_per_clip(
        self, unified_metrics_df: pl.DataFrame
    ) -> None:
        """Test getting number of rollouts per clip."""
        udf = UnprocessedMetricsDFs(unified_metrics_df)

        # Test without specifying run
        result = udf.n_rollouts_simulated_per_clip()
        assert isinstance(result, pl.DataFrame)
        assert "n_rollouts" in result.columns
        assert "run_uuid" in result.columns
        assert "run_name" in result.columns
        # Each run should have 2 rollouts based on the test data (rollout1 and rollout2)
        assert result.select("n_rollouts").to_series().unique().to_list() == [2]

        # Test with specific run_name
        n_rollouts = udf.n_rollouts_simulated_per_clip(run_name="test_run_1")
        assert isinstance(n_rollouts, int)
        assert n_rollouts == 2

    def test_time_aggregation_preserves_multipliers(
        self, unified_metrics_df: pl.DataFrame
    ) -> None:
        """Test that df_wide_avg_t (time aggregation only) preserves run/clip/rollout multipliers."""
        processed = aggregate_and_write_metrics_results_txt(unified_metrics_df)
        df_wide_avg_t = processed.df_wide_avg_t

        # Should have 8 trajectories: 2 runs × 2 clips × 2 rollouts each
        assert df_wide_avg_t.height == 8

        # Check run1 trajectories
        run1_trajectories = df_wide_avg_t.filter(pl.col("run_name") == "test_run_1")
        run1_clip1 = run1_trajectories.filter(pl.col("clipgt_id") == "clip1")
        run1_clip2 = run1_trajectories.filter(pl.col("clipgt_id") == "clip2")

        # Check run2 trajectories
        run2_trajectories = df_wide_avg_t.filter(pl.col("run_name") == "test_run_2")
        run2_clip1 = run2_trajectories.filter(pl.col("clipgt_id") == "clip1")
        run2_clip2 = run2_trajectories.filter(pl.col("clipgt_id") == "clip2")

        # Verify all trajectory groups exist (2 rollouts each)
        assert run1_clip1.height == 2
        assert run1_clip2.height == 2
        assert run2_clip1.height == 2
        assert run2_clip2.height == 2

        # Test rollout multiplier within same run/clip
        # rollout2 should be 2x rollout1 for continuous metrics
        run1_clip1_rollout1 = run1_clip1.filter(pl.col("rollout_id") == "rollout1")
        run1_clip1_rollout2 = run1_clip1.filter(pl.col("rollout_id") == "rollout2")
        rollout1_metric_a = run1_clip1_rollout1["metric_a"][0]
        rollout2_metric_a = run1_clip1_rollout2["metric_a"][0]
        rollout_ratio = rollout2_metric_a / rollout1_metric_a
        assert (
            abs(rollout_ratio - 2.0) < 0.01
        ), f"Rollout ratio should be 2.0, got {rollout_ratio}"

        # Verify clip multiplier effects within same run (average across rollouts)
        # clip2 should be 10x clip1 for continuous metrics
        run1_clip1_avg_metric_a = run1_clip1["metric_a"].mean()
        run1_clip2_avg_metric_a = run1_clip2["metric_a"].mean()
        clip_ratio_run1 = run1_clip2_avg_metric_a / run1_clip1_avg_metric_a
        assert (
            abs(clip_ratio_run1 - 10.0) < 0.01
        ), f"Clip ratio in run1 should be 10.0, got {clip_ratio_run1}"

        # Verify run multiplier effects within same clip (average across rollouts)
        # run2 should be 100x run1 for same clip
        run1_clip1_avg_metric_a = run1_clip1["metric_a"].mean()
        run2_clip1_avg_metric_a = run2_clip1["metric_a"].mean()
        run_ratio_clip1 = run2_clip1_avg_metric_a / run1_clip1_avg_metric_a
        assert (
            abs(run_ratio_clip1 - 100.0) < 0.01
        ), f"Run ratio in clip1 should be 100.0, got {run_ratio_clip1}"

    def test_clip_aggregation_preserves_multipliers(
        self, unified_metrics_df: pl.DataFrame
    ) -> None:
        """Test that df_wide_avg_t_clip (time + clip aggregation) preserves run multipliers."""
        processed = aggregate_and_write_metrics_results_txt(unified_metrics_df)
        df_wide_avg_t_clip = processed.df_wide_avg_t_clip

        # Should have 4 rows: 2 runs × 2 rollouts each (averaged across clips)
        assert df_wide_avg_t_clip.height == 4

        # Get values for each run at clip level (should have 2 rollouts each)
        run1_clip_data = df_wide_avg_t_clip.filter(pl.col("run_uuid") == "uuid1")
        run2_clip_data = df_wide_avg_t_clip.filter(pl.col("run_uuid") == "uuid2")

        assert run1_clip_data.height == 2  # 2 rollouts
        assert run2_clip_data.height == 2  # 2 rollouts

        # Check that all rollouts have n_clips = 2
        assert all(n_clips == 2 for n_clips in run1_clip_data["n_clips"].to_list())
        assert all(n_clips == 2 for n_clips in run2_clip_data["n_clips"].to_list())

        # Values should be averaged across clips but maintain 100x ratio between runs
        # Compare averages across rollouts
        run1_clip_avg_metric_a = run1_clip_data["metric_a"].mean()
        run2_clip_avg_metric_a = run2_clip_data["metric_a"].mean()
        clip_level_ratio = run2_clip_avg_metric_a / run1_clip_avg_metric_a
        assert (
            abs(clip_level_ratio - 100.0) < 0.01
        ), f"Run ratio at clip level should be 100.0, got {clip_level_ratio}"

    def test_full_aggregation_preserves_multipliers(
        self, unified_metrics_df: pl.DataFrame
    ) -> None:
        """Test that df_wide_avg_t_clip_rollout (full aggregation) preserves run multipliers."""
        processed = aggregate_and_write_metrics_results_txt(unified_metrics_df)
        final_results = processed.df_wide_avg_t_clip_rollout

        # Get metric values for each run
        run1_data = final_results.filter(pl.col("run_name") == "test_run_1")
        run2_data = final_results.filter(pl.col("run_name") == "test_run_2")

        # Verify we have data for both runs
        assert run1_data.height == 1
        assert run2_data.height == 1

        # Get values for comparison
        run1_dist = run1_data["dist_traveled_m"][0]
        run2_dist = run2_data["dist_traveled_m"][0]
        run1_metric_a = run1_data["metric_a"][0]
        run2_metric_a = run2_data["metric_a"][0]

        # Verify scaled metrics maintain correct relationships
        # run2 should be exactly 100x run1 for scaled metrics (run_multiplier: 1 vs 100)
        assert (
            run2_dist > run1_dist
        ), f"run2 dist ({run2_dist}) should be > run1 dist ({run1_dist})"
        assert (
            run2_metric_a > run1_metric_a
        ), f"run2 metric_a ({run2_metric_a}) should be > run1 metric_a ({run1_metric_a})"

        # Check that the ratio is exactly 100 (our run_multiplier difference)
        dist_ratio = run2_dist / run1_dist
        metric_ratio = run2_metric_a / run1_metric_a

        assert (
            abs(dist_ratio - 100.0) < 0.01
        ), f"Distance ratio should be 100.0, got {dist_ratio}"
        assert (
            abs(metric_ratio - 100.0) < 0.01
        ), f"Metric ratio should be 100.0, got {metric_ratio}"

        # Binary metrics should be the same across runs (not scaled)
        run1_collision = run1_data["collision_any"][0]
        run2_collision = run2_data["collision_any"][0]
        assert run1_collision == run2_collision, "Binary metrics should not be scaled"

        # Verify that n_clips is 2 for both runs (we have clip1 and clip2)
        assert run1_data["n_clips"][0] == 2
        assert run2_data["n_clips"][0] == 2

        # Verify that n_rollouts is 2 for both runs (we have rollout1 and rollout2)
        assert run1_data["n_rollouts"][0] == 2
        assert run2_data["n_rollouts"][0] == 2


class TestAddRolloutAndTrajectoryUids:
    """Test the add_rollout_and_trajectory_uids function."""

    def test_basic_functionality(self, unified_metrics_df: pl.DataFrame) -> None:
        """Test basic UID addition."""
        df, trajectory_uid_df = add_rollout_and_trajectory_uids(unified_metrics_df)

        # Check that UIDs were added
        assert "trajectory_uid" in df.columns
        assert "rollout_uid" not in df.columns  # Should be dropped from main df

        # Check trajectory_uid_df
        assert "trajectory_uid" in trajectory_uid_df.columns
        assert "rollout_uid" in trajectory_uid_df.columns
        assert "run_name" in trajectory_uid_df.columns

    def test_multiple_trajectories(self, unified_metrics_df: pl.DataFrame) -> None:
        """Test with multiple runs and clips."""
        df, trajectory_uid_df = add_rollout_and_trajectory_uids(unified_metrics_df)

        # Should have exactly 8 unique trajectory UIDs (2 runs × 2 clips × 2 rollouts)
        unique_trajectories = df["trajectory_uid"].unique()
        assert len(unique_trajectories) == 8

        # Each trajectory should map to correct metadata
        assert trajectory_uid_df.height == len(unique_trajectories)


class TestAggregateOverClips:
    """Test the aggregate_over_clips function."""

    def test_basic_aggregation(self, trajectory_uid_df: pl.DataFrame) -> None:
        """Test basic clip aggregation."""
        # Create sample wide dataframe with required distance columns
        df_wide_avg_t = pl.DataFrame(
            {
                "trajectory_uid": [1, 2],
                "metric_a": [1.0, 2.0],
                "metric_b": [3.0, 4.0],
                "dist_traveled_m": [1000.0, 2000.0],
                "offroad_or_collision": [1.0, 0.0],
                "offroad_or_collision_at_fault": [0.0, 0.0],
            }
        )

        result = aggregate_over_clips(df_wide_avg_t, trajectory_uid_df)

        # Should have rollout_uid and run_uuid columns
        assert "rollout_uid" in result.columns
        assert "run_uuid" in result.columns
        assert "n_clips" in result.columns


class TestGetAvgDistBetweenIncidents:
    """Test the get_avg_dist_between_incidents function."""

    def test_basic_calculation(self) -> None:
        """Test basic incident distance calculation."""
        df = pl.DataFrame(
            {
                "run_uuid": ["uuid1", "uuid1"],
                "rollout_uid": [1, 2],
                "dist_traveled_m": [1000.0, 2000.0],
                "offroad_or_collision": [1.0, 2.0],
                "offroad_or_collision_at_fault": [0.5, 1.0],
            }
        )

        result = get_avg_dist_between_incidents(df)

        assert "avg_dist_between_incidents" in result.columns
        assert "avg_dist_between_incidents_at_fault" in result.columns

        # Check calculations
        expected_dist = 1000.0 / 1.0 / 1000  # dist / incidents / 1000 (to km)
        assert result["avg_dist_between_incidents"][0] == expected_dist

        # Check at-fault calculations for second rollout
        expected_dist_at_fault_2nd = (
            2000.0 / 1.0 / 1000
        )  # dist / at_fault_incidents / 1000 (to km)
        assert (
            result["avg_dist_between_incidents_at_fault"][1]
            == expected_dist_at_fault_2nd
        )


class TestAggregateAndWriteMetricsResultsTxt:
    """Test the main aggregation function."""

    def test_basic_aggregation(self, unified_metrics_df: pl.DataFrame) -> None:
        """Test basic end-to-end aggregation."""
        result = aggregate_and_write_metrics_results_txt(unified_metrics_df)

        assert isinstance(result, ProcessedMetricDFs)

        # Check all expected dataframes are present
        assert hasattr(result, "df_wide")
        assert hasattr(result, "df_wide_avg_t")
        assert hasattr(result, "df_wide_avg_t_clip")
        assert hasattr(result, "df_wide_avg_t_clip_rollout")
        assert hasattr(result, "trajectory_uid_df")
        assert hasattr(result, "agg_function_df")

    def test_force_same_run_uniqueness_requirements(
        self, unified_metrics_df: pl.DataFrame
    ) -> None:
        """Test force_same_run behavior with unique/duplicate clipgt_id values.

        Documents that clipgt_id must be unique across run_uuid values, but run_name can be duplicated.
        """
        base_df = unified_metrics_df.filter(
            (pl.col("run_name") == "test_run_1") & (pl.col("clipgt_id") == "clip1")
        )

        # Case 1: Same run_name, different clipgt_id - should work
        valid_df = pl.concat(
            [
                base_df,
                base_df.with_columns(
                    pl.col("run_uuid").str.replace("uuid1", "uuid3"),
                    pl.col("clipgt_id").str.replace("clip1", "clip3"),
                ),
            ]
        )

        result = aggregate_and_write_metrics_results_txt(valid_df, force_same_run=True)
        assert result.combined_run_uuids is not None
        assert len(result.df_wide_avg_t_clip_rollout["run_uuid"].unique()) == 1

        # Case 2: Same clipgt_id - should fail
        invalid_df = pl.concat(
            [
                base_df,
                base_df.with_columns(pl.col("run_uuid").str.replace("uuid1", "uuid3")),
            ]
        )

        with pytest.raises(pl.exceptions.ComputeError):
            aggregate_and_write_metrics_results_txt(invalid_df, force_same_run=True)

    def test_force_same_run_combined_uuid_is_deterministic_across_hash_seeds(
        self, unified_metrics_df: pl.DataFrame, tmp_path: pathlib.Path
    ) -> None:
        base_df = unified_metrics_df.filter(
            (pl.col("run_name") == "test_run_1") & (pl.col("clipgt_id") == "clip1")
        )
        valid_df = pl.concat(
            [
                base_df,
                base_df.with_columns(
                    pl.col("run_uuid").str.replace("uuid1", "uuid3"),
                    pl.col("clipgt_id").str.replace("clip1", "clip3"),
                ),
            ]
        )

        input_path = tmp_path / "metrics.parquet"
        valid_df.write_parquet(input_path)

        eval_src_path = pathlib.Path(__file__).resolve().parents[1] / "src"
        script = "\n".join(
            [
                "import polars as pl",
                "from eval.aggregation.processing import aggregate_and_write_metrics_results_txt",
                f"df = pl.read_parquet(r'{input_path}')",
                "result = aggregate_and_write_metrics_results_txt(df, force_same_run=True)",
                "print(result.combined_run_uuids)",
            ]
        )

        outputs: list[str] = []
        for seed in ("1", "2"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{eval_src_path}:{existing_pythonpath}"
                if existing_pythonpath
                else str(eval_src_path)
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            outputs.append(completed.stdout.strip().splitlines()[-1])

        assert outputs[0] == outputs[1]

    def test_with_modifiers(self, unified_metrics_df: pl.DataFrame) -> None:
        """Test with custom modifiers."""
        custom_modifier = AddCombinedEvent(
            event=pl.col("collision_any") > 0.0,
            name="test_event",
            time_aggregation="max",
        )

        result = aggregate_and_write_metrics_results_txt(
            unified_metrics_df, additional_modifiers=[custom_modifier]
        )

        # Check that modifier was applied
        assert result.modifiers is not None
        assert len(result.modifiers) > len(
            [custom_modifier]
        )  # Should include defaults + custom

    @patch("eval.aggregation.processing.write_metrics_results_txt")
    @patch("eval.aggregation.processing.plot_metrics_results")
    def test_with_output_path(
        self,
        mock_plot: MagicMock,
        mock_write: MagicMock,
        unified_metrics_df: pl.DataFrame,
        temp_directory: pathlib.Path,
    ) -> None:
        """Test with output path specified."""
        _ = aggregate_and_write_metrics_results_txt(
            unified_metrics_df, output_path=str(temp_directory)
        )

        # Check that output functions were called
        mock_write.assert_called_once()
        mock_plot.assert_called_once()
        result_summary = temp_directory / "results-summary.json"
        assert result_summary.exists()
        summary_payload = json.loads(result_summary.read_text())
        assert summary_payload["schema_version"] == 1
        assert summary_payload["rollouts"]
        assert summary_payload["metrics_results"]

    @patch("eval.aggregation.processing.plot_metrics_results")
    def test_results_summary_json_contains_rollout_pass_fail_and_metrics_results(
        self,
        mock_plot: MagicMock,
        unified_metrics_df: pl.DataFrame,
        temp_directory: pathlib.Path,
    ) -> None:
        """Test that results-summary.json reports per-rollout status and aggregate metrics."""
        del mock_plot
        metrics_df = unified_metrics_df.with_columns(
            pl.when(
                (pl.col("run_name") == "test_run_1")
                & (pl.col("clipgt_id") == "clip1")
                & (pl.col("rollout_id") == "rollout1")
                & (pl.col("name") == "progress_rel_to_total")
            )
            .then(0.5)
            .otherwise(pl.col("values"))
            .alias("values")
        )
        metrics_df = metrics_df.with_columns(
            pl.when(
                (pl.col("run_name") == "test_run_1")
                & (pl.col("clipgt_id") == "clip1")
                & (pl.col("rollout_id") == "rollout2")
                & (pl.col("name") == "dist_to_gt_trajectory")
            )
            .then(11.0)
            .otherwise(pl.col("values"))
            .alias("values")
        )

        processed = aggregate_and_write_metrics_results_txt(
            metrics_df,
            output_path=str(temp_directory),
        )

        result_summary = temp_directory / "results-summary.json"
        metrics_results = pl.read_parquet(temp_directory / "metrics_results.parquet")
        payload = json.loads(result_summary.read_text())

        assert len(payload["rollouts"]) == processed.df_wide_avg_t.height
        assert len(payload["metrics_results"]) == metrics_results.height
        assert set(payload["metrics_results"][0]).issuperset(metrics_results.columns)

        partial = [
            row
            for row in payload["rollouts"]
            if row["run_name"] == "test_run_1"
            and row["clipgt_id"] == "clip1"
            and row["rollout_id"] == "rollout1"
        ][0]
        assert partial["status"] == "pass"
        assert partial["passed"] is True
        assert partial["score"] == pytest.approx(0.625)
        assert partial["score_metrics"]["progress_clipped_rel"] == 0.5
        assert partial["metrics"]["progress_clipped_rel"] == 0.5
        assert partial["metrics"]["progress"] == 0.5
        assert partial["metrics"]["progress_rel_to_total"] == 0.5

        diverged = [
            row
            for row in payload["rollouts"]
            if row["run_name"] == "test_run_1"
            and row["clipgt_id"] == "clip1"
            and row["rollout_id"] == "rollout2"
        ][0]
        assert diverged["status"] == "pass"
        assert diverged["passed"] is True
        assert diverged["score"] == pytest.approx(0.625)
        assert diverged["failure_reason"] is None

        passed = [
            row
            for row in payload["rollouts"]
            if row["run_name"] == "test_run_1"
            and row["clipgt_id"] == "clip2"
            and row["rollout_id"] == "rollout2"
        ][0]
        assert passed["status"] == "pass"
        assert passed["passed"] is True
        assert passed["score"] == pytest.approx(0.625)

    @patch("eval.aggregation.processing.plot_metrics_results")
    def test_scene_score_ignores_collision_and_offroad_after_deviation_cutoff(
        self,
        mock_plot: MagicMock,
        unified_metrics_df: pl.DataFrame,
        temp_directory: pathlib.Path,
    ) -> None:
        del mock_plot
        target = (
            (pl.col("run_name") == "test_run_1")
            & (pl.col("clipgt_id") == "clip1")
            & (pl.col("rollout_id") == "rollout1")
        )
        metrics_df = unified_metrics_df.with_columns(
            pl.when(target & (pl.col("name") == "collision_any"))
            .then(0.0)
            .when(target & (pl.col("name") == "dist_to_gt_trajectory"))
            .then(pl.when(pl.col("timestamps_us") >= 2000).then(4.0).otherwise(0.0))
            .otherwise(pl.col("values"))
            .alias("values")
        )

        aggregate_and_write_metrics_results_txt(
            metrics_df,
            output_path=str(temp_directory),
            additional_modifiers=[
                RemoveTimestepsAfterEvent(pl.col("dist_to_gt_trajectory") >= 4.0)
            ],
        )

        payload = json.loads((temp_directory / "results-summary.json").read_text())
        rollout = [
            row
            for row in payload["rollouts"]
            if row["run_name"] == "test_run_1"
            and row["clipgt_id"] == "clip1"
            and row["rollout_id"] == "rollout1"
        ][0]

        assert rollout["status"] == "pass"
        assert rollout["passed"] is True
        assert rollout["failure_reason"] is None
        assert rollout["score"] == pytest.approx(0.625)
        assert rollout["score_metrics"]["collision_at_fault"] == 0.0
        assert rollout["score_metrics"]["offroad"] == 0.0

    @patch("eval.aggregation.processing.plot_metrics_results")
    def test_scene_score_requires_score_metrics(
        self,
        mock_plot: MagicMock,
        unified_metrics_df: pl.DataFrame,
        temp_directory: pathlib.Path,
    ) -> None:
        """Test that enabled scene scoring fails fast when inputs are missing."""
        del mock_plot
        metrics_df = unified_metrics_df.filter(pl.col("name") != "gt_dist_traveled_m")

        with pytest.raises(ValueError, match="missing metric 'gt_dist_traveled_m'"):
            aggregate_and_write_metrics_results_txt(
                metrics_df,
                output_path=str(temp_directory),
            )

    @patch("eval.aggregation.processing.plot_metrics_results")
    def test_clamped_long_scene_does_not_get_short_gt_distance_override(
        self,
        mock_plot: MagicMock,
        temp_directory: pathlib.Path,
    ) -> None:
        """Large-deviation clipping must not change the full GT distance."""
        del mock_plot
        timestamps = [1000, 2000, 3000]
        metrics_config = [
            ("eval_relevant", "max", [1.0, 1.0, 1.0]),
            ("collision_any", "max", [0.0, 0.0, 0.0]),
            ("collision_front", "max", [0.0, 0.0, 0.0]),
            ("collision_lateral", "max", [0.0, 0.0, 0.0]),
            ("offroad", "max", [0.0, 0.0, 0.0]),
            ("img_is_black", "max", [0.0, 0.0, 0.0]),
            ("dist_traveled_m", "last", [0.0, 10.0, 100.0]),
            ("dist_to_gt_trajectory", "max", [0.0, 11.0, 11.0]),
            ("progress", "last", [0.0, 0.0, 1.0]),
            ("progress_rel_to_total", "last", [0.0, 0.0, 1.0]),
            ("progress_rel", "min", [1.0, 0.0, 1.0]),
            ("gt_dist_traveled_m", "last", [100.0, 100.0, 100.0]),
        ]
        metrics_df = pl.DataFrame(
            [
                {
                    "timestamps_us": ts,
                    "values": value,
                    "valid": True,
                    "name": name,
                    "time_aggregation": time_aggregation,
                    "clipgt_id": "long-scene",
                    "rollout_id": "rollout-1",
                    "run_uuid": "run-uuid",
                    "run_name": "run-name",
                }
                for name, time_aggregation, values in metrics_config
                for ts, value in zip(timestamps, values)
            ]
        )

        aggregate_and_write_metrics_results_txt(
            metrics_df,
            output_path=str(temp_directory),
            additional_modifiers=[
                RemoveTimestepsAfterEvent(pl.col("dist_to_gt_trajectory") >= 4.0)
            ],
        )

        payload = json.loads((temp_directory / "results-summary.json").read_text())
        rollout = payload["rollouts"][0]

        assert rollout["metrics"]["gt_dist_traveled_m"] == 100.0
        assert rollout["metrics"]["progress_clipped_rel"] == 0.0
        assert rollout["score_metrics"]["progress_score"] == 0.0
        assert rollout["score"] == 0.0

    @patch("eval.aggregation.processing.plot_metrics_results")
    def test_scene_score_can_be_disabled_with_missing_score_metrics(
        self,
        mock_plot: MagicMock,
        unified_metrics_df: pl.DataFrame,
        temp_directory: pathlib.Path,
    ) -> None:
        """Test that disabled scene scoring does not require score inputs."""
        del mock_plot
        metrics_df = unified_metrics_df.filter(
            pl.col("name") != "progress_rel_to_total"
        )

        aggregate_and_write_metrics_results_txt(
            metrics_df,
            output_path=str(temp_directory),
            scene_score_config=SceneScoreConfig(enabled=False),
            failed_rollouts=[
                {
                    "run_name": "test_run_1",
                    "clipgt_id": "clip_failed",
                    "rollout_id": "failed-0",
                    "error": "Runtime failure",
                }
            ],
        )

        payload = json.loads((temp_directory / "results-summary.json").read_text())
        assert payload["scene_score_enabled"] is False
        assert "score_criteria" not in payload
        unscored = [row for row in payload["rollouts"] if row["status"] == "unscored"][
            0
        ]
        assert unscored["passed"] is None
        assert unscored["score"] is None
        assert unscored["score_metrics"] is None
        failed = [
            row for row in payload["rollouts"] if row["clipgt_id"] == "clip_failed"
        ][0]
        assert failed["status"] == "fail"
        assert failed["passed"] is False
        assert "score" not in failed

    @patch("eval.aggregation.processing.plot_metrics_results")
    def test_results_summary_json_includes_failed_rollouts_without_metrics(
        self,
        mock_plot: MagicMock,
        unified_metrics_df: pl.DataFrame,
        temp_directory: pathlib.Path,
    ) -> None:
        """Test that runtime rollout failures are reported even without metrics rows."""
        del mock_plot

        aggregate_and_write_metrics_results_txt(
            unified_metrics_df,
            output_path=str(temp_directory),
            failed_rollouts=[
                {
                    "run_name": "test_run_1",
                    "clipgt_id": "clip_failed",
                    "rollout_id": "failed-0",
                    "error": "Maximum allowed size exceeded",
                }
            ],
        )

        payload = json.loads((temp_directory / "results-summary.json").read_text())
        failed = [
            row for row in payload["rollouts"] if row["clipgt_id"] == "clip_failed"
        ][0]

        assert failed["status"] == "fail"
        assert failed["passed"] is False
        assert failed["failure_reason"] == "Maximum allowed size exceeded"
        assert failed["score"] == 0.0
        assert failed["score_metrics"] == {
            "progress_clipped_rel": None,
            "progress_rel": None,
            "progress_score": 0.0,
            "collision_at_fault": None,
            "offroad": None,
            "dist_to_gt_trajectory": None,
            "gt_dist_traveled_m": None,
        }


class TestProcessedMetricDFs:
    """Test the ProcessedMetricDFs class methods."""

    @pytest.fixture
    def processed_dfs(self, unified_metrics_df: pl.DataFrame) -> ProcessedMetricDFs:
        """Create processed dataframes for testing."""
        return aggregate_and_write_metrics_results_txt(unified_metrics_df)

    @pytest.fixture
    def processed_dfs_with_img_black(
        self, modified_metrics_df_with_img_black: pl.DataFrame
    ) -> ProcessedMetricDFs:
        """Create processed dataframes with img_black trajectory removed."""
        return aggregate_and_write_metrics_results_txt(
            modified_metrics_df_with_img_black
        )

    def test_get_removed_rows_quantitative(
        self, processed_dfs: ProcessedMetricDFs
    ) -> None:
        """Test getting removed rows with specific quantitative checks."""
        removed = processed_dfs.get_removed_rows()

        # With unified_metrics_df fixture and DEFAULT_MODIFIERS:
        # - RemoveTimestepsAfterEvent(offroad_or_collision > 0) removes timestamp 3000 for all trajectories
        # - offroad_or_collision = collision_any OR offroad, which is 1.0 at timestamp 2000
        # - This removes timestamp 3000 for all 8 trajectories = 8 rows
        assert removed.height == 8

        # All removed rows should be from timestamp 3000
        assert removed["timestamps_us"].unique().to_list() == [3000]

        # Check that removed rows have valid trajectory metadata
        assert removed["run_uuid"].null_count() == 0
        assert removed["run_name"].null_count() == 0
        assert removed["clipgt_id"].null_count() == 0

        # Should have valid run names from the test data
        valid_run_names = {"test_run_1", "test_run_2"}
        assert set(removed["run_name"].unique().to_list()).issubset(valid_run_names)

    def test_get_removed_trajectories_quantitative(
        self,
        processed_dfs: ProcessedMetricDFs,
        processed_dfs_with_img_black: ProcessedMetricDFs,
    ) -> None:
        """Test getting removed trajectories with specific counts."""
        # Test with original data - should have 0 removed trajectories
        removed_trajectories = processed_dfs.get_removed_trajectories()
        assert removed_trajectories.height == 0

        # Test with modified data where one trajectory has img_is_black > 0
        removed_trajectories_modified = (
            processed_dfs_with_img_black.get_removed_trajectories()
        )

        # Should have exactly 1 removed trajectory: test_run_1 + clip1 + rollout1
        assert removed_trajectories_modified.height == 1

        # Check that the removed trajectory is exactly the one we expect
        expected_removed = removed_trajectories_modified.select(
            "run_name", "clipgt_id", "rollout_id"
        )

        assert expected_removed["run_name"][0] == "test_run_1"
        assert expected_removed["clipgt_id"][0] == "clip1"
        assert expected_removed["rollout_id"][0] == "rollout1"

        # Verify all required columns are present
        expected_columns = {
            "run_uuid",
            "run_name",
            "clipgt_id",
            "rollout_id",
        }
        assert set(removed_trajectories_modified.columns).issuperset(expected_columns)

        # Verify the remaining trajectories count
        # Original: 8 trajectories, Removed: 1 trajectory, Remaining: 7 trajectories
        assert processed_dfs_with_img_black.df_wide_avg_t.shape[0] == 7

    def test_get_rollouts_per_clip_quantitative(
        self,
        processed_dfs: ProcessedMetricDFs,
        processed_dfs_with_img_black: ProcessedMetricDFs,
    ) -> None:
        """Test rollouts per clip with expected counts from unified_metrics_df."""
        # Test original data - all clips should have 2 rollouts
        rollouts_per_clip = processed_dfs.get_rollouts_per_clip()

        # All clips in original data should have 2 rollouts
        assert all(
            n_rollouts == 2 for n_rollouts in rollouts_per_clip["n_rollouts"].to_list()
        )

        # Test modified data - one specific clip should have 1 rollout, others should have 2
        rollouts_per_clip_modified = (
            processed_dfs_with_img_black.get_rollouts_per_clip()
        )

        # Convert to a more convenient format for checking specific values
        rollouts_dict = {}
        for row in rollouts_per_clip_modified.iter_rows(named=True):
            key = (row["run_name"], row["clipgt_id"])
            rollouts_dict[key] = row["n_rollouts"]

        # Check specific expected values:
        # test_run_1 + clip1: 1 rollout (rollout1 was removed, rollout2 remains)
        assert rollouts_dict[("test_run_1", "clip1")] == 1

        # All other clips should have 2 rollouts
        assert rollouts_dict[("test_run_1", "clip2")] == 2
        assert rollouts_dict[("test_run_2", "clip1")] == 2
        assert rollouts_dict[("test_run_2", "clip2")] == 2
