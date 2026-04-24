# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import polars as pl
import pytest

from eval.aggregation.modifiers import (
    AddCombinedEvent,
    RemoveTimestepsAfterEvent,
    RemoveTimestepsBeforeEvent,
    RemoveTrajectoryWithEvent,
    get_removed_rows,
    remove_timesteps_before_or_after_event,
)


# Fixtures for commonly used dataframes
@pytest.fixture
def basic_df() -> pl.DataFrame:
    """Basic dataframe with trajectory_uid, timestamps_us, and metric_a."""
    return pl.DataFrame(
        {
            "trajectory_uid": [1, 1, 2, 2, 3],
            "timestamps_us": [1000, 2000, 1000, 2000, 1000],
            "metric_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def collision_event_df() -> pl.DataFrame:
    """Dataframe with collision events for testing event-based modifiers."""
    return pl.DataFrame(
        {
            "trajectory_uid": [1, 1, 1, 2, 2, 2, 3, 3],
            "timestamps_us": [1000, 2000, 3000, 1000, 2000, 3000, 1000, 2000],
            "collision": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "other_metric": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )


@pytest.fixture
def two_trajectory_df() -> pl.DataFrame:
    """Generic two-trajectory dataframe for collision-related tests."""
    return pl.DataFrame(
        {
            "trajectory_uid": [1, 1, 2, 2],
            "timestamps_us": [1000, 2000, 1000, 2000],
            "collision": [0.0, 0.0, 0.0, 0.0],
            "other_metric": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def empty_df() -> pl.DataFrame:
    """Empty dataframe with correct schema."""
    return pl.DataFrame(
        {
            "trajectory_uid": [],
            "timestamps_us": [],
            "collision": [],
        },
        schema={
            "trajectory_uid": pl.Int64,
            "timestamps_us": pl.Int64,
            "collision": pl.Float64,
        },
    )


@pytest.fixture
def single_row_df() -> pl.DataFrame:
    """Single row dataframe for edge case testing."""
    return pl.DataFrame(
        {
            "trajectory_uid": [1],
            "timestamps_us": [1000],
            "collision": [1.0],
        }
    )


@pytest.fixture
def multi_event_df() -> pl.DataFrame:
    """Dataframe with multiple events in same trajectory."""
    return pl.DataFrame(
        {
            "trajectory_uid": [1, 1, 1, 1],
            "timestamps_us": [1000, 2000, 3000, 4000],
            "collision": [0.0, 1.0, 0.0, 1.0],
        }
    )


class TestGetRemovedRows:
    """Test the get_removed_rows utility function."""

    def test_get_removed_rows_basic(self, basic_df: pl.DataFrame) -> None:
        """Test basic functionality of get_removed_rows."""
        basic_df_subset = pl.DataFrame(
            {
                "trajectory_uid": [1, 2],
                "timestamps_us": [1000, 1000],
                "metric_a": [1.0, 3.0],
            }
        )

        removed = get_removed_rows(basic_df, basic_df_subset)

        expected = pl.DataFrame(
            {
                "trajectory_uid": [1, 2, 3],
                "timestamps_us": [2000, 2000, 1000],
                "metric_a": [2.0, 4.0, 5.0],
            }
        )

        assert removed.equals(expected)

    def test_get_removed_rows_empty_removal(self, basic_df: pl.DataFrame) -> None:
        """Test when no rows are removed."""
        removed = get_removed_rows(basic_df, basic_df)
        assert removed.height == 0
        assert removed.columns == ["trajectory_uid", "timestamps_us", "metric_a"]

    def test_get_removed_rows_all_removed(self, basic_df: pl.DataFrame) -> None:
        """Test when all rows are removed."""
        df_after = pl.DataFrame(
            {
                "trajectory_uid": [],
                "timestamps_us": [],
                "metric_a": [],
            },
            schema=basic_df.schema,
        )

        removed = get_removed_rows(basic_df, df_after)
        assert removed.equals(basic_df)


class TestRemoveTimestepsBeforeOrAfterEvent:
    """Test the remove_timesteps_before_or_after_event function."""

    def test_remove_after_event(self, collision_event_df: pl.DataFrame) -> None:
        """Test removing timesteps after first event occurrence."""
        event = pl.col("collision") > 0.0

        result = remove_timesteps_before_or_after_event(
            collision_event_df, event, remove_before=False
        )

        # For trajectory_uid=1: keep timestamps 1000, 2000 (stop at collision at 2000)
        # For trajectory_uid=2: keep timestamps 1000, 2000, 3000 (stop at collision at 3000)
        # For trajectory_uid=3: keep all timestamps (no collision)
        expected_rows = [
            (1, 1000, 0.0, 10),
            (1, 2000, 1.0, 20),
            (2, 1000, 0.0, 40),
            (2, 2000, 0.0, 50),
            (2, 3000, 1.0, 60),
            (3, 1000, 0.0, 70),
            (3, 2000, 0.0, 80),
        ]

        result_sorted = result.sort(["trajectory_uid", "timestamps_us"])

        assert result_sorted.height == len(expected_rows)
        for i, (traj_uid, timestamp, collision, other) in enumerate(expected_rows):
            row = result_sorted.row(i)
            assert row[0] == traj_uid
            assert row[1] == timestamp
            assert row[2] == collision
            assert row[3] == other

    def test_remove_before_event(self, collision_event_df: pl.DataFrame) -> None:
        """Test removing timesteps before first event occurrence."""
        event = pl.col("collision") > 0.0

        result = remove_timesteps_before_or_after_event(
            collision_event_df, event, remove_before=True
        )

        # For trajectory_uid=1: keep timestamps 2000, 3000 (start from collision at 2000)
        # For trajectory_uid=2: keep timestamp 3000 (start from collision at 3000)
        # For trajectory_uid=3: keep all timestamps (no collision)
        expected_rows = [
            (1, 2000, 1.0, 20),
            (1, 3000, 0.0, 30),
            (2, 3000, 1.0, 60),
            (3, 1000, 0.0, 70),
            (3, 2000, 0.0, 80),
        ]

        result_sorted = result.sort(["trajectory_uid", "timestamps_us"])

        assert result_sorted.height == len(expected_rows)
        for i, (traj_uid, timestamp, collision, other) in enumerate(expected_rows):
            row = result_sorted.row(i)
            assert row[0] == traj_uid
            assert row[1] == timestamp
            assert row[2] == collision
            assert row[3] == other

    def test_no_event_occurs(self, two_trajectory_df: pl.DataFrame) -> None:
        """Test when the event never occurs for any trajectory."""
        event = pl.col("collision") > 0.0

        # When no event occurs, all data should be kept
        result_after = remove_timesteps_before_or_after_event(
            two_trajectory_df, event, remove_before=False
        )
        result_before = remove_timesteps_before_or_after_event(
            two_trajectory_df, event, remove_before=True
        )

        assert result_after.equals(
            two_trajectory_df.sort(["trajectory_uid", "timestamps_us"])
        )
        assert result_before.equals(
            two_trajectory_df.sort(["trajectory_uid", "timestamps_us"])
        )

    def test_event_at_first_timestamp(self) -> None:
        """Test when event occurs at the first timestamp."""
        df = pl.DataFrame(
            {
                "trajectory_uid": [1, 1, 1],
                "timestamps_us": [1000, 2000, 3000],
                "collision": [1.0, 0.0, 0.0],
                "other_metric": [10, 20, 30],
            }
        )

        event = pl.col("collision") > 0.0

        # Remove after: should keep only first timestamp
        result_after = remove_timesteps_before_or_after_event(
            df, event, remove_before=False
        )
        assert result_after.height == 1
        assert result_after.row(0) == (1, 1000, 1.0, 10)

        # Remove before: should keep all timestamps
        result_before = remove_timesteps_before_or_after_event(
            df, event, remove_before=True
        )
        assert result_before.height == 3

    def test_event_at_last_timestamp(self) -> None:
        """Test when event occurs at the last timestamp."""
        df = pl.DataFrame(
            {
                "trajectory_uid": [1, 1, 1],
                "timestamps_us": [1000, 2000, 3000],
                "collision": [0.0, 0.0, 1.0],
                "other_metric": [10, 20, 30],
            }
        )

        event = pl.col("collision") > 0.0

        # Remove after: should keep all timestamps
        result_after = remove_timesteps_before_or_after_event(
            df, event, remove_before=False
        )
        assert result_after.height == 3

        # Remove before: should keep only last timestamp
        result_before = remove_timesteps_before_or_after_event(
            df, event, remove_before=True
        )
        assert result_before.height == 1
        assert result_before.row(0) == (1, 3000, 1.0, 30)


class TestRemoveTimestepsAfterEvent:
    """Test the RemoveTimestepsAfterEvent modifier."""

    def test_remove_timesteps_after_event(
        self, collision_event_df: pl.DataFrame
    ) -> None:
        """Test removing timesteps after an event."""
        modifier = RemoveTimestepsAfterEvent(pl.col("collision") > 0.0)

        # Create a simple agg_function_df for the __call__ method to test tracking
        agg_df = pl.DataFrame(
            {
                "name": ["collision"],
                "time_aggregation": ["max"],
            }
        )

        # Use __call__ method to test both functionality and tracking
        result, agg_result = modifier(collision_event_df, agg_df)

        # Should keep timestamps up to and including the collision event
        # For trajectory_uid=1: keep timestamps 1000, 2000 (stop at collision at 2000)
        # For trajectory_uid=2: keep timestamps 1000, 2000, 3000 (stop at collision at 3000)
        # For trajectory_uid=3: keep all timestamps (no collision)
        expected_rows = [
            (1, 1000, 0.0, 10),
            (1, 2000, 1.0, 20),
            (2, 1000, 0.0, 40),
            (2, 2000, 0.0, 50),
            (2, 3000, 1.0, 60),
            (3, 1000, 0.0, 70),
            (3, 2000, 0.0, 80),
        ]

        result_sorted = result.sort(["trajectory_uid", "timestamps_us"])
        assert result_sorted.height == len(expected_rows)

        for i, expected_row in enumerate(expected_rows):
            actual_row = result_sorted.row(i)
            assert actual_row == expected_row

        # Test tracking: 1 row removed (trajectory 1's timestamp 3000), 1 trajectory affected
        assert modifier.n_last_modified_rows == 1
        assert modifier.n_last_modified_trajectories == 1

    def test_multiple_events_same_trajectory(
        self, multi_event_df: pl.DataFrame
    ) -> None:
        """Test behavior when multiple events occur in the same trajectory."""
        event = pl.col("collision") > 0.0
        modifier = RemoveTimestepsAfterEvent(event)
        result = modifier.apply(multi_event_df)

        # Should only consider the FIRST event occurrence
        # Should keep up to first event at timestamp 2000
        assert result.height == 2
        assert result["timestamps_us"].to_list() == [1000, 2000]


class TestRemoveTimestepsBeforeEvent:
    """Test the RemoveTimestepsBeforeEvent modifier."""

    def test_remove_timesteps_before_event(
        self, collision_event_df: pl.DataFrame
    ) -> None:
        """Test removing timesteps before an event."""
        modifier = RemoveTimestepsBeforeEvent(pl.col("collision") > 0.0)

        # Create a simple agg_function_df for the __call__ method to test tracking
        agg_df = pl.DataFrame(
            {
                "name": ["collision"],
                "time_aggregation": ["max"],
            }
        )

        # Use __call__ method to test both functionality and tracking
        result, agg_result = modifier(collision_event_df, agg_df)

        # Should keep timestamps from the collision event onwards
        # For trajectory_uid=1: keep timestamps 2000, 3000 (start from collision at 2000)
        # For trajectory_uid=2: keep timestamp 3000 (start from collision at 3000)
        # For trajectory_uid=3: keep all timestamps (no collision)
        expected_rows = [
            (1, 2000, 1.0, 20),
            (1, 3000, 0.0, 30),
            (2, 3000, 1.0, 60),
            (3, 1000, 0.0, 70),
            (3, 2000, 0.0, 80),
        ]

        result_sorted = result.sort(["trajectory_uid", "timestamps_us"])
        assert result_sorted.height == len(expected_rows)

        for i, expected_row in enumerate(expected_rows):
            actual_row = result_sorted.row(i)
            assert actual_row == expected_row

        # Test tracking: 3 rows removed, 2 trajectories affected
        assert modifier.n_last_modified_rows == 3
        assert modifier.n_last_modified_trajectories == 2

    def test_multiple_events_same_trajectory(
        self, multi_event_df: pl.DataFrame
    ) -> None:
        """Test behavior when multiple events occur in the same trajectory."""
        event = pl.col("collision") > 0.0
        modifier = RemoveTimestepsBeforeEvent(event)
        result = modifier.apply(multi_event_df)

        # Should only consider the FIRST event occurrence
        # Should keep from first event at timestamp 2000 onwards
        assert result.height == 3
        assert result["timestamps_us"].to_list() == [2000, 3000, 4000]


class TestAddCombinedEvent:
    """Test the AddCombinedEvent modifier."""

    def test_add_combined_event_basic(self) -> None:
        """Test adding a combined event column."""
        collision_offroad_df = pl.DataFrame(
            {
                "trajectory_uid": [1, 1, 2, 2],
                "timestamps_us": [1000, 2000, 1000, 2000],
                "collision": [0.0, 1.0, 1.0, 0.0],
                "offroad": [1.0, 0.0, 0.0, 1.0],
            }
        )

        event = (pl.col("collision") > 0.0) | (pl.col("offroad") > 0.0)
        modifier = AddCombinedEvent(event, "collision_or_offroad", "max")

        # Create a simple agg_function_df for the __call__ method to test tracking
        agg_df = pl.DataFrame(
            {
                "name": ["collision"],
                "time_aggregation": ["max"],
            }
        )

        # Use __call__ method to test both functionality and tracking
        result, agg_result = modifier(collision_offroad_df, agg_df)

        # Check that the new column is added
        assert "collision_or_offroad" in result.columns
        assert result["collision_or_offroad"].dtype == pl.Float64

        # Check the values
        expected_values = [1.0, 1.0, 1.0, 1.0]  # All should be 1.0 due to OR condition
        assert result["collision_or_offroad"].to_list() == expected_values

        # Test tracking: AddCombinedEvent doesn't remove rows, only adds columns
        assert modifier.n_last_modified_rows == 0
        assert modifier.n_last_modified_trajectories == 0

        # Verify the agg function was added
        assert agg_result.height == 2  # Original + new event
        assert "collision_or_offroad" in agg_result["name"].to_list()

    def test_add_combined_event_agg_function_df(self) -> None:
        """Test modification of aggregation function dataframe."""
        event = pl.col("collision") > 0.0
        modifier = AddCombinedEvent(event, "new_event", "sum")

        agg_df = pl.DataFrame(
            {
                "name": ["existing_metric"],
                "time_aggregation": ["mean"],
            }
        )

        result = modifier.apply_agg_function_df(agg_df)

        # Should add the new event to the aggregation functions
        expected = pl.DataFrame(
            {
                "name": ["existing_metric", "new_event"],
                "time_aggregation": ["mean", "sum"],
            }
        )

        assert result.equals(expected)

    def test_add_combined_event_complex_expression(
        self, collision_event_df: pl.DataFrame
    ) -> None:
        """Test with a more complex expression."""
        # Create a complex event: collision OR high other_metric value
        event = (pl.col("collision") > 0.0) | (pl.col("other_metric") > 50)
        modifier = AddCombinedEvent(event, "collision_or_high_metric", "max")

        result = modifier.apply(collision_event_df)

        # Expected values based on collision_event_df:
        # trajectory_uid=1: [0.0, 1.0, 0.0] -> collision=False, True, False; other_metric=10, 20, 30 -> all < 50
        # So: [False, True, False] -> [0.0, 1.0, 0.0]
        # trajectory_uid=2: [0.0, 0.0, 1.0] -> collision=False, False, True; other_metric=40, 50, 60 -> 60 > 50
        # So: [False, False, True] -> [0.0, 0.0, 1.0]
        # trajectory_uid=3: [0.0, 0.0] -> collision=False, False; other_metric=70, 80 -> both > 50
        # So: [True, True] -> [1.0, 1.0]
        expected_values = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        assert result["collision_or_high_metric"].to_list() == expected_values


class TestRemoveTrajectoryWithEvent:
    """Test the RemoveTrajectoryWithEvent modifier."""

    def test_filter_trajectory_with_event(
        self, collision_event_df: pl.DataFrame
    ) -> None:
        """Test filtering trajectories that contain an event."""
        # Filter out trajectories that have any collision
        modifier = RemoveTrajectoryWithEvent(pl.col("collision") > 0.0)
        result = modifier.apply(collision_event_df)

        # Should keep only trajectory 3 (no collisions)
        expected_trajectory_uids = [3, 3]
        actual_trajectory_uids = result["trajectory_uid"].to_list()
        assert actual_trajectory_uids == expected_trajectory_uids

    def test_filter_trajectory_no_events(self, two_trajectory_df: pl.DataFrame) -> None:
        """Test when no trajectories have events."""
        modifier = RemoveTrajectoryWithEvent(pl.col("collision") > 0.0)
        result = modifier.apply(two_trajectory_df)

        # Should keep all trajectories since none have collisions
        assert result.height == 4
        assert result.equals(two_trajectory_df)

    def test_filter_trajectory_all_have_events(self) -> None:
        """Test when all trajectories have events."""
        all_collision_df = pl.DataFrame(
            {
                "trajectory_uid": [1, 1, 2, 2],
                "timestamps_us": [1000, 2000, 1000, 2000],
                "collision": [1.0, 0.0, 0.0, 1.0],
                "other_metric": [10, 20, 30, 40],
            }
        )

        modifier = RemoveTrajectoryWithEvent(pl.col("collision") > 0.0)
        result = modifier.apply(all_collision_df)

        # Should filter out all trajectories
        assert result.height == 0
        assert result.columns == all_collision_df.columns
