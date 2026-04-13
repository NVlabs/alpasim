# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging
from abc import ABC, abstractmethod

import polars as pl

logger = logging.getLogger(__name__)


def get_removed_rows(
    df_wide_before: pl.DataFrame, df_wide_after: pl.DataFrame
) -> pl.DataFrame:
    """Get the filtered dataframe."""
    return df_wide_before.join(
        df_wide_after,
        on=["trajectory_uid", "timestamps_us"],
        how="anti",
    )


class MetricAggregationModifiers(ABC):

    @abstractmethod
    def apply(self, df_wide: pl.DataFrame) -> pl.DataFrame:
        """Apply the filter to the dataframe.

        Args:
          df_wide: The dataframe to filter.
              Index: trajectory_uid, timestamps_us
              Other columns: metrics

        Returns:
          The filtered dataframe.
        """
        pass

    def apply_agg_function_df(self, agg_function_df: pl.DataFrame) -> pl.DataFrame:
        """Modify the agg function df."""
        return agg_function_df

    def __init__(self, event: pl.Expr):
        self.event = event
        self.n_last_modified_rows = 0
        self.n_last_modified_trajectories = 0

    def __call__(
        self, df_wide_before: pl.DataFrame, agg_function_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        df_wide = self.apply(df_wide_before)
        agg_function_df = self.apply_agg_function_df(agg_function_df)
        self.n_last_modified_rows = len(df_wide_before) - len(df_wide)
        removed_rows = (
            get_removed_rows(df_wide_before, df_wide).select("trajectory_uid").unique()
        )
        self.n_last_modified_trajectories = len(removed_rows)
        logger.info(
            "Applying modification: %s",
            self,
        )
        return df_wide, agg_function_df

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.event}) "
            f"±({self.n_last_modified_rows} rows, "
            f"{self.n_last_modified_trajectories} trajectories)"
        )


def remove_timesteps_before_or_after_event(
    df_wide: pl.DataFrame, event: pl.Expr, remove_before: bool = False
) -> pl.DataFrame:
    """Remove timestamps before or after the first occurrence of the event.

    Args:
      df_wide: The dataframe to filter.
          Index: trajectory_uid, timestamps_us
          Other columns: metrics
      event: The event to filter after, e.g. `pl.col("offroad_or_collision") > 0.0`
      remove_before: Whether to remove the rows before the event. If False, we
          keep the rows before the event and remove the rows after the event.

    Returns:
      The filtered dataframe.
    """
    df_wide = df_wide.sort(["trajectory_uid", "timestamps_us"])
    df_wide = df_wide.with_columns(
        pl.col("timestamps_us")
        .filter(event)
        .first()
        .over("trajectory_uid")
        .alias("first_event_timestamp")
    )

    if remove_before:
        filter_event = pl.col("timestamps_us") >= pl.col("first_event_timestamp")
    else:
        filter_event = pl.col("timestamps_us") <= pl.col("first_event_timestamp")

    return df_wide.filter(
        filter_event | pl.col("first_event_timestamp").is_null()
    ).drop("first_event_timestamp")


class RemoveTimestepsAfterEvent(MetricAggregationModifiers):

    def apply(self, df_wide: pl.DataFrame) -> pl.DataFrame:
        return remove_timesteps_before_or_after_event(
            df_wide, self.event, remove_before=False
        )


class RemoveTimestepsBeforeEvent(MetricAggregationModifiers):

    def apply(self, df_wide: pl.DataFrame) -> pl.DataFrame:
        return remove_timesteps_before_or_after_event(
            df_wide, self.event, remove_before=True
        )


class AddCombinedEvent(MetricAggregationModifiers):

    def __init__(self, event: pl.Expr, name: str, time_aggregation: str):
        super().__init__(event)
        self.name = name
        self.time_aggregation = time_aggregation

    def apply(self, df_wide: pl.DataFrame) -> pl.DataFrame:
        return df_wide.with_columns(self.event.alias(self.name).cast(pl.Float64))

    def apply_agg_function_df(self, agg_function_df: pl.DataFrame) -> pl.DataFrame:
        return pl.concat(
            [
                agg_function_df,
                pl.DataFrame(
                    {"name": [self.name], "time_aggregation": [self.time_aggregation]}
                ),
            ],
            how="vertical",
        )

    def __str__(self):
        return f"'{self.name}': " + super().__str__()


class RemoveTrajectoryWithEvent(MetricAggregationModifiers):
    """Remove entire trajectories where `event` occurs at least once.

    A trajectory is kept only if **all** timesteps evaluate to `False`
    (or `NULL`) for the provided `event` expression.
    """

    def apply(self, df_wide: pl.DataFrame) -> pl.DataFrame:
        event_per_traj = (
            pl.coalesce(self.event, pl.lit(False))  # NULL → False
            .max()
            .over("trajectory_uid")
        )
        return df_wide.filter(~event_per_traj)


class RemoveTimestepsBeforeFirstDriven(MetricAggregationModifiers):
    """Drop rows whose timestamp precedes the ego's first driven step.

    Uses the ``first_driven_timestamp_us`` column (attached to the metrics
    dataframe by ``create_metrics_dataframe`` when the caller provides it)
    to filter out prerun / warmup timesteps that otherwise produce spurious
    offroad / collision events on the aggregated score.

    Acts as a no-op for dataframes without the column, which keeps this
    safe to include in the default pipeline while also letting callers
    (e.g. ground-truth baseline runs) opt out by simply not populating it.
    """

    def __init__(self) -> None:
        super().__init__(event=pl.lit(True))

    def apply(self, df_wide: pl.DataFrame) -> pl.DataFrame:
        if "first_driven_timestamp_us" not in df_wide.columns:
            return df_wide
        return df_wide.filter(
            pl.col("first_driven_timestamp_us").is_null()
            | (pl.col("timestamps_us") >= pl.col("first_driven_timestamp_us"))
        )
