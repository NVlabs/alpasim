# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import seaborn_polars as snl
from rich.console import Console
from rich.table import Table

from eval.aggregation.modifiers import (
    AddCombinedEvent,
    MetricAggregationModifiers,
    RemoveTimestepsAfterEvent,
    RemoveTrajectoryWithEvent,
    get_removed_rows,
)
from eval.data import AggregationType

DEFAULT_MODIFIERS = [
    AddCombinedEvent(
        event=(pl.col("collision_any") > 0.0) | (pl.col("offroad") > 0.0),
        name="offroad_or_collision",
        time_aggregation="max",
    ),
    AddCombinedEvent(
        event=(pl.col("collision_front") > 0.0) | (pl.col("collision_lateral") > 0),
        name="collision_at_fault",
        time_aggregation="max",
    ),
    AddCombinedEvent(
        event=(pl.col("collision_at_fault") > 0.0) | (pl.col("offroad") > 0.0),
        name="offroad_or_collision_at_fault",
        time_aggregation="max",
    ),
    AddCombinedEvent(
        event=(
            pl.col("timestamps_us")
            - pl.col("timestamps_us").min().over("trajectory_uid")
        )
        / 20e6,
        name="duration_frac_20s",
        time_aggregation="last",
    ),
    RemoveTrajectoryWithEvent(pl.col("img_is_black") > 0.0),
    RemoveTimestepsAfterEvent(pl.col("offroad_or_collision") > 0.0),
]


def add_rollout_and_trajectory_uids(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add rollout and trajectory uids to the per_timestep_df.

    The rollout_uid (not to be confused with `rollout_id`!) is a unique id
    across scenes. So if `n_rollouts` was greater than one, we can use this
    to compute e.g. the std by treating each rollout as a separate simulation.

    The trajectory_uid is a unique id across scenes and runs. It's used for
    easier grouped computations in the internal processing of this script, but
    not needed by the end user.

    Returns:
        combined_per_timestep_df: The input dataframe with trajectory uids.
        trajectory_uid_df: A dataframe with the trajectory uids, rollout uids,
        and other metadata. Used to later join in the the rollout_uid and run
        name, after processing based on the trajectory_uid.
    """
    df = df.with_columns(
        # Different scenes share the same rollout_uid within a run.
        # Allows for computing the std across rollouts, averaging over clips.
        # Note: batch_id is excluded because it varies across jobs and doesn't
        # indicate a different rollout - only rollout_id determines rollout identity.
        pl.concat_str(["run_uuid", "rollout_id"], separator="_").alias("rollout_uid"),
        # Unique even across scenes, i.e. truely unique per rollout.
        pl.concat_str(
            ["run_uuid", "clipgt_id", "batch_id", "rollout_id"], separator="_"
        ).alias("trajectory_uid"),
    ).with_columns(
        pl.col("rollout_uid").rank("dense").over("clipgt_id").alias("rollout_uid"),
    )

    trajectory_uid_df = df.select(
        pl.col("trajectory_uid"),
        pl.col("rollout_uid"),
        pl.col("run_name"),
        pl.col("run_uuid"),
        pl.col("clipgt_id"),
        pl.col("batch_id"),
        pl.col("rollout_id"),
    ).unique()

    df = df.drop(
        ["run_name", "run_uuid", "clipgt_id", "batch_id", "rollout_id", "rollout_uid"]
    )

    return df, trajectory_uid_df


def aggregate_over_clips(
    df_wide_avg_t: pl.DataFrame, trajectory_uid_df: pl.DataFrame
) -> pl.DataFrame:
    """Aggregate the metrics over clips.

    Note that we don't yet aggregate over "rollout_uids", allowing treating
    those as a separate simulation, hence allow to compute error bars.

    Args:
        df_wide_avg_t: The dataframe with the aggregated metrics over time.
        trajectory_uid_df: The dataframe with the trajectory uids.

    Returns:
        df_wide_avg_t_clip: The dataframe with the aggregated metrics over time
        and clips.
    """
    # Join "run_uuid" and "rollout_uid" back into the dataframe:
    df_wide_avg_t = df_wide_avg_t.join(
        trajectory_uid_df.select(
            pl.col("trajectory_uid"), pl.col("rollout_uid"), pl.col("run_uuid")
        ),
        on="trajectory_uid",
        how="left",
    ).drop("trajectory_uid")

    # Average over clips, `rollout_uid` is unique for each [run, batch, rollout]
    df_wide_avg_t_clip = df_wide_avg_t.group_by("run_uuid", "rollout_uid").agg(
        pl.col("*").mean(),
        pl.len().alias("n_clips"),
    )
    df_wide_avg_t_clip = df_wide_avg_t_clip.join(
        get_avg_dist_between_incidents(df_wide_avg_t),
        on=["run_uuid", "rollout_uid"],
        how="left",
    )
    return df_wide_avg_t_clip


def write_metrics_results_txt(
    df_wide_avg_t_clip_rollout: pl.DataFrame,
    agg_function_df: pl.DataFrame,
    output_path: str,
    modifiers: list[MetricAggregationModifiers] | None = None,
) -> None:
    """Write the metrics results to a txt file.

    Args:
        df_wide_avg_t_clip_rollout: The dataframe with metrics aggregated over
            time,clips and rollouts, i.e. with one line per run.
        agg_function_df: The dataframe with the aggregation functions.
        output_path: The path to the output folder.
    """
    console = Console(record=True, width=160)

    if modifiers:
        console.rule("Modifiers applied:")
        for modifier in modifiers:
            console.print(modifier)

    console.rule("Results")

    column_names = df_wide_avg_t_clip_rollout.columns
    metric_names = [
        col
        for col in column_names
        if col not in ["run_name", "run_uuid", "n_rollouts", "n_clips"]
        and not col.endswith("_std")  # These are paired, each metric has a std.
    ]
    metric_names.sort()
    for row in df_wide_avg_t_clip_rollout.iter_rows(named=True):

        table = Table(
            title=f"Run: {row['run_name']}",
            caption=(
                f"n_clips: {row['n_clips']}, n_rollouts/clip: {row['n_rollouts']}\n"
                "Std deviation is over rollouts."
            ),
        )
        table.add_column("Metric Name", justify="left")
        table.add_column("Metric Value", justify="center")
        table.add_column("Time Aggregation", justify="center")
        for col_name in metric_names:
            val = row[col_name]
            agg_str = agg_function_df.filter(pl.col("name") == col_name)[
                "time_aggregation"
            ].to_list()
            agg_str = agg_str[0] if agg_str else None
            val_str = f"{val:.2f}" if val is not None else "N/A"
            table.add_row(
                col_name,
                (
                    f"{val_str} ± {row[f'{col_name}_std']:.2f}"
                    if row[f"{col_name}_std"] is not None
                    else val_str
                ),
                agg_str,
            )

        console.print(table)

    output_file = os.path.join(output_path, "metrics_results.txt")
    console.save_text(output_file)

    print(f"Results saved to {output_file}.")


def rename_legend_handles(ax: plt.Axes, trajectory_uid_df: pl.DataFrame) -> None:
    """Rename the legend handles to the run names.

    Motivation: Run names aren't guaranteed to be unique. By doing the renaming
    only in the legend, and using run_uuids until then, we guarantee that
    nothing is meshed together by mistake.

    Args:
        ax: The axis to rename the legend handles for.
        trajectory_uid_df: The dataframe with the trajectory uids and run names.
    """
    run_uuid_to_name_dicts = (
        trajectory_uid_df.select(pl.col("run_uuid"), pl.col("run_name"))
        .unique()
        .to_dicts()
    )
    run_uuid_to_name = {d["run_uuid"]: d["run_name"] for d in run_uuid_to_name_dicts}

    # Get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Apply mapping to create new labels
    new_labels = [run_uuid_to_name.get(label, label) for label in labels]

    # Create new legend
    ax.legend(handles=handles, labels=new_labels)


def plot_metrics_results(
    df_wide_avg_t_clip: pl.DataFrame,
    trajectory_uid_df: pl.DataFrame,
    output_path: str,
    errorbar: str = "ci",
) -> None:
    """Plot the metrics results.

    Args:
        df_wide_avg_t_clip: The dataframe with metrics aggregated over
            time and clips, NOT over rollouts (to allow for error bars).
        trajectory_uid_df: The dataframe with the trajectory uids and run names.
        output_path: The path to the output folder.
        errorbar: The type of error bar to plot.
    """
    sns.set_theme()
    df_long_avg = df_wide_avg_t_clip.drop("n_clips").unpivot(
        index=["run_uuid", "rollout_uid"],
    )

    # We do the renamin back to run_name last and only for plotting to not
    fig, ax = plt.subplots()

    snl.barplot(
        df_long_avg, x="variable", y="value", hue="run_uuid", errorbar=errorbar, ax=ax
    )
    plt.xticks(rotation=45, ha="right")
    rename_legend_handles(ax, trajectory_uid_df)
    ax.set_xlabel("")
    ax.set_ylabel("")
    output_file = os.path.join(output_path, "metrics_results.png")
    fig.savefig(output_file)
    print(f"Results plotted to {output_file}.")


@dataclass
class UnprocessedMetricsDFs:
    """Unprocessed metrics dataframe."""

    # Original metrics. Long format, unaggregated.
    unprocessed_df: pl.DataFrame

    def save_to(self, directory: pathlib.Path) -> None:
        self.unprocessed_df.write_parquet(directory / "metrics_unprocessed.parquet")

    @staticmethod
    def load_from(directory: pathlib.Path) -> "UnprocessedMetricsDFs":
        """Load the unprocessed metrics dataframe from a directory."""
        df = pl.read_parquet(directory / "metrics_unprocessed.parquet")

        # Backwards compatibility from when we weren't writing the
        # unprocessed df to parquet.
        if "run_name" not in df.columns:
            trajectory_uid_df = pl.read_parquet(directory / "trajectory_uid.parquet")
            df = df.join(
                trajectory_uid_df,
                on="trajectory_uid",
                how="left",
            ).drop("trajectory_uid", "rollout_uid")

        # Merge in: run_name, run_uuid, clipgt_id, batch_id, rollout_id
        # return aggregate_and_write_metrics_results_txt(df)
        return UnprocessedMetricsDFs(df)

    @staticmethod
    def concat(
        pdfs: list["UnprocessedMetricsDFs"],
        rename_run_names: Optional[dict[str, str]] = None,
    ) -> "UnprocessedMetricsDFs":
        """Concatenate a list of unprocessed metrics dataframes."""
        columns = pdfs[0].unprocessed_df.columns
        metric_df = pl.concat([pdf.unprocessed_df.select(columns) for pdf in pdfs])
        if rename_run_names:
            metric_df = metric_df.with_columns(
                pl.col("run_name").replace(rename_run_names)
            )
        return UnprocessedMetricsDFs(metric_df)

    def process(
        self,
        force_same_run: bool = False,
        output_path: Optional[str] = None,
        additional_modifiers: Optional[list[MetricAggregationModifiers]] = None,
    ) -> "ProcessedMetricDFs":
        """Process the unprocessed metrics dataframe."""
        return aggregate_and_write_metrics_results_txt(
            self.unprocessed_df, force_same_run, output_path, additional_modifiers
        )

    def n_rollouts_simulated_per_clip(
        self, run_name: str | None = None, run_uuid: str | None = None
    ) -> int | pl.DataFrame:
        """Get the number of rollouts simulated per clip (before filtering).

        Args:
            run_name: The name of the run.
            run_uuid: The uuid of the run.

        Returns:
            The number of rollouts simulated per clip. If `run_name` or `run_uuid`
            is provided, or only a single run is present, returns the number of
            rollouts per clip for that run. Otherwise, returns a dataframe with
            `run_name`, `run_uuid`, and `n_rollouts`.
        """

        run_name_and_uuid_to_n_rollouts = (
            self.unprocessed_df.select(
                "run_uuid",
                "run_name",
                "clipgt_id",
                "batch_id",
                "rollout_id",
                "timestamps_us",
            )
            .unique()  # Remove duplicate metrics
            .group_by("run_uuid", "run_name", "clipgt_id", "timestamps_us")
            .agg(pl.col("rollout_id").count().alias("n_rollouts"))
            .select("run_uuid", "run_name", "n_rollouts")
            .unique()
            # .max()
        )
        if run_name is not None:
            return run_name_and_uuid_to_n_rollouts.filter(
                pl.col("run_name") == run_name
            )["n_rollouts"].item()
        if run_uuid is not None:
            return run_name_and_uuid_to_n_rollouts.filter(
                pl.col("run_uuid") == run_uuid
            )["n_rollouts"].item()
        if len(run_name_and_uuid_to_n_rollouts) == 1:
            return run_name_and_uuid_to_n_rollouts["n_rollouts"].item()
        return run_name_and_uuid_to_n_rollouts

    def __repr__(self) -> str:
        nr_trajectories = (
            self.unprocessed_df.select(
                "run_uuid", "clipgt_id", "batch_id", "rollout_id"
            )
            .unique()
            .shape[0]
        )
        return (
            f"{self.__class__.__name__}: \n"
            f"  Runs: {self.unprocessed_df['run_name'].unique().to_list()}\n"
            f"  Nr. Rows: {len(self.unprocessed_df)}\n"
            f"  Nr. Trajectories: {nr_trajectories}\n"
            f"  Nr. Rollouts per clip: {self.n_rollouts_simulated_per_clip()}\n"
            f"  Nr. Metrics: {len(self.unprocessed_df['name'].unique().to_list())}"
        )


@dataclass
class ProcessedMetricDFs(UnprocessedMetricsDFs):
    # Not aggregated yet, but with rollout and trajectory uids.
    df_long: pl.DataFrame
    df_wide: pl.DataFrame
    # Already modified - all further dfs are also modified!
    # Modifications are `_modifiers.MetricAggregationModifiers` applied in sequence.
    df_wide_modified: pl.DataFrame
    # Mapping of `trajectory_uid` to `run_uuid`, `run_name`, etc.
    trajectory_uid_df: pl.DataFrame
    # Wide format, aggregated over time.
    df_wide_avg_t: pl.DataFrame
    # Wide format, aggregated over time and clips.
    df_wide_avg_t_clip: pl.DataFrame
    # Wide format, aggregated over time, clips and rollouts (fully aggregated).
    df_wide_avg_t_clip_rollout: pl.DataFrame
    # Mapping of metric name to aggregation function used to aggregate over time.
    agg_function_df: pl.DataFrame
    # Combined run uuids. If `force_same_run` is True, the run uuids of the
    # individual array jobs are combined into a single run uuid.
    combined_run_uuids: str | None
    modifiers: list[MetricAggregationModifiers] | None

    def __repr__(self) -> str:
        nr_trajectories = (
            self.df_wide_modified.select("trajectory_uid").unique().shape[0]
        )
        return (
            f"{self.__class__.__name__}: \n"
            f"  Runs: {self.unprocessed_df['run_name'].unique().to_list()}\n"
            f"  Nr. Trajectories: {nr_trajectories}\n"
            f"  Nr. Rollouts per clip: {self.n_rollouts_simulated_per_clip()}\n"
            f"  Nr. Metrics: {len(self.unprocessed_df['name'].unique().to_list())}"
            f"\n  Modifiers: {self.modifiers}"
        )

    def get_removed_rows(self) -> pl.DataFrame:
        """Get the rows that are removed by the modifiers."""
        df = get_removed_rows(self.df_wide, self.df_wide_modified)
        return df.join(self.trajectory_uid_df, on="trajectory_uid", how="left").drop(
            ["trajectory_uid", "rollout_uid"]
        )

    def get_removed_trajectories(self) -> pl.DataFrame:
        """Get the trajectories that are removed by the modifiers.

        Returns:
            df with `run_uuid`, `run_name`, `clipgt_id`, `batch_id`, `rollout_id` of the
            trajectories that are removed by the modifiers.
        """
        touched_trajectories = (
            self.get_removed_rows()
            .select(["run_uuid", "run_name", "clipgt_id", "batch_id", "rollout_id"])
            .unique()
        )
        # Remove the ones that are still in the df_wide_avg_t
        return touched_trajectories.join(
            self.df_wide_avg_t,
            on=["run_uuid", "run_name", "clipgt_id", "batch_id", "rollout_id"],
            how="anti",
        )

    def get_rollouts_per_clip(
        self, run_name: str | None = None, run_uuid: str | None = None
    ) -> pl.DataFrame:
        """Get the number of rollouts per clip.

        Args:
            run_name: The name of the run.
            run_uuid: The uuid of the run.

        Returns:
            A dataframe with the number of rollouts per clip. If `run_name` or
            `run_uuid` is provided, returns the number of rollouts per clip for
            that run. Otherwise, returns a dataframe with `run_name`, `run_uuid`,
            and `n_rollouts`.
        """

        def _get_n_rollouts_per_clip(df: pl.DataFrame) -> pl.DataFrame:
            """Returns df with `run_uuid`, `run_name`, `clipgt_id`, and `n_rollouts`."""
            return (
                df.select("run_uuid", "run_name", "clipgt_id", "batch_id", "rollout_id")
                .unique()
                .group_by("run_uuid", "run_name", "clipgt_id")
                .agg(pl.col("rollout_id").count().alias("n_rollouts"))
            )

        # Assign 0 rollouts to the removed trajectories. This is only true for those
        # that do not appear at all in the df_wide_avg_t.
        removed_trajectories = _get_n_rollouts_per_clip(
            self.get_removed_trajectories()
        ).with_columns(pl.lit(0).alias("n_rollouts"))
        remaining_trajectories = _get_n_rollouts_per_clip(self.df_wide_avg_t)
        df = pl.concat(
            [
                remaining_trajectories,
                removed_trajectories.join(  # Only concat those not in `remaining_trajectories`
                    remaining_trajectories,
                    on=["run_uuid", "run_name", "clipgt_id"],
                    how="anti",
                ),
            ],
            how="vertical_relaxed",
        )
        if run_name is not None:
            return df.filter(pl.col("run_name") == run_name)
        if run_uuid is not None:
            return df.filter(pl.col("run_uuid") == run_uuid)
        return df


def get_avg_dist_between_incidents(df_wide_avg_t: pl.DataFrame) -> pl.DataFrame:
    """Computes Kms Per Incident."""

    df = df_wide_avg_t.group_by(["run_uuid", "rollout_uid"]).agg(
        pl.col("dist_traveled_m").sum().alias("sum_dist_traveled_m"),
        pl.col("offroad_or_collision").sum().alias("sum_offroad_or_collision"),
        pl.col("offroad_or_collision_at_fault")
        .sum()
        .alias("sum_offroad_or_collision_at_fault"),
    )

    return df.with_columns(
        (
            pl.col("sum_dist_traveled_m") / pl.col("sum_offroad_or_collision") / 1000
        ).alias("avg_dist_between_incidents"),
        (
            pl.col("sum_dist_traveled_m")
            / pl.col("sum_offroad_or_collision_at_fault")
            / 1000
        ).alias("avg_dist_between_incidents_at_fault"),
    ).drop(
        "sum_dist_traveled_m",
        "sum_offroad_or_collision",
        "sum_offroad_or_collision_at_fault",
    )


def aggregate_and_write_metrics_results_txt(
    metrics_df: pl.DataFrame,
    force_same_run: bool = False,
    output_path: Optional[str] = None,
    additional_modifiers: Optional[list[MetricAggregationModifiers]] = None,
) -> ProcessedMetricDFs:
    """
    Evaluate the eval parquet file.

    Args:
        eval_df: The eval parquet file. Expected columns:
            timestamps_us[int]: The timestamps of the metric.
            value[float]: The value of the metric at that timestamp.
            valid[bool]: Whether the metric is valid at that timestamp.
            metric_name[str]: The name of the metric.
            clipgt_id[str]: The clipgt id of the rollout.
            batch_id[str]: The batch id of the rollout.
            rollout_id[str]: The rollout id of the rollout.
            run_uuid[str]: The run uuid of the simulation run (unique).
            run_name[str]: The name of the simulation run (not necessarily unique).
            time_aggregation[str]: How the metric should be aggregated over
            time.
        force_same_run: If True, the metrics will be aggregated over different
            run-uuids. This is useful for aggregating array jobs. Note that this
            requiers that clipgt_ids are unique across different run_uuids.
        output_path: The path to the output folder. Writes the metrics results to
            a txt file and a png file. If None, no outputs are written.
    """

    combined_run_uuids = None
    if force_same_run:
        # Combine the run_uuids of the individual array jobs.
        combined_run_uuids = str(hash(tuple(metrics_df["run_uuid"].unique().sort())))
        metrics_df = metrics_df.with_columns(
            pl.lit(combined_run_uuids).alias("run_uuid")
        )

    df_long, trajectory_uid_df = add_rollout_and_trajectory_uids(metrics_df)

    agg_function_df = df_long.group_by("name").agg(
        pl.col("time_aggregation").first().alias("time_aggregation"),
    )

    # Convert to wide format - easier for some computations. Sorting is important.
    df_wide = df_long.pivot(
        values="values",
        index=["trajectory_uid", "timestamps_us"],
        on="name",
    ).sort(["trajectory_uid", "timestamps_us"])

    df_wide_modified = df_wide
    additional_modifiers = DEFAULT_MODIFIERS + (additional_modifiers or [])
    for modifier in additional_modifiers:
        df_wide_modified, agg_function_df = modifier(df_wide_modified, agg_function_df)

    # Aggregate over time
    df_wide_avg_t = df_wide_modified.group_by(["trajectory_uid"]).agg(
        *[
            AggregationType(row["time_aggregation"]).get_polars_agg_expr(row["name"])
            for row in agg_function_df.iter_rows(named=True)
        ]
    )

    # Also adds "KPI" and "KPI_at_fault" to the df_wide_avg_t
    df_wide_avg_t_clip = aggregate_over_clips(df_wide_avg_t, trajectory_uid_df)

    # Average over rollouts -> One line per run
    df_wide_avg_t_clip_rollout = (
        df_wide_avg_t_clip.drop("rollout_uid")
        .group_by("run_uuid")
        .agg(
            pl.col("*").mean(),
            pl.col("*").std().name.suffix("_std"),
            pl.len().alias("n_rollouts"),
        )
    )

    # Join back the run_name for easier interpretation of results
    df_wide_avg_t_clip_rollout = df_wide_avg_t_clip_rollout.join(
        trajectory_uid_df.select(pl.col("run_uuid"), pl.col("run_name")).unique(),
        on="run_uuid",
        how="left",
    )

    if output_path:
        write_metrics_results_txt(
            df_wide_avg_t_clip_rollout,
            agg_function_df,
            output_path,
            additional_modifiers,
        )
        plot_metrics_results(df_wide_avg_t_clip, trajectory_uid_df, output_path)

    df_wide_avg_t = df_wide_avg_t.join(
        trajectory_uid_df.select(
            pl.col("trajectory_uid"),
            pl.col("rollout_uid"),
            pl.col("run_name"),
            pl.col("run_uuid"),
            pl.col("clipgt_id"),
            pl.col("batch_id"),
            pl.col("rollout_id"),
        ).unique(),
        on="trajectory_uid",
        how="left",
    )

    return ProcessedMetricDFs(
        unprocessed_df=metrics_df,
        df_long=df_long,
        df_wide=df_wide,
        df_wide_modified=df_wide_modified,
        trajectory_uid_df=trajectory_uid_df,
        df_wide_avg_t=df_wide_avg_t,
        df_wide_avg_t_clip=df_wide_avg_t_clip,
        df_wide_avg_t_clip_rollout=df_wide_avg_t_clip_rollout,
        agg_function_df=agg_function_df,
        combined_run_uuids=combined_run_uuids,
        modifiers=additional_modifiers,
    )
