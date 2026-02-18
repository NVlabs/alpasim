# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Generate metrics analysis plots from Prometheus .prom files.

This module produces a 3x3 grid of plots for analyzing simulation performance:
1. RPC Duration histogram
2. RPC Blocking histogram
3. RPC Queue Depth histogram
4. Rollout Duration histogram
5. Step Duration histogram
6. Service Configuration summary
7. CPU Utilization boxplot
8. GPU Utilization boxplot
9. GPU Memory boxplot
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import yaml
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")


# Methods to plot (by service)
METHODS_TO_PLOT = {
    "controller": ["run_controller_and_vehicle"],
    "driver": [
        "drive",
        "submit_egomotion_observation",
        "submit_image_observation",
        "submit_route",
    ],
    "physics": ["ground_intersection"],
    "sensorsim": ["render_rgb"],
}


# --- Prometheus parsing ---


@dataclass
class HistogramData:
    """Parsed histogram data with buckets and summary stats."""

    name: str
    labels: dict[str, str]
    buckets: list[tuple[float, int]]  # (le, cumulative_count)
    sum: float
    count: int


@dataclass
class GaugeData:
    """Parsed gauge data."""

    name: str
    labels: dict[str, str]
    value: float


def _load_prometheus_file(path: Path) -> Iterator[Metric]:
    """Load a .prom file and yield metric families."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    yield from text_string_to_metric_families(text)


def _extract_histograms(path: Path) -> list[HistogramData]:
    """Extract all histograms from a .prom file."""
    results = []

    for family in _load_prometheus_file(path):
        if family.type != "histogram":
            continue

        # Group samples by label set (excluding 'le')
        by_labels: dict[tuple, dict] = {}

        for sample in family.samples:
            # Extract labels without 'le' for grouping
            labels = {k: v for k, v in sample.labels.items() if k != "le"}
            key = tuple(sorted(labels.items()))

            if key not in by_labels:
                by_labels[key] = {
                    "labels": labels,
                    "buckets": [],
                    "sum": 0.0,
                    "count": 0,
                }

            if sample.name.endswith("_bucket"):
                le = float(sample.labels.get("le", "inf"))
                by_labels[key]["buckets"].append((le, int(sample.value)))
            elif sample.name.endswith("_sum"):
                by_labels[key]["sum"] = sample.value
            elif sample.name.endswith("_count"):
                by_labels[key]["count"] = int(sample.value)

        for data in by_labels.values():
            results.append(
                HistogramData(
                    name=family.name,
                    labels=data["labels"],
                    buckets=sorted(data["buckets"]),
                    sum=data["sum"],
                    count=data["count"],
                )
            )

    return results


def _extract_gauges(path: Path) -> list[GaugeData]:
    """Extract all gauges from a .prom file."""
    results = []

    for family in _load_prometheus_file(path):
        if family.type != "gauge":
            continue

        for sample in family.samples:
            results.append(
                GaugeData(
                    name=family.name,
                    labels=dict(sample.labels),
                    value=sample.value,
                )
            )

    return results


# --- DataFrame building ---


def _build_histograms_dataframe(histograms: list[HistogramData]) -> pl.DataFrame:
    """Convert histogram data to a polars DataFrame with per-bucket counts."""
    # Collect all unique label keys
    all_label_keys: set[str] = set()
    for h in histograms:
        all_label_keys.update(h.labels.keys())

    # Create rows
    rows = []
    for h in histograms:
        for bucket_le, bucket_count in h.buckets:
            row: dict[str, Any] = {
                "metric": h.name,
                "bucket_le": bucket_le,
                "bucket_count": bucket_count,
                "sum": h.sum,
                "count": h.count,
            }
            for key in all_label_keys:
                row[key] = h.labels.get(key)
            rows.append(row)

    df = pl.DataFrame(rows, infer_schema_length=None)

    # Keep original numeric bucket values for calculations, and create formatted
    # labels for display (using ≤ prefix prevents matplotlib from treating them
    # as parsable floats which would trigger a warning)
    df = df.with_columns(
        pl.col("bucket_le").alias("bucket_le_num"),
        pl.col("bucket_le")
        .map_elements(lambda x: f" {x}", return_dtype=pl.Utf8)
        .alias("bucket_le"),
    )

    # Compute per-bucket counts (convert cumulative to individual)
    # Sort by numeric bucket value to ensure correct ordering
    sort_cols = ["metric", "service", "method", "worker_id", "bucket_le_num"]
    # Only use columns that exist
    sort_cols = [c for c in sort_cols if c in df.columns]
    group_cols = [c for c in sort_cols if c != "bucket_le_num"]

    df = df.sort(sort_cols).with_columns(
        pl.col("bucket_count")
        .diff()
        .over(group_cols)
        .fill_null(pl.col("bucket_count"))
        .alias("per_bucket_count")
    )

    return df


def _build_gauges_dataframe(gauges: list[GaugeData]) -> pl.DataFrame:
    """Convert gauge data to a polars DataFrame."""
    # Collect all unique label keys
    all_label_keys: set[str] = set()
    for g in gauges:
        all_label_keys.update(g.labels.keys())

    # Create rows
    rows = []
    for g in gauges:
        row: dict[str, Any] = {
            "metric": g.name,
            "value": g.value,
        }
        for key in all_label_keys:
            row[key] = g.labels.get(key)
        rows.append(row)

    df = pl.DataFrame(rows, infer_schema_length=None)

    # Filter out "created" metrics
    df = df.filter(
        ~pl.col("metric").is_in(
            [
                "rpc_duration_seconds_created",
                "rpc_blocking_seconds_created",
                "rpc_queue_depth_at_start_created",
                "rollout_duration_seconds_created",
                "step_duration_seconds_created",
            ]
        )
    )

    return df


# --- Plotting helpers ---


def _normalize_histogram_by_hue(df: pl.DataFrame, hue_col: str | None) -> pl.DataFrame:
    """Normalize histogram counts per hue group so each group sums to 1.

    Args:
        df: DataFrame containing the histogram data
        hue_col: Column name to group by. If None, normalize across all data.
    Returns:
        DataFrame with normalized histogram counts
    """
    if df.is_empty():
        return df

    if hue_col is None:
        # Normalize across all data
        total = df["per_bucket_count"].sum()
        if total > 0:
            return df.with_columns(
                (pl.col("per_bucket_count") / total).alias("per_bucket_count")
            )
        return df

    # Normalize within each hue group
    return df.with_columns(
        (
            pl.col("per_bucket_count") / pl.col("per_bucket_count").sum().over(hue_col)
        ).alias("per_bucket_count")
    )


def _plot_cpu_boxplots(ax: plt.Axes, gauges_df: pl.DataFrame) -> None:
    """Plot CPU utilization boxplots from pre-computed stats."""
    cpu_stats = gauges_df.filter(pl.col("metric") == "process_cpu_utilization")

    if cpu_stats.is_empty() or "name" not in cpu_stats.columns:
        ax.text(
            0.5, 0.5, "No CPU data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_ylabel("CPU Utilization")
        return

    names = cpu_stats["name"].unique()
    plotted_names = []
    boxes = []

    for name in names:
        s = {
            row["stat"]: row["value"]
            for row in cpu_stats.filter(pl.col("name") == name).to_dicts()
        }
        if not s or s.get("mean", 0) < 50:
            continue
        plotted_names.append(name)
        box_stat = dict(
            whislo=s.get("p05", 0),
            q1=s.get("p25", 0),
            med=s.get("p50", 0),
            q3=s.get("p75", 0),
            whishi=s.get("p95", 0),
            mean=s.get("mean", 0),
            fliers=[s.get("min", 0), s.get("max", 0)],
        )
        boxes.append(box_stat)

    if not boxes:
        ax.text(
            0.5, 0.5, "No CPU data", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_ylabel("CPU Utilization (%)")
        return

    bp = ax.bxp(boxes, showmeans=True, meanline=True, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="steelblue", alpha=0.7)
    for mean in bp["means"]:
        mean.set(color="red", linestyle="--", linewidth=2)

    ax.set_ylabel("CPU Utilization (%)")
    ax.set_xticklabels(plotted_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)


def _plot_gpu_boxplots(
    ax_util: plt.Axes, ax_mem: plt.Axes, gauges_df: pl.DataFrame
) -> None:
    """Plot GPU utilization and memory boxplots from pre-computed stats."""
    gpu_memory_stats = gauges_df.filter(pl.col("metric") == "gpu_memory_used_bytes")
    gpu_util_stats = gauges_df.filter(pl.col("metric") == "gpu_utilization")
    gpu_memory_total = gauges_df.filter(pl.col("metric") == "gpu_memory_total_bytes")

    if gpu_memory_stats.is_empty() or "gpu" not in gpu_memory_stats.columns:
        ax_util.text(
            0.5,
            0.5,
            "No GPU data",
            ha="center",
            va="center",
            transform=ax_util.transAxes,
        )
        ax_mem.text(
            0.5,
            0.5,
            "No GPU data",
            ha="center",
            va="center",
            transform=ax_mem.transAxes,
        )
        ax_util.set_ylabel("GPU Utilization (%)")
        ax_mem.set_ylabel("GPU Memory Used (GB)")
        return

    gpus = sorted(gpu_memory_stats["gpu"].unique())
    plotted_labels = [f"GPU {gpu}" for gpu in gpus]

    bytes_to_gb = 1024**3

    # Get total memory per GPU (for horizontal line)
    total_memory_by_gpu = {}
    if not gpu_memory_total.is_empty() and "gpu" in gpu_memory_total.columns:
        for row in gpu_memory_total.to_dicts():
            total_memory_by_gpu[row["gpu"]] = row["value"] / bytes_to_gb

    # GPU Memory plot
    memory_boxes = []
    for gpu in gpus:
        s = {
            row["stat"]: row["value"]
            for row in gpu_memory_stats.filter(pl.col("gpu") == gpu).to_dicts()
        }
        box_stat = dict(
            whislo=s.get("p05", 0) / bytes_to_gb,
            q1=s.get("p25", 0) / bytes_to_gb,
            med=s.get("p50", 0) / bytes_to_gb,
            q3=s.get("p75", 0) / bytes_to_gb,
            whishi=s.get("p95", 0) / bytes_to_gb,
            mean=s.get("mean", 0) / bytes_to_gb,
            fliers=[s.get("min", 0) / bytes_to_gb, s.get("max", 0) / bytes_to_gb],
        )
        memory_boxes.append(box_stat)

    if memory_boxes:
        bp1 = ax_mem.bxp(memory_boxes, showmeans=True, meanline=True, patch_artist=True)
        for box in bp1["boxes"]:
            box.set(facecolor="seagreen", alpha=0.7)
        for mean in bp1["means"]:
            mean.set(color="red", linestyle="--", linewidth=2)
        ax_mem.set_xticklabels(plotted_labels)

        # Draw horizontal line at max (total) GPU memory
        if total_memory_by_gpu:
            # Use the max total memory across all GPUs for the line
            max_total_gb = max(total_memory_by_gpu.values())
            ax_mem.axhline(
                y=max_total_gb,
                color="darkred",
                linestyle=":",
                linewidth=2,
                label=f"Total: {max_total_gb:.1f} GB",
            )
            ax_mem.legend(loc="upper right", fontsize=8)

    ax_mem.set_ylabel("GPU Memory Used (GB)")
    ax_mem.grid(axis="y", alpha=0.3)

    # GPU Utilization plot
    util_boxes = []
    for gpu in gpus:
        s = {
            row["stat"]: row["value"]
            for row in gpu_util_stats.filter(pl.col("gpu") == gpu).to_dicts()
        }
        box_stat = dict(
            whislo=s.get("p05", 0),
            q1=s.get("p25", 0),
            med=s.get("p50", 0),
            q3=s.get("p75", 0),
            whishi=s.get("p95", 0),
            mean=s.get("mean", 0),
            fliers=[s.get("min", 0), s.get("max", 0)],
        )
        util_boxes.append(box_stat)

    if util_boxes:
        bp2 = ax_util.bxp(util_boxes, showmeans=True, meanline=True, patch_artist=True)
        for box in bp2["boxes"]:
            box.set(facecolor="steelblue", alpha=0.7)
        for mean in bp2["means"]:
            mean.set(color="red", linestyle="--", linewidth=2)
        ax_util.set_xticklabels(plotted_labels)
    ax_util.set_ylabel("GPU Utilization (%)")
    ax_util.grid(axis="y", alpha=0.3)


def _compute_summary_stats(gauges_df: pl.DataFrame) -> tuple[float, float]:
    """Compute simulation summary statistics from gauges."""
    # Only group by columns that exist in the DataFrame
    group_cols = ["metric", "method", "service", "name", "gpu", "stat"]
    existing_cols = [c for c in group_cols if c in gauges_df.columns]
    gauges_avg = gauges_df.group_by(existing_cols, maintain_order=True).agg(
        pl.col("value").mean().alias("avg_value"),
        pl.col("value").sum().alias("sum"),
    )

    gauge_avg = {row["metric"]: row["avg_value"] for row in gauges_avg.to_dicts()}
    gauge_sum = {row["metric"]: row["sum"] for row in gauges_avg.to_dicts()}

    # Simulation seconds per rollout
    total_seconds = gauge_avg.get("simulation_total_seconds", 0)
    rollout_count = gauge_sum.get("simulation_rollout_count", 1)
    sim_seconds_per_rollout = total_seconds / rollout_count if rollout_count else 0

    # Event loop idle percentage
    idle_time = gauge_avg.get("event_loop_idle_seconds_total", 0)
    work_time = gauge_avg.get("event_loop_work_seconds_total", 0)
    poll_time = gauge_avg.get("event_loop_poll_seconds_total", 0)
    total_time = idle_time + work_time + poll_time
    idle_percentage = idle_time / total_time if total_time > 0 else 0

    return idle_percentage, sim_seconds_per_rollout


def _load_service_config(metrics_path: Path) -> dict[str, dict[str, Any]] | None:
    """Load service configuration from wizard_config.yaml.

    Args:
        metrics_path: Path to the metrics .prom file

    Returns:
        Dictionary mapping service names to their configuration, or None if not found
    """
    # Look for wizard_config.yaml in log_dir (metrics_path is <log_dir>/telemetry/metrics.prom)
    config_path = metrics_path.parent.parent / "wizard-config.yaml"

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.warning(
            "Failed to load wizard_config.yaml from %s: %s", config_path.absolute(), e
        )
        return None

    if not config:
        return None

    services_config = config.get("services", {})
    runtime_config = config.get("runtime", {})
    endpoints_config = runtime_config.get("endpoints", {})

    result = {}
    for service_name in ["sensorsim", "driver", "physics", "controller", "trafficsim"]:
        svc = services_config.get(service_name, {})
        endpoint = endpoints_config.get(service_name, {})

        gpus = svc.get("gpus")
        # nr_gpus: count of GPU list, or 1 for CPU-only services (null gpus)
        if isinstance(gpus, list):
            nr_gpus = len(gpus)
        elif gpus is None:
            nr_gpus = 1  # CPU-only service like controller
        else:
            nr_gpus = 0

        replicas = svc.get("replicas_per_container", 1)
        concurrent = endpoint.get("n_concurrent_rollouts", 1)
        skip = endpoint.get("skip", False)

        result[service_name] = {
            "nr_gpus": nr_gpus,
            "replicas_per_container": replicas,
            "n_concurrent_rollouts": concurrent,
            "skip": skip,
            "total": nr_gpus * replicas * concurrent,
        }

    return result


def _plot_service_config(ax: plt.Axes, metrics_path: Path) -> None:
    """Plot service configuration summary in the given axes."""
    config = _load_service_config(metrics_path)

    if config is None:
        ax.text(
            0.5,
            0.5,
            "No wizard_config.yaml found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            family="monospace",
        )
        ax.axis("off")
        return

    service_abbrevs = {
        "sensorsim": "SENS",
        "driver": "DRIV",
        "physics": "PHYS",
        "controller": "CONT",
        "trafficsim": "TRAF",
    }

    # Build table data
    col_labels = ["Service", "GPUs", "Replicas", "Concurrent", "Total"]
    table_data = []

    for service_name, abbrev in service_abbrevs.items():
        svc = config.get(service_name, {})
        if svc.get("skip", False):
            table_data.append([abbrev, "-", "-", "-", "(skipped)"])
        else:
            nr_gpus = svc.get("nr_gpus", 0)
            replicas = svc.get("replicas_per_container", 0)
            concurrent = svc.get("n_concurrent_rollouts", 0)
            total = svc.get("total", 0)
            table_data.append(
                [abbrev, str(nr_gpus), str(replicas), str(concurrent), str(total)]
            )

    ax.axis("off")
    ax.set_title(
        "Service Configuration and \n number of parallel rollouts",
        fontsize=11,
        fontweight="bold",
        loc="center",
        y=0.80,
    )

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Style header row
    for col_idx in range(len(col_labels)):
        table[(0, col_idx)].set_facecolor("#d0d0d0")
        table[(0, col_idx)].set_text_props(fontweight="bold")

    # Style data rows with wheat background (same as legends)
    for row_idx in range(1, len(table_data) + 1):
        for col_idx in range(len(col_labels)):
            table[(row_idx, col_idx)].set_facecolor("wheat")


# --- Main plotting function ---


def generate_metrics_plot(
    metrics_path: Path, output_path: Path | None = None, run_name: str | None = None
) -> Path:
    """
    Generate metrics analysis plot from a Prometheus .prom file.

    Args:
        metrics_path: Path to the merged metrics.prom file
        output_path: Optional output path for the PNG. Defaults to metrics_path.parent / "metrics_plot.png"

    Returns:
        Path to the generated PNG file
    """
    if output_path is None:
        output_path = metrics_path.parent / "metrics_plot.png"

    logger.info("Loading metrics from: %s", metrics_path)
    histograms = _extract_histograms(metrics_path)
    gauges = _extract_gauges(metrics_path)

    if not histograms and not gauges:
        logger.warning("No metrics found in file")
        return output_path

    histograms_df = _build_histograms_dataframe(histograms)
    gauges_df = _build_gauges_dataframe(gauges)

    # Compute summary stats
    idle_percentage, sim_seconds_per_rollout = _compute_summary_stats(gauges_df)

    # Configure matplotlib
    plt.rcParams["figure.figsize"] = (16, 12)
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["legend.facecolor"] = "wheat"
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.framealpha"] = 0.7
    plt.rcParams["legend.fontsize"] = 8

    # Get methods to filter
    all_methods = [m for methods in METHODS_TO_PLOT.values() for m in methods]

    # Filter data for RPC plots
    df_duration = histograms_df.filter(
        pl.col("metric") == "rpc_duration_seconds",
        (
            pl.col("method").is_in(all_methods)
            if "method" in histograms_df.columns
            else True
        ),
    )
    df_blocking = histograms_df.filter(
        pl.col("metric") == "rpc_blocking_seconds",
        (
            pl.col("method").is_in(all_methods)
            if "method" in histograms_df.columns
            else True
        ),
    )

    # Create figure
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(16, 12))
    axs = axs.flatten()

    # Plot 0: RPC Duration
    ax = axs[0]
    hue_col = "method" if "method" in df_duration.columns else None
    df_duration_norm = _normalize_histogram_by_hue(df_duration, hue_col)
    if sns and not df_duration_norm.is_empty():
        sns.barplot(
            data=df_duration_norm.to_pandas(),
            y="per_bucket_count",
            x="bucket_le",
            hue=hue_col,
            dodge=True,
            errorbar=None,
            ax=ax,
        )
        if ax.get_legend():
            ax.get_legend().remove()
    ax.set_ylabel("RPC Duration (s)")
    ax.set_xlabel("")

    # Plot 1: RPC Blocking
    ax = axs[1]
    hue_col = "method" if "method" in df_blocking.columns else None
    df_blocking_norm = _normalize_histogram_by_hue(df_blocking, hue_col)
    if sns and not df_blocking_norm.is_empty():
        sns.barplot(
            data=df_blocking_norm.to_pandas(),
            y="per_bucket_count",
            x="bucket_le",
            hue=hue_col,
            dodge=True,
            errorbar=None,
            ax=ax,
        )
    ax.set_ylabel("RPC Blocking (s)")
    ax.set_xlabel("")

    # Plot 2: RPC Queue Depth
    ax = axs[2]
    queue_depth_df = histograms_df.filter(
        pl.col("metric") == "rpc_queue_depth_at_start"
    )
    hue_col = "service" if "service" in queue_depth_df.columns else None
    queue_depth_norm = _normalize_histogram_by_hue(queue_depth_df, hue_col)
    if sns and not queue_depth_norm.is_empty():
        sns.barplot(
            data=queue_depth_norm.to_pandas(),
            x="bucket_le",
            y="per_bucket_count",
            hue=hue_col,
            ax=ax,
        )
    ax.set_ylabel("RPC Queue Depth (count)")
    ax.set_xlabel("")

    # Plot 3: Rollout Duration
    ax = axs[3]
    rollout_df = histograms_df.filter(pl.col("metric") == "rollout_duration_seconds")
    if not rollout_df.is_empty():
        # Limit data to last non-zero bucket (filter before plotting since x is categorical)
        max_nonzero = (
            rollout_df.filter(pl.col("per_bucket_count") > 0)
            .select(pl.col("bucket_le_num").max())
            .item()
        )
        if max_nonzero is not None and math.isfinite(max_nonzero):
            rollout_df = rollout_df.filter(pl.col("bucket_le_num") <= max_nonzero)

        sns.barplot(
            data=rollout_df.to_pandas(),
            x="bucket_le",
            y="per_bucket_count",
            ax=ax,
            errorbar=None,
            color="grey",
        )

        # Show at most max_labels tick labels (categorical axis uses bar indices 0, 1, 2, ...)
        num_bars = rollout_df.height
        max_labels = 10
        if num_bars > max_labels:
            every_nth_label = max(1, num_bars // max_labels)
            for idx, label in enumerate(ax.xaxis.get_ticklabels()):
                if (idx + 1) % every_nth_label != 0:
                    label.set_visible(False)
    ax.set_ylabel("Rollout Duration (s)")
    ax.set_xlabel("")
    # Format x-axis ticks as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    # Plot 4: Step Duration
    ax = axs[4]
    step_df = histograms_df.filter(pl.col("metric") == "step_duration_seconds")
    if sns and not step_df.is_empty():
        sns.barplot(
            data=step_df.to_pandas(),
            x="bucket_le",
            y="per_bucket_count",
            ax=ax,
            color="grey",
        )
    ax.set_ylabel("Step Duration (s)")
    ax.set_xlabel("")

    # Plot 5: Service configuration summary
    _plot_service_config(axs[5], metrics_path)

    # Plot 6: CPU boxplots
    _plot_cpu_boxplots(axs[6], gauges_df)

    # Plot 7-8: GPU boxplots
    _plot_gpu_boxplots(axs[7], axs[8], gauges_df)

    # Clear x labels and rotate ticks
    for ax in axs:
        ax.set_xlabel("")

        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    run_name = run_name or "Unknown Run"
    # Add title with summary stats
    fig.suptitle(
        (
            f"Run: {run_name}"
            f" - Async worker idle percentage: {idle_percentage:.2%}, "
            f"Sim seconds per rollout: {sim_seconds_per_rollout:.2f}"
        ),
        fontsize=16,
    )
    plt.subplots_adjust(top=0.93)

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Metrics plot saved to: %s", output_path)
    return output_path
