# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Plot the Grafana dashboard's summary queries from Prometheus."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yaml

logger = logging.getLogger(__name__)

DASHBOARD_NAME = "alpasim-runtime-dashboard.json"
SUMMARY_ROW = "Metrics plot summary"
_LEGEND_LABEL = re.compile(r"{{([^{}]+)}}")
_REGEX_META = re.compile(r"([\\.*+?()\[\]{}|^$])")
_UNRESOLVED_VARIABLE = re.compile(r"\$(?:__)?[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class PrometheusClient:
    base_url: str
    timeout_s: float = 5.0

    def query_range(
        self,
        query: str,
        *,
        start: float,
        end: float,
        step: str,
    ) -> list[dict[str, Any]]:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"Unsupported Prometheus URL '{self.base_url}'. "
                "Expected http(s)://host[:port]"
            )
        params = urlencode({"query": query, "start": start, "end": end, "step": step})
        url = f"{self.base_url.rstrip('/')}/api/v1/query_range?{params}"
        with urlopen(url, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload["status"] != "success":
            raise RuntimeError(f"Prometheus query failed: {payload}")
        if payload["data"]["resultType"] != "matrix":
            raise RuntimeError(f"Expected Prometheus matrix response: {payload}")
        return payload["data"]["result"]


def _load_run_metadata(log_dir: Path) -> dict[str, Any]:
    with open(log_dir / "run_metadata.yaml", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    if not isinstance(metadata, dict):
        raise TypeError("run_metadata.yaml must contain a mapping")
    return metadata


def _load_summary_panels() -> list[dict[str, Any]]:
    dashboard_path = resource_files("alpasim_utils.telemetry").joinpath(DASHBOARD_NAME)
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    panels = []
    found_summary = False
    for panel in dashboard["panels"]:
        if panel["type"] == "row":
            if found_summary:
                break
            found_summary = panel["title"] == SUMMARY_ROW
            continue
        if found_summary:
            if panel["type"] not in {"timeseries", "heatmap", "histogram"}:
                raise ValueError(f"Unsupported summary panel type: {panel['type']}")
            panels.append(panel)
    if not panels:
        raise ValueError(f"Dashboard row '{SUMMARY_ROW}' has no panels")
    return sorted(
        panels, key=lambda panel: (panel["gridPos"]["y"], panel["gridPos"]["x"])
    )


def _promql_regex(value: object) -> str:
    quoted = _REGEX_META.sub(r"\\\1", str(value))
    return json.dumps(quoted)[1:-1]


def _resolve_query(
    query: str,
    *,
    run_uuid: object,
    run_name: object,
    range_seconds: int,
) -> str:
    resolved = query.replace("$run_uuid", _promql_regex(run_uuid))
    resolved = resolved.replace("$run_name", _promql_regex(run_name))
    resolved = resolved.replace("$__range", f"{range_seconds}s")
    unresolved = _UNRESOLVED_VARIABLE.search(resolved)
    if unresolved:
        raise ValueError(f"Unsupported dashboard variable: {unresolved.group()}")
    return resolved


def _legend(legend_format: str, metric: dict[str, str]) -> str:
    if legend_format:
        return _LEGEND_LABEL.sub(
            lambda match: metric.get(match.group(1), match.group(0)),
            legend_format,
        )
    labels = [
        f"{name}={value}"
        for name, value in sorted(metric.items())
        if name not in {"__name__", "run_name", "run_uuid"}
    ]
    return ", ".join(labels) or metric.get("__name__", "value")


def _series_points(series: dict[str, Any]) -> tuple[list[float], list[float]]:
    points = [
        (
            float(mdates.date2num(datetime.fromtimestamp(float(timestamp)))),
            float(value),
        )
        for timestamp, value in series["values"]
        if math.isfinite(float(value))
    ]
    return [point[0] for point in points], [point[1] for point in points]


def _plot_timeseries(
    ax: plt.Axes,
    panel: dict[str, Any],
    results: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    lines = 0
    for target, series_list in results:
        for series in series_list:
            timestamps, values = _series_points(series)
            if not timestamps:
                continue
            ax.plot(
                timestamps,
                values,
                label=_legend(target.get("legendFormat", ""), series["metric"]),
            )
            lines += 1
    if lines:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(panel["title"])
    ax.grid(axis="both", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def _plot_heatmap(
    ax: plt.Axes,
    panel: dict[str, Any],
    results: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    if len(results) != 1:
        raise ValueError(f"Heatmap '{panel['title']}' must have exactly one target")

    bucket_values: dict[float, dict[float, float]] = {}
    for series in results[0][1]:
        boundary = float(series["metric"]["le"])
        if math.isinf(boundary):
            continue
        if boundary in bucket_values:
            raise ValueError(
                f"Heatmap '{panel['title']}' has duplicate bucket {boundary}"
            )
        bucket_values[boundary] = {
            float(timestamp): float(value) for timestamp, value in series["values"]
        }

    if not bucket_values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(panel["title"])
        return

    boundaries = sorted(bucket_values)
    timestamps = sorted(
        {timestamp for values in bucket_values.values() for timestamp in values}
    )
    cumulative = np.array(
        [
            [bucket_values[boundary].get(timestamp, np.nan) for timestamp in timestamps]
            for boundary in boundaries
        ]
    )
    counts = np.maximum(
        np.diff(cumulative, axis=0, prepend=np.zeros((1, len(timestamps)))),
        0,
    )
    image = ax.imshow(counts, aspect="auto", origin="lower", interpolation="nearest")
    ax.figure.colorbar(image, ax=ax, label="observations/s")

    tick_indices = np.unique(
        np.linspace(0, len(timestamps) - 1, min(6, len(timestamps)), dtype=int)
    )
    ax.set_xticks(
        tick_indices,
        [
            datetime.fromtimestamp(timestamps[index]).strftime("%H:%M")
            for index in tick_indices
        ],
    )
    ax.set_yticks(range(len(boundaries)), [f"{boundary:g}" for boundary in boundaries])
    ax.set_ylabel("seconds")
    ax.set_title(panel["title"])


def _plot_histogram(
    ax: plt.Axes,
    panel: dict[str, Any],
    results: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    histograms = []
    for target, series_list in results:
        for series in series_list:
            _, values = _series_points(series)
            if values:
                histograms.append(
                    (
                        _legend(target.get("legendFormat", ""), series["metric"]),
                        values,
                    )
                )
    if not histograms:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(panel["title"])
        return

    bins = np.histogram_bin_edges(
        [value for _, values in histograms for value in values], bins="auto"
    )
    for label, values in histograms:
        ax.hist(values, bins=bins, alpha=0.5, label=label)

    ax.legend(fontsize=7)
    ax.set_ylabel("frequency")
    ax.set_title(panel["title"])


def generate_metrics_plot(
    *,
    prometheus_url: str,
    output_path: Path,
) -> Path:
    """Generate a time-based plot from the dashboard's summary queries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _load_run_metadata(output_path.parent)
    start = datetime.strptime(
        str(metadata["run_time"]), "%Y-%m-%d %H:%M:%S"
    ).timestamp()
    end = time.time()
    range_seconds = max(1, math.ceil(end - start))
    step = f"{max(5, math.ceil(range_seconds / 1200))}s"

    client = PrometheusClient(prometheus_url)
    panels = _load_summary_panels()
    rows = math.ceil(len(panels) / 3)
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(24, 4 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    try:
        for ax, panel in zip(flat_axes, panels, strict=False):
            results = []
            for target in panel["targets"]:
                query = _resolve_query(
                    target["expr"],
                    run_uuid=metadata["run_uuid"],
                    run_name=metadata["run_name"],
                    range_seconds=range_seconds,
                )
                series = client.query_range(
                    query,
                    start=start,
                    end=end,
                    step=step,
                )
                results.append((target, series))
            if panel["type"] == "timeseries":
                _plot_timeseries(ax, panel, results)
            elif panel["type"] == "heatmap":
                _plot_heatmap(ax, panel, results)
            else:
                _plot_histogram(ax, panel, results)

        for ax in flat_axes[len(panels) :]:
            ax.axis("off")
        for ax in flat_axes:
            ax.tick_params(axis="x", rotation=30)

        start_text = datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S")
        end_text = datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S")
        fig.suptitle(f"{metadata['run_name']}\n{start_text} – {end_text}", fontsize=14)
        fig.savefig(output_path, dpi=150)
    finally:
        plt.close(fig)

    logger.info("Generated telemetry metrics plot: %s", output_path)
    return output_path
