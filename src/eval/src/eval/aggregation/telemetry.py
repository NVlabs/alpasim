# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Helpers for extracting aggregate telemetry from runtime Prometheus files."""

from __future__ import annotations

import logging
import pathlib
import re

logger = logging.getLogger(__name__)

_PROM_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_PROM_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _unescape_prometheus_label_value(value: str) -> str:
    result = []
    idx = 0
    while idx < len(value):
        char = value[idx]
        if char != "\\" or idx + 1 >= len(value):
            result.append(char)
            idx += 1
            continue

        escaped = value[idx + 1]
        if escaped == "n":
            result.append("\n")
        elif escaped in {'"', "\\"}:
            result.append(escaped)
        else:
            result.append(f"\\{escaped}")
        idx += 2
    return "".join(result)


def _parse_labels(raw_labels: str | None) -> dict[str, str]:
    if not raw_labels:
        return {}
    return {
        match.group(1): _unescape_prometheus_label_value(match.group(2))
        for match in _PROM_LABEL_RE.finditer(raw_labels)
    }


def _iter_prometheus_samples(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = _PROM_SAMPLE_RE.match(line)
            if not match:
                continue
            yield (
                match.group("name"),
                _parse_labels(match.group("labels")),
                float(match.group("value")),
            )


def extract_driver_drive_rpc_latency(
    metrics_paths: list[pathlib.Path],
    *,
    tag: str = "default",
) -> dict[str, object] | None:
    """Compute aggregate driver ``drive`` RPC latency from Prometheus files.

    The runtime writes ``rpc_duration_seconds_sum`` and
    ``rpc_duration_seconds_count`` counters per worker. The mean latency is the
    sum of all matching durations divided by the sum of all matching counts.
    """

    total_sum_s = 0.0
    total_count = 0.0
    files_used: list[str] = []

    for metrics_path in metrics_paths:
        if not metrics_path.exists():
            continue

        file_sum_s = 0.0
        file_count = 0.0
        file_has_sum = False
        file_has_count = False
        try:
            for name, labels, value in _iter_prometheus_samples(metrics_path):
                if labels.get("service") != "driver":
                    continue
                if labels.get("method") != "drive":
                    continue
                if labels.get("tag", "default") != tag:
                    continue
                if name == "rpc_duration_seconds_sum":
                    file_has_sum = True
                    file_sum_s += value
                elif name == "rpc_duration_seconds_count":
                    file_has_count = True
                    file_count += value
        except OSError as exc:
            logger.warning(
                "Could not read telemetry metrics from %s: %s", metrics_path, exc
            )
            continue

        if file_has_sum and file_has_count and file_count > 0:
            files_used.append(str(metrics_path))
            total_sum_s += file_sum_s
            total_count += file_count

    if total_count <= 0:
        return None

    return {
        "driver_drive_rpc_duration_mean_s": total_sum_s / total_count,
        "driver_drive_rpc_duration_sum_s": total_sum_s,
        "driver_drive_rpc_duration_count": int(total_count),
        "source": "telemetry/metrics.prom",
        "tag": tag,
        "files_used": files_used,
    }


def collect_driver_drive_rpc_latency(
    job_dirs: list[pathlib.Path],
    *,
    tag: str = "default",
) -> dict[str, object] | None:
    metrics_paths = [job_dir / "telemetry" / "metrics.prom" for job_dir in job_dirs]
    return extract_driver_drive_rpc_latency(metrics_paths, tag=tag)


def run_level_metrics_from_summary(
    telemetry_summary: dict[str, object] | None,
) -> dict[str, object] | None:
    if not telemetry_summary:
        return None
    return {
        key: telemetry_summary[key]
        for key in (
            "driver_drive_rpc_duration_mean_s",
            "driver_drive_rpc_duration_sum_s",
            "driver_drive_rpc_duration_count",
        )
        if key in telemetry_summary
    }
