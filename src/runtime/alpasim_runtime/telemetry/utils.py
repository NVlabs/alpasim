# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Telemetry utility functions.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def merge_metrics_files(metrics_dir: str) -> None:
    """Combines individual worker metric files into single aggregate file.

    Groups all samples by metric family to ensure valid Prometheus text format.
    The Prometheus parser expects all samples of a metric to appear contiguously
    after the HELP/TYPE declarations.

    Note: This method is vibe-coded and not thoroughly checked, but seems to work.
    """
    prom_files = sorted(
        [
            f
            for f in os.listdir(metrics_dir)
            if f.startswith("metrics_worker_")
            and f.endswith(".prom")
            and f != "metrics.prom"
        ]
    )
    if not prom_files:
        return

    merged_prom = os.path.join(metrics_dir, "metrics.prom")

    # Regex to match metric lines: name{labels} value [timestamp]
    # Group 1: metric name (including _bucket, _sum, _count suffixes)
    # Group 2: existing labels (optional)
    # Group 3: value + timestamp
    metric_pattern = re.compile(r"^([a-zA-Z0-9_:]+)(?:\{([^}]*)\})?( .+)$")

    # Extract base metric family name (strip _bucket, _sum, _count suffixes for histograms)
    # Note: _created and _total are NOT stripped - they are separate metric families
    def get_metric_family(name: str) -> str:
        for suffix in ("_bucket", "_sum", "_count"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    # Collect all data grouped by metric family
    # Structure: {family_name: {"help": str, "type": str, "samples": [str]}}
    metrics_by_family: dict[str, dict] = {}
    family_order: list[str] = []  # Preserve order of first appearance

    for fname in prom_files:
        # Determine worker ID from filename (metrics_worker_N.prom -> N)
        worker_id = fname.replace("metrics_worker_", "").replace(".prom", "")
        worker_label = f'worker_id="{worker_id}"'
        current_family: str | None = None

        with open(os.path.join(metrics_dir, fname), "r") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("# HELP "):
                    # Extract family name from HELP line
                    parts = line.split(" ", 3)
                    if len(parts) >= 3:
                        current_family = parts[2]
                        if current_family not in metrics_by_family:
                            metrics_by_family[current_family] = {
                                "help": line,
                                "type": None,
                                "samples": [],
                            }
                            family_order.append(current_family)
                    continue

                if line.startswith("# TYPE "):
                    # Extract family name and type from TYPE line
                    parts = line.split(" ", 4)
                    if len(parts) >= 4:
                        current_family = parts[2]
                        if current_family not in metrics_by_family:
                            metrics_by_family[current_family] = {
                                "help": None,
                                "type": line,
                                "samples": [],
                            }
                            family_order.append(current_family)
                        else:
                            metrics_by_family[current_family]["type"] = line
                    continue

                # Regular metric line
                match = metric_pattern.match(line)
                if match:
                    name, labels, rest = match.groups()
                    family = get_metric_family(name)

                    # Add worker_id label
                    if labels:
                        new_labels = f"{labels},{worker_label}"
                    else:
                        new_labels = worker_label

                    sample_line = f"{name}{{{new_labels}}}{rest}"

                    if family not in metrics_by_family:
                        # Orphan metric without HELP/TYPE (shouldn't happen normally)
                        metrics_by_family[family] = {
                            "help": None,
                            "type": None,
                            "samples": [],
                        }
                        family_order.append(family)

                    metrics_by_family[family]["samples"].append(sample_line)

    # Write merged file with metrics grouped by family
    with open(merged_prom, "w") as outfile:
        for family in family_order:
            data = metrics_by_family[family]
            if data["help"]:
                outfile.write(data["help"] + "\n")
            if data["type"]:
                outfile.write(data["type"] + "\n")
            for sample in data["samples"]:
                outfile.write(sample + "\n")

    logger.info(f"Merged {len(prom_files)} Prometheus files into {merged_prom}")
