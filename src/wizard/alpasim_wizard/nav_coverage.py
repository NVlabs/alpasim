# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check recorded-navigation coverage before launching services."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any

from alpasim_utils.nav_route import find_route_indices

from .services import VolumeMount

logger = logging.getLogger(__name__)


def _host_path(container_path: str, mounts: list[VolumeMount]) -> Path:
    target = PurePosixPath(container_path)
    matches: list[tuple[int, Path]] = []
    for mount in mounts:
        try:
            relative = target.relative_to(PurePosixPath(mount.container))
        except ValueError:
            continue
        matches.append(
            (len(PurePosixPath(mount.container).parts), Path(mount.host) / relative)
        )
    if not matches:
        raise ValueError(
            f"recorded-navigation path {container_path!r} is not exposed by any "
            "services.runtime.volumes mount"
        )
    return max(matches, key=lambda match: match[0])[1]


def check_recorded_navigation_coverage(
    simulation_config: dict[str, Any],
    clip_ids: set[str],
    runtime_volume_specs: list[str],
) -> None:
    """Warn about uncovered clips and fail when no requested clip is covered."""
    nav_config = simulation_config.get("nav_instruction")
    if nav_config is None or nav_config["source"] != "recorded":
        return

    mounts = [VolumeMount.from_str(spec) for spec in runtime_volume_specs]
    route_index_dir = _host_path(nav_config["route_index_dir"], mounts)
    available = find_route_indices(route_index_dir, clip_ids)
    missing_details = [
        f"- {clip_id}" for clip_id in sorted(clip_ids - available.keys())
    ]
    if missing_details:
        logger.warning(
            "Recorded navigation is available for %d/%d configured clips. "
            "The following %d clips have no navigation input and will fail "
            "during rollout setup:\n%s",
            len(available),
            len(clip_ids),
            len(missing_details),
            "\n".join(missing_details),
        )
    if clip_ids and not available:
        raise ValueError(
            f"none of the {len(clip_ids)} configured clips has a route index in "
            f"{route_index_dir}"
        )
