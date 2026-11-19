# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Load scene suites from one canonical CSV per suite."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import polars as pl
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene
from alpasim_wizard.scenes.csv_utils import normalize_scene_suite_catalog
from alpasim_wizard.schema import SceneSuiteConfig


def find_scene_suite_csv(
    suite_id: str,
    scene_suite_dirs: Sequence[str],
) -> Path:
    """Find the unique checked-in CSV for a scene-suite ID."""
    if not suite_id or Path(suite_id).name != suite_id:
        raise ValueError(f"Invalid scene-suite ID: {suite_id!r}")

    filename = f"{suite_id}.csv"
    matches = sorted(
        {
            path.absolute()
            for directory in scene_suite_dirs
            for path in Path(directory).rglob(filename)
            if path.is_file()
        }
    )
    if not matches:
        raise ValueError(f"Unknown scene-suite ID: {suite_id!r}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise ValueError(f"Scene-suite ID {suite_id!r} is ambiguous; found: {paths}")
    return matches[0]


def scene_suite_id_from_csv_path(csv_path: str | Path) -> str:
    """Return the suite ID encoded by its CSV filename."""
    path = Path(csv_path)
    if path.suffix != ".csv" or not path.stem:
        raise ValueError(f"Scene-suite CSV must be named <scene-suite-id>.csv: {path}")
    return path.stem


def resolve_scene_suites(
    configs: list[SceneSuiteConfig],
    clip_catalog: pl.DataFrame,
) -> list[Scene]:
    """Resolve suites in authored order and deduplicate by scene key."""
    if not configs:
        raise ValueError("scenes.suite_configs must not be empty")

    suite_ids: set[str] = set()
    rows: list[tuple[str, str, str, int]] = []

    for config in configs:
        path = Path(config.csv_path)
        suite_id = scene_suite_id_from_csv_path(path)
        if suite_id in suite_ids:
            raise ValueError(
                f"Duplicate scene_suite_id in scenes.suite_configs: {suite_id}"
            )
        suite_ids.add(suite_id)

        frame = normalize_scene_suite_catalog(
            pl.read_csv(path, infer_schema_length=0),
            source=str(path),
        )
        rows.extend(
            (*row, config.t0_offset_us)
            for row in frame.select("clip_id", "uuid", "event_t0_us").iter_rows()
        )

    uuid_by_clip: dict[str, str] = {}
    ordered_pins: list[tuple[str, str]] = []
    scenes_by_key: dict[str, Scene] = {}
    for clip_id, uuid, event_t0_us, t0_offset_us in rows:
        previous_uuid = uuid_by_clip.get(clip_id)
        if previous_uuid is not None and previous_uuid != uuid:
            raise ValueError(
                f"Selected scene suites pin multiple UUIDs for clip_id {clip_id!r}: "
                f"{previous_uuid!r} and {uuid!r}"
            )
        if previous_uuid is None:
            uuid_by_clip[clip_id] = uuid
            ordered_pins.append((clip_id, uuid))

        policy_start_us = (
            T0_AUTO if event_t0_us == T0_AUTO else int(event_t0_us) + t0_offset_us
        )
        scene = Scene(
            source=ClipDataSource(clip_id=clip_id, usdz_uuid=uuid),
            policy_start_us=policy_start_us,
        )
        scenes_by_key.setdefault(scene.scene_key, scene)

    catalog_pins = set(clip_catalog.select("clip_id", "uuid").iter_rows())
    missing = [pin for pin in ordered_pins if pin not in catalog_pins]
    if missing:
        raise ValueError(f"Failed to find pinned (clip_id, uuid) pair(s): {missing}")

    return list(scenes_by_key.values())
