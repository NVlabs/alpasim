# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Read clip IDs and their renderer texture banks from an external YAML file."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import yaml


@dataclass(frozen=True)
class EgoMaskOverride:
    """A renderer bank directory and the cameras with patches in that directory."""

    rig_config_id: str
    camera_logical_ids: tuple[str, ...]


def load_ego_mask_overrides(path: str) -> dict[str, EgoMaskOverride]:
    """Load and validate the external map during runtime startup.

    YAML keys are exact source clip IDs. Each value contains ``rig_config_id``
    (a directory under the renderer's hood directory) and ``camera_logical_ids``.
    An empty mapping is valid. Unlisted clips keep the configured default bank.
    """
    with Path(path).open() as stream:
        mapping = yaml.safe_load(stream)
    if not isinstance(mapping, dict):
        raise ValueError(f"Ego mask overrides must be a mapping: {path}")
    overrides = {}
    for clip_id, entry in mapping.items():
        if not _is_identifier(clip_id):
            raise ValueError(f"Invalid source clip ID: {clip_id!r}")
        if "@t0=" in clip_id:
            raise ValueError(
                f"Use a source ID, not a scene key, for ego mask override {clip_id!r}"
            )
        try:
            bare_uuid = UUID(clip_id)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Bare UUID {clip_id!r} is not a ClipGT source ID; use 'clipgt-{bare_uuid}'"
            )
        if not isinstance(entry, dict) or set(entry) != {
            "rig_config_id",
            "camera_logical_ids",
        }:
            raise ValueError(
                f"Expected rig_config_id and camera_logical_ids for {clip_id}"
            )
        bank = entry["rig_config_id"]
        cameras = entry["camera_logical_ids"]
        if not _is_identifier(bank):
            raise ValueError(f"Empty or invalid rig_config_id for {clip_id}")
        if (
            not isinstance(cameras, list)
            or not cameras
            or any(not _is_identifier(camera) for camera in cameras)
            or len(cameras) != len(set(cameras))
        ):
            raise ValueError(
                f"camera_logical_ids must be nonempty, unique strings for {clip_id}"
            )
        overrides[clip_id] = EgoMaskOverride(bank, tuple(cameras))
    return overrides


def _is_identifier(value: object) -> bool:
    """Accept a nonempty, trimmed name without path separators or dot segments."""
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
    )
