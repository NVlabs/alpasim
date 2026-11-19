# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Discover clip-specific recorded-navigation files."""

import filecmp
import json
from pathlib import Path


def find_route_indices(root: Path, clip_ids: set[str]) -> dict[str, Path]:
    """Find route indices beneath a mounted navigation version root.

    Args:
        root: Directory containing ``<bare_clip_id>.pkl`` files in suite
            subtrees, or published ``route_indices*.json`` files mapping bare clip
            IDs to route indices. Other files are ignored.
        clip_ids: Requested source IDs, with or without the ``clipgt-`` prefix.

    Returns:
        Requested IDs mapped to the file holding their route index; uncovered clips
        are omitted. Identical copies resolve to the lexicographically first path.

    Raises:
        NotADirectoryError: The version root does not exist or is not a directory.
        ValueError: A requested clip has conflicting route payloads.
    """
    if not root.is_dir():
        raise NotADirectoryError(root)
    keys = {clip_id.removeprefix("clipgt-") for clip_id in clip_ids}
    paths: dict[str, Path] = {}
    published: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix == ".pkl" and path.stem in keys:
            previous = paths.setdefault(path.stem, path)
            if previous != path and (
                previous.suffix != ".pkl"
                or not filecmp.cmp(previous, path, shallow=False)
            ):
                raise ValueError(
                    f"clip {path.stem!r} has conflicting route indices: "
                    f"{previous} and {path}"
                )
        elif path.suffix == ".json" and path.name.startswith("route_indices"):
            for key, payload in json.loads(path.read_text()).items():
                if key not in keys:
                    continue
                previous = paths.setdefault(key, path)
                if previous != path and published.get(key) != payload:
                    raise ValueError(
                        f"clip {key!r} has conflicting route indices: "
                        f"{previous} and {path}"
                    )
                published[key] = payload
    return {
        clip_id: paths[key]
        for clip_id in clip_ids
        if (key := clip_id.removeprefix("clipgt-")) in paths
    }
