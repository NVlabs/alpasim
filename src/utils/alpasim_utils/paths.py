# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Path utilities for working with rollout directory structures.

This module provides shared path parsing utilities used across alpasim components.
"""

from pathlib import Path


def find_repo_root(start_path: str | Path) -> Path:
    """Find the nearest git repository root for a path.

    The lookup walks up parent directories until it finds a `.git` marker.
    This works for both regular repositories (`.git/`) and git worktrees
    (`.git` file).
    """
    start = Path(start_path).resolve()
    probe = start if start.is_dir() else start.parent

    for candidate in (probe, *probe.parents):
        git_marker = candidate / ".git"
        if git_marker.exists():
            return candidate

    raise FileNotFoundError(f"Could not find git repository root from {start}")


def image_to_sqsh_basename(image: str) -> str:
    """Return the canonical .sqsh basename for a docker image URL."""
    return Path(image).name.replace(":", "_").replace("-", "_") + ".sqsh"


def extract_ids_from_path(file_path: str) -> tuple[str, str]:
    """
    Extract clipgt_id and rollout_id from a file path.

    Parses the rollout directory layout produced by the runtime::

        <log_dir>/rollouts/<clipgt_id>/<rollout_uuid>/
            rollout.asl
            metrics.parquet
            *.mp4

    Example::

        >>> extract_ids_from_path(".../rollouts/clipgt-01d5…3554/4fe6…dbfc/rollout.asl")
        ('clipgt-01d5…3554', '4fe6…dbfc')

    Args:
        file_path: Path to any file inside a rollout directory.

    Returns:
        ``(clipgt_id, rollout_id)`` extracted from the two parent directories.
        Returns ``("unknown", "unknown")`` if the path is too short.
    """
    path = Path(file_path)
    parts = path.parts

    if len(parts) >= 3:
        return parts[-3], parts[-2]

    return "unknown", "unknown"
