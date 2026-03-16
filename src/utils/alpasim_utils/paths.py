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


def extract_ids_from_path(file_path: str) -> tuple[str, str, str]:
    """
    Extract clipgt_id, batch_id, and rollout_id from a file path.

    This function parses paths in the unified rollout directory structure used
    by both post-eval (ASL files) and runtime-eval (metrics files).

    Expected path format:
        .../rollouts/<clipgt_id>/<batch_id>/<filename>.<ext>

    Examples:
        >>> extract_ids_from_path("/data/rollouts/scene_001/0/rollout.asl")
        ('scene_001', '0', 'rollout')

        >>> extract_ids_from_path("/logs/rollouts/clip42/batch_1/metrics.parquet")
        ('clip42', 'batch_1', 'metrics')

    Args:
        file_path: Path to a file within the rollouts directory structure.
                   Can be a string path on any OS.

    Returns:
        Tuple of (clipgt_id, batch_id, rollout_id) where:
        - clipgt_id: The clip/ground-truth identifier (parent's parent directory)
        - batch_id: The batch identifier (parent directory)
        - rollout_id: The file stem (filename without extension)

        Returns ("unknown", "0", "unknown") if path has fewer than 3 parts.
    """
    # Use pathlib for OS-agnostic path parsing
    path = Path(file_path)
    parts = path.parts

    if len(parts) >= 3:
        clipgt_id = parts[-3]
        batch_id = parts[-2]
        rollout_id = path.stem  # filename without extension
        return clipgt_id, batch_id, rollout_id

    return "unknown", "0", "unknown"
