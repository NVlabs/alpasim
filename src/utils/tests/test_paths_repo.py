# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path

import pytest
from alpasim_utils.paths import find_repo_root, image_to_sqsh_basename


def test_find_repo_root_from_directory(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    (repo_root / ".git").write_text("gitdir: /tmp/worktree\n")

    assert find_repo_root(nested) == repo_root


def test_find_repo_root_from_file_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    marker_file = nested / "config.yaml"
    marker_file.write_text("x: 1\n")
    (repo_root / ".git").mkdir()

    assert find_repo_root(marker_file) == repo_root


def test_find_repo_root_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path / "missing")


def test_find_repo_root_plus_src_from_nested_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "src" / "eval" / "src" / "eval"
    nested.mkdir(parents=True)
    (repo_root / ".git").write_text("gitdir: /tmp/worktree\n")

    assert find_repo_root(nested) / "src" == repo_root / "src"


def test_image_to_sqsh_basename() -> None:
    image = "nvcr.io/nvidian/alpamayo/alpasim-runtime:0.32.0-abc123"
    assert image_to_sqsh_basename(image) == "alpasim_runtime_0.32.0_abc123.sqsh"
