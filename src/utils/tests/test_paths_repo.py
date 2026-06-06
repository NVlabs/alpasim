# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path

import pytest
from alpasim_utils.paths import (
    find_repo_root,
    image_to_sqsh_basename,
    is_alpasim_repo_root,
)


def test_find_repo_root_from_directory(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    (repo_root / "src" / "wizard").mkdir(parents=True)
    (repo_root / "src" / "utils").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "alpasim_workspace"\n')
    (repo_root / ".git").write_text("gitdir: /tmp/worktree\n")

    assert find_repo_root(nested) == repo_root


def test_find_repo_root_from_file_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    marker_file = nested / "config.yaml"
    marker_file.write_text("x: 1\n")
    (repo_root / "src" / "wizard").mkdir(parents=True)
    (repo_root / "src" / "utils").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "alpasim_workspace"\n')
    (repo_root / ".git").mkdir()

    assert find_repo_root(marker_file) == repo_root


def test_find_repo_root_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path / "missing")


def test_find_repo_root_from_packaged_source_tree(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "src" / "wizard" / "alpasim_wizard"
    nested.mkdir(parents=True)
    (repo_root / "src" / "utils").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "alpasim_workspace"\n')

    assert find_repo_root(nested) == repo_root


def test_is_alpasim_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "src" / "wizard").mkdir(parents=True)
    (repo_root / "src" / "utils").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "alpasim_workspace"\n')

    assert is_alpasim_repo_root(repo_root)
    assert not is_alpasim_repo_root(tmp_path / "other")


def test_is_alpasim_repo_root_rejects_member_pyproject(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "wizard"
    package_dir.mkdir(parents=True)
    (package_dir / "pyproject.toml").write_text('[project]\nname = "alpasim_wizard"\n')

    assert not is_alpasim_repo_root(package_dir)


def test_find_repo_root_plus_src_from_nested_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "src" / "eval" / "src" / "eval"
    nested.mkdir(parents=True)
    (repo_root / "src" / "wizard").mkdir(parents=True)
    (repo_root / "src" / "utils").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "alpasim_workspace"\n')
    (repo_root / ".git").write_text("gitdir: /tmp/worktree\n")

    assert find_repo_root(nested) / "src" == repo_root / "src"


def test_image_to_sqsh_basename() -> None:
    image = "nvcr.io/nvidian/alpamayo/alpasim-runtime:0.32.0-abc123"
    assert image_to_sqsh_basename(image) == "alpasim_runtime_0.32.0_abc123.sqsh"
