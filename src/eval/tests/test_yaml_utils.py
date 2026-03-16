# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path

import pytest
from alpasim_utils.yaml_utils import load_yaml_dict


def test_load_yaml_dict_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\nb: test\n", encoding="utf-8")

    assert load_yaml_dict(path) == {"a": 1, "b": "test"}


def test_load_yaml_dict_missing_ok_returns_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    assert load_yaml_dict(path, missing_ok=True) == {}


def test_load_yaml_dict_missing_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml_dict(path)
