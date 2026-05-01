# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from dataclasses import dataclass, field
from pathlib import Path

import omegaconf
import pytest
import yaml
from alpasim_utils.yaml_utils import load_yaml_dict, typed_parse_config
from omegaconf import MISSING


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


@dataclass
class _Inner:
    weight: float = 1.0


@dataclass
class _Config:
    name: str = MISSING
    count: int = 3
    inner: _Inner = field(default_factory=_Inner)


@dataclass
class _OtherConfig:
    other_field: str = MISSING


def test_typed_parse_config_valid(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("name: hello\ninner:\n  weight: 2.5\n", encoding="utf-8")

    cfg = typed_parse_config(path, _Config)

    assert cfg.name == "hello"
    assert cfg.count == 3  # default preserved
    assert cfg.inner.weight == 2.5


def test_typed_parse_config_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("name: hello\nother_field: nope\n", encoding="utf-8")

    with pytest.raises(omegaconf.errors.ConfigKeyError):
        typed_parse_config(path, _Config)


def test_typed_parse_config_schema_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("name: hello\n", encoding="utf-8")

    with pytest.raises(omegaconf.errors.ConfigKeyError):
        typed_parse_config(path, _OtherConfig)


def test_typed_parse_config_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        typed_parse_config(tmp_path / "missing.yaml", _Config)


def test_typed_parse_config_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "not_yaml.txt"
    path.write_text("&&&this is not a yaml file\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        typed_parse_config(path, _Config)
