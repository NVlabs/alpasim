# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for ``alpasim_utils.services.validate_renderer_config``.

The validator lives in ``alpasim_utils.services`` so the wizard can call it
at submit time without taking a hard dependency on the runtime package; the
runtime is still its primary consumer (worker startup), so the test lives
in the runtime test suite.
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from alpasim_runtime import config
from alpasim_runtime.scene_loader import trajdata_provider_config_to_params
from alpasim_utils.services import validate_renderer_config
from omegaconf import errors as omega_errors


@dataclass
class _FakeRendererConfig:
    """Sample plugin config dataclass used by validator tests."""

    fps: int = 30
    label: str = "demo"


class _FakeServiceWithSchema:
    @classmethod
    def get_config_schema(cls) -> type:
        return _FakeRendererConfig


class _FakeServiceNoSchema:
    pass


class _FakeRegistry:
    """Stand-in PluginRegistry used by the validator tests."""

    # ClassVar so the test file can reassign it per-test without ruff RUF012.
    mapping: ClassVar[dict[str, object]] = {}

    def __init__(self, group: str):
        assert group == "alpasim.services"

    def get(self, name: str):
        if name not in self.mapping:
            raise LookupError(name)
        return self.mapping[name]


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[str, object]
) -> None:
    from tests.conftest import patch_plugin_registry

    _FakeRegistry.mapping = mapping
    patch_plugin_registry(monkeypatch, _FakeRegistry)


def test_validate_renderer_config_sensorsim_passthrough() -> None:
    # Sensorsim has no plugin schema; config passes through untouched.
    result = validate_renderer_config("sensorsim", {"anything": 1})
    assert result == {"anything": 1}


def test_validate_renderer_config_none_input() -> None:
    result = validate_renderer_config("video_model", None)
    assert result is None


def test_validate_renderer_config_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_registry(monkeypatch, {"video_model": _FakeServiceWithSchema})
    result = validate_renderer_config("video_model", {"fps": 24, "label": "hello"})
    assert result == {"fps": 24, "label": "hello"}


def test_validate_renderer_config_unknown_field_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_registry(monkeypatch, {"video_model": _FakeServiceWithSchema})
    with pytest.raises(ValueError, match="Unknown field"):
        validate_renderer_config("video_model", {"not_a_field": 7})


def test_validate_renderer_config_wrong_type_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_registry(monkeypatch, {"video_model": _FakeServiceWithSchema})
    with pytest.raises(omega_errors.OmegaConfBaseException):
        validate_renderer_config("video_model", {"fps": "not-an-int"})


def test_validate_renderer_config_plugin_without_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_registry(monkeypatch, {"video_model": _FakeServiceNoSchema})
    # Plugin exists but doesn't expose a schema: validator no-ops.
    result = validate_renderer_config("video_model", {"foo": "bar"})
    assert result == {"foo": "bar"}


def test_validate_renderer_config_plugin_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_registry(monkeypatch, {})
    # Plugin not installed locally: validator defers to the runtime container.
    result = validate_renderer_config("video_model", {"foo": "bar"})
    assert result == {"foo": "bar"}


def test_usdz_provider_config_defaults() -> None:
    cfg = config.UsdzProviderConfig(data_dir="/data/usdz")

    assert cfg.data_dir == "/data/usdz"
    assert cfg.artifact_cache_size is None


def test_trajdata_provider_config_to_trajdata_params() -> None:
    cfg = config.TrajdataProviderConfig(
        cache_location="/tmp/cache",
        rebuild_cache=True,
        num_workers=8,
        desired_dt=0.05,
        load_vector_map=True,
        dataset=config.TrajdataDatasetConfig(
            name="nuplan",
            data_dir="/data/nuplan",
            extra_params={"config_dir": "/configs"},
        ),
    )

    params = trajdata_provider_config_to_params(cfg)

    assert params["desired_data"] == ["nuplan"]
    assert params["data_dirs"] == {"nuplan": "/data/nuplan"}
    assert params["cache_location"] == "/tmp/cache"
    assert params["rebuild_cache"] is True
    assert params["num_workers"] == 8
    assert params["desired_dt"] == 0.05
    assert params["incl_vector_map"] is True
    assert params["dataset_kwargs"] == {"nuplan": {"config_dir": "/configs"}}


def test_trajdata_provider_config_includes_dataset_extra_params() -> None:
    cfg = config.TrajdataProviderConfig(
        cache_location="/tmp/cache",
        dataset=config.TrajdataDatasetConfig(
            name="nuplan",
            data_dir="/data/nuplan",
            extra_params={
                "config_dir": "/configs",
                "num_timesteps_before": 30,
                "num_timesteps_after": 80,
            },
        ),
    )

    params = trajdata_provider_config_to_params(cfg)

    assert params["desired_data"] == ["nuplan"]
    assert params["data_dirs"]["nuplan"] == "/data/nuplan"
    assert params["dataset_kwargs"]["nuplan"]["config_dir"] == "/configs"
    assert params["dataset_kwargs"]["nuplan"]["num_timesteps_before"] == 30
    assert params["dataset_kwargs"]["nuplan"]["num_timesteps_after"] == 80


def test_trajdata_provider_config_requires_dataset_name() -> None:
    cfg = config.TrajdataProviderConfig(
        cache_location="/tmp/cache",
        dataset=config.TrajdataDatasetConfig(data_dir="/data/nuplan"),
    )

    with pytest.raises(ValueError, match=r"dataset\.name"):
        trajdata_provider_config_to_params(cfg)


def test_trajdata_provider_config_requires_dataset_data_dir() -> None:
    cfg = config.TrajdataProviderConfig(
        cache_location="/tmp/cache",
        dataset=config.TrajdataDatasetConfig(name="nuplan"),
    )

    with pytest.raises(ValueError, match=r"dataset\.data_dir"):
        trajdata_provider_config_to_params(cfg)


def test_trajdata_provider_config_requires_dataset() -> None:
    cfg = config.TrajdataProviderConfig(
        cache_location="/tmp/cache",
    )

    with pytest.raises(ValueError, match=r"scene_provider\.trajdata\.dataset"):
        trajdata_provider_config_to_params(cfg)
