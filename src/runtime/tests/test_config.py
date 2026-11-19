# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for runtime configuration helpers."""

from dataclasses import fields
from pathlib import Path

import pytest
from alpasim_runtime import config
from alpasim_runtime.data_source_loader import trajdata_provider_config_to_params
from alpasim_utils.yaml_utils import typed_parse_config
from omegaconf import OmegaConf


def test_rollout_retry_default() -> None:
    assert config.UserSimulatorConfig().max_rollout_retries == 2


def test_usdz_provider_config_defaults() -> None:
    cfg = config.UsdzProviderConfig(data_dir="/data/usdz")

    assert cfg.data_dir == "/data/usdz"
    assert cfg.artifact_cache_size is None


def test_runtime_schema_keeps_data_source_provider_config_key() -> None:
    schema = OmegaConf.structured(config.UserSimulatorConfig)

    assert "data_source_provider" in schema.keys()
    assert "scene_provider" not in schema.keys()
    assert "clip_provider" not in schema.keys()
    assert "clip_affine_dispatch" in schema.keys()
    field_types = {
        field.name: field.type for field in fields(config.UserSimulatorConfig)
    }
    assert field_types["data_source_provider"] is config.DataSourceProviderConfig
    assert field_types["clip_affine_dispatch"] is config.ClipAffineDispatchConfig


def test_generated_user_config_parses_data_source_and_clip_affinity() -> None:
    user_config = typed_parse_config(
        Path(__file__).parent / "data" / "mock" / "user-config.yaml",
        config.UserSimulatorConfig,
    )

    assert user_config.data_source_provider.kind == "usdz"
    assert user_config.clip_affine_dispatch.enabled is False
    assert len(user_config.scenes) == 1


def test_clip_affine_dispatch_config_defaults() -> None:
    cfg = config.UserSimulatorConfig().clip_affine_dispatch

    assert cfg.enabled is False
    assert cfg.max_renderers_per_clip == 2
    assert cfg.max_clips_per_renderer is None


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

    with pytest.raises(ValueError, match=r"data_source_provider\.trajdata\.dataset"):
        trajdata_provider_config_to_params(cfg)
