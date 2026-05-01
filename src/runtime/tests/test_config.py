# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import pytest
from alpasim_runtime import config
from alpasim_runtime.scene_loader import trajdata_provider_config_to_params


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

    with pytest.raises(ValueError, match="dataset.name"):
        trajdata_provider_config_to_params(cfg)


def test_trajdata_provider_config_requires_dataset_data_dir() -> None:
    cfg = config.TrajdataProviderConfig(
        cache_location="/tmp/cache",
        dataset=config.TrajdataDatasetConfig(name="nuplan"),
    )

    with pytest.raises(ValueError, match="dataset.data_dir"):
        trajdata_provider_config_to_params(cfg)


def test_trajdata_provider_config_requires_dataset() -> None:
    cfg = config.TrajdataProviderConfig(
        cache_location="/tmp/cache",
    )

    with pytest.raises(ValueError, match="scene_provider.trajdata.dataset"):
        trajdata_provider_config_to_params(cfg)
