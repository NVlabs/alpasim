# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import omegaconf
import pytest
import yaml
from alpasim_runtime import config


def test_typed_parse_config_valid():
    user_cfg = config.typed_parse_config(
        "tests/data/valid_user_config.yaml", config.UserSimulatorConfig
    )
    assert user_cfg.simulation_config.force_gt_duration_us == 1700000

    default = config.SimulationConfig()
    assert user_cfg.simulation_config.control_timestep_us == default.control_timestep_us

    assert len(user_cfg.scenes) == 1
    assert user_cfg.scenes[0].scene_id == "clipgt-f94a6ae5-019e-4467-840f-5376b5255828"

    network_cfg = config.typed_parse_config(
        "tests/data/valid_network_config.yaml", config.NetworkSimulatorConfig
    )
    assert network_cfg.sensorsim.addresses[0] == "nre:6000"
    assert network_cfg.trafficsim.addresses[0] == "trafficsim:6200"


def test_typed_parse_config_invalid_config_type():
    # attempt to create a user config from a network config file
    with pytest.raises(omegaconf.errors.ConfigKeyError):
        config.typed_parse_config(
            "tests/data/valid_network_config.yaml", config.UserSimulatorConfig
        )


def test_typed_parse_config_file_not_found():
    # attempt to create a user config from a non-existent file
    with pytest.raises(FileNotFoundError):
        config.typed_parse_config("non_existent_file.yaml", config.UserSimulatorConfig)


def test_typed_parse_config_invalid_yaml(tmp_path):
    not_yaml = tmp_path / "not_yaml.txt"
    not_yaml.write_text("&&&this is not a yaml file\n")

    with pytest.raises(yaml.YAMLError):
        config.typed_parse_config(not_yaml, config.UserSimulatorConfig)


# TODO(mwatson, mtyszkiewicz): What should happen when the config is empty? Currently,
# no error is raised, and we return an empty config object. Is this the desired behavior?


def test_data_source_config_defaults():
    """Test that DataSourceConfig has sensible defaults."""
    cfg = config.DataSourceConfig(
        cache_location="/tmp/cache",
        sources={"usdz": config.GenericSourceConfig(data_dir="/data/usdz")},
    )
    assert cfg.cache_location == "/tmp/cache"
    assert cfg.rebuild_cache is False
    assert cfg.rebuild_maps is False
    assert cfg.num_workers == 1
    assert "usdz" in cfg.sources
    assert cfg.sources["usdz"].data_dir == "/data/usdz"
    assert cfg.sources["usdz"].extra_params == {}


def test_data_source_config_to_trajdata_params():
    """Test conversion from hierarchical config to flat trajdata parameters."""
    cfg = config.DataSourceConfig(
        cache_location="/tmp/cache",
        rebuild_cache=True,
        num_workers=8,
        desired_dt=0.05,
        incl_vector_map=True,
        sources={
            "usdz": config.GenericSourceConfig(
                data_dir="/data/usdz",
                extra_params={"asset_base_path": "/assets"},
            )
        },
    )
    params = cfg.to_trajdata_params()

    assert params["desired_data"] == ["usdz"]
    assert params["data_dirs"] == {"usdz": "/data/usdz"}
    assert params["cache_location"] == "/tmp/cache"
    assert params["rebuild_cache"] is True
    assert params["num_workers"] == 8
    assert params["desired_dt"] == 0.05
    assert params["incl_vector_map"] is True
    assert params["dataset_kwargs"] == {"usdz": {"asset_base_path": "/assets"}}


def test_data_source_config_multiple_sources():
    """Test configuration with multiple data sources enabled."""
    cfg = config.DataSourceConfig(
        cache_location="/tmp/cache",
        sources={
            "usdz": config.GenericSourceConfig(data_dir="/data/usdz"),
            "nuplan": config.GenericSourceConfig(
                data_dir="/data/nuplan",
                extra_params={
                    "config_dir": "/configs",
                    "num_timesteps_before": 30,
                    "num_timesteps_after": 80,
                },
            ),
        },
    )
    params = cfg.to_trajdata_params()

    assert set(params["desired_data"]) == {"usdz", "nuplan"}
    assert params["data_dirs"]["usdz"] == "/data/usdz"
    assert params["data_dirs"]["nuplan"] == "/data/nuplan"
    assert params["dataset_kwargs"]["nuplan"]["config_dir"] == "/configs"
    assert params["dataset_kwargs"]["nuplan"]["num_timesteps_before"] == 30
    assert params["dataset_kwargs"]["nuplan"]["num_timesteps_after"] == 80


def test_data_source_config_no_sources_enabled():
    """Test that error is raised when no data sources are enabled."""
    cfg = config.DataSourceConfig(
        cache_location="/tmp/cache",
        # All sources are None, so no sources enabled
    )

    with pytest.raises(ValueError, match="No data sources enabled"):
        cfg.to_trajdata_params()
