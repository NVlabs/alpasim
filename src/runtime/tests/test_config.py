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
        desired_data=["nuplan_mini"],
        data_dirs={"nuplan_mini": "/data/nuplan"},
        cache_location="/tmp/cache",
    )
    assert cfg.desired_data == ["nuplan_mini"]
    assert cfg.data_dirs == {"nuplan_mini": "/data/nuplan"}
    assert cfg.cache_location == "/tmp/cache"
    assert cfg.incl_vector_map is True
    assert cfg.rebuild_cache is False
    assert cfg.rebuild_maps is False
    assert cfg.desired_dt == 0.1
    assert cfg.num_workers == 1
    assert cfg.num_timesteps_before == 30
    assert cfg.num_timesteps_after == 80
    assert cfg.config_dir is None
    assert cfg.asset_base_path is None


def test_data_source_config_optional_fields():
    """Test that DataSourceConfig optional fields work correctly."""
    cfg = config.DataSourceConfig(
        desired_data=["usdz"],
        data_dirs={"usdz": "/data/usdz"},
        cache_location="/tmp/cache",
        config_dir="/configs",
        asset_base_path="/assets",
        rebuild_cache=True,
        desired_dt=0.05,
    )
    assert cfg.config_dir == "/configs"
    assert cfg.asset_base_path == "/assets"
    assert cfg.rebuild_cache is True
    assert cfg.desired_dt == 0.05
