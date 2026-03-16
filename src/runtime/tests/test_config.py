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
