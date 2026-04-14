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


def test_simulation_config_parses_driver_backends(tmp_path):
    cfg_path = tmp_path / "user.yaml"
    cfg_path.write_text(
        """
endpoints:
  sensorsim: {n_concurrent_rollouts: 1}
  driver: {n_concurrent_rollouts: 1}
  physics: {n_concurrent_rollouts: 1}
  trafficsim: {n_concurrent_rollouts: 1}
  controller: {n_concurrent_rollouts: 1}
simulation_config:
  n_sim_steps: 1
  n_rollouts: 1
  observation_cache_size: 12
  observation_window_summary_size: 6
  driver_backends:
    - backend_id: ar1_backend
      model_type: ar1
      priority: 5
    - backend_id: pdm_backend
      model_type: pdm
      priority: 1
scenes:
  - scene_id: clipgt-a
"""
    )

    user_cfg = config.typed_parse_config(cfg_path, config.UserSimulatorConfig)

    assert len(user_cfg.simulation_config.driver_backends) == 2
    assert user_cfg.simulation_config.driver_backends[1].model_type == "pdm"
    assert user_cfg.simulation_config.observation_cache_size == 12
    assert user_cfg.simulation_config.observation_window_summary_size == 6


# TODO(mwatson, mtyszkiewicz): What should happen when the config is empty? Currently,
# no error is raised, and we return an empty config object. Is this the desired behavior?
