# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Data preprocessing module for building trajdata cache.

This module provides tools for preparing scene data before running simulations.
It supports:

1. Basic preprocessing - Build trajdata cache for all scenes in a dataset
2. YAML config preprocessing - Batch process specific scenes based on YAML configs
3. Central token mode - Process scenes around specific central tokens (NuPlan)

Main functions:
    - preprocess_basic: Basic preprocessing for all scenes
    - preprocess_from_yaml_configs: Batch preprocessing from YAML configs
    - load_yaml_configs: Load YAML configuration files

Example usage:

    from alpasim_runtime.prepare_data import preprocess_basic, preprocess_from_yaml_configs

    # Basic preprocessing
    preprocess_basic(
        desired_data=["nuplan_test"],
        data_dirs={"nuplan_test": "/path/to/nuplan"},
        cache_location="/path/to/cache",
    )

    # YAML config batch preprocessing
    preprocess_from_yaml_configs(
        config_dir=Path("/path/to/configs"),
        cache_location="/path/to/cache",
        data_dirs={"nuplan_test": "/path/to/nuplan"},
        env_name="nuplan_test",
        num_timesteps_before=30,
        num_timesteps_after=80,
    )

CLI usage:
    python -m alpasim_runtime.prepare_data --help
"""

from alpasim_runtime.prepare_data.__main__ import (
    load_yaml_configs,
    main,
    preprocess_basic,
    preprocess_from_yaml_configs,
)

__all__ = [
    "preprocess_basic",
    "preprocess_from_yaml_configs",
    "load_yaml_configs",
    "main",
]
