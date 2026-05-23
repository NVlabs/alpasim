# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Data preprocessing module for building trajdata cache.

This module provides two approaches for preparing scene data before simulations:

1. **User Config Path** (Recommended for complex scenarios):
   - Load configuration from YAML file
   - Supports multiple data sources with individual settings
   - Automatic NuPlan YAML batch processing when config_dir is provided
   - Full control over per-dataset parameters

2. **CLI Path** (For simple, quick preprocessing):
   - Specify parameters via command line
   - Uniform parameters applied to all datasets
   - Good for testing or simple caching tasks

Main exports:
    - preprocess_basic: Unified preprocessing function (handles both basic and NuPlan YAML modes)
    - process_nuplan_yaml_configs: Process NuPlan YAML configs into central_tokens format
    - load_yaml_configs: Load and parse YAML configuration files
    - PrepareDataConfig: Configuration class for CLI mode
    - main: CLI entry point

Example usage (programmatic):

    from prepare_trajdata import preprocess_basic, PrepareDataConfig

    # Simple preprocessing with CLI config
    config = PrepareDataConfig(
        desired_data=["waymo"],
        data_dirs={"waymo": "/path/to/waymo"},
        cache_location="/path/to/cache",
        desired_dt=0.1,
    )
    preprocess_basic(config, verbose=True)

    # Or use user config (recommended for production)
    from alpasim_runtime.config import UserSimulatorConfig
    from alpasim_utils.yaml_utils import typed_parse_config
    user_config = typed_parse_config("user.yaml", UserSimulatorConfig)
    preprocess_basic(user_config.data_source, verbose=True)

CLI usage:
    # Simple mode
    python -m prepare_trajdata \\
        --desired-data waymo \\
        --data-dir /path/to/waymo \\
        --cache-location /path/to/cache

    # Complex mode with user config
    python -m prepare_trajdata \\
        --user-config user.yaml \\
        --rebuild-cache
"""

from prepare_trajdata.cli import (
    PrepareDataConfig,
    load_yaml_configs,
    main,
    preprocess_basic,
    process_nuplan_yaml_configs,
)

__all__ = [
    "preprocess_basic",
    "process_nuplan_yaml_configs",
    "load_yaml_configs",
    "PrepareDataConfig",
    "main",
]
