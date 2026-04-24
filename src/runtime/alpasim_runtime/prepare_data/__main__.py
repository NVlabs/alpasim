# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Data preprocessing CLI for building trajdata cache.

This module provides two clear paths for data preprocessing:

1. **User Config Path** (--user-config): For complex scenarios
   - Load full configuration from YAML file
   - Supports multiple data sources, hierarchical config
   - Supports YAML batch mode (NuPlan central_tokens)
   - CLI overrides limited to: --rebuild-cache, --rebuild-maps, --verbose

2. **CLI Path**: For simple, quick preprocessing
   - Specify all parameters via command line
   - Single dataset preprocessing only
   - Basic preprocessing mode only (no YAML batch mode)
   - Good for testing or simple caching tasks

Usage Examples:

    # Complex: Use user config with optional overrides
    python -m alpasim_runtime.prepare_data --user-config user.yaml --rebuild-cache

    # Simple: Direct CLI parameters for basic preprocessing
    python -m alpasim_runtime.prepare_data \\
        --desired-data nuplan_test \\
        --data-dir /path/to/nuplan \\
        --cache-location /path/to/cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from alpasim_runtime.config import UserSimulatorConfig, typed_parse_config
from omegaconf import OmegaConf
from trajdata.dataset import UnifiedDataset

logger = logging.getLogger(__name__)


@dataclass
class PrepareDataConfig:
    """Configuration for CLI-based data preprocessing.

    This is used ONLY for CLI mode. User config mode uses DataSourceConfig directly.

    Note: CLI mode only supports basic preprocessing. For YAML batch mode (NuPlan),
    use user-config files with config_dir in extra_params.
    """

    # Data source parameters (required)
    desired_data: List[str]
    data_dirs: Dict[str, str]
    cache_location: str

    # Optional preprocessing parameters
    rebuild_cache: bool = False
    rebuild_maps: bool = False
    incl_vector_map: bool = True
    desired_dt: float = 0.1
    num_workers: int = 1

    def to_trajdata_params(self) -> dict:
        """Convert to flat parameters for trajdata's UnifiedDataset.

        Returns:
            Dictionary with keys expected by UnifiedDataset constructor
        """
        return {
            "desired_data": self.desired_data,
            "data_dirs": self.data_dirs,
            "cache_location": self.cache_location,
            "rebuild_cache": self.rebuild_cache,
            "rebuild_maps": self.rebuild_maps,
            "num_workers": self.num_workers,
            "desired_dt": self.desired_dt,
            "incl_vector_map": self.incl_vector_map,
        }


def load_yaml_configs(config_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    Load all yaml configuration files and group them by central_log.

    Supports both simple YAML files and NuPlan-generated files with Python object tags.
    Uses a custom loader to handle Python objects without requiring module imports.

    Args:
        config_dir: Directory containing yaml configuration files.

    Returns:
        Dict where key is central_log and value is the list of central_tokens configs for that log.
    """
    configs_by_log = defaultdict(list)

    yaml_files = list(config_dir.glob("*.yaml"))
    logger.info(f"Found {len(yaml_files)} yaml configuration files.")

    # Custom YAML loader that converts unknown Python objects to dicts
    class SafeLoaderWithObjects(yaml.SafeLoader):
        """Custom YAML loader that treats Python objects as plain dicts."""

        pass

    def python_object_constructor(loader, tag_suffix, node):
        """Convert Python object tags to plain dicts.

        Args:
            loader: YAML loader instance
            tag_suffix: Tag suffix (for multi_constructor, ignored for single constructor)
            node: YAML node to construct
        """
        return loader.construct_mapping(node, deep=True)

    def python_tuple_constructor(loader, tag_suffix, node):
        """Convert Python tuple tags to lists.

        Args:
            loader: YAML loader instance
            tag_suffix: Tag suffix (ignored)
            node: YAML node to construct
        """
        return loader.construct_sequence(node, deep=True)

    # Register constructors for Python objects and tuples
    # Note: add_multi_constructor passes 3 args (loader, tag_suffix, node)
    SafeLoaderWithObjects.add_multi_constructor(
        "tag:yaml.org,2002:python/object", python_object_constructor
    )
    SafeLoaderWithObjects.add_multi_constructor(
        "tag:yaml.org,2002:python/tuple", python_tuple_constructor
    )

    for yaml_file in yaml_files:
        try:
            # Load YAML with custom loader that handles Python objects
            config = yaml.load(yaml_file.read_text(), Loader=SafeLoaderWithObjects)

            # Support both attribute-style (config.central_log) and dict-style access
            if hasattr(config, "central_log"):
                central_log = config.central_log
                central_tokens = config.central_tokens
            else:
                central_log = config.get("central_log", "")
                central_tokens = config.get("central_tokens", [])

            if not central_log or not central_tokens:
                logger.warning(
                    f"{yaml_file.name} is missing central_log or central_tokens, skipping."
                )
                continue

            for token in central_tokens:
                configs_by_log[central_log].append(
                    {
                        "central_token": token,
                        "logfile": central_log,
                        "yaml_file": str(yaml_file),
                    }
                )

        except Exception as e:
            logger.error(f"Failed to load {yaml_file.name}: {e}")
            continue

    logger.info(
        f"\nAfter grouping by central_log, there are {len(configs_by_log)} different log files."
    )
    for log, configs in configs_by_log.items():
        logger.info(f"  {log}: {len(configs)} central tokens")

    return dict(configs_by_log)


def process_nuplan_yaml_configs(
    dataset_name: str, extra_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Process NuPlan YAML configuration files into central_tokens_config format.

    Args:
        dataset_name: Name of the NuPlan dataset (e.g., 'nuplan_mini', 'nuplan_test')
        extra_params: Dictionary containing 'config_dir' and optional timestep parameters

    Returns:
        Processed dataset kwargs with central_tokens_config, or None if no configs found
    """
    logger.info(f"Processing NuPlan YAML configs for {dataset_name}")
    config_dir = Path(extra_params["config_dir"])

    # Load YAML configs
    configs_by_log = load_yaml_configs(config_dir)

    if not configs_by_log:
        logger.warning(f"No valid YAML configs found in {config_dir}")
        return None

    # Build central_tokens_config list
    all_central_tokens_config: List[Dict[str, Any]] = []
    for _, configs in configs_by_log.items():
        for cfg in configs:
            all_central_tokens_config.append(
                {
                    "central_token": cfg["central_token"],
                    "logfile": cfg["logfile"],
                }
            )

    logger.info(f"  Found {len(all_central_tokens_config)} central tokens")

    # Return processed config
    return {
        "central_tokens_config": all_central_tokens_config,
        "num_timesteps_before": extra_params.get("num_timesteps_before", 30),
        "num_timesteps_after": extra_params.get("num_timesteps_after", 80),
    }


def preprocess_basic(config: Any, verbose: bool = True) -> bool:
    """Basic preprocessing - build trajdata cache for all scenes.

    Args:
        config: Configuration (supports both PrepareDataConfig from CLI and
                DataSourceConfig from user config).
        verbose: Whether to show verbose output from trajdata.

    Returns:
        True if successful, False otherwise.
    """
    params = config.to_trajdata_params()

    logger.info("Data source configuration:")
    logger.info(f"  cache_location: {params['cache_location']}")
    logger.info(f"  desired_dt: {params['desired_dt']}")
    logger.info(f"  rebuild_cache: {params['rebuild_cache']}")
    logger.info(f"  rebuild_maps: {params['rebuild_maps']}")
    logger.info(f"  desired_data: {params['desired_data']}")
    logger.info(f"  data_dirs: {params['data_dirs']}")

    # Process NuPlan-specific YAML configs if present
    dataset_kwargs = params.get("dataset_kwargs", {})
    if dataset_kwargs:
        for dataset_name, extra_params in dataset_kwargs.items():
            # Check if this is a NuPlan dataset (nuplan, nuplan_mini, nuplan_test, etc.)
            if "nuplan" in dataset_name.lower() and "config_dir" in extra_params:
                processed_config = process_nuplan_yaml_configs(
                    dataset_name, extra_params
                )
                if processed_config:
                    dataset_kwargs[dataset_name] = processed_config

    # Create cache directory
    cache_path = Path(params["cache_location"])
    cache_path.mkdir(parents=True, exist_ok=True)

    # Build UnifiedDataset (this triggers cache building)
    logger.info("Creating UnifiedDataset...")
    start_time = time.perf_counter()

    try:
        dataset = UnifiedDataset(
            desired_data=params["desired_data"],
            data_dirs=params["data_dirs"],
            cache_location=params["cache_location"],
            incl_vector_map=params["incl_vector_map"],
            rebuild_cache=params["rebuild_cache"],
            rebuild_maps=params["rebuild_maps"],
            desired_dt=params["desired_dt"],
            num_workers=params["num_workers"],
            dataset_kwargs=dataset_kwargs,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Failed to create UnifiedDataset: {e}")
        import traceback

        traceback.print_exc()
        return False

    elapsed = time.perf_counter() - start_time
    logger.info(f"UnifiedDataset created in {elapsed:.2f} seconds")

    # Get scene count
    num_scenes = dataset.num_scenes()
    logger.info(f"Scene files (logs): {num_scenes}")

    logger.info("Data preparation complete!")
    return True


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for prepare_data CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare scene data and build trajdata cache for alpasim simulations.\n\n"
            "Two modes:\n"
            "  1. User config (--user-config): Complex scenarios with full YAML config\n"
            "  2. CLI mode (--desired-data + --data-dir): Simple, direct preprocessing"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_argument_group("Mode Selection")
    mode_group.add_argument(
        "--user-config",
        type=str,
        help="Path to user config YAML file containing data_source configuration",
    )

    # Data source parameters
    data_group = parser.add_argument_group("Data Source")
    data_group.add_argument(
        "--desired-data",
        type=str,
        nargs="+",
        help="List of dataset names to prepare (e.g., nuplan_test, waymo_val, usdz)",
    )
    data_group.add_argument(
        "--data-dir",
        type=str,
        action="append",
        dest="data_dirs",
        help="Data directory (format: dataset_name=/path/to/data or just /path/to/data)",
    )
    data_group.add_argument(
        "--cache-location",
        type=str,
        help="Path to trajdata cache directory",
    )

    # Preprocessing options
    preprocess_group = parser.add_argument_group("Preprocessing Options")
    preprocess_group.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force rebuild cache even if it already exists",
    )
    preprocess_group.add_argument(
        "--rebuild-maps",
        action="store_true",
        help="Force rebuild map cache",
    )
    preprocess_group.add_argument(
        "--desired-dt",
        type=float,
        default=0.1,
        help="Desired timestep duration in seconds (default: 0.1)",
    )
    preprocess_group.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    preprocess_group.add_argument(
        "--no-vector-map",
        action="store_true",
        help="Exclude vector maps (default: include)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    output_group.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show verbose output (default: enabled)",
    )

    return parser


def parse_data_dirs(
    data_dirs_args: Optional[List[str]], desired_data: List[str]
) -> Dict[str, str]:
    """Parse data directory arguments into a dict.

    Supports two formats:
    - "dataset_name=/path/to/data" - explicit mapping
    - "/path/to/data" - auto-map to desired_data entries in order

    Args:
        data_dirs_args: List of data directory arguments
        desired_data: List of dataset names

    Returns:
        Dictionary mapping dataset names to data directories
    """
    if not data_dirs_args:
        return {}

    result: Dict[str, str] = {}

    for i, arg in enumerate(data_dirs_args):
        if "=" in arg:
            # Explicit mapping: dataset_name=/path/to/data
            parts = arg.split("=", 1)
            result[parts[0]] = parts[1]
        else:
            # Implicit mapping: use desired_data order
            if i < len(desired_data):
                result[desired_data[i]] = arg
            else:
                # Use as default for remaining datasets (with warning)
                for ds in desired_data[len(result) :]:
                    if ds not in result:
                        logger.warning(
                            f"Dataset '{ds}' has no explicit data_dir, using last provided: '{arg}'"
                        )
                        result[ds] = arg

    return result


def run_from_user_config(config_path: str, args: argparse.Namespace) -> bool:
    """Run preprocessing from user config file with minimal CLI overrides.

    Args:
        config_path: Path to user config YAML file
        args: Parsed command line arguments (used only for overrides)

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Loading configuration from: {config_path}")
    user_config = typed_parse_config(config_path, UserSimulatorConfig)
    user_config = OmegaConf.to_object(user_config)

    config = user_config.data_source

    # Apply minimal CLI overrides (only top-level flags)
    if args.rebuild_cache:
        config.rebuild_cache = True
        logger.info("CLI override: rebuild_cache=True")
    if args.rebuild_maps:
        config.rebuild_maps = True
        logger.info("CLI override: rebuild_maps=True")

    # Use unified preprocessing (handles both basic and YAML batch mode)
    return preprocess_basic(config, verbose=args.verbose)


def run_from_cli(args: argparse.Namespace) -> bool:
    """Run preprocessing from CLI arguments directly.

    CLI mode only supports basic preprocessing with uniform parameters applied
    to all datasets. For dataset-specific parameters (e.g., different
    smooth_trajectories per dataset) or YAML batch mode, use --user-config.

    Args:
        args: Parsed command line arguments

    Returns:
        True if successful, False otherwise
    """
    # Validate required arguments
    if not args.desired_data:
        logger.error("--desired-data is required when not using --user-config")
        return False
    if not args.data_dirs:
        logger.error("--data-dir is required when not using --user-config")
        return False
    if not args.cache_location:
        logger.error("--cache-location is required when not using --user-config")
        return False

    # Build simple configuration
    data_dirs = parse_data_dirs(args.data_dirs, args.desired_data)

    # Validate that all datasets have data directories
    missing = set(args.desired_data) - set(data_dirs.keys())
    if missing:
        logger.error(f"Missing data directories for datasets: {missing}")
        logger.error("Provide --data-dir for each dataset: dataset=/path or in order")
        return False

    incl_vector_map = not args.no_vector_map

    config = PrepareDataConfig(
        desired_data=args.desired_data,
        data_dirs=data_dirs,
        cache_location=args.cache_location,
        incl_vector_map=incl_vector_map,
        rebuild_cache=args.rebuild_cache,
        rebuild_maps=args.rebuild_maps,
        desired_dt=args.desired_dt,
        num_workers=args.num_workers,
    )

    logger.info("Mode: CLI-based basic preprocessing")
    return preprocess_basic(config, verbose=args.verbose)


def main(arg_list: Optional[List[str]] = None) -> int:
    """Main entry point for prepare_data CLI."""
    parser = create_arg_parser()
    args = parser.parse_args(arg_list)

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Alpasim Data Preparation Tool")
    logger.info("=" * 60)

    # Route to appropriate mode
    try:
        if args.user_config:
            success = run_from_user_config(args.user_config, args)
        else:
            success = run_from_cli(args)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
