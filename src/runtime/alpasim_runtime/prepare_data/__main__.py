# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Data preprocessing CLI for building trajdata cache.

This module provides a command-line interface for preparing scene data before
running simulations. It supports:

1. Basic preprocessing - Build trajdata cache for all scenes in a dataset
2. YAML config preprocessing - Batch process specific scenes based on YAML configs
3. Central token mode - Process scenes around specific central tokens (NuPlan)

Usage Examples:

    # Basic preprocessing using user-config
    python -m alpasim_runtime.prepare_data --user-config user.yaml

    # Basic preprocessing with explicit parameters
    python -m alpasim_runtime.prepare_data \\
        --desired-data nuplan_test \\
        --data-dir /path/to/nuplan \\
        --cache-location /path/to/cache

    # Rebuild cache even if it exists
    python -m alpasim_runtime.prepare_data --user-config user.yaml --rebuild-cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Optional trajdata import
try:
    from trajdata.dataset import UnifiedDataset

    TRAJDATA_AVAILABLE = True

except ImportError:
    TRAJDATA_AVAILABLE = False
    UnifiedDataset = None
    env_utils = None


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

            # Extract first central_token
            configs_by_log[central_log].append(
                {
                    "central_token": central_tokens[0],
                    "logfile": central_log,
                    "yaml_file": str(yaml_file),
                }
            )
            # Extract every central_token
            # for token in central_tokens:
            #     configs_by_log[central_log].append({
            #         'central_token': token,
            #         'logfile': central_log,
            #         'yaml_file': str(yaml_file),
            #     })

        except Exception as e:
            logger.error(f"Failed to load {yaml_file.name}: {e}")
            continue

    logger.info(
        f"\nAfter grouping by central_log, there are {len(configs_by_log)} different log files."
    )
    for log, configs in configs_by_log.items():
        logger.info(f"  {log}: {len(configs)} central tokens")

    return dict(configs_by_log)


def preprocess_from_yaml_configs(
    config_dir: Path,
    cache_location: str,
    data_dirs: Dict[str, str],
    env_name: str = "nuplan_test",
    rebuild_cache: bool = True,
    rebuild_maps: bool = False,
    num_workers: int = 1,
    desired_dt: float = 0.5,
    num_timesteps_before: int = 30,
    num_timesteps_after: int = 80,
    verbose: bool = True,
) -> bool:
    """
    Batch preprocess data based on YAML configuration files.

    This function reads YAML config files containing central_log and central_tokens,
    and processes only those specific scenes. This is useful for processing
    specific scenarios without loading the entire dataset.

    Args:
        config_dir: Directory containing YAML configuration files.
        cache_location: Path to cache directory.
        data_dirs: Dictionary of dataset name to data directory paths.
        env_name: Environment name (e.g., "nuplan_test").
        rebuild_cache: Whether to rebuild cache.
        rebuild_maps: Whether to rebuild maps.
        num_workers: Number of worker processes.
        desired_dt: Desired timestep duration in seconds.
        num_timesteps_before: Number of timesteps before the central token.
        num_timesteps_after: Number of timesteps after the central token.
        verbose: Whether to show verbose logs.

    Returns:
        True if successful, False otherwise.
    """
    if not TRAJDATA_AVAILABLE:
        logger.error("trajdata is not installed. Please install it first.")
        return False

    # Load all YAML configs
    configs_by_log = load_yaml_configs(config_dir)

    if not configs_by_log:
        logger.error("No valid YAML configuration files found.")
        return False

    # Merge all configs into a single central_tokens_config list
    all_central_tokens_config: List[Dict[str, Any]] = []
    for _, configs in configs_by_log.items():
        for cfg in configs:
            all_central_tokens_config.append(
                {
                    "central_token": cfg["central_token"],
                    "logfile": cfg["logfile"],
                    "num_timesteps_before": num_timesteps_before,
                    "num_timesteps_after": num_timesteps_after,
                }
            )

    logger.info(f"Total {len(all_central_tokens_config)} central tokens to process")

    try:
        # Create cache directory
        Path(cache_location).mkdir(parents=True, exist_ok=True)

        # Create UnifiedDataset (this triggers cache building)
        logger.info("Creating UnifiedDataset with YAML configs...")
        start_time = time.perf_counter()

        dataset = UnifiedDataset(
            dataset_kwargs={
                "central_tokens_config": all_central_tokens_config,
                "num_timesteps_before": num_timesteps_before,
                "num_timesteps_after": num_timesteps_after,
            },
            desired_data=[env_name],
            cache_location=cache_location,
            rebuild_cache=rebuild_cache,
            rebuild_maps=rebuild_maps,
            require_map_cache=False,
            num_workers=num_workers,
            desired_dt=desired_dt,
            verbose=verbose,
            data_dirs=data_dirs,
        )

        elapsed = time.perf_counter() - start_time

        # Get both scene index count and total dataset length
        num_scenes = dataset.num_scenes()
        logger.info("=" * 80)
        logger.info("Preprocessing completed!")
        logger.info(f"  Num Scenes: {num_scenes}")
        logger.info(f"  Time elapsed: {elapsed:.2f} seconds")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


def preprocess_basic(
    desired_data: List[str],
    data_dirs: Dict[str, str],
    cache_location: str,
    rebuild_cache: bool = False,
    rebuild_maps: bool = False,
    incl_vector_map: bool = True,
    desired_dt: float = 0.1,
    num_workers: int = 1,
    verbose: bool = True,
    list_scenes: bool = False,
    smooth_trajectories: bool = True,
) -> bool:
    """
    Basic preprocessing - build trajdata cache for all scenes.

    Args:
        desired_data: List of dataset names to load.
        data_dirs: Dict mapping dataset names to their data directories.
        cache_location: Path to trajdata cache directory.
        rebuild_cache: Whether to force rebuild cache.
        rebuild_maps: Whether to rebuild map cache.
        incl_vector_map: Whether to include vector maps.
        desired_dt: Desired time delta between frames in seconds.
        num_workers: Number of workers for data loading.
        verbose: Whether to show verbose output.
        list_scenes: Whether to list all available scenes after preparation.

    Returns:
        True if successful, False otherwise.
    """
    if not TRAJDATA_AVAILABLE:
        logger.error("trajdata is not installed. Please install it first.")
        return False

    logger.info("Data source configuration:")
    logger.info(f"  desired_data: {desired_data}")
    logger.info(f"  data_dirs: {data_dirs}")
    logger.info(f"  cache_location: {cache_location}")
    logger.info(f"  rebuild_cache: {rebuild_cache}")
    logger.info(f"  desired_dt: {desired_dt}")
    logger.info(f"  smooth_trajectories: {smooth_trajectories}")

    # Create cache directory
    cache_path = Path(cache_location)
    cache_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_path}")

    # Build UnifiedDataset (this triggers cache building)
    logger.info("Creating UnifiedDataset (this may take a while)...")
    start_time = time.perf_counter()

    try:
        dataset = UnifiedDataset(
            desired_data=desired_data,
            data_dirs=data_dirs,
            cache_location=cache_location,
            incl_vector_map=incl_vector_map,
            rebuild_cache=rebuild_cache,
            rebuild_maps=rebuild_maps,
            # require_map_cache=False,
            desired_dt=desired_dt,
            num_workers=num_workers,
            verbose=verbose,
            dataset_kwargs={"smooth_trajectories": smooth_trajectories},
        )
    except Exception as e:
        logger.error(f"Failed to create UnifiedDataset: {e}")
        import traceback

        traceback.print_exc()
        return False

    elapsed = time.perf_counter() - start_time
    logger.info(f"UnifiedDataset created in {elapsed:.2f} seconds")

    # Get both scene index count and total dataset length
    num_scenes = dataset.num_scenes()
    logger.info(f"Scene files (logs): {num_scenes}")

    # List scenes if requested
    if list_scenes and num_scenes > 0:
        logger.info("Available scenes:")
        max_display = min(num_scenes, 100)
        for i in range(max_display):
            try:
                scene = dataset.get_scene(i)
                logger.info(f"  [{i}] {scene.name}")
            except Exception as e:
                logger.warning(f"  [{i}] (failed to load: {e})")

        if num_scenes > 100:
            logger.info(f"  ... and {num_scenes - 100} more scenes")

    logger.info("Data preparation complete!")
    return True


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for prepare_data CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare scene data and build trajdata cache for alpasim simulations.",
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
    preprocess_group.add_argument(
        "--smooth-trajectories",
        type=str,
        choices=["true", "false", "True", "False"],
        default=None,
        help="Enable/disable cubic spline smoothing for trajectories (default: True)",
    )

    # YAML config mode options (NuPlan specific)
    yaml_group = parser.add_argument_group("YAML Config Mode (NuPlan)")
    yaml_group.add_argument(
        "--num-timesteps-before",
        type=int,
        default=30,
        help="Number of timesteps before central token (default: 30)",
    )
    yaml_group.add_argument(
        "--num-timesteps-after",
        type=int,
        default=80,
        help="Number of timesteps after central token (default: 80)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without building cache",
    )
    output_group.add_argument(
        "--list-scenes",
        action="store_true",
        help="List all available scenes after preparation",
    )
    output_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show verbose output (default: True)",
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load data source configuration from user config file."""
    from alpasim_runtime.config import UserSimulatorConfig, typed_parse_config

    user_config = typed_parse_config(config_path, UserSimulatorConfig)

    if user_config.data_source is None:
        raise ValueError(
            f"No data_source configuration found in {config_path}. "
            "Please add a 'data_source' section to your user config."
        )

    ds = user_config.data_source
    return {
        "desired_data": ds.desired_data,
        "data_dirs": ds.data_dirs,
        "cache_location": ds.cache_location,
        "config_dir": ds.config_dir,
        "incl_vector_map": ds.incl_vector_map,
        "rebuild_cache": ds.rebuild_cache,
        "rebuild_maps": ds.rebuild_maps,
        "desired_dt": ds.desired_dt,
        "num_workers": ds.num_workers,
        "num_timesteps_before": ds.num_timesteps_before,
        "num_timesteps_after": ds.num_timesteps_after,
        "smooth_trajectories": user_config.smooth_trajectories,
    }


def parse_data_dirs(
    data_dirs_args: Optional[List[str]], desired_data: List[str]
) -> Dict[str, str]:
    """
    Parse data directory arguments into a dict.

    Supports two formats:
    - "dataset_name=/path/to/data" - explicit mapping
    - "/path/to/data" - auto-map to desired_data entries in order
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
                # Use as default for remaining datasets
                for ds in desired_data[len(result) :]:
                    if ds not in result:
                        result[ds] = arg

    return result


def run_yaml_batch_preprocessing(
    config_dir: str,
    desired_data: List[str],
    data_dirs: Dict[str, str],
    cache_location: str,
    rebuild_cache: bool = False,
    rebuild_maps: bool = False,
    num_workers: int = 1,
    desired_dt: float = 0.5,
    num_timesteps_before: int = 30,
    num_timesteps_after: int = 80,
    verbose: bool = True,
) -> int:
    """
    Execute YAML batch preprocessing mode.

    This is a unified entry point for YAML-based batch preprocessing,
    whether triggered from user config file or CLI arguments.

    Args:
        config_dir: Directory containing YAML scene config files.
        desired_data: List of dataset names (first one will be used as env_name).
        data_dirs: Dictionary mapping dataset names to data directories.
        cache_location: Path to cache directory.
        rebuild_cache: Whether to rebuild cache.
        rebuild_maps: Whether to rebuild maps.
        num_workers: Number of worker processes.
        desired_dt: Desired timestep duration in seconds.
        num_timesteps_before: Number of timesteps before central token.
        num_timesteps_after: Number of timesteps after central token.
        verbose: Whether to show verbose output.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Using YAML config batch preprocessing mode")

    success = preprocess_from_yaml_configs(
        config_dir=Path(config_dir),
        cache_location=cache_location,
        data_dirs=data_dirs,
        env_name=desired_data[0],  # Use first dataset as environment name
        rebuild_cache=rebuild_cache,
        rebuild_maps=rebuild_maps,
        num_workers=num_workers,
        desired_dt=desired_dt,
        num_timesteps_before=num_timesteps_before,
        num_timesteps_after=num_timesteps_after,
        verbose=verbose,
    )

    return 0 if success else 1


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

    verbose = args.verbose and not args.quiet

    logger.info("=" * 60)
    logger.info("Alpasim Data Preparation Tool")
    logger.info("=" * 60)

    if not TRAJDATA_AVAILABLE:
        logger.error("trajdata is not installed. Please install it first:")
        logger.error("pip install trajdata")
        return 1

    # Determine mode and load configuration
    try:
        if args.user_config:
            logger.info(f"Loading configuration from: {args.user_config}")
            config = load_config_from_file(args.user_config)

            # Override with command line args if provided
            if args.rebuild_cache:
                config["rebuild_cache"] = True
            if args.rebuild_maps:
                config["rebuild_maps"] = True
            if args.smooth_trajectories is not None:
                smooth_value = args.smooth_trajectories.lower() in ["true", "1", "yes"]
                config["smooth_trajectories"] = smooth_value
                logger.info(
                    f"Command line overrides config: smooth_trajectories={smooth_value}"
                )

            # Check if user config contains config_dir -> automatically use Mode 2
            if config.get("config_dir") is not None:
                logger.info("Detected 'config_dir' in user config")
                return run_yaml_batch_preprocessing(
                    config_dir=config["config_dir"],
                    desired_data=config["desired_data"],
                    data_dirs=config["data_dirs"],
                    cache_location=config["cache_location"],
                    rebuild_cache=config.get("rebuild_cache", False),
                    rebuild_maps=config.get("rebuild_maps", False),
                    num_workers=config.get("num_workers", 8),
                    desired_dt=config.get("desired_dt", 0.5),
                    num_timesteps_before=config.get("num_timesteps_before", 30),
                    num_timesteps_after=config.get("num_timesteps_after", 80),
                    verbose=verbose,
                )
            # Otherwise continue with Mode 1 (basic mode)

        else:
            if not args.desired_data:
                logger.error("Either --user-config, or --desired-data must be provided")
                return 1
            if not args.data_dirs:
                logger.error("--data-dir is required when not using --user-config")
                return 1
            if not args.cache_location:
                logger.error(
                    "--cache-location is required when not using --user-config"
                )
                return 1

            data_dirs = parse_data_dirs(args.data_dirs, args.desired_data)
            incl_vector_map = not args.no_vector_map

            if args.smooth_trajectories is not None:
                smooth_value = args.smooth_trajectories.lower() in ["true", "1", "yes"]
            else:
                smooth_value = True

            config = {
                "desired_data": args.desired_data,
                "data_dirs": data_dirs,
                "cache_location": args.cache_location,
                "incl_vector_map": incl_vector_map,
                "rebuild_cache": args.rebuild_cache,
                "rebuild_maps": args.rebuild_maps,
                "desired_dt": args.desired_dt,
                "num_workers": args.num_workers,
                "smooth_trajectories": smooth_value,
            }

    except Exception as e:
        logger.error(f"Configuration error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Run basic preprocessing
    success = preprocess_basic(
        desired_data=config["desired_data"],
        data_dirs=config["data_dirs"],
        cache_location=config["cache_location"],
        rebuild_cache=config.get("rebuild_cache", False),
        rebuild_maps=config.get("rebuild_maps", False),
        incl_vector_map=config.get("incl_vector_map", True),
        desired_dt=config.get("desired_dt", 0.1),
        num_workers=config.get("num_workers", 8),
        verbose=verbose,
        list_scenes=args.list_scenes,
        smooth_trajectories=config.get("smooth_trajectories", True),
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
