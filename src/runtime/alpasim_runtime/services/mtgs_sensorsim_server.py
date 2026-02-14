# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Server script for MTGS Sensorsim Service.

This script starts a gRPC server that exposes the MTGS renderer
as a SensorsimService, allowing it to replace the default sensorsim in alpasim.
"""

import argparse
import json
import logging
from concurrent import futures
from pathlib import Path

import grpc

from alpasim_grpc.v0.sensorsim_pb2_grpc import add_SensorsimServiceServicer_to_server
from alpasim_runtime.services.mtgs_sensorsim_service import (
    MTGSSensorsimService,
)
from alpasim_utils.data_source_loader import load_data_sources

logger = logging.getLogger(__name__)

# Mapping from trajdata dataset names to MTGS asset folder subdirectories
# Format: "nuplan_<split>" -> "data_nav<split>"
TRAJDATA_TO_ASSET_FOLDER_MAPPING = {
    "nuplan_test": "navtest",
    "nuplan_train": "navtrain",
    # Add more mappings as needed
}


def parse_args(arg_list: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MTGS Sensorsim Service Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    artifact_group = parser.add_mutually_exclusive_group(required=True)
    artifact_group.add_argument(
        "--artifact-glob",
        type=str,
        help="Glob expression to find USDZ artifacts. Must end in .usdz",
    )
    artifact_group.add_argument(
        "--trajdata-config",
        type=str,
        help="Path to trajdata config YAML/JSON file",
    )
    parser.add_argument(
        "--asset-folder-path",
        type=str,
        help="Path to MTGS asset folder containing scene assets",
        required=True,
    )
    parser.add_argument(
        "--scene-id-to-asset-id-mapping",
        type=str,
        help="Optional JSON file mapping scene_id to asset_id",
        default=None,
    )
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=2,
        help="Number of renderer instances to keep in LRU cache",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for rendering",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args, overrides = parser.parse_known_args(arg_list)
    return args, overrides


def load_scene_id_mapping(mapping_path: str | None) -> dict[str, str] | None:
    """Load scene_id to asset_id mapping from JSON file."""
    if mapping_path is None:
        return None
    
    mapping_file = Path(mapping_path)
    if not mapping_file.exists():
        logger.warning(f"Mapping file not found: {mapping_path}")
        return None
    
    try:
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        logger.info(f"Loaded {len(mapping)} scene_id mappings from {mapping_path}")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load mapping file: {e}")
        return None


def resolve_asset_folder_path(
    base_asset_folder_path: str,
    trajdata_config_path: str | None = None,
) -> str:
    """
    Resolve the full asset folder path based on trajdata config.
    
    If trajdata_config_path is provided, reads the desired_data from the config
    and automatically appends the corresponding subdirectory to the base path.
    
    Args:
        base_asset_folder_path: Base path to MTGS asset folder
        trajdata_config_path: Optional path to trajdata config YAML/JSON file
    
    Returns:
        Resolved asset folder path (with subdirectory appended if applicable)
    
    Examples:
        If base_asset_folder_path="/path/to/assets" and trajdata config has
        desired_data=["nuplan_test"], returns "/path/to/assets/data_navtest"
    """
    base_path = Path(base_asset_folder_path)
    
    # If no trajdata config, return base path as-is
    if trajdata_config_path is None:
        return str(base_path)
    
    # Load trajdata config to extract desired_data
    config_path = Path(trajdata_config_path)
    if not config_path.exists():
        logger.warning(
            f"Trajdata config file not found: {config_path}. "
            f"Using base asset folder path: {base_path}"
        )
        return str(base_path)
    
    try:
        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                import yaml
                config = yaml.safe_load(f)
            else:
                # Assume JSON
                config = json.load(f)
        
        # Extract desired_data from config
        desired_data = config.get("desired_data", [])
        if not desired_data:
            logger.warning(
                f"No 'desired_data' found in trajdata config. "
                f"Using base asset folder path: {base_path}"
            )
            return str(base_path)
        
        # Use the first dataset name in desired_data
        # (typically there's only one, but if multiple, use the first)
        dataset_name = desired_data[0] if isinstance(desired_data, list) else desired_data
        
        # Look up the corresponding asset folder subdirectory
        asset_subdir = TRAJDATA_TO_ASSET_FOLDER_MAPPING.get(dataset_name)
        
        if asset_subdir is None:
            logger.warning(
                f"No mapping found for dataset '{dataset_name}'. "
                f"Available mappings: {list(TRAJDATA_TO_ASSET_FOLDER_MAPPING.keys())}. "
                f"Using base asset folder path: {base_path}"
            )
            return str(base_path)
        
        # Construct full path: base_path / asset_subdir
        full_path = base_path / asset_subdir / 'assets'
        
        # Verify the path exists
        if not full_path.exists():
            logger.warning(
                f"Asset folder subdirectory does not exist: {full_path}. "
                f"Using base asset folder path: {base_path}"
            )
            return str(base_path)
        
        logger.info(
            f"Auto-resolved asset folder path: {base_path} -> {full_path} "
            f"(based on dataset '{dataset_name}' -> '{asset_subdir}')"
        )
        return str(full_path)
        
    except Exception as e:
        logger.error(
            f"Failed to resolve asset folder path from trajdata config: {e}. "
            f"Using base asset folder path: {base_path}"
        )
        return str(base_path)


def main(arg_list: list[str] | None = None) -> None:
    """Main entry point for the server."""
    args, _ = parse_args(arg_list)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )
    
    logger.info("Starting MTGS Sensorsim Service")
    
    # Resolve asset folder path based on trajdata config if provided
    # This automatically appends the correct subdirectory (e.g., data_navtest) 
    # based on the desired_data in trajdata config
    resolved_asset_folder_path = resolve_asset_folder_path(
        base_asset_folder_path=args.asset_folder_path,
        trajdata_config_path=args.trajdata_config if not args.artifact_glob else None,
    )
    
    # Load artifacts from either USDZ glob or trajdata config
    if args.artifact_glob:
        logger.info(f"Loading artifacts from USDZ glob: {args.artifact_glob}")
        artifacts = load_data_sources(usdz_glob=args.artifact_glob)
    else:
        logger.info(f"Loading artifacts from trajdata config: {args.trajdata_config}")
        artifacts = load_data_sources(trajdata_config_path=args.trajdata_config)
    
    logger.info(f"Asset folder path: {resolved_asset_folder_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Loaded {len(artifacts)} scenes")
    
    # Load scene_id to asset_id mapping if provided
    scene_id_mapping = load_scene_id_mapping(args.scene_id_to_asset_id_mapping)
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)
    
    # Create and register service
    service = MTGSSensorsimService(
        server=server,
        artifacts=artifacts,
        asset_folder_path=resolved_asset_folder_path,
        scene_id_to_asset_id_mapping=scene_id_mapping,
        cache_size=args.cache_size,
        device=args.device,
    )
    add_SensorsimServiceServicer_to_server(service, server)
    
    logger.info(f"Serving MTGS Sensorsim Service on {address}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        server.stop(0)


if __name__ == "__main__":
    main()