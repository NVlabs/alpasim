# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Server script for DigitalTwin Sensorsim Service.

This script starts a gRPC server that exposes the worldengine DigitalTwin renderer
as a SensorsimService, allowing it to replace the default sensorsim in alpasim.
"""

import argparse
import json
import logging
from concurrent import futures
from pathlib import Path

import grpc

from alpasim_grpc.v0.sensorsim_pb2_grpc import add_SensorsimServiceServicer_to_server
from alpasim_runtime.services.digitaltwin_sensorsim_service import (
    DigitalTwinSensorsimService,
)

logger = logging.getLogger(__name__)


def parse_args(arg_list: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DigitalTwin Sensorsim Service Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact-glob",
        type=str,
        help="Glob expression to find USDZ artifacts. Must end in .usdz",
        required=True,
    )
    parser.add_argument(
        "--asset-folder-path",
        type=str,
        help="Path to DigitalTwin asset folder containing scene assets",
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


def main(arg_list: list[str] | None = None) -> None:
    """Main entry point for the server."""
    args, _ = parse_args(arg_list)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )
    
    logger.info("Starting DigitalTwin Sensorsim Service")
    logger.info(f"Artifact glob: {args.artifact_glob}")
    logger.info(f"Asset folder path: {args.asset_folder_path}")
    logger.info(f"Device: {args.device}")
    
    # Load scene_id to asset_id mapping if provided
    scene_id_mapping = load_scene_id_mapping(args.scene_id_to_asset_id_mapping)
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)
    
    # Create and register service
    service = DigitalTwinSensorsimService(
        server=server,
        artifact_glob=args.artifact_glob,
        asset_folder_path=args.asset_folder_path,
        scene_id_to_asset_id_mapping=scene_id_mapping,
        cache_size=args.cache_size,
        device=args.device,
    )
    add_SensorsimServiceServicer_to_server(service, server)
    
    logger.info(f"Serving DigitalTwin Sensorsim Service on {address}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        server.stop(0)


if __name__ == "__main__":
    main()
