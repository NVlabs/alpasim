# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Standalone MTGS Sensorsim Service server.

Starts a gRPC server that exposes the MTGS renderer as a SensorsimService.

Usage:
    alpasim-mtgs-server --user-config user_config.yaml --host 0.0.0.0 --port 8080
"""

import argparse
import functools
import logging
from concurrent import futures
from pathlib import Path
from typing import Callable

from alpasim_grpc.v0.sensorsim_pb2_grpc import add_SensorsimServiceServicer_to_server
from alpasim_mtgs.server.servicer import MTGSSensorsimService
from alpasim_runtime.config import UserSimulatorConfig
from alpasim_runtime.scene_loader import (
    scene_name_to_index_from_dataset,
    trajdata_provider_config_to_params,
)
from alpasim_utils.yaml_utils import typed_parse_config

import grpc

try:
    from alpasim_utils.trajdata_data_source import TrajdataDataSource

    from trajdata.dataset import UnifiedDataset

    TRAJDATA_AVAILABLE = True
except ImportError:
    TRAJDATA_AVAILABLE = False
    UnifiedDataset = None
    TrajdataDataSource = None

logger = logging.getLogger(__name__)

DATASET_NAME_MAPPING = {
    "nuplan_test": "navtest",
    "nuplan_mini": "navtest",
}


def _asset_folder_map_from_extra_params(extra_params) -> dict[str, str]:
    """Return a validated scene-to-MTGS-asset mapping from dataset config."""
    raw_mapping = extra_params.get("asset_folder_map")
    if raw_mapping is None:
        return {}
    if not hasattr(raw_mapping, "items"):
        raise ValueError(
            "scene_provider.trajdata.dataset.extra_params.asset_folder_map "
            "must be a mapping of scene IDs to asset folder IDs."
        )

    mapping: dict[str, str] = {}
    for scene_id, asset_folder in raw_mapping.items():
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("asset_folder_map scene IDs must be non-empty strings.")
        if not isinstance(asset_folder, str) or not asset_folder:
            raise ValueError(
                f"asset_folder_map[{scene_id!r}] must be a non-empty string."
            )
        mapping[scene_id] = asset_folder
    return mapping


def parse_args(arg_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MTGS Sensorsim Service Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--user-config", type=str, required=True, help="Path to user config YAML"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=2,
        help="LRU cache size for renderer instances and scene data sources",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(arg_list)


def create_get_scene_function(
    user_config: UserSimulatorConfig,
    cache_size: int = 2,
) -> tuple[Callable, Callable]:
    if not TRAJDATA_AVAILABLE:
        raise ImportError("trajdata is required for MTGS sensorsim server.")

    scene_provider = user_config.scene_provider
    if scene_provider is None or scene_provider.trajdata is None:
        raise ValueError("scene_provider.trajdata is required in user config.")

    trajdata_config = scene_provider.trajdata
    dataset_config = trajdata_config.dataset
    if dataset_config is None:
        raise ValueError("scene_provider.trajdata.dataset is required in user config.")

    extra_params = dataset_config.extra_params or {}
    asset_base_path_config = extra_params.get("asset_base_path")
    if not asset_base_path_config:
        raise ValueError(
            "scene_provider.trajdata.dataset.extra_params.asset_base_path "
            "is required for MTGS rendering."
        )

    mapped_name = DATASET_NAME_MAPPING.get(dataset_config.name, dataset_config.name)
    mtgs_asset_base_path = str(Path(asset_base_path_config) / mapped_name / "assets")
    logger.info(f"MTGS asset path: {mtgs_asset_base_path}")
    asset_folder_map = _asset_folder_map_from_extra_params(extra_params)
    if asset_folder_map:
        logger.info(
            "Loaded %d explicit scene-to-asset-folder mappings",
            len(asset_folder_map),
        )

    params = trajdata_provider_config_to_params(trajdata_config)
    logger.info("Creating UnifiedDataset from config")
    dataset = UnifiedDataset(**params)
    logger.info(f"Created UnifiedDataset with {dataset.num_scenes()} scenes")

    scene_id_to_idx = scene_name_to_index_from_dataset(dataset)
    logger.info(f"Built scene_id mapping for {len(scene_id_to_idx)} scenes")

    @functools.lru_cache(maxsize=cache_size)
    def get_scene(scene_id: str) -> TrajdataDataSource:
        scene_idx = scene_id_to_idx.get(scene_id)
        if scene_idx is None:
            raise KeyError(f"Scene {scene_id} not found in dataset")

        scene = dataset.get_scene(scene_idx)
        if scene is None:
            raise KeyError(f"Scene at index {scene_idx} not found")

        data_source = TrajdataDataSource(
            scene=scene,
            scene_cache=dataset.get_scene_cache(scene),
            map_api=dataset.map_api,
            vector_map_params=dataset.vector_map_params,
            smooth_trajectories=user_config.smooth_trajectories,
            asset_base_path=mtgs_asset_base_path,
            asset_folder_resolver=(
                (
                    lambda resolved_scene: asset_folder_map.get(
                        resolved_scene.name, resolved_scene.name
                    )
                )
                if asset_folder_map
                else None
            ),
        )

        logger.info(f"Loaded scene {scene_id}, asset_path={data_source.asset_path}")
        return data_source

    def get_available_scene_ids() -> list[str]:
        return list(scene_id_to_idx.keys())

    return get_scene, get_available_scene_ids


def main(arg_list: list[str] | None = None) -> None:
    args = parse_args(arg_list)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("MTGS Sensorsim Service Server")
    logger.info(
        f"Config: {args.user_config}, Host: {args.host}:{args.port}, Device: {args.device}"
    )

    try:
        user_config = typed_parse_config(args.user_config, UserSimulatorConfig)
    except Exception as e:
        logger.error(f"Failed to load user config: {e}")
        return

    if (
        user_config.scene_provider is None
        or user_config.scene_provider.trajdata is None
    ):
        logger.error("scene_provider.trajdata is required in user config.")
        return

    try:
        get_scene, get_available_scene_ids = create_get_scene_function(
            user_config, cache_size=args.cache_size
        )
    except Exception as e:
        logger.error(f"Failed to create get_scene function: {e}")
        return

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)

    service = MTGSSensorsimService(
        server=server,
        get_scene=get_scene,
        get_available_scene_ids=get_available_scene_ids,
        cache_size=args.cache_size,
        device=args.device,
    )
    add_SensorsimServiceServicer_to_server(service, server)

    logger.info(f"Server ready on {address}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(0)


if __name__ == "__main__":
    main()
