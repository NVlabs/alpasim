# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Standalone MTGS Sensorsim Service server.

Starts a gRPC server that exposes the MTGS renderer as a SensorsimService.

Usage:
    alpasim-mtgs-server --user-config user_config.yaml --host 0.0.0.0 --port 8080
"""

import argparse
import logging
from concurrent import futures
from pathlib import Path
from typing import Callable, Optional

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
    "nuplan_private": "private",
}


def _build_token_to_asset_folder(configs_dir: Path) -> dict:
    """Read MTGS config YAMLs to build a central_token → road_block_name mapping.

    Each YAML lists all central_tokens sharing one rendered asset folder
    (road_block_name = central_log + '-' + central_tokens[0]).  This mapping
    lets the MTGS server resolve any token to its shared asset folder at
    runtime, regardless of the trajdata cache state.
    """
    import yaml

    class _SafeLoader(yaml.SafeLoader):
        pass

    _SafeLoader.add_multi_constructor(
        "tag:yaml.org,2002:python/object",
        lambda loader, tag, node: loader.construct_mapping(node, deep=True),
    )
    _SafeLoader.add_multi_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, tag, node: loader.construct_sequence(node, deep=True),
    )

    mapping: dict = {}
    if not configs_dir.exists():
        logger.warning("MTGS configs dir not found: %s", configs_dir)
        return mapping

    for yaml_file in configs_dir.glob("*.yaml"):
        try:
            cfg = yaml.load(yaml_file.read_text(), Loader=_SafeLoader)
            if not isinstance(cfg, dict):
                continue
            central_log = cfg.get("central_log", "")
            central_tokens = cfg.get("central_tokens", [])
            if not central_tokens:
                continue
            road_block_name = cfg.get("road_block_name", "")
            if not road_block_name and central_log:
                road_block_name = f"{central_log}-{central_tokens[0]}"
            if not road_block_name:
                continue
            for token in central_tokens:
                mapping[str(token)] = road_block_name
        except Exception as exc:
            logger.warning("Failed to load %s: %s", yaml_file.name, exc)

    logger.info(
        "Built asset-folder mapping for %d tokens from %s", len(mapping), configs_dir
    )
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
        help="LRU cache size for renderer instances",
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

    configs_dir = Path(asset_base_path_config) / mapped_name / "configs"
    token_to_asset_folder = _build_token_to_asset_folder(configs_dir)

    def _asset_folder_resolver(scene) -> str:
        token = scene.name.rsplit("-", 1)[-1]
        return token_to_asset_folder.get(str(token), scene.name)

    asset_folder_resolver: Optional[Callable] = (
        _asset_folder_resolver if token_to_asset_folder else None
    )

    params = trajdata_provider_config_to_params(trajdata_config)
    logger.info("Creating UnifiedDataset from config")
    dataset = UnifiedDataset(**params)
    logger.info(f"Created UnifiedDataset with {dataset.num_scenes()} scenes")

    scene_id_to_idx = scene_name_to_index_from_dataset(dataset)
    logger.info(f"Built scene_id mapping for {len(scene_id_to_idx)} scenes")

    scene_cache = {}

    def get_scene(scene_id: str) -> TrajdataDataSource:
        if scene_id in scene_cache:
            return scene_cache[scene_id]

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
            asset_folder_resolver=asset_folder_resolver,
        )

        scene_cache[scene_id] = data_source
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
        get_scene, get_available_scene_ids = create_get_scene_function(user_config)
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
