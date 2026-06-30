# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from concurrent import futures
from pathlib import Path

import hydra
from alpasim_grpc.v0 import traffic_pb2_grpc
from alpasim_trafficsim.grpc.config import (
    TrafficServerConfig,
    resolve_traffic_server_config,
)
from alpasim_trafficsim.grpc.servicer import TrafficServiceServicer
from loguru import logger
from omegaconf import DictConfig

import grpc


def _validate_usdz_folder(cfg: TrafficServerConfig) -> Path:
    usdz_folder = Path(cfg.catk.loader.usdz_folder)
    if not usdz_folder.is_dir():
        raise FileNotFoundError(
            f"catk.loader.usdz_folder does not exist or is not a directory: {usdz_folder}"
        )
    if not any(usdz_folder.rglob("*.usdz")):
        raise FileNotFoundError(
            f"catk.loader.usdz_folder contains no .usdz files recursively: {usdz_folder}"
        )
    return usdz_folder


def serve(cfg: TrafficServerConfig) -> grpc.Server:
    usdz_folder = _validate_usdz_folder(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg.server.max_workers))
    servicer = TrafficServiceServicer(
        server,
        usdz_folder=usdz_folder,
        catk_config=cfg.catk,
    )
    traffic_pb2_grpc.add_TrafficServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{cfg.server.host}:{cfg.server.port}")
    logger.info(
        "Starting CATK traffic gRPC server on {}:{}",
        cfg.server.host,
        cfg.server.port,
    )
    server.start()
    server.wait_for_termination()
    return server


@hydra.main(version_base=None, config_path="../config", config_name="server")
def main(hydra_cfg: DictConfig) -> None:
    cfg = resolve_traffic_server_config(hydra_cfg)

    log_file_str = cfg.server.log_file
    if log_file_str is not None:
        log_path = Path(log_file_str)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, enqueue=True)

    serve(cfg)


if __name__ == "__main__":
    main()
