# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Typed configuration for the CATK traffic gRPC service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from omegaconf import MISSING, OmegaConf


def _default_map_element_names() -> list[str]:
    return [
        "lane_lines",
        "lane_centers",
        "road_boundaries",
        "road_islands",
        "crosswalks",
        "wait_lines",
    ]


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 6200
    max_workers: int = 1
    log_file: str | None = None


@dataclass
class CatkLoaderConfig:
    usdz_folder: str = MISSING
    num_history_steps: int = 16
    minimum_future_steps: int = 5
    time_step: float = 0.1

    map_element_names: list[str] | None = field(
        default_factory=_default_map_element_names
    )
    map_polyline_length_k: int = 4
    map_resample_interval_m: float | None = 1.0
    map_polyline_filter_mode: str = "v_to_ego_and_obs"
    map_max_pts_to_ego_distance: float = 25.0
    map_polyline_number_control_mode: str = "adv"
    map_adv_max_lane_polylines_num: int = 1000
    map_adv_max_road_boundary_num: int = 750
    map_adv_max_other_polylines_num: int = 250

    map_distance_x: float = 0.0
    map_distance_y: float = 0.0


@dataclass
class CatkModelConfig:
    config_path: str = MISSING
    ckpt_path: str = MISSING
    token_pkl_dir: str = MISSING
    disable_sub_plyline_type: bool = True
    use_downsampled_lines: bool = False


@dataclass
class CatkConfig:
    device: str = "cuda"
    filter_distance_th: float = 100.0
    predict_static: bool = False
    min_valid_history_steps: int = 5
    loader: CatkLoaderConfig = field(default_factory=CatkLoaderConfig)
    model: CatkModelConfig = field(default_factory=CatkModelConfig)


@dataclass
class TrafficServerConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    catk: CatkConfig = field(default_factory=CatkConfig)


def resolve_traffic_server_config(cfg: Any) -> TrafficServerConfig:
    """Merge a Hydra/OmegaConf config with the typed traffic service schema."""
    schema = OmegaConf.structured(TrafficServerConfig)
    merged = OmegaConf.merge(schema, cfg)
    OmegaConf.resolve(merged)
    return cast(TrafficServerConfig, OmegaConf.to_object(merged))
