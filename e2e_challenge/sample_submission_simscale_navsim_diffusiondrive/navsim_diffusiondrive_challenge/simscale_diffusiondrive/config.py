# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Dependency-free inference configuration for SimScale DiffusionDrive."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrajectorySampling:
    """Sampling metadata required by the trajectory head."""

    num_poses: int = 8
    time_horizon: float = 4.0
    interval_length: float = 0.5


@dataclass
class DiffusionDriveConfig:
    """Inference-only configuration matching the released NAVHARD model."""

    trajectory_sampling: TrajectorySampling = field(default_factory=TrajectorySampling)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    bkb_path: str = ""

    latent: bool = True
    lidar_seq_len: int = 1
    use_ground_plane: bool = False

    lidar_max_x: float = 32.0
    lidar_max_y: float = 32.0
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = 8
    img_horz_anchors: int = 32
    lidar_vert_anchors: int = 8
    lidar_horz_anchors: int = 8

    block_exp: int = 4
    n_layer: int = 2
    n_head: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    perspective_downsample_factor: int = 1
    transformer_decoder_join: bool = True
    detect_boxes: bool = True
    use_bev_semantic: bool = True
    use_semantic: bool = False
    use_depth: bool = False
    add_features: bool = True

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    num_bounding_boxes: int = 30
    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
