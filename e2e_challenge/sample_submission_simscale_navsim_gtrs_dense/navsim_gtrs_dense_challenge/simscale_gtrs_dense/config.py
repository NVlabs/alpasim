# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GTRSDenseConfig:
    num_poses: int = 40
    vocab_size: int = 8192
    vocab_dropout: bool = True
    normalize_vocab_pos: bool = True
    num_ego_status: int = 1
    use_nerf: bool = False
    use_back_view: bool = False

    backbone_type: str = "resnet"
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    latent: bool = True
    use_ground_plane: bool = False
    lidar_seq_len: int = 4

    camera_width: int = 2048
    camera_height: int = 512
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256
    img_vert_anchors: int = 16
    img_horz_anchors: int = 64
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
    use_bev_semantic: bool = False
    use_semantic: bool = False
    use_depth: bool = False
    add_features: bool = True

    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0
    vadv2_head_nhead: int = 8
    vadv2_head_nlayers: int = 3

    num_bounding_boxes: int = 30
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    def __post_init__(self) -> None:
        if self.backbone_type not in {"resnet", "vov"}:
            raise ValueError("backbone_type must be 'resnet' or 'vov'")
        if self.num_poses != 40:
            raise ValueError("GTRS-Dense release trajectories require 40 poses")
        if self.use_nerf or self.use_back_view:
            raise ValueError("NeRF and rear-view variants are not supported")
