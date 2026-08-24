# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Configuration for the inference-only SimScale LTF model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransfuserConfig:
    """Parameters that determine the released TransFuser architecture."""

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    latent: bool = True
    lidar_seq_len: int = 1
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
    use_ground_plane: bool = False
