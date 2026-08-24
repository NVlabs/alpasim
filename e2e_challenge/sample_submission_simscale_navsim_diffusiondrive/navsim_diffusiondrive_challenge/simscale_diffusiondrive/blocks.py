# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math

import numpy as np
import torch
import torch.nn as nn


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/"""
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class GridSampleCrossBEVAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        num_levels=1,
        in_bev_dims=64,
        num_points=8,
        config=None,
    ):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)

        self.value_proj = nn.Sequential(
            nn.Conv2d(
                in_bev_dims,
                256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """

        bs, num_queries, num_points, _ = traj_points.shape

        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = (
            normalized_trajectory[..., 0] / self.config.lidar_max_y
        )
        normalized_trajectory[..., 1] = (
            normalized_trajectory[..., 1] / self.config.lidar_max_x
        )

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y

        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(
            -1
        )

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )  # bs, C, num_queries, num_points

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries
