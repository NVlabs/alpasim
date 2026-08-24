# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import GTRSDenseConfig
from .vov import VoVNet


class VoVBackbone(nn.Module):
    def __init__(self, config: GTRSDenseConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type
        self.image_encoder = VoVNet(
            spec_name="V-99-eSE",
            out_features=["stage4", "stage5"],
            norm_eval=True,
            with_cp=False,
        )
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (config.img_vert_anchors, config.img_horz_anchors)
        )
        self.img_feat_c = 1024

    def forward(self, image: Tensor) -> Tensor:
        image_features = self.image_encoder(image)[-1]
        return self.avgpool_img(image_features)


class TransfuserBackbone(nn.Module):
    def __init__(self, config: GTRSDenseConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = timm.create_model(
            config.image_architecture,
            pretrained=False,
            features_only=True,
        )
        in_channels = (
            2 * config.lidar_seq_len
            if config.use_ground_plane
            else config.lidar_seq_len
        )
        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn(
                    1,
                    in_channels,
                    config.lidar_resolution_width,
                    config.lidar_resolution_height,
                )
            )

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (config.img_vert_anchors, config.img_horz_anchors)
        )
        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (config.lidar_vert_anchors, config.lidar_horz_anchors)
        )
        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)

        start_index = int(len(self.image_encoder.return_layers) > 4)
        image_info = self.image_encoder.feature_info.info
        lidar_info = self.lidar_encoder.feature_info.info
        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=image_info[start_index + index]["num_chs"],
                    config=config,
                    lidar_time_frames=1,
                )
                for index in range(4)
            ]
        )
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    lidar_info[start_index + index]["num_chs"],
                    image_info[start_index + index]["num_chs"],
                    kernel_size=1,
                )
                for index in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    image_info[start_index + index]["num_chs"],
                    lidar_info[start_index + index]["num_chs"],
                    kernel_size=1,
                )
                for index in range(4)
            ]
        )

        self.num_image_features = image_info[start_index + 3]["num_chs"]
        self.perspective_upsample_factor = (
            image_info[start_index + 3]["reduction"]
            // config.perspective_downsample_factor
        )
        if config.transformer_decoder_join:
            self.num_features = lidar_info[start_index + 3]["num_chs"]
        elif config.add_features:
            self.lidar_to_img_features_end = nn.Linear(
                lidar_info[start_index + 3]["num_chs"],
                image_info[start_index + 3]["num_chs"],
            )
            self.num_features = image_info[start_index + 3]["num_chs"]
        else:
            self.num_features = (
                image_info[start_index + 3]["num_chs"]
                + lidar_info[start_index + 3]["num_chs"]
            )

        self.relu = nn.ReLU(inplace=True)
        if config.detect_boxes or config.use_bev_semantic:
            channel = config.bev_features_channels
            self.upsample = nn.Upsample(
                scale_factor=config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    config.lidar_resolution_height // config.bev_down_sample_factor,
                    config.lidar_resolution_width // config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )
            self.up_conv5 = nn.Conv2d(channel, channel, 3, padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, 3, padding=1)
            self.c5_conv = nn.Conv2d(lidar_info[start_index + 3]["num_chs"], channel, 1)

    def top_down(self, features: Tensor) -> Tensor:
        p5 = self.relu(self.c5_conv(features))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        return self.relu(self.up_conv4(self.upsample2(p4)))

    def forward(
        self,
        image: Tensor,
        lidar: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor, Tensor | None]:
        image_features = image
        lidar_features = lidar
        if self.config.latent:
            lidar_features = self.lidar_latent.repeat(image.shape[0], 1, 1, 1)
        if lidar_features is None:
            raise ValueError("lidar input is required when latent LiDAR is disabled")

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())
        if len(self.image_encoder.return_layers) > 4:
            image_features = self._forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self._forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )

        for index in range(4):
            image_features = self._forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
            lidar_features = self._forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )
            image_features, lidar_features = self._fuse_features(
                image_features, lidar_features, index
            )

        image_feature_grid = (
            image_features
            if self.config.use_semantic or self.config.use_depth
            else None
        )
        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            pooled_image = torch.flatten(self.global_pool_img(image_features), 1)
            pooled_lidar = torch.flatten(self.global_pool_lidar(lidar_features), 1)
            if self.config.add_features:
                fused_features = pooled_image + self.lidar_to_img_features_end(
                    pooled_lidar
                )
            else:
                fused_features = torch.cat((pooled_image, pooled_lidar), dim=1)

        upscaled = (
            self.top_down(lidar_features)
            if self.config.detect_boxes or self.config.use_bev_semantic
            else None
        )
        return upscaled, fused_features, image_feature_grid

    @staticmethod
    def _forward_layer_block(
        layers: Iterator[tuple[str, nn.Module]],
        return_layers: Mapping[str, str],
        features: Tensor,
    ) -> Tensor:
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def _fuse_features(
        self,
        image_features: Tensor,
        lidar_features: Tensor,
        layer_index: int,
    ) -> tuple[Tensor, Tensor]:
        image_embedding = self.avgpool_img(image_features)
        lidar_embedding = self.avgpool_lidar(lidar_features)
        lidar_embedding = self.lidar_channel_to_img[layer_index](lidar_embedding)
        image_delta, lidar_delta = self.transformers[layer_index](
            image_embedding, lidar_embedding
        )
        lidar_delta = self.img_channel_to_lidar[layer_index](lidar_delta)
        image_delta = F.interpolate(
            image_delta,
            size=image_features.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        lidar_delta = F.interpolate(
            lidar_delta,
            size=lidar_features.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return image_features + image_delta, lidar_features + lidar_delta


class GPT(nn.Module):
    def __init__(
        self,
        n_embd: int,
        config: GTRSDenseConfig,
        lidar_time_frames: int,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = 1
        self.lidar_seq_len = config.lidar_seq_len
        self.config = config
        self.lidar_time_frames = lidar_time_frames
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                config.img_vert_anchors * config.img_horz_anchors
                + lidar_time_frames
                * config.lidar_vert_anchors
                * config.lidar_horz_anchors,
                n_embd,
            )
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(
        self, image_tensor: Tensor, lidar_tensor: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch = lidar_tensor.shape[0]
        lidar_height, lidar_width = lidar_tensor.shape[2:4]
        image_height, image_width = image_tensor.shape[2:4]
        image_tokens = (
            image_tensor.permute(0, 2, 3, 1).contiguous().view(batch, -1, self.n_embd)
        )
        lidar_tokens = (
            lidar_tensor.permute(0, 2, 3, 1).contiguous().view(batch, -1, self.n_embd)
        )
        output = self.ln_f(
            self.blocks(
                self.drop(self.pos_emb + torch.cat((image_tokens, lidar_tokens), dim=1))
            )
        )
        image_count = self.config.img_vert_anchors * self.config.img_horz_anchors
        image_output = (
            output[:, :image_count]
            .view(batch, image_height, image_width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_output = (
            output[:, image_count:]
            .view(batch, lidar_height, lidar_width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return image_output, lidar_output


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError("embedding dimension must be divisible by heads")
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, inputs: Tensor) -> Tensor:
        batch, tokens, channels = inputs.shape
        head_channels = channels // self.n_head
        key = (
            self.key(inputs)
            .view(batch, tokens, self.n_head, head_channels)
            .transpose(1, 2)
        )
        query = (
            self.query(inputs)
            .view(batch, tokens, self.n_head, head_channels)
            .transpose(1, 2)
        )
        value = (
            self.value(inputs)
            .view(batch, tokens, self.n_head, head_channels)
            .transpose(1, 2)
        )
        weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        weights = self.attn_drop(F.softmax(weights, dim=-1))
        output = (
            (weights @ value).transpose(1, 2).contiguous().view(batch, tokens, channels)
        )
        return self.resid_drop(self.proj(output))


class Block(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_exp: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attn(self.ln1(inputs))
        return inputs + self.mlp(self.ln2(inputs))
