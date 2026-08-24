# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Implements the TransFuser vision backbone.
"""

import copy
import math

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .config import TransfuserConfig


class TransfuserBackbone(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config: TransfuserConfig):

        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model(
            config.image_architecture,
            pretrained=False,
            features_only=True,
        )
        if config.use_ground_plane:
            in_channels = 2 * config.lidar_seq_len
        else:
            in_channels = config.lidar_seq_len

        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn(
                    (
                        1,
                        in_channels,
                        config.lidar_resolution_width,
                        config.lidar_resolution_height,
                    ),
                    requires_grad=True,
                )
            )

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (self.config.lidar_vert_anchors, self.config.lidar_horz_anchors)
        )
        lidar_time_frames = [1, 1, 1, 1]

        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        # Some networks have a stem layer
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[start_index + i][
                        "num_chs"
                    ],
                    config=config,
                    # lidar_video=self.lidar_video,
                    lidar_time_frames=lidar_time_frames[i],
                )
                for i in range(4)
            ]
        )
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )

        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3][
            "num_chs"
        ]
        # Typical encoders down-sample by a factor of 32
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        if self.config.transformer_decoder_join:
            self.num_features = self.lidar_encoder.feature_info.info[start_index + 3][
                "num_chs"
            ]
        else:
            if self.config.add_features:
                self.lidar_to_img_features_end = nn.Linear(
                    self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"],
                )
                # Number of features the encoder produces.
                self.num_features = self.image_encoder.feature_info.info[
                    start_index + 3
                ]["num_chs"]
            else:
                # Number of features the encoder produces.
                self.num_features = (
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
                    + self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
                )

        # FPN fusion
        channel = self.config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        # top down
        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height
                    // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width
                    // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

            # lateral
            self.c5_conv = nn.Conv2d(
                self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                channel,
                (1, 1),
            )

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def forward(self, image, lidar):
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        """
        image_features, lidar_features = image, lidar

        if self.config.latent and lidar_features is None:
            batch_size = image.shape[0]
            lidar_features = self.lidar_latent.repeat(batch_size, 1, 1, 1)

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # Stem layer.
        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )

        # Loop through the 4 blocks of the network.
        for i in range(4):
            image_features = self.forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )

            image_features, lidar_features = self.fuse_features(
                image_features, lidar_features, i
            )

        if self.config.detect_boxes or self.config.use_bev_semantic:
            x4 = lidar_features

        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features

        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            image_features = self.global_pool_img(image_features)
            image_features = torch.flatten(image_features, 1)
            lidar_features = self.global_pool_lidar(lidar_features)
            lidar_features = torch.flatten(lidar_features, 1)

            if self.config.add_features:
                lidar_features = self.lidar_to_img_features_end(lidar_features)
                fused_features = image_features + lidar_features
            else:
                fused_features = torch.cat((image_features, lidar_features), dim=1)

        if self.config.detect_boxes or self.config.use_bev_semantic:
            features = self.top_down(x4)
        else:
            features = None

        return features, fused_features, image_feature_grid

    def forward_layer_block(self, layers, return_layers, features):
        """
        Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block
        :param layers: Iterator starting at the current layer block
        :param return_layers: TIMM dictionary describing at which intermediate layers features are returned.
        :param features: Input features
        :return: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(self, image_features, lidar_features, layer_idx):
        """
        Perform a TransFuser feature fusion block using a Transformer module.
        :param image_features: Features from the image branch
        :param lidar_features: Features from the LiDAR branch
        :param layer_idx: Transformer layer index.
        :return: image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)

        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer, lidar_embd_layer
        )
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer
        )

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    """The full GPT language backbone, with a context size of block_size."""

    # def __init__(self, n_embd, config, lidar_video, lidar_time_frames):
    def __init__(self, n_embd, config, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1
        self.lidar_seq_len = config.lidar_seq_len
        self.config = config
        self.lidar_time_frames = lidar_time_frames

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len
                * self.config.img_vert_anchors
                * self.config.img_horz_anchors
                + lidar_time_frames
                * self.config.lidar_vert_anchors
                * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )

        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for layer in range(config.n_layer)
            ]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
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

    def forward(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
        """

        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]

        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = (
            image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        lidar_tensor = (
            lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[
                :,
                : self.seq_len
                * self.config.img_vert_anchors
                * self.config.img_horz_anchors,
                :,
            ]
            .view(bz * self.seq_len, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[
                :,
                self.seq_len
                * self.config.img_vert_anchors
                * self.config.img_horz_anchors :,
                :,
            ]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the
    end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = (
            self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        q = (
            self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        v = (
            self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class MultiheadAttentionWithAttention(nn.Module):
    """
    MultiheadAttention that also return attention weights
    """

    def __init__(self, n_embd, n_head, pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, q_in, k_in, v_in):
        b, t, c = q_in.size()
        _, t_mem, _ = k_in.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q = (
            self.query(q_in).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        k = (
            self.key(k_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        v = (
            self.value(v_in)
            .view(b, t_mem, self.n_head, c // self.n_head)
            .transpose(1, 2)
        )  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        attention = torch.mean(att, dim=1)  # Average attention over heads
        return y, attention


class TransformerDecoderLayerWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, tgt, memory):
        x = tgt
        tmp, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(tmp))
        tmp, attention = self.multihead_attn(x, memory, memory)
        x = self.norm2(x + self.dropout2(tmp))
        tmp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(tmp))

        return x, attention


class TransformerDecoderWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(self, layers, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layers) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, queries, memory):
        output = queries
        attentions = []
        for mod in self.layers:
            output, attention = mod(output, memory)
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        return output, avg_attention
