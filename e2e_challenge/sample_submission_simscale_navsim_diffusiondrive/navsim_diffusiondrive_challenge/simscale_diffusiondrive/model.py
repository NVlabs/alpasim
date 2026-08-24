# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import copy
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TransfuserBackbone
from .blocks import (
    GridSampleCrossBEVAttention,
    bias_init_with_prob,
    gen_sineembed_for_position,
    linear_relu_ln,
)
from .config import DiffusionDriveConfig
from .scheduler import DDIMScheduler


class BoundingBox2DIndex:
    """Indices for the five released bounding-box regression values."""

    POINT = slice(0, 2)
    HEADING = 2

    @staticmethod
    def size() -> int:
        return 5


class StateSE2Index:
    """Indices for local x, y, heading poses."""

    HEADING = 2


class SinusoidalPosEmb(nn.Module):
    """Timestep embedding used by the released trajectory head."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=x.device) * -scale)
        embedding = x[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class DiffusionDriveModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(
        self,
        config: DiffusionDriveConfig,
        plan_anchor: np.ndarray,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        if plan_anchor.shape != (20, config.trajectory_sampling.num_poses, 2):
            raise ValueError(
                "plan anchor shape must be "
                f"(20, {config.trajectory_sampling.num_poses}, 2), got {plan_anchor.shape}"
            )
        if plan_anchor.dtype != np.float32:
            raise ValueError(f"plan anchor must use float32, got {plan_anchor.dtype}")
        if not np.isfinite(plan_anchor).all():
            raise ValueError("plan anchor must contain only finite values")

        self._backbone = (
            backbone if backbone is not None else TransfuserBackbone(config)
        )

        self._keyval_embedding = nn.Embedding(
            8**2 + 1, config.tf_d_model
        )  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor=plan_anchor,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1, 320),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        if isinstance(status_feature, list):
            status_feature = status_feature[-1]
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]

        batch_size = status_feature.shape[0]

        # bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = (
            keyval[:, :-1]
            .permute(0, 2, 1)
            .contiguous()
            .view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        )
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(
            concat_cross_bev,
            size=bev_spatial_shape,
            mode="bilinear",
            align_corners=False,
        )
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(
            cross_bev_feature.flatten(-2, -1).permute(0, 2, 1)
        )
        cross_bev_feature = (
            cross_bev_feature.permute(0, 2, 1)
            .contiguous()
            .view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        )
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        trajectory = self._trajectory_head(
            trajectory_query,
            agents_query,
            cross_bev_feature,
            bev_spatial_shape,
            status_encoding[:, None],
            noise=noise,
            global_img=None,
        )

        # Keep release heads in the strict state dict even though service inference
        # consumes only the selected trajectory.
        _ = bev_semantic_map
        _ = self._agent_head(agents_query)
        return trajectory


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls


class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2, 3).permute(0, 2, 1).contiguous()
            global_feature = torch.cat([global_img, global_feature], axis=-1)

        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        num_poses,
        d_model,
        d_ffn,
        config,
    ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model, 256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
        global_img=None,
    ):
        traj_feature = self.cross_bev_attention(
            traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape
        )
        traj_feature = traj_feature + self.dropout(
            self.cross_agent_attention(traj_feature, agents_query, agents_query)[0]
        )
        traj_feature = self.norm1(traj_feature)

        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(
            self.cross_ego_attention(traj_feature, ego_query, ego_query)[0]
        )
        traj_feature = self.norm2(traj_feature)

        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(
            traj_feature, time_embed, global_cond=None, global_img=global_img
        )

        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature)  # bs,20,8,3; bs,20
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = (
            poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi
        )

        return poses_reg, poses_cls


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self,
        traj_feature,
        noisy_traj_points,
        bev_feature,
        bev_spatial_shape,
        agents_query,
        ego_query,
        time_embed,
        status_encoding,
        global_img=None,
    ):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(
                traj_feature,
                traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[..., :2].clone().detach()
        return poses_reg_list, poses_cls_list


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(
        self,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        plan_anchor: np.ndarray,
        config: DiffusionDriveConfig,
    ) -> None:
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        self.plan_anchor = nn.Parameter(
            torch.from_numpy(plan_anchor.copy()),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2 * (odo_info_fut_x + 1.2) / 56.9 - 1
        odo_info_fut_y = 2 * (odo_info_fut_y + 20) / 46 - 1
        odo_info_fut_head = 2 * (odo_info_fut_head + 2) / 3.9 - 1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1) / 2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def forward(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: tuple[int, int],
        status_encoding: torch.Tensor,
        noise: torch.Tensor | None = None,
        global_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the released two-step inference path."""
        return self.forward_test(
            ego_query,
            agents_query,
            bev_feature,
            bev_spatial_shape,
            status_encoding,
            noise,
            global_img,
        )

    def forward_test(
        self,
        ego_query: torch.Tensor,
        agents_query: torch.Tensor,
        bev_feature: torch.Tensor,
        bev_spatial_shape: tuple[int, int],
        status_encoding: torch.Tensor,
        noise: torch.Tensor | None,
        global_img: torch.Tensor | None,
    ) -> torch.Tensor:
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (
            (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        )
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        img = self.norm_odo(plan_anchor)
        if noise is None:
            noise = torch.randn(img.shape, device=device, dtype=img.dtype)
        elif noise.shape != img.shape:
            raise ValueError(
                f"noise must have shape {tuple(img.shape)}, got {tuple(noise.shape)}"
            )
        else:
            noise = noise.to(device=device, dtype=img.dtype)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(
            original_samples=img, noise=noise, timesteps=trunc_timesteps
        )
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(
                noisy_traj_points, hidden_dim=64
            )
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor(
                    [timesteps], dtype=torch.long, device=img.device
                )
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)

            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs, 1, -1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(
                traj_feature,
                noisy_traj_points,
                bev_feature,
                bev_spatial_shape,
                agents_query,
                ego_query,
                time_embed,
                status_encoding,
                global_img,
            )
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[..., :2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start, timestep=k, sample=img
            ).prev_sample
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses, 3)
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        return best_reg
