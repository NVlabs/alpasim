# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical


@torch.no_grad()
def cal_polygon_contour(
    pos: Tensor,  # [n_agent, n_step, n_target, 2]
    head: Tensor,  # [n_agent, n_step, n_target]
    width_length: Tensor,  # [n_agent, 1, 1, 2]
) -> Tensor:  # [n_agent, n_step, n_target, 4, 2]
    x, y = pos[..., 0], pos[..., 1]  # [n_agent, n_step, n_target]
    width, length = width_length[..., 0], width_length[..., 1]  # [n_agent, 1 ,1]

    half_cos = 0.5 * head.cos()  # [n_agent, n_step, n_target]
    half_sin = 0.5 * head.sin()  # [n_agent, n_step, n_target]
    length_cos = length * half_cos  # [n_agent, n_step, n_target]
    length_sin = length * half_sin  # [n_agent, n_step, n_target]
    width_cos = width * half_cos  # [n_agent, n_step, n_target]
    width_sin = width * half_sin  # [n_agent, n_step, n_target]

    left_front_x = x + length_cos - width_sin
    left_front_y = y + length_sin + width_cos
    left_front = torch.stack((left_front_x, left_front_y), dim=-1)

    right_front_x = x + length_cos + width_sin
    right_front_y = y + length_sin - width_cos
    right_front = torch.stack((right_front_x, right_front_y), dim=-1)

    right_back_x = x - length_cos + width_sin
    right_back_y = y - length_sin - width_cos
    right_back = torch.stack((right_back_x, right_back_y), dim=-1)

    left_back_x = x - length_cos - width_sin
    left_back_y = y - length_sin + width_cos
    left_back = torch.stack((left_back_x, left_back_y), dim=-1)

    polygon_contour = torch.stack(
        (left_front, right_front, right_back, left_back), dim=-2
    )

    return polygon_contour


def transform_to_global(
    pos_local: Tensor,  # [n_agent, n_step, 2]
    head_local: Optional[Tensor],  # [n_agent, n_step]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos

    pos_global = torch.bmm(pos_local, rot_mat)  # [n_agent, n_step, 2]*[n_agent, 2, 2]
    pos_global = pos_global + pos_now.unsqueeze(1)
    if head_local is None:
        head_global = None
    else:
        head_global = head_local + head_now.unsqueeze(1)
    return pos_global, head_global


def transform_to_local(
    pos_global: Tensor,  # [n_agent, n_step, 2]
    head_global: Optional[Tensor],  # [n_agent, n_step]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos

    pos_local = pos_global - pos_now.unsqueeze(1)
    pos_local = torch.bmm(pos_local, rot_mat)  # [n_agent, n_step, 2]*[n_agent, 2, 2]
    if head_global is None:
        head_local = None
    else:
        head_local = head_global - head_now.unsqueeze(1)
    return pos_local, head_local


def sample_next_token_traj(
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    token_traj_all: Tensor,  # [n_agent, n_token, 6, 4, 2]
    sampling_scheme: DictConfig,
    # ! for most-likely sampling
    next_token_logits: Tensor,  # [n_agent, n_token], with grad
    # ! for nearest-pos sampling, sampling near to GT
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
    pos_next_gt: Tensor | None = None,  # [n_agent, 2]
    head_next_gt: Tensor | None = None,  # [n_agent]
    valid_next_gt: Tensor | None = None,  # [n_agent]
    token_agent_shape: Tensor | None = None,  # [n_agent, 2]
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
        next_token_idx: [n_agent], without grad
        next_token_traj_all: [n_agent, 6, 4, 2], local coord
        sample_logits: [n_agent, n_token], without grad
    """
    range_a = torch.arange(next_token_logits.shape[0])
    next_token_logits = next_token_logits.detach()

    if (
        sampling_scheme.criterium == "topk_prob"
        or sampling_scheme.criterium == "topk_prob_sampled_with_dist"
    ):
        topk_logits, topk_indices = torch.topk(
            next_token_logits, sampling_scheme.num_k, dim=-1, sorted=False
        )
        if sampling_scheme.criterium == "topk_prob_sampled_with_dist":
            # gt_contour: [n_agent, 4, 2] in global coord
            gt_contour = cal_polygon_contour(
                pos_next_gt, head_next_gt, token_agent_shape
            )
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]
            token_world_sample = token_traj[range_a.unsqueeze(1), topk_indices]
            token_world_sample = transform_to_global(
                pos_local=token_world_sample.flatten(1, 2),
                head_local=None,
                pos_now=pos_now,  # [n_agent, 2]
                head_now=head_now,  # [n_agent]
            )[0].view(*token_world_sample.shape)

            # dist: [n_agent, n_token]
            dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
            topk_logits = topk_logits.masked_fill(
                valid_next_gt.unsqueeze(1), 0.0
            ) - 1.0 * dist.masked_fill(~valid_next_gt.unsqueeze(1), 0.0)
    elif sampling_scheme.criterium == "topk_dist_sampled_with_prob":
        # gt_contour: [n_agent, 4, 2] in global coord
        gt_contour = cal_polygon_contour(pos_next_gt, head_next_gt, token_agent_shape)
        gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]
        token_world_sample = transform_to_global(
            pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
            head_local=None,
            pos_now=pos_now,  # [n_agent, 2]
            head_now=head_now,  # [n_agent]
        )[0].view(*token_traj.shape)

        _invalid = ~valid_next_gt
        # dist: [n_agent, n_token]
        dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
        _logits = -1.0 * dist.masked_fill(_invalid.unsqueeze(1), 0.0)

        if _invalid.any():
            _logits[_invalid] = next_token_logits[_invalid]
        _, topk_indices = torch.topk(
            _logits, sampling_scheme.num_k, dim=-1, sorted=False
        )  # [n_agent, K]
        topk_logits = next_token_logits[range_a.unsqueeze(1), topk_indices]

    else:
        raise ValueError(f"Invalid criterium: {sampling_scheme.criterium}")

    sample_logits = torch.zeros_like(next_token_logits) - float("inf")
    sample_logits[range_a.unsqueeze(1), topk_indices] = topk_logits.clone()

    # topk_logits, topk_indices: [n_agent, K]
    topk_logits = topk_logits / sampling_scheme.temp
    samples = Categorical(logits=topk_logits).sample()  # [n_agent] in K
    next_token_idx = topk_indices[range_a, samples]
    next_token_traj_all = token_traj_all[range_a, next_token_idx]

    return next_token_idx, next_token_traj_all, sample_logits
