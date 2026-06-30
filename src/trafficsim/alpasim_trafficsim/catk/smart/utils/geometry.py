# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math

import torch


def angle_between_2d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1]
        - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)
