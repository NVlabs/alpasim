# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared image preprocessing operations for trajectory models."""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF


def resize_and_center_crop_batch(
    images: list[np.ndarray],
    *,
    target_height: int,
    target_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize and center-crop HWC uint8 images as one device batch."""
    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)

    height, width = batch.shape[2:]
    scale = target_height / height
    resized_width = int(width * scale)
    if height != target_height or width != resized_width:
        batch = TF.resize(batch, [target_height, resized_width], antialias=True)

    if batch.shape[3] > target_width:
        left = (batch.shape[3] - target_width) // 2
        batch = batch[:, :, :, left : left + target_width]
    elif batch.shape[3] < target_width:
        raise ValueError(
            f"Image width {batch.shape[3]} too small after resize, need {target_width}"
        )

    return batch
