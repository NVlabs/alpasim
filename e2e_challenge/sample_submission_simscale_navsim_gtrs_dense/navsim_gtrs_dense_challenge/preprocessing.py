# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import torch

CAMERA_IDS = ("CAM_L0", "CAM_F0", "CAM_R0")
EXPECTED_HEIGHT = 1080
EXPECTED_WIDTH = 1920
OUTPUT_HEIGHT = 512
OUTPUT_WIDTH = 2048


def _validated_image(camera_id: str, image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    expected_shape = (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3)
    if array.shape != expected_shape:
        raise ValueError(
            f"{camera_id} image must be 1920x1080 with shape {expected_shape}; "
            f"got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(
            f"{camera_id} image must have dtype uint8; got dtype {array.dtype}"
        )
    return array


def preprocess_images(
    images: Mapping[str, np.ndarray],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not dtype.is_floating_point:
        raise ValueError(
            f"preprocess_images requires a floating-point dtype; got {dtype}"
        )

    missing = [camera_id for camera_id in CAMERA_IDS if camera_id not in images]
    if missing:
        raise ValueError(f"missing required cameras: {', '.join(missing)}")

    left = _validated_image("CAM_L0", images["CAM_L0"])[28:-28, 416:-416]
    front = _validated_image("CAM_F0", images["CAM_F0"])[28:-28]
    right = _validated_image("CAM_R0", images["CAM_R0"])[28:-28, 416:-416]

    panorama = np.concatenate((left, front, right), axis=1)
    resized = cv2.resize(
        panorama,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().to(dtype=dtype)
    return tensor.div_(255.0)


def _vector(name: str, values: Sequence[float] | np.ndarray, size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},); got shape {vector.shape}")
    return vector


def build_status_feature(
    *,
    command_one_hot: Sequence[float] | np.ndarray,
    velocity_xy: Sequence[float] | np.ndarray,
    acceleration_xy: Sequence[float] | np.ndarray,
) -> torch.Tensor:
    status = np.concatenate(
        (
            _vector("command_one_hot", command_one_hot, 4),
            _vector("velocity_xy", velocity_xy, 2),
            _vector("acceleration_xy", acceleration_xy, 2),
        )
    )
    return torch.from_numpy(status)
