# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from navsim_gtrs_dense_challenge.preprocessing import (
    build_status_feature,
    preprocess_images,
)

IMAGE_SHAPE = (1080, 1920, 3)


def _images_with_borders() -> dict[str, np.ndarray]:
    images = {
        "CAM_L0": np.full(IMAGE_SHAPE, 32, dtype=np.uint8),
        "CAM_F0": np.full(IMAGE_SHAPE, 128, dtype=np.uint8),
        "CAM_R0": np.full(IMAGE_SHAPE, 224, dtype=np.uint8),
    }
    for image in images.values():
        image[:28] = 255
        image[-28:] = 255
    for camera_id in ("CAM_L0", "CAM_R0"):
        images[camera_id][:, :416] = 255
        images[camera_id][:, -416:] = 255
    return images


def _gradient_image(base: int) -> np.ndarray:
    rows = np.arange(1080, dtype=np.uint16)[:, None, None]
    columns = np.arange(1920, dtype=np.uint16)[None, :, None]
    channel_offsets = np.array([0, 41, 97], dtype=np.uint16)[None, None, :]
    row_scales = np.array([1, 3, 7], dtype=np.uint16)[None, None, :]
    column_scales = np.array([3, 5, 11], dtype=np.uint16)[None, None, :]
    return (
        (base + rows * row_scales + columns * column_scales + channel_offsets) % 256
    ).astype(np.uint8)


def test_preprocess_images_crops_stitches_resizes_and_normalizes() -> None:
    actual = preprocess_images(_images_with_borders())

    assert actual.shape == (3, 512, 2048)
    assert actual.dtype == torch.float32
    assert actual[0, 256, 256].item() == pytest.approx(32 / 255, abs=1 / 255)
    assert actual[0, 256, 1024].item() == pytest.approx(128 / 255, abs=1 / 255)
    assert actual[0, 256, 1792].item() == pytest.approx(224 / 255, abs=1 / 255)
    assert actual.min().item() >= 0.0
    assert actual.max().item() <= 1.0


def test_preprocess_images_supports_float16_output() -> None:
    actual = preprocess_images(_images_with_borders(), dtype=torch.float16)

    assert actual.dtype == torch.float16
    assert actual[0, 256, 256].item() == pytest.approx(32 / 255, abs=1e-3)


def test_preprocess_images_rejects_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match=r"floating-point dtype"):
        preprocess_images(_images_with_borders(), dtype=torch.uint8)


def test_preprocess_images_matches_pixel_level_reference() -> None:
    images = {
        "CAM_L0": _gradient_image(11),
        "CAM_F0": _gradient_image(67),
        "CAM_R0": _gradient_image(149),
    }

    actual = preprocess_images(images)
    stitched = np.concatenate(
        (
            images["CAM_L0"][28:-28, 416:-416],
            images["CAM_F0"][28:-28],
            images["CAM_R0"][28:-28, 416:-416],
        ),
        axis=1,
    )
    resized = cv2.resize(stitched, (2048, 512), interpolation=cv2.INTER_LINEAR)
    expected = (
        torch.from_numpy(resized).permute(2, 0, 1).contiguous().to(torch.float32)
        / 255.0
    )

    torch.testing.assert_close(actual, expected)


def test_preprocess_images_rejects_wrong_camera_shape() -> None:
    images = _images_with_borders()
    images["CAM_R0"] = np.zeros((720, 1280, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"CAM_R0.*1920x1080"):
        preprocess_images(images)


def test_build_status_feature_concatenates_float32_vectors() -> None:
    actual = build_status_feature(
        command_one_hot=[0, 1, 0, 0],
        velocity_xy=[2.5, -0.5],
        acceleration_xy=[0.25, -0.25],
    )

    expected = torch.tensor([0, 1, 0, 0, 2.5, -0.5, 0.25, -0.25], dtype=torch.float32)
    assert actual.shape == (8,)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("values", "parameter_name"),
    [
        (
            {
                "command_one_hot": [0, 1, 0],
                "velocity_xy": [4, -0.5],
                "acceleration_xy": [0.2, 0.1],
            },
            "command_one_hot",
        ),
        (
            {
                "command_one_hot": [0, 1, 0, 0],
                "velocity_xy": [4],
                "acceleration_xy": [0.2, 0.1],
            },
            "velocity_xy",
        ),
        (
            {
                "command_one_hot": [0, 1, 0, 0],
                "velocity_xy": [4, -0.5],
                "acceleration_xy": [0.2, 0.1, 0.0],
            },
            "acceleration_xy",
        ),
    ],
)
def test_build_status_feature_rejects_wrong_vector_shapes(
    values: dict[str, list[float]], parameter_name: str
) -> None:
    with pytest.raises(ValueError, match=parameter_name):
        build_status_feature(**values)


def test_preprocess_images_requires_all_cameras() -> None:
    images = _images_with_borders()
    del images["CAM_R0"]

    with pytest.raises(ValueError, match=r"missing required cameras.*CAM_R0"):
        preprocess_images(images)


def test_preprocess_images_rejects_non_uint8_images() -> None:
    images = _images_with_borders()
    images["CAM_L0"] = images["CAM_L0"].astype(np.float32)

    with pytest.raises(ValueError, match=r"CAM_L0.*dtype.*float32"):
        preprocess_images(images)
