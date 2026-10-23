# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Deterministic test image shared by the fixture scripts and the tests."""

import numpy as np


def smooth_image(height: int, width: int) -> np.ndarray:
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.stack(
        [
            127 + 120 * np.sin(u / 37.0),
            127 + 120 * np.cos(v / 23.0),
            (u + v) % 256,
        ],
        axis=-1,
    )
    return image.astype(np.uint8)
