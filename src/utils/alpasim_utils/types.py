# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Shared data types used across alpasim packages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageWithMetadata:
    """Metadata for a camera image captured during simulation.

    This class is defined in alpasim_utils to avoid circular dependencies
    between runtime and eval packages.
    """

    start_timestamp_us: int
    end_timestamp_us: int
    image_bytes: bytes
    camera_logical_id: str

    def __repr__(self) -> str:
        return (
            "ImageWithMetadata("
            f"start_timestamp_us={self.start_timestamp_us:_d}, "
            f"end_timestamp_us={self.end_timestamp_us:_d}, "
            f"camera_logical_id={self.camera_logical_id}, "
            f"len(image_bytes)={len(self.image_bytes)})"
        )
