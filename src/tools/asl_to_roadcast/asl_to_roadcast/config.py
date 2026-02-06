# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Local configuration classes for the asl_to_roadcast tool.

These are simplified versions of the runtime config classes, containing only
what's needed for roadcast conversion.
"""

from dataclasses import dataclass


@dataclass
class VehicleConfig:
    """Vehicle configuration for roadcast output.
    This data should be sourced from the artifact's rig configuration.
    """

    # AABB dimensions in meters
    aabb_x_m: float
    aabb_y_m: float
    aabb_z_m: float

    # AABB offsets from the rig origin
    aabb_x_offset_m: float
    aabb_y_offset_m: float
    aabb_z_offset_m: float


@dataclass
class RoadCastConfig:
    """Configuration for RoadCast output"""

    # Radial distance from the ego to find lanes to write to RoadCast.
    lane_radius: float = 50.0
