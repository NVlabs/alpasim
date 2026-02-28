# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import pytest
from alpasim_runtime.config import VehicleConfig
from alpasim_runtime.loop import get_ds_rig_to_aabb_center_transform


def test_get_ds_rig_to_aabb_center_transform():
    vehicle_config = VehicleConfig(
        aabb_x_m=6.0, aabb_y_m=2.0, aabb_z_m=1.0, aabb_x_offset_m=-2.5
    )
    pose = get_ds_rig_to_aabb_center_transform(vehicle_config)
    assert pose.quat.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])  # no rotation
    assert pose.vec3.tolist() == [0.5, 0.0, 0.5]
