# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the force-GT cache render-config signature."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.camera_catalog import CameraDefinition
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.force_gt_cache_signature import build_force_gt_render_signature
from alpasim_utils.geometry import Pose


def _camera_definition(
    logical_id: str = "cam_front", res_h: int = 320, res_w: int = 512
) -> CameraDefinition:
    spec = sensorsim_pb2.CameraSpec(
        logical_id=logical_id, resolution_h=res_h, resolution_w=res_w
    )
    return CameraDefinition(
        logical_id=logical_id,
        intrinsics=spec,
        rig_to_camera=Pose(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
    )


def _version_ids(version_id: str = "nre-26.04", git_hash: str = "abc123"):
    return RolloutMetadata.VersionIds(
        sensorsim_version=VersionId(
            version_id=version_id,
            git_hash=git_hash,
            grpc_api_version=VersionId.APIVersion(major=0, minor=1, patch=0),
        ),
    )


def _unbound(**overrides) -> SimpleNamespace:
    base = dict(
        image_format=2,  # JPEG
        ego_mask_rig_config_id="hyperion_8_1",
        physics_update_mode=PhysicsUpdateMode.EGO_ONLY,
        force_gt_duration_us=1_700_000,
        control_timestep_us=100_000,
        render_start_timestamp_us=0,
        vehicle_config=SimpleNamespace(
            aabb_x_m=4.0,
            aabb_y_m=2.0,
            aabb_z_m=1.5,
            aabb_x_offset_m=0.0,
            aabb_y_offset_m=0.0,
            aabb_z_offset_m=0.0,
        ),
        version_ids=_version_ids(),
        force_gt_frame_cache_extra_key=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _signature(**kwargs) -> str:
    kwargs.setdefault("camera_definitions", {"cam_front": _camera_definition()})
    kwargs.setdefault("smooth_trajectories", True)
    kwargs.setdefault("unbound", _unbound())
    return build_force_gt_render_signature(**kwargs)


def test_signature_is_deterministic() -> None:
    assert _signature() == _signature()
    assert _signature().startswith("jpeg-")


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_format": 1},
        {"physics_update_mode": PhysicsUpdateMode.NONE},
        {"force_gt_duration_us": 2_000_000},
        {"force_gt_frame_cache_extra_key": "renderer-xyz"},
        {"version_ids": _version_ids(git_hash="deadbeef")},
    ],
)
def test_signature_changes_with_unbound_config(overrides: dict) -> None:
    assert _signature() != _signature(unbound=_unbound(**overrides))


def test_signature_changes_with_resolution_and_smoothing() -> None:
    bigger = {"cam_front": _camera_definition(res_h=640, res_w=1024)}
    assert _signature() != _signature(camera_definitions=bigger)
    assert _signature() != _signature(smooth_trajectories=False)
