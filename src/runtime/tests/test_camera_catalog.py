# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import numpy as np
import pytest
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_runtime.camera_catalog import CameraCatalog, CameraDefinition
from alpasim_runtime.config import (
    CameraDefinitionConfig,
    CameraIntrinsicsConfig,
    OpenCVPinholeConfig,
    PoseConfig,
    RuntimeCameraConfig,
)


def _make_local_camera_definition(logical_id: str) -> CameraDefinitionConfig:
    return CameraDefinitionConfig(
        logical_id=logical_id,
        rig_to_camera=PoseConfig(
            translation_m=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                focal_length=(800.0, 800.0),
                principal_point=(400.0, 200.0),
                radial=(0.0,) * 6,
                tangential=(0.0,) * 2,
                thin_prism=(0.0,) * 4,
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )


@pytest.mark.asyncio
async def test_register_scene_appends_local_definition() -> None:
    local_cfg = _make_local_camera_definition("camera_local")
    catalog = CameraCatalog([local_cfg])

    sensorsim_camera_front = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id="camera_front",
        intrinsics=sensorsim_pb2.CameraSpec(
            logical_id="camera_front",
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        ),
    )
    sensorsim_camera_front.rig_to_camera.quat.w = 1.0
    sensorsim_camera_local = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id="camera_local",
        intrinsics=sensorsim_pb2.CameraSpec(
            logical_id="camera_local",
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        ),
    )
    sensorsim_camera_local.rig_to_camera.quat.w = 1.0

    sensorsim_cameras = [sensorsim_camera_front, sensorsim_camera_local]

    await catalog.merge_local_and_sensorsim_cameras("scene_a", sensorsim_cameras)

    merged = catalog.get_camera_definitions("scene_a")
    assert set(merged.keys()) == {"camera_front", "camera_local"}
    assert (
        merged["camera_local"].intrinsics.shutter_type
        == sensorsim_pb2.ShutterType.ROLLING_TOP_TO_BOTTOM
    )

    await catalog.merge_local_and_sensorsim_cameras("scene_a", sensorsim_cameras)
    refreshed = catalog.get_camera_definitions("scene_a")
    assert refreshed is not merged
    assert set(refreshed.keys()) == {"camera_front", "camera_local"}
    for key, original_def in merged.items():
        refreshed_def = refreshed[key]
        assert refreshed_def is not original_def
        assert refreshed_def.logical_id == original_def.logical_id
        assert (
            refreshed_def.intrinsics.SerializeToString()
            == original_def.intrinsics.SerializeToString()
        )

        assert np.allclose(
            refreshed_def.rig_to_camera.vec3, original_def.rig_to_camera.vec3
        )
        assert np.allclose(
            refreshed_def.rig_to_camera.quat, original_def.rig_to_camera.quat
        )


@pytest.mark.asyncio
async def test_register_scene_overwrites_intrinsics() -> None:
    local_cfg = _make_local_camera_definition("camera_front")
    catalog = CameraCatalog([local_cfg])

    sensorsim_camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id="camera_front",
        intrinsics=sensorsim_pb2.CameraSpec(
            logical_id="camera_front",
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        ),
    )
    sensorsim_camera.rig_to_camera.quat.w = 1.0

    await catalog.merge_local_and_sensorsim_cameras("scene_b", [sensorsim_camera])

    merged = catalog.get_camera_definitions("scene_b")
    assert merged["camera_front"].intrinsics.shutter_type == (
        sensorsim_pb2.ShutterType.ROLLING_TOP_TO_BOTTOM
    )


@pytest.mark.asyncio
async def test_ensure_camera_defined_missing_definition_raises() -> None:
    catalog = CameraCatalog([])

    await catalog.merge_local_and_sensorsim_cameras("scene_a", [])

    scenario_camera = RuntimeCameraConfig(
        logical_id="camera_front",
        height=320,
        width=512,
        frame_interval_us=40_000,
        shutter_duration_us=20_000,
        first_frame_offset_us=1_000,
    )

    with pytest.raises(KeyError):
        catalog.ensure_camera_defined("scene_a", scenario_camera.logical_id)


def test_duplicate_local_definitions_raise() -> None:
    cfg = _make_local_camera_definition("camera_local")
    with pytest.raises(ValueError):
        CameraCatalog([cfg, cfg])


def test_pinhole_coeff_validations() -> None:
    base_kwargs = dict(
        focal_length=(800.0, 800.0),
        principal_point=(400.0, 200.0),
    )
    valid_radial = (0.0,) * 6
    valid_tangential = (0.0,) * 2
    valid_thin_prism = (0.0,) * 4

    bad_radial = CameraDefinitionConfig(
        logical_id="cam_bad_radial",
        rig_to_camera=PoseConfig(
            translation_m=(0.0, 0.0, 0.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                **base_kwargs,
                radial=(0.1,),
                tangential=valid_tangential,
                thin_prism=valid_thin_prism,
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )

    with pytest.raises(ValueError, match="radial must provide exactly 6"):
        CameraDefinition.from_config(bad_radial)

    bad_tangential = CameraDefinitionConfig(
        logical_id="cam_bad_tangential",
        rig_to_camera=bad_radial.rig_to_camera,
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                **base_kwargs,
                radial=valid_radial,
                tangential=(0.1,),
                thin_prism=valid_thin_prism,
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )

    with pytest.raises(ValueError, match="tangential must provide exactly 2"):
        CameraDefinition.from_config(bad_tangential)

    bad_thin_prism = CameraDefinitionConfig(
        logical_id="cam_bad_thin_prism",
        rig_to_camera=bad_radial.rig_to_camera,
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                **base_kwargs,
                radial=valid_radial,
                tangential=valid_tangential,
                thin_prism=(0.1,),
            ),
        ),
        resolution_hw=(400, 800),
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )

    with pytest.raises(ValueError, match="thin_prism must provide exactly 4"):
        CameraDefinition.from_config(bad_thin_prism)


# --- Partial override tests ---


def _make_sensorsim_camera(
    logical_id: str = "camera_front",
) -> sensorsim_pb2.AvailableCamerasReturn.AvailableCamera:
    """Create a sensorsim camera proto with known values for testing."""
    camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id=logical_id,
        intrinsics=sensorsim_pb2.CameraSpec(
            logical_id=logical_id,
            resolution_h=720,
            resolution_w=1280,
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        ),
    )
    # Set some intrinsics
    camera.intrinsics.opencv_pinhole_param.focal_length_x = 1000.0
    camera.intrinsics.opencv_pinhole_param.focal_length_y = 1000.0
    camera.intrinsics.opencv_pinhole_param.principal_point_x = 640.0
    camera.intrinsics.opencv_pinhole_param.principal_point_y = 360.0
    # Set rig_to_camera pose
    camera.rig_to_camera.vec.x = 1.0
    camera.rig_to_camera.vec.y = 2.0
    camera.rig_to_camera.vec.z = 3.0
    camera.rig_to_camera.quat.w = 1.0
    camera.rig_to_camera.quat.x = 0.0
    camera.rig_to_camera.quat.y = 0.0
    camera.rig_to_camera.quat.z = 0.0
    return camera


@pytest.mark.asyncio
async def test_partial_override_rig_to_camera_only() -> None:
    """Override only rig_to_camera, preserving intrinsics from sensorsim."""
    override = CameraDefinitionConfig(
        logical_id="camera_front",
        rig_to_camera=PoseConfig(
            translation_m=(10.0, 20.0, 30.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        # intrinsics, resolution_hw, shutter_type are None (not overridden)
    )
    catalog = CameraCatalog([override])

    sensorsim_camera = _make_sensorsim_camera("camera_front")
    await catalog.merge_local_and_sensorsim_cameras("scene", [sensorsim_camera])

    merged = catalog.get_camera_definition("scene", "camera_front")

    # rig_to_camera should be overridden
    assert merged.rig_to_camera.vec3[0] == 10.0
    assert merged.rig_to_camera.vec3[1] == 20.0
    assert merged.rig_to_camera.vec3[2] == 30.0

    # Intrinsics should be preserved from sensorsim
    assert merged.intrinsics.opencv_pinhole_param.focal_length_x == 1000.0
    assert merged.intrinsics.opencv_pinhole_param.focal_length_y == 1000.0
    assert merged.intrinsics.resolution_h == 720
    assert merged.intrinsics.resolution_w == 1280
    assert merged.intrinsics.shutter_type == sensorsim_pb2.ShutterType.GLOBAL


@pytest.mark.asyncio
async def test_partial_override_resolution_only() -> None:
    """Override only resolution_hw, preserving other fields from sensorsim."""
    override = CameraDefinitionConfig(
        logical_id="camera_front",
        resolution_hw=(1080, 1920),
    )
    catalog = CameraCatalog([override])

    sensorsim_camera = _make_sensorsim_camera("camera_front")
    await catalog.merge_local_and_sensorsim_cameras("scene", [sensorsim_camera])

    merged = catalog.get_camera_definition("scene", "camera_front")

    # Resolution should be overridden
    assert merged.intrinsics.resolution_h == 1080
    assert merged.intrinsics.resolution_w == 1920

    # rig_to_camera should be preserved from sensorsim
    assert merged.rig_to_camera.vec3[0] == 1.0
    assert merged.rig_to_camera.vec3[1] == 2.0
    assert merged.rig_to_camera.vec3[2] == 3.0

    # Intrinsics should be preserved from sensorsim
    assert merged.intrinsics.opencv_pinhole_param.focal_length_x == 1000.0


@pytest.mark.asyncio
async def test_partial_override_shutter_type_only() -> None:
    """Override only shutter_type, preserving other fields from sensorsim."""
    override = CameraDefinitionConfig(
        logical_id="camera_front",
        shutter_type="ROLLING_TOP_TO_BOTTOM",
    )
    catalog = CameraCatalog([override])

    sensorsim_camera = _make_sensorsim_camera("camera_front")
    await catalog.merge_local_and_sensorsim_cameras("scene", [sensorsim_camera])

    merged = catalog.get_camera_definition("scene", "camera_front")

    # Shutter type should be overridden
    assert (
        merged.intrinsics.shutter_type
        == sensorsim_pb2.ShutterType.ROLLING_TOP_TO_BOTTOM
    )

    # Other fields should be preserved from sensorsim
    assert merged.rig_to_camera.vec3[0] == 1.0
    assert merged.intrinsics.resolution_h == 720
    assert merged.intrinsics.opencv_pinhole_param.focal_length_x == 1000.0


@pytest.mark.asyncio
async def test_partial_override_intrinsics_only() -> None:
    """Override only intrinsics, preserving rig_to_camera from sensorsim."""
    override = CameraDefinitionConfig(
        logical_id="camera_front",
        intrinsics=CameraIntrinsicsConfig(
            model="opencv_pinhole",
            opencv_pinhole=OpenCVPinholeConfig(
                focal_length=(500.0, 500.0),
                principal_point=(320.0, 180.0),
                radial=(0.0,) * 6,
                tangential=(0.0,) * 2,
                thin_prism=(0.0,) * 4,
            ),
        ),
    )
    catalog = CameraCatalog([override])

    sensorsim_camera = _make_sensorsim_camera("camera_front")
    await catalog.merge_local_and_sensorsim_cameras("scene", [sensorsim_camera])

    merged = catalog.get_camera_definition("scene", "camera_front")

    # Intrinsics should be overridden
    assert merged.intrinsics.opencv_pinhole_param.focal_length_x == 500.0
    assert merged.intrinsics.opencv_pinhole_param.focal_length_y == 500.0
    assert merged.intrinsics.opencv_pinhole_param.principal_point_x == 320.0
    assert merged.intrinsics.opencv_pinhole_param.principal_point_y == 180.0

    # rig_to_camera should be preserved from sensorsim
    assert merged.rig_to_camera.vec3[0] == 1.0
    assert merged.rig_to_camera.vec3[1] == 2.0
    assert merged.rig_to_camera.vec3[2] == 3.0


@pytest.mark.asyncio
async def test_partial_override_multiple_fields() -> None:
    """Override rig_to_camera and resolution_hw together."""
    override = CameraDefinitionConfig(
        logical_id="camera_front",
        rig_to_camera=PoseConfig(
            translation_m=(5.0, 6.0, 7.0),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        resolution_hw=(480, 640),
    )
    catalog = CameraCatalog([override])

    sensorsim_camera = _make_sensorsim_camera("camera_front")
    await catalog.merge_local_and_sensorsim_cameras("scene", [sensorsim_camera])

    merged = catalog.get_camera_definition("scene", "camera_front")

    # Both overridden fields should be applied
    assert merged.rig_to_camera.vec3[0] == 5.0
    assert merged.rig_to_camera.vec3[1] == 6.0
    assert merged.rig_to_camera.vec3[2] == 7.0
    assert merged.intrinsics.resolution_h == 480
    assert merged.intrinsics.resolution_w == 640

    # Non-overridden fields should be preserved from sensorsim
    assert merged.intrinsics.opencv_pinhole_param.focal_length_x == 1000.0
    assert merged.intrinsics.shutter_type == sensorsim_pb2.ShutterType.GLOBAL


@pytest.mark.asyncio
async def test_no_override_preserves_sensorsim_values() -> None:
    """A camera with only logical_id should not modify any sensorsim values."""
    # This is a valid config now (all optional fields are None)
    override = CameraDefinitionConfig(logical_id="camera_front")
    catalog = CameraCatalog([override])

    sensorsim_camera = _make_sensorsim_camera("camera_front")
    await catalog.merge_local_and_sensorsim_cameras("scene", [sensorsim_camera])

    merged = catalog.get_camera_definition("scene", "camera_front")

    # All values should be preserved from sensorsim
    assert merged.rig_to_camera.vec3[0] == 1.0
    assert merged.rig_to_camera.vec3[1] == 2.0
    assert merged.rig_to_camera.vec3[2] == 3.0
    assert merged.intrinsics.resolution_h == 720
    assert merged.intrinsics.resolution_w == 1280
    assert merged.intrinsics.opencv_pinhole_param.focal_length_x == 1000.0
    assert merged.intrinsics.shutter_type == sensorsim_pb2.ShutterType.GLOBAL
