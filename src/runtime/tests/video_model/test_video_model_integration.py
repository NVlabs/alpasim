# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""End-to-end integration tests for the built-in video-model renderer.

These tests exercise the renderer against:

- An in-process ``grpc.aio`` server implementing
  ``WorldModelServiceServicer``, so we catch wire-compatibility regressions
  (proto package name, method paths, request/response field handling) before
  they surface against a real GPU server.
- A synthetic USDZ archive containing a realistic ``clipgt/calibration_estimate.parquet``,
  HD map parquets, and per-camera first frames, so we catch calibration parsing
  and HD-map extraction regressions of the kind that produced the recent
  "underground HD map" rendering bug.

We do not exercise the actual flashdreams server here -- the goal is to
guarantee that the renderer's wiring, contract, and data-extraction code paths
are healthy. End-to-end tests against a live model still require a GPU.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import yaml
from alpasim_grpc.v0 import sensorsim_pb2, video_model_pb2_grpc
from alpasim_grpc.v0.common_pb2 import (
    Empty,
    Pose,
    PoseAtTime,
    Quat,
    Trajectory,
    Vec3,
    VersionId,
)
from alpasim_grpc.v0.video_model_pb2 import (
    ActorClassId,
    CameraOutput,
    DynamicActor,
    DynamicWorldState,
    Image,
    ImageFormat,
    SessionCloseRequest,
    SessionId,
    SessionRequest,
    VideoChunkReturn,
)
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.camera_catalog import CameraCatalog, CameraDefinition
from alpasim_runtime.config import (
    CameraDefinitionConfig,
    CameraIntrinsicsConfig,
    OpenCVPinholeConfig,
    PoseConfig,
    VideoModelConfig,
)
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.video_model_service import VideoModelService
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_runtime.video_model.usdz_calibration import parse_cameras_from_usdz
from alpasim_runtime.video_model.utils import (
    build_camera_specs_and_initial_frames,
    extract_first_frames,
    extract_hdmap_for_video_model,
)
from alpasim_utils import geometry
from alpasim_utils.scenario import AABB

import grpc
import grpc.aio

# ---------------------------------------------------------------------------
# Synthetic USDZ builders
# ---------------------------------------------------------------------------


def _build_clipgt_calibration_dict(sensors: list[dict]) -> dict:
    """Build a MADS_RIG_V2-shaped calibration dict accepted by the parser."""
    return {"rig": {"sensors": sensors}}


def _make_ftheta_sensor(
    *,
    name: str,
    width: int = 1280,
    height: int = 640,
    cx: float | None = None,
    cy: float | None = None,
    polynomial: str = "0.0 1.0",
    polynomial_type: str = "pixeldistance-to-angle",
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation_m: tuple[float, float, float] = (0.0, 0.0, 1.5),
) -> dict:
    """Build a single MADS_RIG_V2 ``camera`` sensor entry."""
    return {
        "name": name,
        "properties": {
            "Model": "ftheta",
            "cx": cx if cx is not None else width / 2.0,
            "cy": cy if cy is not None else height / 2.0,
            "width": width,
            "height": height,
            "polynomial": polynomial,
            "polynomial-type": polynomial_type,
            "linear-c": 1.0,
            "linear-d": 0.0,
            "linear-e": 0.0,
        },
        "nominalSensor2Rig_FLU": {
            "roll-pitch-yaw": list(rpy_deg),
            "t": list(translation_m),
        },
        "correction_sensor_R_FLU": {"roll-pitch-yaw": [0.0, 0.0, 0.0]},
        "correction_rig_T": [0.0, 0.0, 0.0],
    }


def _build_synthetic_usdz(
    out_path: Path,
    *,
    scene_id: str,
    sensors: list[dict],
    hdmap_files: dict[str, bytes] | None = None,
    frame_archive_entries: dict[str, bytes] | None = None,
) -> Path:
    """Write a minimal USDZ that exercises the video model's USDZ-parsing surface.

    Contents:
    - ``metadata.yaml`` with a ``scene_id`` (required by HD-map extraction)
    - ``clipgt/calibration_estimate.parquet`` (parsed by ``usdz_calibration``)
    - ``clipgt/<file>`` HD-map fixtures (re-packed by ``extract_hdmap_for_video_model``)
    - ``frames/<camera>/<timestamp>.jpeg`` first-frame fixtures
    """
    df = pd.DataFrame(
        {"calibration_estimate": [json.dumps(_build_clipgt_calibration_dict(sensors))]}
    )
    parquet_buf = io.BytesIO()
    df.to_parquet(parquet_buf, index=False)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump({"scene_id": scene_id}))
        zf.writestr("clipgt/calibration_estimate.parquet", parquet_buf.getvalue())
        for name, content in (hdmap_files or {}).items():
            zf.writestr(f"clipgt/{name}", content)
        for name, content in (frame_archive_entries or {}).items():
            zf.writestr(name, content)

    return out_path


# ---------------------------------------------------------------------------
# Section 1: USDZ calibration parsing
# ---------------------------------------------------------------------------


def test_parse_cameras_from_usdz_extracts_intrinsics_and_extrinsics(
    tmp_path: Path,
) -> None:
    """Calibration-parquet -> CameraDefinition has real intrinsics + pose.

    Regression: the previous fallback synthesized a 60deg equidistant FTheta
    centered at the rig origin, which made HD-map renders look underground.
    """
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-deadbeef",
        sensors=[
            _make_ftheta_sensor(
                name="camera:front:wide:120fov",
                width=1920,
                height=1208,
                polynomial="0.0 0.0008 0.0",
                rpy_deg=(0.0, 0.0, 0.0),
                translation_m=(2.1, 0.0, 1.55),
            ),
            _make_ftheta_sensor(
                name="camera:cross:left:120fov",
                width=1920,
                height=1208,
                polynomial="0.0 0.00075 0.0",
                rpy_deg=(0.0, 0.0, 90.0),
                translation_m=(2.1, 0.85, 1.55),
            ),
        ],
    )

    cameras = parse_cameras_from_usdz(usdz)

    assert set(cameras.keys()) == {
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
    }

    front = cameras["camera_front_wide_120fov"]
    assert isinstance(front, CameraDefinition)
    assert front.intrinsics.resolution_w == 1920
    assert front.intrinsics.resolution_h == 1208
    assert front.intrinsics.shutter_type == sensorsim_pb2.ShutterType.GLOBAL
    assert front.intrinsics.HasField("ftheta_param")
    ft = front.intrinsics.ftheta_param
    # Both polynomial directions are populated (the parser inverts whichever
    # one is missing from the parquet).
    assert len(ft.pixeldist_to_angle_poly) > 0
    assert len(ft.angle_to_pixeldist_poly) > 0
    # Extrinsics: translation comes from nominalSensor2Rig_FLU (no corrections).
    front_t = front.rig_to_camera.vec3
    assert math.isclose(float(front_t[0]), 2.1, abs_tol=1e-4)
    assert math.isclose(float(front_t[2]), 1.55, abs_tol=1e-4)
    # Rotation differs between front (yaw=0) and cross-left (yaw=90).
    left = cameras["camera_cross_left_120fov"]
    assert front.rig_to_camera.quat.tolist() != left.rig_to_camera.quat.tolist()


def test_parse_cameras_from_usdz_skips_non_ftheta_and_non_camera(
    tmp_path: Path,
) -> None:
    """Non-camera entries and non-ftheta cameras are silently skipped."""
    sensors = [
        _make_ftheta_sensor(name="camera:front:wide:120fov"),
        # Non-camera sensor (lidar): ignored.
        {
            "name": "lidar:top",
            "properties": {"Model": "spinning"},
            "nominalSensor2Rig_FLU": {"roll-pitch-yaw": [0, 0, 0], "t": [0, 0, 0]},
        },
        # Non-ftheta camera (pinhole): ignored with a warning.
        {
            "name": "camera:rear:fisheye",
            "properties": {"Model": "pinhole"},
            "nominalSensor2Rig_FLU": {"roll-pitch-yaw": [0, 0, 0], "t": [0, 0, 0]},
        },
    ]
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz", scene_id="clipgt-x", sensors=sensors
    )
    cameras = parse_cameras_from_usdz(usdz)
    assert list(cameras.keys()) == ["camera_front_wide_120fov"]


def test_parse_cameras_from_usdz_missing_calibration_raises(tmp_path: Path) -> None:
    """USDZ without clipgt/calibration_estimate.parquet -> FileNotFoundError."""
    bad = tmp_path / "no_calib.usdz"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("metadata.yaml", yaml.safe_dump({"scene_id": "clipgt-x"}))
    with pytest.raises(FileNotFoundError, match="calibration_estimate"):
        parse_cameras_from_usdz(bad)


# ---------------------------------------------------------------------------
# Section 2: USDZ first-frame and HD-map extraction
# ---------------------------------------------------------------------------


def test_extract_first_frames_from_timestamped_frames_directory(
    tmp_path: Path,
) -> None:
    """frames/<camera>/<timestamp>.jpeg files are picked up + ID-normalized."""
    front_bytes = b"\xff\xd8\xffFRONT_BYTES"
    left_bytes = b"\xff\xd8\xffLEFT_BYTES"
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-x",
        sensors=[_make_ftheta_sensor(name="camera:front:wide:120fov")],
        frame_archive_entries={
            "frames/camera:front:wide:120fov/200000.jpeg": front_bytes,
            "frames/camera:cross:left:120fov/300000.jpeg": left_bytes,
        },
    )

    generic, per_camera = extract_first_frames(usdz)
    assert generic is None
    # Colon-separated camera IDs are normalized to underscore form.
    assert per_camera["camera_front_wide_120fov"] == (front_bytes, "JPEG")
    assert per_camera["camera_cross_left_120fov"] == (left_bytes, "JPEG")


def test_extract_first_frames_prefers_earliest_timestamp(
    tmp_path: Path,
) -> None:
    """Multiple JPEG timestamps are allowed; the earliest one wins."""
    later_bytes = b"\xff\xd8\xffLATER_BYTES"
    earliest_bytes = b"\xff\xd8\xffEARLIEST_BYTES"
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-x",
        sensors=[_make_ftheta_sensor(name="camera:front:wide:120fov")],
        frame_archive_entries={
            "frames/camera_front_wide_120fov/300000.jpeg": later_bytes,
            "frames/camera_front_wide_120fov/200000.jpeg": earliest_bytes,
        },
    )

    _generic, per_camera = extract_first_frames(usdz)

    assert per_camera["camera_front_wide_120fov"] == (earliest_bytes, "JPEG")


def test_extract_first_frames_rejects_non_jpeg_frame(
    tmp_path: Path,
) -> None:
    """Only frames/<camera>/<timestamp>.jpeg is supported."""
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-x",
        sensors=[_make_ftheta_sensor(name="camera:front:wide:120fov")],
        frame_archive_entries={
            "frames/camera_front_wide_120fov/200000.png": b"\x89PNG\r\n\x1a\nPNG",
        },
    )

    with pytest.raises(ValueError, match="Unsupported first-frame image path"):
        extract_first_frames(usdz)


def test_extract_first_frames_rejects_duplicate_camera_timestamp(
    tmp_path: Path,
) -> None:
    """A single camera timestamp must map to exactly one image."""
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-x",
        sensors=[_make_ftheta_sensor(name="camera:front:wide:120fov")],
    )
    with zipfile.ZipFile(usdz, "a") as zf:
        zf.writestr("frames/camera_front_wide_120fov/200000.jpeg", b"\xff\xd8\xffA")
        zf.writestr("frames/camera_front_wide_120fov/200000.jpeg", b"\xff\xd8\xffB")

    with pytest.raises(ValueError, match="Duplicate first-frame image"):
        extract_first_frames(usdz)


def test_extract_hdmap_for_video_model_renames_clipgt_entries(tmp_path: Path) -> None:
    """clipgt/foo.parquet -> <clip_id>.foo.parquet inside the returned zip."""
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-deadbeef",
        sensors=[_make_ftheta_sensor(name="camera:front:wide:120fov")],
        hdmap_files={
            "lanes.parquet": b"LANE_PARQUET",
            "intersections.parquet": b"INT_PARQUET",
        },
    )

    hdmap_zip_bytes = extract_hdmap_for_video_model(usdz)
    with zipfile.ZipFile(io.BytesIO(hdmap_zip_bytes), "r") as zf:
        names = set(zf.namelist())
        # clip_id strips the 'clipgt-' prefix.
        assert "deadbeef.lanes.parquet" in names
        assert "deadbeef.intersections.parquet" in names
        # The original calibration parquet rides along too (it lives under clipgt/).
        assert "deadbeef.calibration_estimate.parquet" in names
        assert zf.read("deadbeef.lanes.parquet") == b"LANE_PARQUET"


def test_extract_hdmap_missing_metadata_raises(tmp_path: Path) -> None:
    """metadata.yaml is required to derive the clip_id used as renamed prefix."""
    bad = tmp_path / "no_meta.usdz"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("clipgt/lanes.parquet", b"x")
    with pytest.raises(ValueError, match="metadata.yaml"):
        extract_hdmap_for_video_model(bad)


# ---------------------------------------------------------------------------
# Section 3: VideoModelService._register_scene_cameras + _apply_override
# ---------------------------------------------------------------------------


def _make_service(camera_catalog: CameraCatalog | None = None) -> VideoModelService:
    return VideoModelService(
        address="unused:0",
        config=VideoModelConfig(),
        skip=True,
        camera_catalog=camera_catalog,
    )


def test_register_scene_cameras_uses_usdz_calibration_and_applies_resolution_override(
    tmp_path: Path,
) -> None:
    """USDZ supplies real calibration; local overrides may patch resolution.

    Regression: previously the override path silently dropped the USDZ
    extrinsics (placing cameras at the rig origin) and produced
    underground-looking renders.
    """
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-deadbeef",
        sensors=[
            _make_ftheta_sensor(
                name="camera:front:wide:120fov",
                width=1920,
                height=1208,
                translation_m=(2.0, 0.0, 1.5),
            ),
        ],
    )

    catalog = CameraCatalog(
        [
            CameraDefinitionConfig(
                logical_id="camera_front_wide_120fov",
                resolution_hw=(848, 1600),
            )
        ]
    )
    service = _make_service(camera_catalog=catalog)

    service._register_scene_cameras(scene_id="clipgt-deadbeef", usdz_path=str(usdz))

    cam = catalog.get_camera_definition("clipgt-deadbeef", "camera_front_wide_120fov")
    # Override applied:
    assert cam.intrinsics.resolution_h == 848
    assert cam.intrinsics.resolution_w == 1600
    # Calibration-sensitive values stay tied to the USDZ first frame.
    assert cam.intrinsics.shutter_type == sensorsim_pb2.ShutterType.GLOBAL
    assert math.isclose(float(cam.rig_to_camera.vec3[0]), 2.0, abs_tol=1e-4)
    assert math.isclose(float(cam.rig_to_camera.vec3[2]), 1.5, abs_tol=1e-4)


@pytest.mark.parametrize(
    ("override", "field"),
    [
        (
            CameraDefinitionConfig(
                logical_id="camera_front_wide_120fov",
                rig_to_camera=PoseConfig(
                    translation_m=(3.0, 0.25, 1.75),
                    rotation_xyzw=(0.0, 0.0, 0.70710678, 0.70710678),
                ),
            ),
            "rig_to_camera",
        ),
        (
            CameraDefinitionConfig(
                logical_id="camera_front_wide_120fov",
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
            ),
            "intrinsics",
        ),
        (
            CameraDefinitionConfig(
                logical_id="camera_front_wide_120fov",
                shutter_type="ROLLING_TOP_TO_BOTTOM",
            ),
            "shutter_type",
        ),
    ],
)
def test_register_scene_cameras_rejects_unsafe_video_model_overrides(
    tmp_path: Path, override: CameraDefinitionConfig, field: str
) -> None:
    """Video-model seed frames must stay aligned with recorded calibration."""
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-deadbeef",
        sensors=[_make_ftheta_sensor(name="camera:front:wide:120fov")],
    )

    catalog = CameraCatalog([override])
    service = _make_service(camera_catalog=catalog)

    with pytest.raises(ValueError, match=field):
        service._register_scene_cameras(scene_id="clipgt-deadbeef", usdz_path=str(usdz))


def test_register_scene_cameras_rejects_override_without_matching_usdz_camera(
    tmp_path: Path,
) -> None:
    """An override with no matching USDZ camera is a configuration error.

    Subset selection belongs in simulation_config.cameras; extra_cameras are
    actual overrides and should fail loudly when they reference an invalid ID.
    """
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-x",
        sensors=[_make_ftheta_sensor(name="camera:front:wide:120fov")],
    )

    catalog = CameraCatalog(
        [
            CameraDefinitionConfig(logical_id="camera_front_wide_120fov"),
            CameraDefinitionConfig(logical_id="camera_does_not_exist"),
        ]
    )
    service = _make_service(camera_catalog=catalog)

    with pytest.raises(ValueError, match="camera_does_not_exist"):
        service._register_scene_cameras(scene_id="clipgt-x", usdz_path=str(usdz))


def test_apply_override_resolution_only() -> None:
    """``_apply_override`` honors only resolution overrides."""
    base_intrinsics = sensorsim_pb2.CameraSpec(
        logical_id="cam",
        resolution_h=600,
        resolution_w=800,
        shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
    )

    base = CameraDefinition(
        logical_id="cam",
        intrinsics=base_intrinsics,
        rig_to_camera=geometry.Pose(
            np.array([1, 2, 3], dtype=np.float32),
            np.array([0, 0, 0, 1], dtype=np.float32),
        ),
    )

    override = CameraDefinitionConfig(
        logical_id="cam",
        resolution_hw=(900, 1600),
    )

    out = VideoModelService._apply_override(base, override)
    assert out.intrinsics.resolution_h == 900
    assert out.intrinsics.resolution_w == 1600
    assert out.intrinsics.shutter_type == sensorsim_pb2.ShutterType.GLOBAL
    assert math.isclose(float(out.rig_to_camera.vec3[0]), 1.0, abs_tol=1e-4)
    assert math.isclose(float(out.rig_to_camera.vec3[1]), 2.0, abs_tol=1e-4)
    assert math.isclose(float(out.rig_to_camera.vec3[2]), 3.0, abs_tol=1e-4)
    # Base must remain unchanged (we should be working on a copy).
    assert base.intrinsics.resolution_h == 600
    assert math.isclose(float(base.rig_to_camera.vec3[0]), 1.0, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Section 4: build_camera_specs_and_initial_frames
# ---------------------------------------------------------------------------


def _runtime_camera(logical_id: str, hw: tuple[int, int] = (1208, 1920)):
    """Build a minimal RuntimeCamera (Clock fields are unused by camera-spec build)."""

    return RuntimeCamera(
        logical_id=logical_id,
        render_resolution_hw=hw,
        clock=Clock(interval_us=33_333, duration_us=10_000_000),
    )


def test_build_camera_specs_and_initial_frames_uses_real_ftheta_and_shutter(
    tmp_path: Path,
) -> None:
    """The CameraSpec built for the wire carries FTheta + shutter from USDZ.

    Regression: shutter_type used to default to UNKNOWN on the wire, which
    silently degraded server output for scenes that need GLOBAL shutter.
    """
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-x",
        sensors=[
            _make_ftheta_sensor(
                name="camera:front:wide:120fov",
                width=1920,
                height=1208,
                polynomial="0.0 0.0008",
                translation_m=(2.0, 0.0, 1.5),
            )
        ],
        frame_archive_entries={
            "frames/camera_front_wide_120fov/200000.jpeg": b"\xff\xd8\xffFRAMEBYTES",
        },
    )

    catalog = CameraCatalog(
        [CameraDefinitionConfig(logical_id="camera_front_wide_120fov")]
    )
    service = _make_service(camera_catalog=catalog)
    service._register_scene_cameras(scene_id="clipgt-x", usdz_path=str(usdz))

    specs, rig_to_camera, frames = build_camera_specs_and_initial_frames(
        runtime_cameras=[_runtime_camera("camera_front_wide_120fov")],
        camera_catalog=catalog,
        scene_id="clipgt-x",
        usdz_path=str(usdz),
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.logical_id == "camera_front_wide_120fov"
    assert spec.resolution_h == 1208
    assert spec.resolution_w == 1920
    assert spec.HasField("ftheta_param")
    assert spec.shutter_type == sensorsim_pb2.ShutterType.GLOBAL
    assert len(rig_to_camera) == 1
    assert math.isclose(rig_to_camera[0].vec.x, 2.0, abs_tol=1e-4)
    assert math.isclose(rig_to_camera[0].vec.z, 1.5, abs_tol=1e-4)

    assert len(frames) == 1
    img_bytes, fmt = frames[0]
    assert img_bytes == b"\xff\xd8\xffFRAMEBYTES"
    assert fmt == ImageFormat.JPEG


def test_build_camera_specs_and_initial_frames_uses_timestamped_frames(
    tmp_path: Path,
) -> None:
    """Video model session init can use frames created by add_first_frames_to_usdz."""
    jpeg_bytes = b"\xff\xd8\xffFRAMEBYTES"
    usdz = _build_synthetic_usdz(
        tmp_path / "scene.usdz",
        scene_id="clipgt-x",
        sensors=[
            _make_ftheta_sensor(
                name="camera:front:wide:120fov",
                width=1920,
                height=1208,
            )
        ],
        frame_archive_entries={
            "frames/camera_front_wide_120fov/200000.jpeg": jpeg_bytes,
        },
    )

    catalog = CameraCatalog(
        [CameraDefinitionConfig(logical_id="camera_front_wide_120fov")]
    )
    service = _make_service(camera_catalog=catalog)
    service._register_scene_cameras(scene_id="clipgt-x", usdz_path=str(usdz))

    _specs, _rig_to_camera, frames = build_camera_specs_and_initial_frames(
        runtime_cameras=[_runtime_camera("camera_front_wide_120fov")],
        camera_catalog=catalog,
        scene_id="clipgt-x",
        usdz_path=str(usdz),
    )

    assert frames == [(jpeg_bytes, ImageFormat.JPEG)]


# ---------------------------------------------------------------------------
# Section 5: gRPC integration via in-process server
# ---------------------------------------------------------------------------


class _CollectingHandler:
    def __init__(self) -> None:
        self.entry_types: list[str] = []

    async def on_message(self, message) -> None:
        self.entry_types.append(message.WhichOneof("log_entry"))


class _RecordingWorldModelServicer(video_model_pb2_grpc.WorldModelServiceServicer):
    """In-process servicer that records requests and emits canned responses.

    Each method captures the latest request in ``self.<method>_request`` and
    returns whatever the test pre-loaded into ``self.<method>_response``.
    """

    def __init__(self) -> None:
        self.start_session_request: SessionRequest | None = None
        self.start_session_response: SessionId = SessionId(
            session_id="server-session-0"
        )

        self.render_request = None
        self.render_response: VideoChunkReturn | None = None

        self.close_session_request: SessionCloseRequest | None = None

    async def get_version(self, request: Empty, context):
        del request, context
        return VersionId(version_id="fake-video-model", git_hash="test")

    async def start_session(self, request: SessionRequest, context):
        self.start_session_request = request
        return self.start_session_response

    async def close_session(self, request: SessionCloseRequest, context):
        self.close_session_request = request
        return Empty()

    async def render_video_chunk(self, request, context):
        self.render_request = request
        if self.render_response is None:
            return VideoChunkReturn()
        return self.render_response


async def _start_world_model_server(
    servicer: _RecordingWorldModelServicer,
) -> tuple[grpc.aio.Server, str]:
    server = grpc.aio.server()
    video_model_pb2_grpc.add_WorldModelServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


@pytest_asyncio.fixture
async def grpc_world_model():
    """In-process ``WorldModelService`` server on an ephemeral port.

    Yields ``(servicer, address)``. The server is shut down on test teardown
    even if the test raises.
    """
    servicer = _RecordingWorldModelServicer()
    server, address = await _start_world_model_server(servicer)
    try:
        yield servicer, address
    finally:
        await server.stop(grace=0.5)


def _pose_at(t_us: int, x: float = 0.0) -> PoseAtTime:
    return PoseAtTime(
        pose=Pose(
            vec=Vec3(x=x, y=0.0, z=0.0),
            quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
        timestamp_us=t_us,
    )


def _trajectory(timestamps_us: list[int]) -> Trajectory:
    return Trajectory(poses=[_pose_at(t) for t in timestamps_us])


@pytest.mark.asyncio
async def test_start_session_round_trip_serializes_request_correctly(
    grpc_world_model,
) -> None:
    """A real gRPC client/server round trip: request fields survive the wire.

    Regression: a proto package mismatch (or stub class wiring drift) would
    fail this test loudly, instead of silently failing at the GPU server.
    """
    servicer, address = grpc_world_model
    service = VideoModelService(address=address, config=VideoModelConfig())
    await service._open_connection()
    log_handler = _CollectingHandler()
    service.session_info = SessionInfo(
        uuid="test-session",
        broadcaster=MessageBroadcaster([log_handler]),
    )

    cam_spec = sensorsim_pb2.CameraSpec(
        logical_id="camera_front_wide_120fov", resolution_h=720, resolution_w=1280
    )
    rig_to_camera = [
        Pose(
            vec=Vec3(x=0.0, y=0.0, z=0.0),
            quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
        )
    ]
    initial_frames = [(b"INITFRAME", ImageFormat.JPEG)]
    session_id = await service.start_session(
        hdmap_bytes=b"HDMAP",
        camera_specs=[cam_spec],
        rig_to_camera=rig_to_camera,
        initial_frames=initial_frames,
        text_prompt_positive="a sunny day",
    )
    service.session_info = None
    await service._close_connection()

    assert session_id.session_id == "server-session-0"
    assert log_handler.entry_types == [
        "video_model_session_request",
        "video_model_session_id",
    ]
    captured = servicer.start_session_request
    assert captured is not None
    assert captured.static_world_map.hdmap_parquets == b"HDMAP"
    assert captured.text_prompt.positive == "a sunny day"
    assert len(captured.camera_specs) == 1
    assert captured.camera_specs[0].logical_id == "camera_front_wide_120fov"
    assert len(captured.rig_to_camera) == 1
    assert len(captured.initial_frames) == 1
    assert captured.initial_frames[0].data == b"INITFRAME"
    assert captured.initial_frames[0].format == ImageFormat.JPEG


@pytest.mark.asyncio
async def test_render_chunk_round_trip_per_camera_dispatch(grpc_world_model) -> None:
    """Multi-camera ``CameraOutput`` is correctly dispatched into the result.

    Verifies per-camera frame routing, RGB + HDMap timestamp resolution, and
    that ``raw_request`` / ``raw_response`` are passed through for downstream
    diagnostics.
    """
    servicer, address = grpc_world_model
    service = VideoModelService(address=address, config=VideoModelConfig(fps=10))
    await service._open_connection()
    log_handler = _CollectingHandler()
    service.session_info = SessionInfo(
        uuid="test-session",
        broadcaster=MessageBroadcaster([log_handler]),
    )

    request_traj = _trajectory([0, 100_000, 200_000])
    servicer.render_response = VideoChunkReturn(
        camera_outputs=[
            CameraOutput(
                camera_logical_id="camera_front_wide_120fov",
                rgb_frames=[
                    Image(data=b"FRONT_RGB_0", format=ImageFormat.JPEG),
                    Image(data=b"FRONT_RGB_1", format=ImageFormat.JPEG),
                    Image(data=b"FRONT_RGB_2", format=ImageFormat.JPEG),
                ],
                hdmap_condition_frames=[
                    Image(data=b"FRONT_HDMAP_0", format=ImageFormat.JPEG),
                ],
            ),
            CameraOutput(
                camera_logical_id="camera_cross_left_120fov",
                rgb_frames=[Image(data=b"LEFT_RGB_0", format=ImageFormat.JPEG)],
            ),
        ],
    )

    service._session_id = SessionId(session_id="active-session")
    chunk = await service.render_chunk(
        trajectory_local_to_rig=request_traj,
        dynamic_state=DynamicWorldState(),
    )
    service.session_info = None
    await service._close_connection()

    assert chunk.raw_request is not None
    assert chunk.raw_response is not None
    assert chunk.raw_request.session_id.session_id == "active-session"
    assert log_handler.entry_types == [
        "video_model_chunk_request",
        "video_model_chunk_return",
    ]

    # Per-camera RGB dispatch.
    front_rgb = chunk.rgb_frames_per_camera["camera_front_wide_120fov"]
    assert [f.image_bytes for f in front_rgb] == [
        b"FRONT_RGB_0",
        b"FRONT_RGB_1",
        b"FRONT_RGB_2",
    ]
    assert [f.start_timestamp_us for f in front_rgb] == [0, 100_000, 200_000]

    left_rgb = chunk.rgb_frames_per_camera["camera_cross_left_120fov"]
    assert len(left_rgb) == 1
    assert left_rgb[0].image_bytes == b"LEFT_RGB_0"

    # HDMap dispatch: ID gets the ``hdmap_<cam>`` prefix.
    hdmap_front = chunk.hdmap_frames_per_camera["camera_front_wide_120fov"]
    assert len(hdmap_front) == 1
    assert hdmap_front[0].camera_logical_id == "hdmap_camera_front_wide_120fov"
    assert hdmap_front[0].image_bytes == b"FRONT_HDMAP_0"


@pytest.mark.asyncio
async def test_render_chunk_serializes_dynamic_world_state(grpc_world_model) -> None:
    """Dynamic actor conditioning is forwarded on the chunk request."""
    servicer, address = grpc_world_model
    service = VideoModelService(address=address, config=VideoModelConfig(fps=10))
    await service._open_connection()

    actor_trajectory = Trajectory(
        poses=[
            _pose_at(100_000, x=4.0),
            _pose_at(200_000, x=5.0),
        ]
    )
    dynamic_state = DynamicWorldState(
        actors=[
            DynamicActor(
                class_id=ActorClassId.CAR,
                bbox_dims=AABB(4.5, 2.0, 1.6).to_grpc(),
                trajectory=actor_trajectory,
            )
        ]
    )

    service._session_id = SessionId(session_id="active-session")
    await service.render_chunk(
        trajectory_local_to_rig=_trajectory([100_000, 200_000]),
        dynamic_state=dynamic_state,
    )
    await service._close_connection()

    captured = servicer.render_request
    assert captured is not None
    assert len(captured.dynamic_state.actors) == 1
    actor = captured.dynamic_state.actors[0]
    assert actor.class_id == ActorClassId.CAR
    assert actor.bbox_dims.size_x == pytest.approx(4.5)
    assert [p.timestamp_us for p in actor.trajectory.poses] == [100_000, 200_000]
    assert [p.pose.vec.x for p in actor.trajectory.poses] == pytest.approx([4.0, 5.0])


@pytest.mark.asyncio
async def test_render_chunk_uses_request_trajectory_for_frame_timestamps(
    grpc_world_model,
) -> None:
    """Returned frames are timestamped from the request trajectory."""
    servicer, address = grpc_world_model
    service = VideoModelService(address=address, config=VideoModelConfig())
    await service._open_connection()

    request_traj = _trajectory([0, 100_000, 200_000, 300_000])
    servicer.render_response = VideoChunkReturn(
        camera_outputs=[
            CameraOutput(
                camera_logical_id="camera_front_wide_120fov",
                rgb_frames=[
                    Image(data=b"A", format=ImageFormat.JPEG),
                    Image(data=b"B", format=ImageFormat.JPEG),
                    Image(data=b"C", format=ImageFormat.JPEG),
                    Image(data=b"D", format=ImageFormat.JPEG),
                ],
            )
        ],
    )
    service._session_id = SessionId(session_id="s")
    chunk = await service.render_chunk(
        trajectory_local_to_rig=request_traj,
        dynamic_state=DynamicWorldState(),
    )
    await service._close_connection()

    rgb = chunk.rgb_frames_per_camera["camera_front_wide_120fov"]
    assert [f.start_timestamp_us for f in rgb] == [0, 100_000, 200_000, 300_000]


@pytest.mark.asyncio
async def test_cleanup_session_calls_close_session_on_server(grpc_world_model) -> None:
    """``_cleanup_session`` triggers a real ``close_session`` RPC."""
    servicer, address = grpc_world_model
    service = VideoModelService(address=address, config=VideoModelConfig())
    await service._open_connection()
    service._session_id = SessionId(session_id="to-close")
    log_handler = _CollectingHandler()
    service.session_info = SessionInfo(
        uuid="test-session",
        broadcaster=MessageBroadcaster([log_handler]),
    )

    await service._cleanup_session(session_info=None)  # type: ignore[arg-type]
    service.session_info = None
    await service._close_connection()

    assert servicer.close_session_request is not None
    assert servicer.close_session_request.session_id == "to-close"
    assert log_handler.entry_types == [
        "video_model_session_close_request",
    ]
    # State was reset.
    assert service._session_id is None


@pytest.mark.asyncio
async def test_cleanup_session_swallows_server_unavailable() -> None:
    """Cleanup tolerates a server that has already gone away.

    Mirrors the production case where the GPU server crashes mid-rollout: we
    log a warning and reset state instead of poisoning the rollout teardown.
    Owns its own server lifecycle so we can stop it before invoking cleanup.
    """
    servicer = _RecordingWorldModelServicer()
    server, address = await _start_world_model_server(servicer)

    service = VideoModelService(address=address, config=VideoModelConfig())
    await service._open_connection()
    service._session_id = SessionId(session_id="orphan")

    await server.stop(grace=0.0)
    await service._cleanup_session(session_info=None)  # type: ignore[arg-type]
    await service._close_connection()

    assert service._session_id is None
