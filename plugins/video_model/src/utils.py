# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Shared utilities for video model initialization and trajectory building."""

from __future__ import annotations

import io
import logging
import math
import zipfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import yaml
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2
from alpasim_grpc.v0.video_model_pb2 import ImageFormat as VideoModelImageFormat
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.types import RuntimeCamera
from alpasim_utils import geometry

logger = logging.getLogger(__name__)


def populate_default_equidistant_ftheta(
    ftheta_param: Any, *, h: int, w: int, half_fov_deg: float = 60.0
) -> None:
    """Populate ``ftheta_param`` with equidistant FTheta defaults (``r = f * theta``).

    Used as a fallback when the source camera intrinsics don't expose an
    FTheta model.  ``ftheta_param`` may be either an
    ``alpasim_grpc.v0.sensorsim_pb2.FthetaCameraParam`` or an equivalent
    wire-compatible FTheta proto: both have the same field layout and
    ``ANGLE_TO_PIXELDIST`` enum value, so ``type(...)`` is used to look up the
    correct enum at runtime.
    """
    half_fov = math.radians(half_fov_deg)
    half_diag = math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
    focal_length = half_diag / half_fov
    ftheta_param.principal_point_x = w / 2.0
    ftheta_param.principal_point_y = h / 2.0
    ftheta_param.reference_poly = type(ftheta_param).PolynomialType.ANGLE_TO_PIXELDIST
    del ftheta_param.angle_to_pixeldist_poly[:]
    ftheta_param.angle_to_pixeldist_poly.extend([0.0, focal_length])
    ftheta_param.max_angle = half_fov


def build_trajectory_for_video_model(
    ego_trajectory: geometry.Trajectory,
    start_us: int,
    num_frames: int,
    frame_interval_us: int,
) -> common_pb2.Trajectory:
    """Build a trajectory message for the video model chunk request."""
    pose_trajectory = (
        ego_trajectory.trajectory()
        if hasattr(ego_trajectory, "trajectory")
        else ego_trajectory
    )
    traj_range = pose_trajectory.time_range_us
    timestamps_us = np.array(
        [start_us + i * frame_interval_us for i in range(num_frames)], dtype=np.uint64
    )
    clamped_timestamps_us = np.clip(
        timestamps_us,
        traj_range.start,
        traj_range.stop - 1,
    )
    sampled_poses = pose_trajectory.interpolate_poses_list(clamped_timestamps_us)
    poses_at_time = []

    for ts, pose in zip(timestamps_us, sampled_poses, strict=True):
        poses_at_time.append(
            common_pb2.PoseAtTime(
                pose=common_pb2.Pose(
                    vec=common_pb2.Vec3(x=pose.vec3[0], y=pose.vec3[1], z=pose.vec3[2]),
                    quat=common_pb2.Quat(
                        w=pose.quat[3], x=pose.quat[0], y=pose.quat[1], z=pose.quat[2]
                    ),
                ),
                timestamp_us=ts,
            )
        )

    return common_pb2.Trajectory(poses=poses_at_time)


def build_camera_specs_and_initial_frames(
    runtime_cameras: list[RuntimeCamera],
    camera_catalog: CameraCatalog,
    scene_id: str,
    usdz_path: str,
) -> tuple[
    list[sensorsim_pb2.CameraSpec], list[common_pb2.Pose], list[tuple[bytes, int]]
]:
    """Build camera intrinsics, extrinsics, and initial frames for a session."""
    _generic_first, per_camera_first = extract_first_frames(usdz_path)

    if not per_camera_first:
        raise FileNotFoundError(
            f"No frames/<camera>/<timestamp>.jpeg files found in {usdz_path}"
        )

    camera_specs: list[sensorsim_pb2.CameraSpec] = []
    rig_to_camera: list[common_pb2.Pose] = []
    initial_frames: list[tuple[bytes, int]] = []

    for camera in runtime_cameras:
        cam_def = camera_catalog.get_camera_definition(scene_id, camera.logical_id)

        # Preserve all recorded intrinsics, including linear_cde. A previous
        # hand-rolled copy dropped linear_cde and caused inconsistent FTheta
        # projection in the video-model server.
        spec = sensorsim_pb2.CameraSpec()
        spec.CopyFrom(cam_def.intrinsics)

        # If the recorded calibration didn't expose an FTheta model, synthesize
        # a default equidistant one (the video model server requires FTheta).
        if not spec.HasField("ftheta_param"):
            h = spec.resolution_h or 640
            w = spec.resolution_w or 1280
            populate_default_equidistant_ftheta(spec.ftheta_param, h=h, w=w)

        # Default shutter to GLOBAL when calibration left it UNKNOWN (which
        # silently degrades output for scenes the server treats as rolling).
        if not spec.shutter_type:
            spec.shutter_type = sensorsim_pb2.ShutterType.GLOBAL

        camera_specs.append(spec)
        rig_to_camera.append(geometry.pose_to_grpc(cam_def.rig_to_camera))

        img_bytes, fmt = _select_first_frame(camera.logical_id, per_camera_first)
        initial_frames.append((img_bytes, VideoModelImageFormat.Value(fmt)))

    return camera_specs, rig_to_camera, initial_frames


def _select_first_frame(
    camera_logical_id: str,
    per_camera_first: dict[str, tuple[bytes, str]],
) -> tuple[bytes, str]:
    """Select the appropriate first frame for a camera."""
    if camera_logical_id in per_camera_first:
        return per_camera_first[camera_logical_id]
    raise FileNotFoundError(
        f"No frames/<camera>/<timestamp>.jpeg available for camera {camera_logical_id}"
    )


# ---------------------------------------------------------------------------
# USDZ extraction utilities (ported from human-driver hdmap_utils)
# ---------------------------------------------------------------------------


def _normalize_camera_id(camera_id: str) -> str:
    return camera_id.replace(":", "_")


def _extract_timestamped_frame_name(name: str) -> tuple[str, int] | None:
    """Return ``(camera_id, timestamp_us)`` for frames/<camera>/<timestamp>.jpeg.

    First frames in USDZ archives are standardized to JPEG (matching the
    format the data lake actually returns).  Mismatched extensions caused
    real bugs (PNG payloads saved with ``.jpeg`` suffix), so we fail loudly
    rather than guess the encoding from the bytes.
    """
    parts = name.split("/")
    if len(parts) != 3 or parts[0] != "frames":
        return None

    camera_id = parts[1]
    frame_path = Path(parts[2])
    if frame_path.suffix.lower() != ".jpeg":
        raise ValueError(
            f"Unsupported first-frame image path in USDZ: {name} "
            f"(only .jpeg is supported; the data lake returns JPEG)."
        )
    try:
        timestamp_us = int(frame_path.stem)
    except ValueError:
        raise ValueError(
            f"Invalid first-frame timestamp in USDZ path: {name}"
        ) from None
    if timestamp_us < 0:
        raise ValueError(
            f"Invalid first-frame timestamp in USDZ path: {name} "
            f"(timestamp must be non-negative)."
        )
    return camera_id, timestamp_us


def _extract_clip_id_from_usdz(usdz: zipfile.ZipFile) -> str:
    if "metadata.yaml" not in usdz.namelist():
        raise ValueError("metadata.yaml not found in USDZ")
    with usdz.open("metadata.yaml") as f:
        metadata = yaml.safe_load(f)
    scene_id = metadata.get("scene_id")
    if not scene_id:
        raise ValueError("scene_id not found in metadata.yaml")
    return scene_id[len("clipgt-") :] if scene_id.startswith("clipgt-") else scene_id


def extract_first_frames(
    usdz_path: Union[str, Path],
) -> tuple[Optional[tuple[bytes, str]], dict[str, tuple[bytes, str]]]:
    """Extract timestamped JPEG first-frame images from a USDZ archive."""
    usdz_path = Path(usdz_path)
    if not usdz_path.exists():
        raise FileNotFoundError(f"USDZ file not found: {usdz_path}")

    per_camera: dict[str, tuple[bytes, str]] = {}

    with zipfile.ZipFile(usdz_path, "r") as usdz:
        timestamped_frames: dict[str, tuple[int, bytes, str]] = {}
        seen_frame_keys: set[tuple[str, int]] = set()

        for name in usdz.namelist():
            if not name.startswith("frames/") or name.endswith("/"):
                continue
            frame_name = _extract_timestamped_frame_name(name)
            if frame_name is None:
                continue
            camera_id, timestamp_us = frame_name
            cam_key = _normalize_camera_id(camera_id)
            frame_key = (cam_key, timestamp_us)
            if frame_key in seen_frame_keys:
                raise ValueError(
                    f"Duplicate first-frame image for camera={cam_key} "
                    f"timestamp_us={timestamp_us}"
                )
            seen_frame_keys.add(frame_key)

            data = usdz.read(name)
            if not data.startswith(b"\xff\xd8\xff"):
                raise ValueError(f"First-frame image is not JPEG: {name}")

            existing = timestamped_frames.get(cam_key)
            if existing is None or timestamp_us < existing[0]:
                timestamped_frames[cam_key] = (timestamp_us, data, name)

        for cam_key, (_timestamp_us, data, _name) in timestamped_frames.items():
            per_camera[cam_key] = (data, "JPEG")

    return None, per_camera


def extract_hdmap_for_video_model(usdz_path: Union[str, Path]) -> bytes:
    """Extract HD map parquets from USDZ, renamed for the video model server."""
    usdz_path = Path(usdz_path)
    if not usdz_path.exists():
        raise FileNotFoundError(f"USDZ file not found: {usdz_path}")

    with zipfile.ZipFile(usdz_path, "r") as usdz:
        clip_id = _extract_clip_id_from_usdz(usdz)
        clipgt_entries = [n for n in usdz.namelist() if n.startswith("clipgt/")]
        if not clipgt_entries:
            raise ValueError(f"Scene {usdz_path} has no clipgt/ directory.")

        logger.info(
            "Extracting %d clipgt files from %s",
            len(clipgt_entries),
            usdz_path.name,
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out_zip:
            for entry in clipgt_entries:
                if entry.endswith("/"):
                    continue
                content = usdz.read(entry)
                parts = entry.split("/", 1)
                new_name = (
                    f"{clip_id}.{parts[1]}" if len(parts) == 2 else f"{clip_id}.{entry}"
                )
                out_zip.writestr(new_name, content)

        hdmap_bytes = buf.getvalue()
        logger.info(
            "Created HD map archive: %d bytes (%.2f MB)",
            len(hdmap_bytes),
            len(hdmap_bytes) / 1024 / 1024,
        )
        return hdmap_bytes
