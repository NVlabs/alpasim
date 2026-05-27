# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Parse camera calibration from ``clipgt/calibration_estimate.parquet``.

This is the canonical path from the USDZ scene's recorded calibration to a
runtime ``CameraDefinition`` carrying real ftheta intrinsics and the actual
sensor-mount pose.  Avoids the previous degenerate fallback in
``VideoModelService._register_scene_cameras`` (synthesized 60deg equidistant
FTheta + ``Pose.identity()`` rig_to_camera), which placed the camera at the
rig origin and made the world model render the scene from ground level.

Schema notes (kept here so future maintainers don't have to re-derive):

The parquet is a single-row table whose ``calibration_estimate`` column
contains either a JSON string or a dict with a nested ``rig_json`` string.
The decoded structure is MADS_RIG_V2 with a ``sensors`` list, where each
camera entry has:

- ``name``: colon-separated identifier (e.g. ``camera:front:wide:120fov``)
- ``properties``: ftheta intrinsics
  (``cx``, ``cy``, ``width``, ``height``, ``polynomial``,
  ``polynomial-type``, ``linear-c``, ``linear-d``, ``linear-e``)
- ``nominalSensor2Rig_FLU``: nominal sensor->rig pose
  (roll-pitch-yaw degrees + translation metres, FLU frame)
- ``correction_sensor_R_FLU``: small rotation correction in sensor frame
- ``correction_rig_T``: small translation correction in rig frame

This module is a direct port of the corresponding helper from the
``dev/gtc-demo-human-driver`` reference branch -- the parsing logic is
identical, only the imports were adapted to use ``sensorsim_pb2.CameraSpec``
to match the public ``CameraDefinition.intrinsics`` type.
"""

from __future__ import annotations

import io
import json
import logging
import math
import zipfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_runtime.camera_catalog import CameraDefinition
from alpasim_utils.geometry import Pose
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def parse_cameras_from_usdz(
    usdz_path: Union[str, Path],
) -> dict[str, CameraDefinition]:
    """Parse all camera definitions from the clipgt calibration in a USDZ file.

    Returns a mapping of ``logical_id`` (underscore-separated, e.g.
    ``camera_front_wide_120fov``) to ``CameraDefinition``.

    Raises ``FileNotFoundError`` if the USDZ or parquet is missing,
    ``ValueError`` if the parquet schema is unexpected.
    """
    usdz_path = Path(usdz_path)
    if not usdz_path.exists():
        raise FileNotFoundError(f"USDZ file not found: {usdz_path}")

    with zipfile.ZipFile(usdz_path, "r") as zf:
        if "clipgt/calibration_estimate.parquet" not in zf.namelist():
            raise FileNotFoundError(
                f"clipgt/calibration_estimate.parquet not found in {usdz_path}"
            )
        with zf.open("clipgt/calibration_estimate.parquet") as f:
            df = pd.read_parquet(io.BytesIO(f.read()))

    if df.empty or "calibration_estimate" not in df.columns:
        raise ValueError(
            f"Unexpected calibration_estimate.parquet schema in {usdz_path}"
        )

    calib = df.iloc[0]["calibration_estimate"]
    if isinstance(calib, str):
        calib = json.loads(calib)

    if "rig_json" in calib:
        rig_json_str = calib["rig_json"]
        rig_data = (
            json.loads(rig_json_str) if isinstance(rig_json_str, str) else rig_json_str
        )
        rig = rig_data.get("rig", rig_data)
    elif "rig" in calib:
        rig = calib["rig"]
    else:
        rig = calib

    sensors = rig.get("sensors", [])
    cameras: dict[str, CameraDefinition] = {}
    for sensor in sensors:
        name: str = sensor.get("name", "")
        if not name.startswith("camera:"):
            continue

        props = sensor.get("properties") or {}
        model = props.get("Model", "")
        if model != "ftheta":
            logger.warning("Skipping camera %s: unsupported model %r", name, model)
            continue

        try:
            cam_def = _build_camera_definition(name, sensor, props)
        except Exception:
            logger.exception("Failed to parse camera %s from clipgt", name)
            continue

        cameras[cam_def.logical_id] = cam_def

    logger.info(
        "Parsed %d camera(s) from clipgt/calibration_estimate.parquet: %s",
        len(cameras),
        list(cameras.keys()),
    )
    return cameras


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clipgt_name_to_logical_id(name: str) -> str:
    """``camera:front:wide:120fov`` -> ``camera_front_wide_120fov``."""
    return name.replace(":", "_")


def _parse_polynomial(poly_str: str) -> list[float]:
    return [float(c) for c in poly_str.strip().split()]


def _invert_polynomial(
    coeffs: list[float],
    *,
    x_max: float,
    n_samples: int = 256,
    degree: int | None = None,
) -> list[float]:
    """Fit the inverse polynomial of ``y = poly(x)`` over ``x in [0, x_max]``.

    The MADS calibration parquet stores either ``pixeldistance-to-angle`` or
    ``angle-to-pixeldistance`` form, never both, but downstream consumers
    (server + alpasim rectifier) ask for whichever direction they need.
    Polynomial inversion in closed form depends on the parent polynomial's
    structure, so we just sample-and-refit at the same degree -- empirically
    this matches the original to <1e-6 over the FOV for both directions of
    the FTheta polynomials we've seen in practice.
    """
    if not coeffs:
        return []
    if degree is None:
        degree = max(1, len(coeffs) - 1)
    xs = np.linspace(0.0, x_max, n_samples)
    ys = np.polynomial.polynomial.polyval(xs, coeffs)
    # Sort by y so polyfit sees a monotonic input (FTheta polynomials are
    # monotonic over the valid FOV by construction).
    order = np.argsort(ys)
    inv_coeffs = np.polynomial.polynomial.polyfit(ys[order], xs[order], degree)
    return inv_coeffs.tolist()


def _compute_corrected_pose(sensor: dict) -> Pose:
    """Compose ``nominalSensor2Rig_FLU`` with the per-sensor corrections.

    - ``nominalSensor2Rig_FLU``: base rotation (rpy) + translation (FLU)
    - ``correction_sensor_R_FLU``: rotation correction in sensor frame
      (right-multiplied)
    - ``correction_rig_T``: translation correction in rig frame (added)
    """
    nominal = sensor.get("nominalSensor2Rig_FLU", {})
    nominal_rpy = nominal.get("roll-pitch-yaw", [0.0, 0.0, 0.0])
    nominal_t = np.array(nominal.get("t", [0.0, 0.0, 0.0]), dtype=np.float32)

    R_nominal = Rotation.from_euler("xyz", nominal_rpy, degrees=True)
    corr_rpy = sensor.get("correction_sensor_R_FLU", {}).get(
        "roll-pitch-yaw", [0.0, 0.0, 0.0]
    )
    R_correction = Rotation.from_euler("xyz", corr_rpy, degrees=True)
    R_final = R_nominal * R_correction

    corr_t = np.array(sensor.get("correction_rig_T", [0.0, 0.0, 0.0]), dtype=np.float32)
    t_final = nominal_t + corr_t

    quat_xyzw = R_final.as_quat().astype(np.float32)
    return Pose(t_final, quat_xyzw)


def _build_camera_definition(name: str, sensor: dict, props: dict) -> CameraDefinition:
    logical_id = _clipgt_name_to_logical_id(name)

    cx = float(props["cx"])
    cy = float(props["cy"])
    width = int(props["width"])
    height = int(props["height"])
    poly_coeffs = _parse_polynomial(props.get("polynomial", ""))
    poly_type_str = props.get("polynomial-type", "pixeldistance-to-angle")
    linear_c = float(props.get("linear-c", 1.0))
    linear_d = float(props.get("linear-d", 0.0))
    linear_e = float(props.get("linear-e", 0.0))

    spec = sensorsim_pb2.CameraSpec(
        logical_id=logical_id,
        resolution_h=height,
        resolution_w=width,
        shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
    )

    ftheta = spec.ftheta_param
    ftheta.principal_point_x = cx
    ftheta.principal_point_y = cy

    # We always populate BOTH polynomial directions on the proto -- the
    # parquet stores only the canonical one (per ``polynomial-type``), but
    # downstream consumers ask for whichever direction they need:
    #   - the video-model server uses the canonical polynomial
    #   - alpasim's driver-side rectifier
    #     (`alpasim_driver/rectification.py:ray_to_pixel`) always reads
    #     `angle_to_pixeldist_poly`
    # so we numerically invert the canonical polynomial and store both.
    half_diag = math.hypot(width / 2.0, height / 2.0)
    if poly_type_str == "pixeldistance-to-angle":
        ftheta.reference_poly = (
            sensorsim_pb2.FthetaCameraParam.PolynomialType.PIXELDIST_TO_ANGLE
        )
        ftheta.pixeldist_to_angle_poly.extend(poly_coeffs)
        ftheta.angle_to_pixeldist_poly.extend(
            _invert_polynomial(poly_coeffs, x_max=half_diag * 1.05)
        )
    elif poly_type_str == "angle-to-pixeldistance":
        ftheta.reference_poly = (
            sensorsim_pb2.FthetaCameraParam.PolynomialType.ANGLE_TO_PIXELDIST
        )
        ftheta.angle_to_pixeldist_poly.extend(poly_coeffs)
        ftheta.pixeldist_to_angle_poly.extend(
            _invert_polynomial(poly_coeffs, x_max=math.radians(80.0))
        )
    else:
        raise ValueError(f"Unknown polynomial-type: {poly_type_str!r}")

    linear = ftheta.linear_cde
    linear.linear_c = linear_c
    linear.linear_d = linear_d
    linear.linear_e = linear_e

    rig_to_camera = _compute_corrected_pose(sensor)

    return CameraDefinition(
        logical_id=logical_id,
        intrinsics=spec,
        rig_to_camera=rig_to_camera,
    )
