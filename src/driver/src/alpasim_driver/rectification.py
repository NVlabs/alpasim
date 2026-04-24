# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Module to rectify f-theta camera renders into pinhole frames.

Note: NuRec renderer will support pinhole rendering natively in the future.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import cv2
import numpy as np
from alpasim_grpc.v0 import sensorsim_pb2

from .schema import RectificationTargetConfig

logger = logging.getLogger(__name__)


def _has_distortion(config: RectificationTargetConfig) -> bool:
    """Return True if any distortion coefficients are provided."""
    return bool(config.radial or config.tangential or config.thin_prism)


def _dist_coeff_vector(config: RectificationTargetConfig) -> np.ndarray:
    """Return a 14-length OpenCV distortion vector from the config."""

    dist = np.zeros(14, dtype=np.float64)
    for idx, value in enumerate(config.radial[:6]):
        dist[[0, 1, 4, 5, 6, 7][idx]] = value
    for idx, value in enumerate(config.tangential[:2]):
        dist[[2, 3][idx]] = value
    for idx, value in enumerate(config.thin_prism[:4]):
        dist[8 + idx] = value
    return dist


def _compute_overscan_scale(config: RectificationTargetConfig) -> float:
    """Estimate overscan scale needed to keep undistorted points inside the frame.

    The heuristic samples a small grid on the *distorted* canvas, undistorts the
    samples, measures how far they would land outside the canvas, and converts
    the maximal overflow into a uniform scale. Clamped by `max_overscan_scale`.

    This is necessary because we first rectify from f-theta to a pinhole frame,
    and then re-introduce distortions of the pinhole camera.
    To not loose needed image content in the rectification step, we rectify
    a larger canvas and only crop in the end.
    """

    if not _has_distortion(config):
        return 1.0

    height, width = config.resolution_hw
    fx, fy = config.focal_length
    cx, cy = config.principal_point
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = _dist_coeff_vector(config)

    # Sample a moderately dense grid to avoid under-estimating overflow; cost is tiny
    # (cv2.undistortPoints on <200 samples is negligible compared to map builds).
    samples_x = np.linspace(0.0, width - 1.0, num=9, dtype=np.float64)
    samples_y = np.linspace(0.0, height - 1.0, num=9, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(samples_x, samples_y)
    distorted_pixels = np.stack((grid_x, grid_y), axis=-1)

    undistorted = cv2.undistortPoints(
        distorted_pixels.reshape(-1, 1, 2),
        cameraMatrix=camera_matrix,
        distCoeffs=dist,
    ).reshape(-1, 2)
    undistorted_pixels = np.empty_like(undistorted)
    undistorted_pixels[:, 0] = undistorted[:, 0] * fx + cx
    undistorted_pixels[:, 1] = undistorted[:, 1] * fy + cy

    min_x = float(np.min(undistorted_pixels[:, 0]))
    max_x = float(np.max(undistorted_pixels[:, 0]))
    min_y = float(np.min(undistorted_pixels[:, 1]))
    max_y = float(np.max(undistorted_pixels[:, 1]))

    overflow_left = max(0.0, -min_x)
    overflow_right = max(0.0, max_x - (width - 1))
    overflow_top = max(0.0, -min_y)
    overflow_bottom = max(0.0, max_y - (height - 1))

    margin_x = max(overflow_left, overflow_right) + float(config.safety_margin_px)
    margin_y = max(overflow_top, overflow_bottom) + float(config.safety_margin_px)

    if margin_x <= 0.0 and margin_y <= 0.0:
        return 1.0

    scale_x = 1.0 + 2.0 * margin_x / float(width)
    scale_y = 1.0 + 2.0 * margin_y / float(height)
    overscan_scale = max(scale_x, scale_y)
    overscan_scale = min(overscan_scale, float(config.max_overscan_scale))
    overscan_scale = max(1.0, overscan_scale)
    return overscan_scale


class _PinholeCamera:
    """Generates unit rays for every pixel in an image."""

    def __init__(
        self,
        focal_length: Tuple[float, float],
        principal_point: Tuple[float, float],
        resolution_hw: Tuple[int, int],
    ) -> None:
        """Cache basic intrinsics for fast ray generation."""

        self._fx, self._fy = focal_length
        self._cx, self._cy = principal_point
        self._height, self._width = resolution_hw

    def unit_rays_grid(self) -> np.ndarray:
        """Return a (H, W, 3) array of unit rays for each image pixel."""

        xs = np.arange(self._width, dtype=np.float64)
        ys = np.arange(self._height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        x = (grid_x - self._cx) / self._fx
        y = (grid_y - self._cy) / self._fy
        z = np.ones_like(x)
        rays = np.stack((x, y, z), axis=-1)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return rays / norms


class _FthetaCamera:
    """Projects unit rays into pixel coordinates using theta→radius polynomials."""

    def __init__(
        self,
        intrinsics: sensorsim_pb2.FthetaCameraParam,
        resolution_hw: Tuple[int, int],
    ) -> None:
        """Precompute polynomial and linear transform state for projections."""

        self._angle_to_pixeldist = np.asarray(
            intrinsics.angle_to_pixeldist_poly, dtype=np.float64
        )
        self._pixeldist_to_angle = np.asarray(
            intrinsics.pixeldist_to_angle_poly, dtype=np.float64
        )
        self._principal_point = np.array(
            [intrinsics.principal_point_x, intrinsics.principal_point_y],
            dtype=np.float64,
        )
        self._max_angle = intrinsics.max_angle
        if intrinsics.HasField("linear_cde"):
            linear_c = intrinsics.linear_cde.linear_c
            linear_d = intrinsics.linear_cde.linear_d
            linear_e = intrinsics.linear_cde.linear_e
        else:
            linear_c, linear_d, linear_e = 1.0, 0.0, 0.0
        self._linear_matrix = np.array(
            [[linear_c, linear_d], [linear_e, 1.0]],
            dtype=np.float64,
        )
        self._source_resolution = resolution_hw

    def ray_to_pixel(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project rays (N,3) into pixel coordinates.

        Returns both the pixel coordinates and a boolean validity mask.
        """

        reshaped = rays.reshape(-1, 3).astype(np.float64)
        xy = reshaped[:, :2]
        z = reshaped[:, 2]
        xy_norm = np.linalg.norm(xy, axis=1)

        theta = np.zeros_like(xy_norm)
        positive_z = z > 0.0
        theta[positive_z] = np.arctan2(xy_norm[positive_z], z[positive_z])
        within_fov = theta <= self._max_angle + 1e-6

        radii = np.polynomial.polynomial.polyval(theta, self._angle_to_pixeldist)
        # Avoid division by zero for rays along the optical axis.
        with np.errstate(divide="ignore", invalid="ignore"):
            scales = np.divide(
                radii, xy_norm, out=np.zeros_like(radii), where=xy_norm > 1e-9
            )

        offsets = xy * scales[:, None]
        pixel_offsets = offsets @ self._linear_matrix.T
        pixels = pixel_offsets + self._principal_point

        height, width = self._source_resolution
        in_image = (
            (pixels[:, 0] >= -0.5)
            & (pixels[:, 0] <= width - 0.5)
            & (pixels[:, 1] >= -0.5)
            & (pixels[:, 1] <= height - 0.5)
        )
        valid = positive_z & within_fov & in_image

        return pixels.reshape(rays.shape[:-1] + (2,)), valid.reshape(rays.shape[:-1])


class FthetaToPinholeRectifier:
    """Caches cv2.remap grids to rectify f-theta renders into a pinhole view."""

    def __init__(
        self,
        source_intrinsics: sensorsim_pb2.FthetaCameraParam,
        source_resolution_hw: Tuple[int, int],
        target_intrinsics: RectificationTargetConfig,
    ) -> None:
        """Precompute remap grids that turn f-theta pixels into pinhole pixels."""

        self._source_resolution = tuple(int(v) for v in source_resolution_hw)
        self._requested_resolution = tuple(
            int(v) for v in target_intrinsics.resolution_hw
        )

        self._overscan_scale = _compute_overscan_scale(target_intrinsics)
        requested_h, requested_w = self._requested_resolution
        if self._overscan_scale > 1.0:
            overscan_h = int(math.ceil(requested_h * self._overscan_scale))
            overscan_w = int(math.ceil(requested_w * self._overscan_scale))
            self._rectify_resolution = (overscan_h, overscan_w)
            margin_x = int(round((overscan_w - requested_w) / 2.0))
            margin_y = int(round((overscan_h - requested_h) / 2.0))
            self._rectify_intrinsics = RectificationTargetConfig(
                focal_length=target_intrinsics.focal_length,
                principal_point=(
                    target_intrinsics.principal_point[0] + margin_x,
                    target_intrinsics.principal_point[1] + margin_y,
                ),
                resolution_hw=self._rectify_resolution,
                radial=target_intrinsics.radial,
                tangential=target_intrinsics.tangential,
                thin_prism=target_intrinsics.thin_prism,
                max_overscan_scale=target_intrinsics.max_overscan_scale,
                safety_margin_px=target_intrinsics.safety_margin_px,
            )
            logger.info(
                "Rectification overscan scale %.3f: %s -> %s",
                self._overscan_scale,
                self._requested_resolution,
                self._rectify_resolution,
            )
        else:
            self._rectify_resolution = self._requested_resolution
            self._rectify_intrinsics = target_intrinsics

        self._crop_needed = self._rectify_resolution != self._requested_resolution
        self._crop_origin: tuple[int, int] = (0, 0)
        if self._crop_needed:
            overscan_h, overscan_w = self._rectify_resolution
            crop_x0 = int(round((overscan_w - requested_w) / 2.0))
            crop_y0 = int(round((overscan_h - requested_h) / 2.0))
            crop_x0 = max(0, min(crop_x0, overscan_w - requested_w))
            crop_y0 = max(0, min(crop_y0, overscan_h - requested_h))
            self._crop_origin = (crop_y0, crop_x0)

        self._map_x, self._map_y = self._build_maps(
            source_intrinsics, self._rectify_intrinsics
        )
        self._distort_map_x: np.ndarray | None = None
        self._distort_map_y: np.ndarray | None = None
        if _has_distortion(self._rectify_intrinsics):
            (
                self._distort_map_x,
                self._distort_map_y,
            ) = self._build_distortion_maps(self._rectify_intrinsics)

    def _build_maps(
        self,
        source_intrinsics: sensorsim_pb2.FthetaCameraParam,
        target_intrinsics: RectificationTargetConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate OpenCV remap tables for the provided camera pair."""

        target_resolution = (
            int(target_intrinsics.resolution_hw[0]),
            int(target_intrinsics.resolution_hw[1]),
        )
        pinhole = _PinholeCamera(
            target_intrinsics.focal_length,
            target_intrinsics.principal_point,
            target_resolution,
        )
        rays = pinhole.unit_rays_grid()
        ftheta = _FthetaCamera(source_intrinsics, self._source_resolution)
        pixels, valid = ftheta.ray_to_pixel(rays)

        map_x = pixels[..., 0].astype(np.float32)
        map_y = pixels[..., 1].astype(np.float32)
        invalid_mask = ~valid
        map_x[invalid_mask] = -1.0
        map_y[invalid_mask] = -1.0

        return map_x, map_y

    def _build_distortion_maps(
        self,
        target_intrinsics: RectificationTargetConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create map to reintroduce OpenCV distortion into the pinhole frame."""

        height, width = (
            int(target_intrinsics.resolution_hw[0]),
            int(target_intrinsics.resolution_hw[1]),
        )
        fx, fy = target_intrinsics.focal_length
        cx, cy = target_intrinsics.principal_point

        xs = np.arange(width, dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        distorted_pixels = np.stack((grid_x, grid_y), axis=-1)

        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        dist = _dist_coeff_vector(target_intrinsics)

        undistorted = cv2.undistortPoints(
            distorted_pixels.reshape(-1, 1, 2),
            cameraMatrix=camera_matrix,
            distCoeffs=dist,
        ).reshape(-1, 2)
        undistorted_pixels = np.empty_like(undistorted)
        undistorted_pixels[:, 0] = undistorted[:, 0] * fx + cx
        undistorted_pixels[:, 1] = undistorted[:, 1] * fy + cy

        map_x = undistorted_pixels[:, 0].reshape(height, width).astype(np.float32)
        map_y = undistorted_pixels[:, 1].reshape(height, width).astype(np.float32)

        valid = (
            (map_x >= -0.5)
            & (map_x <= width - 0.5)
            & (map_y >= -0.5)
            & (map_y <= height - 0.5)
        )
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0

        return map_x, map_y

    def rectify(self, image: np.ndarray) -> np.ndarray:
        """Return a pinhole-rectified copy of the provided f-theta image."""

        logger.debug(
            "Rectifying image %s with source resolution %s",
            image.shape,
            self._source_resolution,
        )
        expected_h, expected_w = self._source_resolution
        actual_h, actual_w = image.shape[0], image.shape[1]
        if actual_h != expected_h or actual_w != expected_w:
            delta_h = actual_h - expected_h
            delta_w = actual_w - expected_w
            if 0 <= delta_h <= 1 and 0 <= delta_w <= 1:
                logger.warning(
                    "Source resolution %s differs from expected %s; cropping to expected",
                    (actual_h, actual_w),
                    self._source_resolution,
                )
                image = image[:expected_h, :expected_w]
            else:
                raise ValueError(
                    "Unexpected source resolution: "
                    f"got {(actual_h, actual_w)}, expected {self._source_resolution}"
                )

        rectified = cv2.remap(
            image,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if self._distort_map_x is not None and self._distort_map_y is not None:
            rectified = cv2.remap(
                rectified,
                self._distort_map_x,
                self._distort_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        if self._crop_needed:
            crop_y0, crop_x0 = self._crop_origin
            crop_y1 = crop_y0 + self._requested_resolution[0]
            crop_x1 = crop_x0 + self._requested_resolution[1]
            rectified = rectified[crop_y0:crop_y1, crop_x0:crop_x1]
        return rectified


def _scale_ftheta_intrinsics_to_resolution(
    intrinsics: sensorsim_pb2.FthetaCameraParam,
    native_resolution_hw: tuple[int, int],
    target_resolution_hw: tuple[int, int],
) -> sensorsim_pb2.FthetaCameraParam:
    """Return a copy of `intrinsics` scaled to `target_resolution_hw`.

    The y-axis scale is applied through the radial terms (angle→pixel distance),
    and the x-axis scale is captured by the linear C/D terms so we can represent
    anisotropic resizing without double-scaling the radius.
    """

    native_h, native_w = native_resolution_hw
    target_h, target_w = target_resolution_hw
    scale_x = target_w / native_w
    scale_y = target_h / native_h
    if not np.isclose(scale_x, scale_y, atol=1e-6):
        logger.warning(
            "Anisotropic f-theta scaling: native=%s, target=%s, scales=(%.6f, %.6f)",
            native_resolution_hw,
            target_resolution_hw,
            scale_x,
            scale_y,
        )

    scaled = sensorsim_pb2.FthetaCameraParam()
    scaled.CopyFrom(intrinsics)

    scaled.principal_point_x *= scale_x
    scaled.principal_point_y *= scale_y

    # Scale the radial terms with the vertical factor; the x-axis scale is
    # injected via the linear C/D terms below to represent anisotropic resizing.
    radial_scale = scale_y
    if len(scaled.angle_to_pixeldist_poly):
        coeffs = (
            np.asarray(scaled.angle_to_pixeldist_poly, dtype=np.float64) * radial_scale
        )
        scaled.angle_to_pixeldist_poly[:] = []
        scaled.angle_to_pixeldist_poly.extend(coeffs.tolist())
    if len(scaled.pixeldist_to_angle_poly):
        powers = np.power(radial_scale, np.arange(len(scaled.pixeldist_to_angle_poly)))
        coeffs = np.asarray(scaled.pixeldist_to_angle_poly, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            coeffs = np.divide(
                coeffs,
                powers,
                out=np.zeros_like(coeffs),
                where=powers != 0,
            )
        scaled.pixeldist_to_angle_poly[:] = []
        scaled.pixeldist_to_angle_poly.extend(coeffs.tolist())

    # Apply the remaining x-axis scaling through the linear terms so we do not
    # double-scale the radius when the resize is anisotropic.
    linear_c = (
        intrinsics.linear_cde.linear_c if intrinsics.HasField("linear_cde") else 1.0
    )
    linear_d = (
        intrinsics.linear_cde.linear_d if intrinsics.HasField("linear_cde") else 0.0
    )
    linear_e = (
        intrinsics.linear_cde.linear_e if intrinsics.HasField("linear_cde") else 0.0
    )
    x_ratio = scale_x / radial_scale
    scaled.linear_cde.linear_c = linear_c * x_ratio
    scaled.linear_cde.linear_d = linear_d * x_ratio
    scaled.linear_cde.linear_e = linear_e

    return scaled


def build_ftheta_rectifier_for_resolution(
    camera_proto: sensorsim_pb2.AvailableCamerasReturn.AvailableCamera,
    target_cfg: RectificationTargetConfig,
    source_resolution_hw: tuple[int, int],
) -> FthetaToPinholeRectifier:
    """Construct a rectifier using `camera_proto` scaled to `source_resolution_hw`."""

    if camera_proto.intrinsics.WhichOneof("camera_param") != "ftheta_param":
        raise ValueError(
            f"Camera {camera_proto.logical_id} does not provide f-theta intrinsics"
        )

    native_resolution_hw = (
        int(camera_proto.intrinsics.resolution_h),
        int(camera_proto.intrinsics.resolution_w),
    )

    # The intrinsics are for the native camera resolution, so we need to scale them
    # to the resolution that the camera was actually rendered at.
    scaled_intrinsics = _scale_ftheta_intrinsics_to_resolution(
        intrinsics=camera_proto.intrinsics.ftheta_param,
        native_resolution_hw=native_resolution_hw,
        target_resolution_hw=source_resolution_hw,
    )
    return FthetaToPinholeRectifier(
        source_intrinsics=scaled_intrinsics,
        source_resolution_hw=source_resolution_hw,
        target_intrinsics=target_cfg,
    )


def build_rectifier_map(
    rectification_cfg: dict[str, RectificationTargetConfig] | None,
    desired_cameras: set[str],
    camera_lookup: dict[str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera],
) -> dict[str, FthetaToPinholeRectifier | None]:
    """Instantiate per-camera rectifiers as requested in the Hydra config."""

    rectifiers: dict[str, FthetaToPinholeRectifier | None] = {
        logical_id: None for logical_id in desired_cameras
    }
    if rectification_cfg is None:
        return rectifiers

    for logical_id in desired_cameras:
        target_cfg = rectification_cfg[logical_id]
        camera_proto = camera_lookup[logical_id]

        rectifiers[logical_id] = build_ftheta_rectifier_for_resolution(
            camera_proto=camera_proto,
            target_cfg=target_cfg,
            source_resolution_hw=(
                int(camera_proto.intrinsics.resolution_h),
                int(camera_proto.intrinsics.resolution_w),
            ),
        )
        logger.info("Enabled f-theta→pinhole rectification for %s", logical_id)

    return rectifiers
