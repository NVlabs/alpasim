# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
from alpasim_driver.rectification import (
    _FthetaCamera,
    _scale_ftheta_intrinsics_to_resolution,
)
from alpasim_grpc.v0 import sensorsim_pb2


def test_resolution_scaling_is_not_applied_twice() -> None:
    """Scaling intrinsics to a lower resolution should scale pixel radii once."""

    native_resolution = (1000, 2000)  # (H, W)
    target_resolution = (500, 1000)  # exact 0.5 scale
    scale = target_resolution[1] / native_resolution[1]

    intrinsics = sensorsim_pb2.FthetaCameraParam()
    intrinsics.principal_point_x = native_resolution[1] / 2
    intrinsics.principal_point_y = native_resolution[0] / 2
    intrinsics.angle_to_pixeldist_poly.extend([0.0, 1000.0])  # linear radius

    # Ray leaning along +X, still inside the FOV.
    theta = 0.2
    ray_x = np.array([[np.sin(theta), 0.0, np.cos(theta)]], dtype=np.float64)

    # Baseline at native resolution.
    native_cam = _FthetaCamera(intrinsics, native_resolution)
    native_pixels, _ = native_cam.ray_to_pixel(ray_x)
    native_dx = native_pixels[0, 0] - intrinsics.principal_point_x

    scaled_intrinsics = _scale_ftheta_intrinsics_to_resolution(
        intrinsics, native_resolution, target_resolution
    )
    scaled_cam = _FthetaCamera(scaled_intrinsics, target_resolution)
    scaled_pixels, _ = scaled_cam.ray_to_pixel(ray_x)
    scaled_dx = scaled_pixels[0, 0] - scaled_intrinsics.principal_point_x

    assert np.isclose(scaled_dx / native_dx, scale, atol=1e-6)

    # And the Y-axis behaves symmetrically.
    ray_y = np.array([[0.0, np.sin(theta), np.cos(theta)]], dtype=np.float64)
    native_pixels_y, _ = native_cam.ray_to_pixel(ray_y)
    native_dy = native_pixels_y[0, 1] - intrinsics.principal_point_y

    scaled_pixels_y, _ = scaled_cam.ray_to_pixel(ray_y)
    scaled_dy = scaled_pixels_y[0, 1] - scaled_intrinsics.principal_point_y

    assert np.isclose(scaled_dy / native_dy, scale, atol=1e-6)


def test_anisotropic_scaling_respects_axes() -> None:
    """Anisotropic resizing should scale x/y offsets independently."""

    native_resolution = (1000, 2000)  # (H, W)
    target_resolution = (500, 2500)  # scales: y=0.5, x=1.25
    scale_y = target_resolution[0] / native_resolution[0]
    scale_x = target_resolution[1] / native_resolution[1]

    intrinsics = sensorsim_pb2.FthetaCameraParam()
    intrinsics.principal_point_x = native_resolution[1] / 2
    intrinsics.principal_point_y = native_resolution[0] / 2
    intrinsics.angle_to_pixeldist_poly.extend([0.0, 1000.0])  # linear radius

    theta = 0.2
    ray_x = np.array([[np.sin(theta), 0.0, np.cos(theta)]], dtype=np.float64)
    ray_y = np.array([[0.0, np.sin(theta), np.cos(theta)]], dtype=np.float64)

    native_cam = _FthetaCamera(intrinsics, native_resolution)
    native_pixels_x, _ = native_cam.ray_to_pixel(ray_x)
    native_dx = native_pixels_x[0, 0] - intrinsics.principal_point_x
    native_pixels_y, _ = native_cam.ray_to_pixel(ray_y)
    native_dy = native_pixels_y[0, 1] - intrinsics.principal_point_y

    scaled_intrinsics = _scale_ftheta_intrinsics_to_resolution(
        intrinsics, native_resolution, target_resolution
    )
    scaled_cam = _FthetaCamera(scaled_intrinsics, target_resolution)
    scaled_pixels_x, _ = scaled_cam.ray_to_pixel(ray_x)
    scaled_dx = scaled_pixels_x[0, 0] - scaled_intrinsics.principal_point_x
    scaled_pixels_y, _ = scaled_cam.ray_to_pixel(ray_y)
    scaled_dy = scaled_pixels_y[0, 1] - scaled_intrinsics.principal_point_y

    assert np.isclose(scaled_dx / native_dx, scale_x, atol=1e-6)
    assert np.isclose(scaled_dy / native_dy, scale_y, atol=1e-6)
