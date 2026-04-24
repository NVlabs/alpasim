#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Fit an f-theta camera model that matches an OpenCV pinhole calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf

_RADIAL_IDXS = (0, 1, 4, 5, 6, 7)


@dataclass(frozen=True)
class PinholeIntrinsics:
    focal_length: Tuple[float, float]
    principal_point: Tuple[float, float]
    radial: Sequence[float]
    tangential: Sequence[float]
    thin_prism: Sequence[float]
    resolution_hw: Tuple[int, int]

    @property
    def camera_matrix(self) -> np.ndarray:
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )


def _build_distortion_vector(pinhole: PinholeIntrinsics) -> np.ndarray:
    coeffs = np.zeros(12, dtype=np.float64)
    for idx, value in zip(_RADIAL_IDXS, pinhole.radial):
        coeffs[idx] = value
    if len(pinhole.tangential) >= 2:
        coeffs[2] = pinhole.tangential[0]
        coeffs[3] = pinhole.tangential[1]
    if len(pinhole.thin_prism) >= 4:
        coeffs[8:12] = np.asarray(pinhole.thin_prism[:4], dtype=np.float64)
    return coeffs


def _sample_theta_radius(
    pinhole: PinholeIntrinsics, grid_w: int, grid_h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample rays across the imager to build theta/radius pairs."""
    height, width = pinhole.resolution_hw
    xs = np.linspace(0.0, width - 1.0, num=grid_w, dtype=np.float64)
    ys = np.linspace(0.0, height - 1.0, num=grid_h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pixels = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)

    undistorted = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        cameraMatrix=pinhole.camera_matrix,
        distCoeffs=_build_distortion_vector(pinhole).reshape(-1, 1),
    )
    xy = undistorted[:, 0, :]
    theta = np.arctan(np.linalg.norm(xy, axis=1))

    principal_point = np.array(pinhole.principal_point, dtype=np.float64)
    radius = np.linalg.norm(pixels - principal_point, axis=1)
    return pixels, theta, radius


def _fit_angle_to_pixel(
    theta: np.ndarray, radius: np.ndarray, max_order: int
) -> np.ndarray:
    if max_order < 1 or max_order % 2 == 0:
        raise ValueError("max_order must be an odd positive integer (e.g. 5).")
    powers = np.arange(1, max_order + 1, 2)
    design = np.stack([theta**power for power in powers], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, radius, rcond=None)
    poly = np.zeros(max_order + 1, dtype=np.float64)
    for power, coeff in zip(powers, coeffs):
        poly[power] = coeff
    return poly


def _fit_pixel_to_angle(
    radius: np.ndarray, theta: np.ndarray, degree: int
) -> np.ndarray:
    coeffs = np.polyfit(radius, theta, degree)[::-1]
    coeffs[0] = 0.0
    return coeffs


def _resize_coeffs(poly: np.ndarray, length: int) -> np.ndarray:
    if poly.shape[0] == length:
        return poly
    if poly.shape[0] > length:
        return poly[:length]
    resized = np.zeros(length, dtype=poly.dtype)
    resized[: poly.shape[0]] = poly
    return resized


def _load_pinhole_from_yaml(path: Path) -> PinholeIntrinsics:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML structure must be a mapping.")

    intrinsics = cfg.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError("YAML must contain an 'intrinsics' mapping.")

    model = intrinsics.get("model")
    if model != "opencv_pinhole":
        raise ValueError(f"Unsupported model '{model}'. Expected 'opencv_pinhole'.")

    details = intrinsics.get("opencv_pinhole", {})
    focal = details.get("focal_length")
    principal = details.get("principal_point")
    radial = details.get("radial", [])
    tangential = details.get("tangential", [])
    thin_prism = details.get("thin_prism", [])

    if not focal or not principal:
        raise ValueError(
            "Pinhole intrinsics must define 'focal_length' and 'principal_point'."
        )

    resolution = intrinsics.get("resolution_hw", cfg.get("resolution_hw"))
    if not resolution:
        raise ValueError("Resolution must be provided as 'resolution_hw'.")

    return PinholeIntrinsics(
        focal_length=tuple(float(v) for v in focal[:2]),
        principal_point=tuple(float(v) for v in principal[:2]),
        radial=[float(v) for v in radial],
        tangential=[float(v) for v in tangential],
        thin_prism=[float(v) for v in thin_prism],
        resolution_hw=tuple(int(v) for v in resolution[:2]),
    )


def _build_output_dict(
    pinhole: PinholeIntrinsics,
    angle_to_pixeldist: np.ndarray,
    pixeldist_to_angle: np.ndarray,
    theta: np.ndarray,
    radius: np.ndarray,
) -> dict:
    fitted = np.polynomial.polynomial.polyval(theta, angle_to_pixeldist)
    residuals = radius - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    max_err = float(np.max(np.abs(residuals)))

    return {
        "intrinsics": {
            "model": "ftheta",
            "ftheta": {
                "principal_point": list(pinhole.principal_point),
                "reference_poly": "angle_to_pixel",
                "pixeldist_to_angle": pixeldist_to_angle.tolist(),
                "angle_to_pixeldist": angle_to_pixeldist.tolist(),
                "max_angle": float(theta.max()),
            },
            "resolution_hw": list(pinhole.resolution_hw),
        },
        "fit_metrics": {
            "rmse_pixels": rmse,
            "max_error_pixels": max_err,
        },
        "sampling": {"num_samples": int(theta.shape[0])},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenCV pinhole intrinsics into an f-theta model.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the YAML file that stores the pinhole intrinsics.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the fitted f-theta intrinsics (YAML).",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=60,
        help="Number of samples along image width (default: 40).",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=45,
        help="Number of samples along image height (default: 30).",
    )
    parser.add_argument(
        "--coeff-count",
        type=int,
        default=6,
        help="Number of coefficients to publish for both polynomials (default: 6).",
    )
    parser.add_argument(
        "--angle-to-pixel-order",
        type=int,
        help="Override highest odd theta power for theta→pixel polynomial. "
        "Defaults to coeff-count - 1.",
    )
    parser.add_argument(
        "--pixel-to-angle-degree",
        type=int,
        help="Override degree for pixel→theta polynomial. Defaults to coeff-count - 1.",
    )
    parser.add_argument(
        "--reference-poly",
        choices=["angle_to_pixel", "pixel_to_angle"],
        default="pixel_to_angle",
        help="Which polynomial the downstream consumer treats as canonical.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.coeff_count < 2:
        raise ValueError("Coefficient count must be >= 2.")

    pinhole = _load_pinhole_from_yaml(args.input)
    _, theta, radius = _sample_theta_radius(pinhole, args.grid_width, args.grid_height)
    default_order = args.coeff_count - 1
    angle_order = (
        args.angle_to_pixel_order
        if args.angle_to_pixel_order is not None
        else default_order
    )
    if angle_order < 1 or angle_order % 2 == 0:
        raise ValueError("Highest theta power must be an odd positive integer.")
    pixel_degree = (
        args.pixel_to_angle_degree
        if args.pixel_to_angle_degree is not None
        else args.coeff_count - 1
    )
    if pixel_degree < 1:
        raise ValueError("Pixel-to-angle degree must be >= 1.")

    angle_to_pix = _fit_angle_to_pixel(theta, radius, angle_order)
    angle_to_pix = _resize_coeffs(angle_to_pix, args.coeff_count)
    pix_to_angle = _fit_pixel_to_angle(radius, theta, pixel_degree)
    pix_to_angle = _resize_coeffs(pix_to_angle, args.coeff_count)
    output = _build_output_dict(pinhole, angle_to_pix, pix_to_angle, theta, radius)
    output["intrinsics"]["ftheta"]["reference_poly"] = args.reference_poly

    print("Angle→pixel polynomial coefficients:")
    for power, coefficient in enumerate(angle_to_pix):
        print(f"  r(theta) coeff[{power}]: {coefficient:.9f}")

    print("\nPixel→angle polynomial coefficients:")
    for power, coefficient in enumerate(pix_to_angle):
        print(f"  theta(r) coeff[{power}]: {coefficient:.12f}")

    metrics = output["fit_metrics"]
    print(
        f"\nFit metrics (angle→pixel): RMSE={metrics['rmse_pixels']:.4f} px, "
        f"max error={metrics['max_error_pixels']:.4f} px"
    )

    if args.output:
        OmegaConf.save(config=OmegaConf.create(output), f=str(args.output))
        print(f"Saved f-theta intrinsics to {args.output}")


if __name__ == "__main__":
    main()
