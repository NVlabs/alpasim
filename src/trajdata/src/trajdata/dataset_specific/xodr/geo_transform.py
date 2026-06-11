# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Coordinate transformation utilities for XODR processing.

This module provides functions for converting between different coordinate
systems (WGS-84, ECEF, ENU) used in OpenDRIVE georeference processing.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np


def lat_lng_alt_2_ecef_ellipsoidal(
    lat_lng_alt: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """Convert WGS-84 coordinates to ECEF (Earth-Centered, Earth-Fixed).

    Args:
        lat_lng_alt: Array of shape [..., 3] where columns are
                     (lat_deg, lon_deg, alt_m)
        a: Semi-major axis of the reference ellipsoid in meters
        b: Semi-minor axis of the reference ellipsoid in meters

    Returns:
        Cartesian ECEF coordinates with shape [..., 3] containing
        (x, y, z) in meters
    """
    phi = np.deg2rad(lat_lng_alt[..., 0])
    lam = np.deg2rad(lat_lng_alt[..., 1])
    h = lat_lng_alt[..., 2]

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)

    e_sq = (a * a - b * b) / (a * a)
    N = a / np.sqrt(1.0 - e_sq * sin_phi * sin_phi)

    x = (N + h) * cos_phi * cos_lam
    y = (N + h) * cos_phi * sin_lam
    z = (N * (b * b) / (a * a) + h) * sin_phi

    return np.stack([x, y, z], axis=-1)


def ecef_2_enu_matrix(ref_lat_lon_alt: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transform from ECEF to local ENU coordinates.

    Creates a transformation matrix that converts ECEF coordinates to
    a local East-North-Up coordinate system centered at the reference point.

    Args:
        ref_lat_lon_alt: Reference point as [latitude, longitude, altitude]
                        where lat/lon are in degrees and altitude in meters

    Returns:
        4x4 transformation matrix from ECEF to ENU coordinates
    """
    # WGS-84 constants
    a = 6378137.0
    flattening = 1.0 / 298.257223563
    b = a * (1.0 - flattening)

    ref_ecef = lat_lng_alt_2_ecef_ellipsoidal(ref_lat_lon_alt[None, :], a, b)[0]

    lat_rad = np.deg2rad(ref_lat_lon_alt[0])
    lon_rad = np.deg2rad(ref_lat_lon_alt[1])

    rot = np.array(
        [
            [-np.sin(lon_rad), np.cos(lon_rad), 0.0],
            [
                -np.sin(lat_rad) * np.cos(lon_rad),
                -np.sin(lat_rad) * np.sin(lon_rad),
                np.cos(lat_rad),
            ],
            [
                np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad),
            ],
        ]
    )

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = -rot @ ref_ecef
    return T


def get_t_rig_enu_from_ecef(t_rig_ecef: np.ndarray, xodr_xml: str) -> np.ndarray:
    """Compute rig-to-ENU transform using OpenDRIVE georeference.

    Extracts the georeference information from the XODR XML and uses it
    to create a transformation from the rig's ECEF coordinates to the
    local ENU coordinate system.

    Args:
        t_rig_ecef: 4x4 transformation matrix for rig in ECEF coordinates
        xodr_xml: OpenDRIVE XML content as string

    Returns:
        4x4 transformation matrix from rig to ENU coordinates.
        Returns identity matrix if no valid georeference is found.
    """
    root = ET.fromstring(xodr_xml)
    geos = root.findall(".//geoReference")
    if not geos or geos[0].text is None:
        return np.eye(4)  # identity fallback

    proj = geos[0].text.strip()
    lat = lon = alt = None

    # Parse proj4 string for lat_0, lon_0, alt_0/h_0
    # Handle malformed tokens like "+=alt_0=0" found in some files
    for part in proj.split():
        part = part.strip()
        if not part or "=" not in part:
            continue

        try:
            if part.startswith("+lat_0"):
                lat = float(part.split("=", 1)[1])
            elif part.startswith("+lon_0"):
                lon = float(part.split("=", 1)[1])
            elif part.startswith("+alt_0") or part.startswith("+h_0"):
                alt = float(part.split("=", 1)[1])
        except (ValueError, IndexError):
            # Skip malformed tokens
            continue

    if lat is None or lon is None:
        return np.eye(4)
    if alt is None:
        alt = 0.0

    # Basic validation
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return np.eye(4)

    ref = np.array([lat, lon, alt])
    T_ecef_enu = ecef_2_enu_matrix(ref)
    return T_ecef_enu @ t_rig_ecef


def apply_transform(xyz: np.ndarray, transform_mat: Optional[np.ndarray]) -> np.ndarray:
    """Apply an optional 4x4 homogeneous transform to xyz coordinates.

    Args:
        xyz: Points to transform, either a single point (3,) or
             array of points (N, 3)
        transform_mat: Optional 4x4 SE(3) transformation matrix.
                      If None, returns xyz unchanged.

    Returns:
        Transformed coordinates in the same shape as input
    """
    if transform_mat is None:
        return xyz

    # Handle single point
    if xyz.ndim == 1:
        xyz1 = np.concatenate([xyz, np.ones(1)])
        return (transform_mat @ xyz1)[:3]

    # Handle array of points
    xyz1 = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    return ((transform_mat @ xyz1.T).T)[:, :3]
