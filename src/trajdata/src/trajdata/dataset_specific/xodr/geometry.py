# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Geometry computation functions for XODR parsing.

This module contains functions for sampling road centerlines, computing lane
widths, generating polylines from geometric descriptions, and calculating headings.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np


def sample_centerline(
    road: ET.Element, resolution: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the road centerline including elevation profile.

    This function processes both the planView (2D geometry) and elevationProfile
    (vertical alignment) to generate a complete 3D road centerline.

    Args:
        road: XML element representing an OpenDRIVE road element
        resolution: Sampling resolution in meters (distance between sample points)

    Returns:
        Tuple containing:
        - x_coords: Array of x-coordinates in the road's coordinate system
        - y_coords: Array of y-coordinates in the road's coordinate system
        - z_coords: Array of z-coordinates (elevations) from elevation profile
        - headings: Array of heading angles in radians at each sample point

        All arrays have the same length and empty arrays if no geometry is found.
    """
    plan_view = road.find("planView")
    if plan_view is None:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

    # Parse elevation profile
    elevation_profile = road.find("elevationProfile")
    elevation_sections = []
    if elevation_profile is not None:
        for elev in elevation_profile.findall("elevation"):
            elevation_sections.append(
                {
                    "s": float(elev.attrib["s"]),
                    "a": float(elev.attrib["a"]),
                    "b": float(elev.attrib["b"]),
                    "c": float(elev.attrib["c"]),
                    "d": float(elev.attrib["d"]),
                }
            )
    elevation_sections.sort(key=lambda x: x["s"])

    center_xs: List[float] = []
    center_ys: List[float] = []
    center_zs: List[float] = []
    headings: List[float] = []
    cumulative_s = 0.0

    for geom in plan_view.findall("geometry"):
        geom_len = float(geom.attrib["length"])
        geom_x = float(geom.attrib["x"])
        geom_y = float(geom.attrib["y"])
        geom_hdg = float(geom.attrib["hdg"])

        # Sample points along *s* at desired resolution.
        num_samples = max(2, int(math.ceil(geom_len / resolution)))
        s_values = np.linspace(0, geom_len, num_samples)

        if geom.find("line") is not None:
            # Straight segment
            dx = np.cos(geom_hdg) * s_values
            dy = np.sin(geom_hdg) * s_values
            xs = geom_x + dx
            ys = geom_y + dy
            hdgs = np.full_like(xs, geom_hdg)
        else:
            # Arc segment
            arc = geom.find("arc")
            if arc is None:
                continue  # Unsupported geometry type.
            curvature = float(arc.attrib["curvature"])

            if abs(curvature) < 1e-6:  # Nearly straight
                dx = np.cos(geom_hdg) * s_values
                dy = np.sin(geom_hdg) * s_values
                xs = geom_x + dx
                ys = geom_y + dy
                hdgs = np.full_like(xs, geom_hdg)
            else:
                radius = 1.0 / curvature
                center_dir = geom_hdg + (math.pi / 2 if curvature > 0 else -math.pi / 2)
                cx = geom_x + math.cos(center_dir) * abs(radius)
                cy = geom_y + math.sin(center_dir) * abs(radius)

                angle_start = geom_hdg - math.pi / 2 + (math.pi if curvature < 0 else 0)
                angles = angle_start + s_values * curvature

                xs = cx + np.cos(angles) * abs(radius)
                ys = cy + np.sin(angles) * abs(radius)

                hdgs = geom_hdg + s_values * curvature

        # Compute elevation for each point
        zs = np.zeros_like(xs)
        for i, s_local in enumerate(s_values):
            s_global = cumulative_s + s_local
            z = _compute_elevation_at_s(s_global, elevation_sections)
            zs[i] = z

        center_xs.extend(xs.tolist())
        center_ys.extend(ys.tolist())
        center_zs.extend(zs.tolist())
        headings.extend(hdgs.tolist())

        cumulative_s += geom_len

    return (
        np.asarray(center_xs),
        np.asarray(center_ys),
        np.asarray(center_zs),
        np.asarray(headings),
    )


def _compute_elevation_at_s(
    s: float, elevation_sections: List[Dict[str, float]]
) -> float:
    """Compute elevation at a given s-coordinate using polynomial sections.

    OpenDRIVE defines elevation as a cubic polynomial:
        elevation(ds) = a + b*ds + c*ds² + d*ds³
    where ds is the distance from the start of the elevation section.

    Args:
        s: The s-coordinate along the road reference line (in meters)
        elevation_sections: List of elevation polynomial sections, each containing:
            - 's': Start position of the section
            - 'a', 'b', 'c', 'd': Polynomial coefficients

    Returns:
        The elevation (z-coordinate) at the given s position in meters.
        Returns 0.0 if no elevation sections are defined.
    """
    if not elevation_sections:
        return 0.0

    # Find the applicable elevation section
    section = None
    for elev_section in reversed(elevation_sections):
        if s >= elev_section["s"]:
            section = elev_section
            break

    if section is None:
        section = elevation_sections[0]

    # Compute elevation using cubic polynomial
    ds = s - section["s"]
    elevation = (
        section["a"]
        + section["b"] * ds
        + section["c"] * ds * ds
        + section["d"] * ds * ds * ds
    )

    return elevation


def compute_polyline_from_width(
    center_x: np.ndarray,
    center_y: np.ndarray,
    widths: np.ndarray,
    sign: int,
    road_headings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return edge polyline given centerline + signed widths using road headings.

    Uses OpenDRIVE lateral offset formula:
    x' = x - t*sin(hdg)
    y' = y + t*cos(hdg)

    Args:
        center_x: X coordinates of centerline
        center_y: Y coordinates of centerline
        widths: Width values at each point
        sign: 1 for left side, -1 for right side
        road_headings: Road heading angles at each point

    Returns:
        Tuple of (edge_x, edge_y) coordinates
    """
    edge_x = center_x - sign * widths * np.sin(road_headings)
    edge_y = center_y + sign * widths * np.cos(road_headings)
    return edge_x, edge_y


def gather_lane_offsets_all_sections(
    lanes_elem: ET.Element,
) -> Tuple[
    List[Tuple[int, List[Tuple[float, float, float, float, float]]]],
    Dict[int, str],
    Dict[int, str],
]:
    """Aggregate <width> polynomials for every lane across all laneSections.

    The sOffset attribute of <width> is relative to its laneSection's starting s,
    so we convert it to a global s along the road (0 at road start) before collecting.

    Args:
        lanes_elem: XML element containing lane sections

    Returns:
        Tuple of:
        - List of (lane_id, width_sections) where width_sections contains
          (s, a, b, c, d) polynomial coefficients
        - Dictionary mapping lane_id to lane type string
        - Dictionary mapping lane_id to lane direction string
    """
    per_lane_sections: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    lane_types: Dict[int, str] = {}
    lane_directions: Dict[int, str] = {}

    for lane_section in lanes_elem.findall("laneSection"):
        section_s0 = float(lane_section.attrib.get("s", "0"))

        for side in ("left", "right"):
            side_elem = lane_section.find(side)
            if side_elem is None:
                continue
            for lane in side_elem.findall("lane"):
                lid = int(lane.attrib["id"])
                lane_types.setdefault(lid, lane.attrib.get("type", "none").lower())
                lane_directions.setdefault(
                    lid, lane.attrib.get("direction", "standard")
                )

                for w in lane.findall("width"):
                    s_rel = float(w.attrib["sOffset"])
                    global_s = section_s0 + s_rel
                    a = float(w.attrib["a"])
                    b = float(w.attrib["b"])
                    c = float(w.attrib["c"])
                    d = float(w.attrib["d"])
                    per_lane_sections.setdefault(lid, []).append((global_s, a, b, c, d))

    lane_offsets: List[Tuple[int, List[Tuple[float, float, float, float, float]]]] = []
    for lid, sections in per_lane_sections.items():
        sections.sort(key=lambda x: x[0])
        lane_offsets.append((lid, sections))

    return lane_offsets, lane_types, lane_directions


def recompute_headings(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute heading angles from x,y coordinate sequences.

    Calculates the heading angle at each point based on the direction
    to the next point. The last point uses the same heading as the
    second-to-last point.

    Args:
        xs: X coordinates of points
        ys: Y coordinates of points

    Returns:
        Array of heading angles in radians, normalized to [-π, π]
    """
    headings = np.zeros_like(xs)
    if xs.shape[0] <= 1:
        return headings

    vec = np.stack([np.diff(xs), np.diff(ys)], axis=1)
    headings[:-1] = np.arctan2(vec[:, 1], vec[:, 0])
    # Normalize to [-π, π]
    headings[:-1] = (headings[:-1] + math.pi) % (2 * math.pi) - math.pi
    headings[-1] = headings[-2]

    return headings


def gather_lane_offsets(
    lane_section: ET.Element,
) -> Tuple[
    List[Tuple[int, List[Tuple[float, float, float, float, float]]]],
    Dict[int, str],
    Dict[int, str],
]:
    """Return per-lane width polynomials and lane type mapping for a single lane section.

    Args:
        lane_section: XML element for a single lane section

    Returns:
        Tuple of:
        - List of (lane_id, width_sections) where width_sections contains
          (s, a, b, c, d) polynomial coefficients
        - Dictionary mapping lane_id to lane type string
        - Dictionary mapping lane_id to lane direction string
    """
    lane_offsets: List[Tuple[int, List[Tuple[float, float, float, float, float]]]] = []
    lane_types: Dict[int, str] = {}
    lane_directions: Dict[int, str] = {}

    for side in ("left", "right"):
        side_elem = lane_section.find(side)
        if side_elem is None:
            continue
        for lane in side_elem.findall("lane"):
            lid = int(lane.attrib["id"])
            lane_types[lid] = lane.attrib.get("type", "none").lower()
            lane_directions[lid] = lane.attrib.get("direction", "standard")
            width_sections = [
                (
                    float(w.attrib["sOffset"]),
                    float(w.attrib["a"]),
                    float(w.attrib["b"]),
                    float(w.attrib["c"]),
                    float(w.attrib["d"]),
                )
                for w in lane.findall("width")
            ]
            width_sections.sort(key=lambda x: x[0])
            lane_offsets.append((lid, width_sections))

    return lane_offsets, lane_types, lane_directions
