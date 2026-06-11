# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Lane processing functions for XODR parsing.

This module handles the core lane geometry processing, including width
calculations, edge computations, and artificial edge creation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np

from .datatypes import LaneGeom
from .geometry import (
    compute_polyline_from_width,
    gather_lane_offsets_all_sections,
    recompute_headings,
    sample_centerline,
)

# Driveable lane types from OpenDRIVE spec
# Reference: ASAM OpenDRIVE Specification - Section 11.7.1 Lane type
# https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/11_lanes/11_07_lane_properties.html#sec-79c983d6-db57-41ad-85f7-4643c25910dc
DRIVEABLE_LANE_TYPES = {
    "driving",
    "exit",
    "entry",
    "onRamp",
    "offRamp",
    "connectingRamp",
    "parking",
}


def process_road(
    road: ET.Element,
    resolution: float,
    lane_geoms: Dict[str, LaneGeom],
    road_edges: Dict[str, np.ndarray],
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    sidewalks: Dict[str, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Populate global containers with lane geometry for a single road.

    Args:
        road: XML element representing a road
        resolution: Sampling resolution in meters
        lane_geoms: Dictionary with lane geometries (updated in place)
        road_edges: Dictionary with road edge polylines (updated in place)
        min_xyz: Minimum extent (updated in place)
        max_xyz: Maximum extent (updated in place)

    Returns:
        Updated (min_xyz, max_xyz) extent arrays
    """
    road_id = road.attrib["id"]
    is_junction_road = road.attrib.get("junction", "-1") != "-1"

    # Sample the road centerline
    center_x, center_y, center_z, road_headings = sample_centerline(road, resolution)
    if center_x.size == 0:
        return min_xyz, max_xyz  # Nothing to do

    # Get lanes element
    lane_elem_lanes = road.find("lanes")
    if lane_elem_lanes is None:
        return min_xyz, max_xyz
    if lane_elem_lanes.find("laneSection") is None:
        return min_xyz, max_xyz

    # Collect lane width polynomials across all laneSection elements
    lane_offsets, lane_types, lane_directions = gather_lane_offsets_all_sections(
        lane_elem_lanes
    )

    # Build width samples along s grid
    s_grid = np.arange(center_x.shape[0]) * resolution

    # Apply lane offset to centerline
    center_x, center_y = _apply_lane_offset(
        lane_elem_lanes, s_grid, center_x, center_y, road_headings
    )

    # Precompute widths per lane
    lane_width_samples = _compute_lane_widths(lane_offsets, s_grid)

    # Process all lanes to create geometry and capture outer road edges
    _process_lane_geometry(
        road_id,
        lane_offsets,
        lane_types,
        lane_directions,
        lane_width_samples,
        center_x,
        center_y,
        center_z,
        road_headings,
        lane_geoms,
        road_edges,
        min_xyz,
        max_xyz,
        sidewalks,
    )

    # Create artificial edges for non-junction roads
    if not is_junction_road:
        _create_artificial_edges(
            road_id,
            lane_offsets,
            center_x,
            center_y,
            road_headings,
            road_edges,
        )

    return min_xyz, max_xyz


def _apply_lane_offset(
    lanes_elem: ET.Element,
    s_grid: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    road_headings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply <laneOffset> to shift the road reference line.

    Args:
        lanes_elem: XML element containing lane sections
        s_grid: Array of s-coordinates along the road
        center_x: X coordinates of centerline
        center_y: Y coordinates of centerline
        road_headings: Heading angles at each point

    Returns:
        Updated (center_x, center_y) coordinates
    """
    lane_offset_sections = []
    for offset_elem in lanes_elem.findall("laneOffset"):
        lane_offset_sections.append(
            (
                float(offset_elem.attrib["s"]),
                float(offset_elem.attrib["a"]),
                float(offset_elem.attrib["b"]),
                float(offset_elem.attrib["c"]),
                float(offset_elem.attrib["d"]),
            )
        )
    lane_offset_sections.sort(key=lambda x: x[0])

    if not lane_offset_sections:
        return center_x, center_y

    # Compute lateral offsets
    lateral_offsets = np.zeros_like(s_grid)
    section_starts = np.array([s[0] for s in lane_offset_sections])
    indices = np.searchsorted(section_starts, s_grid, side="right") - 1
    indices[indices < 0] = 0

    for i in range(len(s_grid)):
        s0, a, b, c, d = lane_offset_sections[indices[i]]
        ds = s_grid[i] - s0
        lateral_offsets[i] = a + b * ds + c * ds**2 + d * ds**3

    # Apply lateral offset to centerline
    if lateral_offsets.any():
        center_x = center_x - lateral_offsets * np.sin(road_headings)
        center_y = center_y + lateral_offsets * np.cos(road_headings)

    return center_x, center_y


def _compute_lane_widths(
    lane_offsets: List[Tuple[int, List]],
    s_grid: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute width samples for each lane along the road.

    Args:
        lane_offsets: List of (lane_id, width_sections) tuples
        s_grid: Array of s-coordinates along the road

    Returns:
        Dictionary mapping lane_id to width array
    """
    lane_width_samples: Dict[int, np.ndarray] = {}

    for lid, sections in lane_offsets:
        widths = np.zeros_like(s_grid)
        if not sections:
            lane_width_samples[lid] = widths
            continue

        # Use searchsorted for efficient lookup
        section_starts = np.array([s[0] for s in sections])
        indices = np.searchsorted(section_starts, s_grid, side="right") - 1
        indices[indices < 0] = 0

        for i in range(len(s_grid)):
            active_section_index = indices[i]
            s0, a, b, c, d = sections[active_section_index]
            ds = s_grid[i] - s0
            widths[i] = a + b * ds + c * ds**2 + d * ds**3

        lane_width_samples[lid] = widths

    return lane_width_samples


def _process_lane_geometry(
    road_id: str,
    lane_offsets: List[Tuple[int, List]],
    lane_types: Dict[int, str],
    lane_directions: Dict[int, str],
    lane_width_samples: Dict[int, np.ndarray],
    center_x: np.ndarray,
    center_y: np.ndarray,
    center_z: np.ndarray,
    road_headings: np.ndarray,
    lane_geoms: Dict[str, LaneGeom],
    road_edges: Dict[str, np.ndarray],
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    sidewalks: Dict[str, np.ndarray] = None,
) -> None:
    """Process lanes to create geometry using edge-chaining approach.

    Args:
        road_id: ID of the road being processed
        lane_offsets: List of (lane_id, width_sections) tuples
        lane_types: Dictionary mapping lane_id to lane type
        lane_directions: Dictionary mapping lane_id to lane direction
        lane_width_samples: Precomputed widths for each lane
        center_x: X coordinates of road centerline
        center_y: Y coordinates of road centerline
        road_headings: Heading angles at each point
        lane_geoms: Dictionary with lane geometries (updated in place)
        road_edges: Dictionary with road edge polylines (updated in place)
        min_xyz: Minimum extent (updated in place)
        max_xyz: Maximum extent (updated in place)
    """
    # Process left side (positive IDs) and right side (negative IDs) separately
    # Both are processed from inside out (centerline -> outer edge)
    left_lane_ids = sorted([lid for lid, _ in lane_offsets if lid > 0])  # [1, 2, 3...]
    right_lane_ids = sorted(
        [lid for lid, _ in lane_offsets if lid < 0], reverse=True
    )  # [-1, -2, -3...]

    for is_left_side, side_ids in [(True, left_lane_ids), (False, right_lane_ids)]:
        if is_left_side:
            # For left lanes, start from reversed centerline since they flow opposite
            current_edge_x = center_x[::-1]
            current_edge_y = center_y[::-1]
            current_center_z = center_z[::-1]
            current_road_headings = road_headings[::-1]
        else:
            # For right lanes, use normal centerline
            current_edge_x = center_x.copy()
            current_edge_y = center_y.copy()
            current_center_z = center_z.copy()
            current_road_headings = road_headings.copy()

        for lid in side_ids:
            if is_left_side:
                # For left lanes, reverse the width array to match reversed geometry
                widths = lane_width_samples[lid][::-1]
            else:
                widths = lane_width_samples[lid]

            lane_type = lane_types.get(lid, "none")
            lane_direction = lane_directions.get(lid, "standard")
            is_driving = lane_type in DRIVEABLE_LANE_TYPES
            sign = 1 if lid > 0 else -1

            # Create lane geometry
            lane_geom = _create_single_lane_geometry(
                road_id,
                lid,
                widths,
                sign,
                current_edge_x,
                current_edge_y,
                current_center_z,
                current_road_headings,
                lane_type,
                lane_direction,
                is_driving,
            )

            unique_id = f"{road_id}_{lid}"

            # Handle sidewalks separately as PedWalkway elements
            if lane_type == "sidewalk" and sidewalks is not None:
                # Create polygon from lane edges
                if lane_geom.left_edge is not None and lane_geom.right_edge is not None:
                    # Create closed polygon: left edge + reversed right edge
                    polygon = np.concatenate(
                        [lane_geom.left_edge, lane_geom.right_edge[::-1]], axis=0
                    )
                    sidewalks[unique_id] = polygon

                    # Still track spatial extent
                    min_xyz[:] = np.minimum(min_xyz, polygon.min(axis=0))
                    max_xyz[:] = np.maximum(max_xyz, polygon.max(axis=0))
            else:
                # Regular lane
                lane_geoms[unique_id] = lane_geom

                # Track spatial extent
                min_xyz[:] = np.minimum(min_xyz, lane_geom.center.min(axis=0))
                max_xyz[:] = np.maximum(max_xyz, lane_geom.center.max(axis=0))

            # Update edge for next iteration (move to outer edge)
            outer_edge_x, outer_edge_y = compute_polyline_from_width(
                current_edge_x, current_edge_y, widths, sign, current_road_headings
            )
            current_edge_x, current_edge_y = outer_edge_x, outer_edge_y

        # After all lanes on this side, current_edge is the outermost road edge
        if side_ids:
            if is_left_side:
                # Reverse back to road reference direction
                edge = np.stack(
                    [current_edge_x[::-1], current_edge_y[::-1], center_z], axis=1
                )
                road_edges[f"{road_id}_L"] = edge
            else:
                edge = np.stack([current_edge_x, current_edge_y, center_z], axis=1)
                road_edges[f"{road_id}_R"] = edge


def _create_single_lane_geometry(
    road_id: str,
    lane_id: int,
    widths: np.ndarray,
    sign: int,
    current_edge_x: np.ndarray,
    current_edge_y: np.ndarray,
    center_z: np.ndarray,
    road_headings: np.ndarray,
    lane_type: str,
    lane_direction: str,
    is_driving: bool,
) -> LaneGeom:
    """Create geometry for a single lane.

    Args:
        road_id: ID of the containing road
        lane_id: ID of the lane
        widths: Width values along the lane
        sign: 1 for left lanes, -1 for right lanes
        current_edge_x: X coordinates of the inner edge
        current_edge_y: Y coordinates of the inner edge
        center_z: Z coordinates along the lane
        road_headings: Heading angles along the road
        lane_type: Lane type from XODR
        lane_direction: Lane direction from XODR ("standard", "reversed", or "both")
        is_driving: Whether this is a driveable lane

    Returns:
        LaneGeom object with computed geometry
    """
    # Centerline is halfway between current edge and new outer edge
    mid_offset_from_edge = widths / 2.0
    mid_x, mid_y = compute_polyline_from_width(
        current_edge_x, current_edge_y, mid_offset_from_edge, sign, road_headings
    )

    # New outer edge is full width away from current edge
    outer_edge_x, outer_edge_y = compute_polyline_from_width(
        current_edge_x, current_edge_y, widths, sign, road_headings
    )

    # Build 3D coordinates using actual elevation data
    xyz_center = np.stack([mid_x, mid_y, center_z], axis=1)

    # Assign edges
    # For right lanes: current edge is inner (left), outer edge is outer (right)
    # For left lanes: with pre-reversed geometry, we use the same assignment
    xyz_left = np.stack([current_edge_x, current_edge_y, center_z], axis=1)
    xyz_right = np.stack([outer_edge_x, outer_edge_y, center_z], axis=1)

    # Compute lane headings from actual lane centerline
    lane_headings = recompute_headings(mid_x, mid_y)

    return LaneGeom(
        lane_id_xml=lane_id,
        unique_id=f"{road_id}_{lane_id}",
        center=xyz_center,
        left_edge=xyz_left,
        right_edge=xyz_right,
        headings=lane_headings,
        road_id=road_id,
        lane_type=lane_type,
        is_driving=is_driving,
        is_left=lane_id > 0,
        direction=lane_direction,
    )


def _create_artificial_edges(
    road_id: str,
    lane_offsets: List[Tuple[int, List]],
    center_x: np.ndarray,
    center_y: np.ndarray,
    road_headings: np.ndarray,
    road_edges: Dict[str, np.ndarray],
) -> None:
    """Create artificial edges for roads with lanes on only one side.

    Args:
        road_id: ID of the road
        lane_offsets: List of (lane_id, width_sections) tuples
        center_x: X coordinates of road centerline
        center_y: Y coordinates of road centerline
        road_headings: Heading angles along the road
        road_edges: Dictionary to update with artificial edges
    """
    left_lane_ids = [lid for lid, _ in lane_offsets if lid > 0]
    right_lane_ids = [lid for lid, _ in lane_offsets if lid < 0]

    has_left_edge = f"{road_id}_L" in road_edges
    has_right_edge = f"{road_id}_R" in road_edges
    has_any_lanes = bool(left_lane_ids or right_lane_ids)

    if not has_any_lanes:
        return

    # Minimal offset to ensure the edge is visible
    minimal_offset = 0.01  # 1cm offset

    # Create artificial left edge if missing
    if not has_left_edge:
        artificial_left_x, artificial_left_y = compute_polyline_from_width(
            center_x,
            center_y,
            np.full_like(center_x, minimal_offset),
            1,
            road_headings,
        )
        artificial_left_edge = np.stack(
            [
                artificial_left_x,
                artificial_left_y,
                np.zeros_like(artificial_left_x),
            ],
            axis=1,
        )
        road_edges[f"{road_id}_L"] = artificial_left_edge

    # Create artificial right edge if missing
    if not has_right_edge:
        artificial_right_x, artificial_right_y = compute_polyline_from_width(
            center_x,
            center_y,
            np.full_like(center_x, minimal_offset),
            -1,
            road_headings,
        )
        artificial_right_edge = np.stack(
            [
                artificial_right_x,
                artificial_right_y,
                np.zeros_like(artificial_right_x),
            ],
            axis=1,
        )
        road_edges[f"{road_id}_R"] = artificial_right_edge
