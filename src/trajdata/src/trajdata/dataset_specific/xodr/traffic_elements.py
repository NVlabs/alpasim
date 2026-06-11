# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Traffic elements parsing for XODR files.

This module handles parsing and association of traffic control elements
including signs, signals, and wait lines with their corresponding lanes.

Wait lines (stop lines, yield lines) are parsed from <object type="roadMark">
elements with explicit geometry. Only actual road surface markings are included.

Note: <signal name="stopline"> elements are NOT parsed as wait lines because
their zOffset values indicate they are elevated/buried infrastructure, not
road surface markings.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

import numpy as np

from .datatypes import LaneGeom
from .geometry import sample_centerline

# Constants
# Maximum distance (meters) to associate traffic sign with lane
TRAFFIC_SIGN_ASSOCIATION_THRESHOLD = 50.0
# Maximum distance (meters) to associate wait line with lane
WAIT_LINE_ASSOCIATION_THRESHOLD = 20.0


def parse_traffic_infrastructure(
    root: ET.Element,
    lane_geoms: Dict[str, LaneGeom],
    resolution: float,
) -> Tuple[Dict[str, Tuple[np.ndarray, str]], Dict[str, Tuple[np.ndarray, str]]]:
    """Parse traffic signs and wait lines from XODR.

    This function parses traffic infrastructure from OpenDRIVE files:
    1. Traffic signs from <signal> elements
    2. Wait lines from <object type="roadMark"> elements with subtype="signalLines"
       or name containing "stop"/"yield"

    Note: <signal name="stopline"> elements are NOT parsed as wait lines because
    their zOffset values indicate they are elevated/buried infrastructure, not
    road surface markings. Only roadMark objects represent actual road markings.

    Args:
        root: Root XML element of the XODR document
        lane_geoms: Dictionary of lane geometries for reference
        resolution: Sampling resolution in meters

    Returns:
        Tuple of:
        - Dictionary mapping sign IDs to (position, type) tuples
        - Dictionary mapping wait line IDs to (polyline, wait_line_type) tuples
          where wait_line_type is either "STOP" or "YIELD"
    """
    traffic_signs: Dict[str, Tuple[np.ndarray, str]] = {}
    wait_lines: Dict[str, Tuple[np.ndarray, str]] = {}

    for road in root.findall("road"):
        road_id = road.attrib["id"]

        # Get the road centerline (reference line)
        center_x, center_y, center_z, road_headings = sample_centerline(
            road, resolution
        )
        if center_x.size == 0:
            continue  # No geometry

        # Create a "virtual" lane_ref that represents the road centerline
        # This ensures object positions are calculated relative to the road reference line
        lane_ref = type(
            "obj",
            (object,),
            {
                "center": np.stack([center_x, center_y, center_z], axis=1),
                "headings": road_headings,
                "road_id": road_id,
            },
        )()

        # Parse roadMark objects that represent wait lines
        objects_elem = road.find("objects")
        if objects_elem is not None:
            for obj in objects_elem.findall("object"):
                if obj.attrib.get("type") == "roadMark":
                    # Check if this is a wait line (by subtype or name)
                    subtype = obj.attrib.get("subtype", "")
                    name = obj.attrib.get("name", "").lower()
                    if subtype == "signalLines" or "stop" in name or "yield" in name:
                        wl_data = _parse_roadmark_object(
                            obj, road_id, lane_ref, resolution
                        )
                        if wl_data:
                            wl_id, wl_points = wl_data
                            # Determine wait line type based on name
                            # Use all caps to match evaluation expectations
                            wl_type = "YIELD" if "yield" in name else "STOP"
                            wait_lines[wl_id] = (wl_points, wl_type)

        # Parse traffic signals
        signals_elem = road.find("signals")
        if signals_elem is not None:
            for sign in signals_elem.findall("signal"):
                # Parse all signals as traffic signs (including those named "stopline")
                sign_data = _parse_traffic_sign(
                    sign, road_id, lane_ref, resolution, len(traffic_signs)
                )
                if sign_data:
                    sign_id, position, sign_type = sign_data
                    traffic_signs[sign_id] = (position, sign_type)

    return traffic_signs, wait_lines


def _parse_traffic_sign(
    sign: ET.Element,
    road_id: str,
    lane_ref: LaneGeom,
    resolution: float,
    sign_count: int,
) -> Tuple[str, np.ndarray, str] | None:
    """Parse a single traffic sign element.

    Args:
        sign: XML element for the signal
        road_id: ID of the containing road
        lane_ref: Reference lane for coordinate conversion
        resolution: Sampling resolution
        sign_count: Current count of signs for ID generation

    Returns:
        Tuple of (sign_id, position, sign_type) or None if parsing fails
    """
    sign_id = sign.attrib.get("id", f"{road_id}_sig_{sign_count}")
    s_pos = float(sign.attrib["s"])
    t_offset = float(sign.attrib.get("t", "0"))

    # Compose a comprehensive type string
    sign_main = sign.attrib.get("type", "unknown")
    sign_sub = sign.attrib.get("subtype", "")
    sign_name = sign.attrib.get("name", "")

    if sign_sub:
        sign_type = f"{sign_main}:{sign_sub}"
    else:
        sign_type = sign_main

    if sign_name:
        sign_type = f"{sign_type}:{sign_name}"

    # Convert s-coordinate to xyz position
    center_xyz = lane_ref.center
    idx_float = s_pos / resolution
    idx_low = int(math.floor(idx_float))
    idx_hi = min(idx_low + 1, center_xyz.shape[0] - 1)
    alpha = idx_float - idx_low
    xyz = (1 - alpha) * center_xyz[idx_low] + alpha * center_xyz[idx_hi]

    # Apply lateral offset
    hdg = lane_ref.headings[idx_low]
    xyz[:2] += np.array([-t_offset * math.sin(hdg), t_offset * math.cos(hdg)])

    return sign_id, xyz, sign_type


def _parse_roadmark_object(
    obj: ET.Element,
    road_id: str,
    lane_ref: LaneGeom,
    resolution: float,
) -> Tuple[str, np.ndarray] | None:
    """Parse a roadMark object that represents a stop line.

    Args:
        obj: XML element for the roadMark object
        road_id: ID of the containing road
        lane_ref: Reference lane for coordinate conversion
        resolution: Sampling resolution

    Returns:
        Tuple of (wait_line_id, points_array) or None if parsing fails
    """
    obj_id = obj.attrib.get("id", f"{road_id}_wl_{obj.attrib.get('s', '0')}")
    s_pos = float(obj.attrib["s"])
    t_offset = float(obj.attrib.get("t", "0"))
    z_offset = float(obj.attrib.get("zOffset", "0"))
    hdg = float(obj.attrib.get("hdg", "0"))

    # Convert s-coordinate to xyz position for object center
    center_xyz = lane_ref.center
    idx_float = s_pos / resolution
    idx_low = int(math.floor(idx_float))
    idx_hi = min(idx_low + 1, center_xyz.shape[0] - 1)
    alpha = idx_float - idx_low
    base_xyz = (1 - alpha) * center_xyz[idx_low] + alpha * center_xyz[idx_hi]

    # Apply lateral offset
    road_hdg = lane_ref.headings[idx_low]
    base_xyz[:2] += np.array(
        [-t_offset * math.sin(road_hdg), t_offset * math.cos(road_hdg)]
    )
    base_xyz[2] += z_offset

    # Parse outline if present
    outline_elem = obj.find("outline")
    if outline_elem is not None:
        corners = []
        for corner in outline_elem.findall("cornerLocal"):
            u = float(corner.attrib["u"])
            v = float(corner.attrib["v"])
            z = float(corner.attrib.get("z", "0"))

            # Transform local coordinates to road coordinates
            # u is along the object's heading, v is perpendicular
            obj_hdg = road_hdg + hdg
            x = base_xyz[0] + u * math.cos(obj_hdg) - v * math.sin(obj_hdg)
            y = base_xyz[1] + u * math.sin(obj_hdg) + v * math.cos(obj_hdg)
            z = base_xyz[2] + z

            corners.append([x, y, z])

        if len(corners) >= 2:
            # For stop lines, we typically want the two endpoints
            # If we have a polygon, extract the line that's perpendicular to the road
            corners_array = np.array(corners)

            # Find the two points that form the line most perpendicular to the road
            # This is typically the first and last points of a rectangle
            if len(corners) == 4:
                # Rectangle outline - use the edge perpendicular to road direction
                edge1 = corners_array[1] - corners_array[0]
                edge2 = corners_array[2] - corners_array[1]

                # Check which edge is more perpendicular to road direction
                road_dir = np.array([math.cos(road_hdg), math.sin(road_hdg), 0])
                dot1 = abs(np.dot(edge1[:2] / np.linalg.norm(edge1[:2]), road_dir[:2]))
                dot2 = abs(np.dot(edge2[:2] / np.linalg.norm(edge2[:2]), road_dir[:2]))

                if dot1 < dot2:
                    # edge1 is more perpendicular
                    return obj_id, np.stack(
                        [corners_array[0], corners_array[1]], axis=0
                    )
                else:
                    # edge2 is more perpendicular
                    return obj_id, np.stack(
                        [corners_array[1], corners_array[2]], axis=0
                    )
            else:
                # Just use first and last points
                return obj_id, np.stack([corners_array[0], corners_array[-1]], axis=0)
    else:
        # No outline provided, fall back to using width/length attributes
        # Default width of 5.0 meters if not specified
        width = float(obj.attrib.get("width", 5.0))

        # Create perpendicular line
        left_pt = base_xyz.copy()
        right_pt = base_xyz.copy()
        obj_hdg = road_hdg + hdg
        perpendicular = np.array([-math.sin(obj_hdg), math.cos(obj_hdg)])

        left_pt[:2] += width / 2 * perpendicular
        right_pt[:2] -= width / 2 * perpendicular

        return obj_id, np.stack([left_pt, right_pt], axis=0)

    return None


def associate_infrastructure_with_lanes(
    lane_geoms: Dict[str, LaneGeom],
    traffic_signs: Dict[str, Tuple[np.ndarray, str]],
    wait_lines: Dict[str, Tuple[np.ndarray, str]],
) -> None:
    """Associate traffic signs and wait lines with their nearest lanes.

    Args:
        lane_geoms: Dictionary of lane geometries (modified in place)
        traffic_signs: Dictionary of traffic signs
        wait_lines: Dictionary of wait lines with type
    """
    # Associate traffic signs with nearby lanes
    for ts_id, (ts_pos, _) in traffic_signs.items():
        closest_lane_id = _find_closest_lane(
            ts_pos, lane_geoms, threshold=TRAFFIC_SIGN_ASSOCIATION_THRESHOLD
        )
        if closest_lane_id:
            lane_geoms[closest_lane_id].traffic_sign_ids.add(ts_id)

    # Associate wait lines with nearby lanes
    for wl_id, (wl_points, _) in wait_lines.items():
        wl_center = np.mean(wl_points, axis=0)
        closest_lane_id = _find_closest_lane(
            wl_center, lane_geoms, threshold=WAIT_LINE_ASSOCIATION_THRESHOLD
        )
        if closest_lane_id:
            lane_geoms[closest_lane_id].wait_line_ids.add(wl_id)


def _find_closest_lane(
    position: np.ndarray,
    lane_geoms: Dict[str, LaneGeom],
    threshold: float,
) -> str | None:
    """Find the closest lane to a given position.

    Args:
        position: 3D position to search from
        lane_geoms: Dictionary of lane geometries
        threshold: Maximum distance threshold

    Returns:
        ID of the closest lane within threshold, or None
    """
    closest_lane_id = None
    min_distance = float("inf")

    for lg_id, lg in lane_geoms.items():
        # Find closest point on lane centerline
        distances = np.linalg.norm(lg.center - position, axis=1)
        min_dist_to_lane = np.min(distances)
        if min_dist_to_lane < min_distance:
            min_distance = min_dist_to_lane
            closest_lane_id = lg_id

    if closest_lane_id and min_distance < threshold:
        return closest_lane_id
    return None
