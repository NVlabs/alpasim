# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Main XODR parser that orchestrates all parsing components.

This module provides the main parse_xodr function that coordinates
all the sub-modules to parse OpenDRIVE files. The parsing process:
1. Process roads to create lane geometries and edges
2. Build connectivity between lanes (adjacency and road-to-road)
3. Parse and associate traffic infrastructure (signs, wait lines)
4. Filter road edges to remove invalid or problematic edges
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Tuple

import numpy as np

from .connectivity import (
    build_lane_adjacency,
    extract_road_connections,
    parse_lane_successors_predecessors,
    process_junction_connections,
)
from .datatypes import LaneGeom, ParsedXodr
from .filtering import filter_road_edges, get_junction_road_ids
from .geometry import sample_centerline
from .lane_processing import process_road
from .traffic_elements import (
    associate_infrastructure_with_lanes,
    parse_traffic_infrastructure,
)


def parse_xodr(xodr_str: str, resolution: float = 0.5) -> ParsedXodr:
    """Parse an OpenDRIVE XML string into lane geometries.

    This is the main entry point that coordinates all parsing steps:
    1. Process roads to create lane geometries
    2. Build connectivity between lanes
    3. Parse traffic infrastructure
    4. Filter road edges

    Args:
        xodr_str: OpenDRIVE XML content as string
        resolution: Sampling resolution along road centerlines in meters

    Returns:
        ParsedXodr object containing all parsed map data
    """
    root = ET.fromstring(xodr_str)
    lane_geoms: Dict[str, LaneGeom] = {}
    road_edges: Dict[str, np.ndarray] = {}
    sidewalks: Dict[str, np.ndarray] = {}
    crosswalks: Dict[str, np.ndarray] = {}

    # Global spatial extent
    min_xyz = np.full((3,), np.inf)
    max_xyz = np.full((3,), -np.inf)

    # First pass: Process roads to create lanes and edges
    junction_road_ids = get_junction_road_ids(root)

    for road in root.findall("road"):
        min_xyz, max_xyz = process_road(
            road, resolution, lane_geoms, road_edges, min_xyz, max_xyz, sidewalks
        )

        # Parse crosswalks from road objects
        min_xyz, max_xyz = _parse_crosswalks(
            road, crosswalks, min_xyz, max_xyz, resolution
        )

    # Second pass: Build connectivity
    # First parse neighbor lanes from XODR
    _parse_neighbor_lanes(root, lane_geoms)

    # Group lanes by road for adjacency
    road_to_lanes: Dict[str, list[LaneGeom]] = {}
    for lg in lane_geoms.values():
        road_to_lanes.setdefault(lg.road_id, []).append(lg)

    # Build lane adjacency within roads
    build_lane_adjacency(road_to_lanes)

    # Extract and apply road-to-road connectivity
    road_connections = extract_road_connections(root)

    # Parse lane-level successor/predecessor connections
    parse_lane_successors_predecessors(root, lane_geoms, road_connections)

    # Process explicit junction connections
    process_junction_connections(root, lane_geoms)

    # Third pass: Parse traffic infrastructure
    traffic_signs, wait_lines = parse_traffic_infrastructure(
        root, lane_geoms, resolution
    )

    # Associate infrastructure with lanes
    associate_infrastructure_with_lanes(lane_geoms, traffic_signs, wait_lines)

    # Fourth pass: Filter road edges
    road_edges = filter_road_edges(road_edges, lane_geoms, junction_road_ids)

    # Create final result
    extent = np.concatenate([min_xyz, max_xyz])
    return ParsedXodr(
        lanes=lane_geoms,
        extent=extent,
        road_edges=road_edges,
        traffic_signs=traffic_signs,
        wait_lines=wait_lines,
        sidewalks=sidewalks,
        crosswalks=crosswalks,
    )


def _parse_crosswalks(
    road: ET.Element,
    crosswalks: Dict[str, np.ndarray],
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse crosswalks from road objects.

    Args:
        road: Road XML element
        crosswalks: Dictionary to populate with crosswalk polygons
        min_xyz: Minimum extent (updated in place)
        max_xyz: Maximum extent (updated in place)
        resolution: Sampling resolution

    Returns:
        Updated (min_xyz, max_xyz) extent arrays
    """
    objects = road.find("objects")
    if objects is None:
        return min_xyz, max_xyz

    road_id = road.attrib["id"]

    # Sample road centerline for s/t to xyz conversion
    center_x, center_y, center_z, road_headings = sample_centerline(road, resolution)
    if center_x.size == 0:
        return min_xyz, max_xyz

    for obj in objects.findall("object"):
        obj_type = obj.attrib.get("type", "").lower()
        obj_name = obj.attrib.get("name", "").lower()

        # Handle both type="crosswalk" and type="roadMark" name="crosswalk"
        is_crosswalk = obj_type in ["crosswalk", "crossing"] or (
            obj_type == "roadmark" and obj_name in ["crosswalk", "crossing"]
        )

        if not is_crosswalk:
            continue

        obj_id = obj.attrib.get("id", f"{road_id}_xwalk_{len(crosswalks)}")

        # Parse outline points (s,t coordinates)
        outline = obj.find("outline")
        polygon_points = []

        if outline is not None:
            # Standard outline format with cornerLocal points
            for corner in outline.findall("cornerLocal"):
                s = float(corner.attrib.get("s", "0"))
                t = float(corner.attrib.get("t", "0"))
                height = float(corner.attrib.get("height", "0"))

                # Convert s,t to xyz
                # Find closest centerline point
                idx = int(np.clip(s / resolution, 0, len(center_x) - 1))
                hdg = road_headings[idx]

                # Apply lateral offset
                x = center_x[idx] - t * np.sin(hdg)
                y = center_y[idx] + t * np.cos(hdg)
                z = center_z[idx] + height  # Add road elevation

                polygon_points.append([x, y, z])
        else:
            # Fallback: create rectangle from s,t,width,length attributes
            s = float(obj.attrib.get("s", "0"))
            t = float(obj.attrib.get("t", "0"))
            width = float(obj.attrib.get("width", "3.0"))  # Default 3m width
            length = float(obj.attrib.get("length", "2.0"))  # Default 2m length
            z = float(obj.attrib.get("zOffset", "0"))

            # Get position and heading at s
            idx = int(np.clip(s / resolution, 0, len(center_x) - 1))
            base_x = center_x[idx]
            base_y = center_y[idx]
            base_z = center_z[idx]
            hdg = road_headings[idx]

            # Center position with lateral offset
            center_x_pos = base_x - t * np.sin(hdg)
            center_y_pos = base_y + t * np.cos(hdg)

            # Create rectangle corners (width along road, length across)
            # Half dimensions
            hw = width / 2.0
            hl = length / 2.0

            # Four corners in local coordinates, then transform
            corners = [
                [-hw, -hl],  # back-left
                [hw, -hl],  # front-left
                [hw, hl],  # front-right
                [-hw, hl],  # back-right
            ]

            for local_s, local_t in corners:
                # Rotate by heading and translate
                x = center_x_pos + local_s * np.cos(hdg) - local_t * np.sin(hdg)
                y = center_y_pos + local_s * np.sin(hdg) + local_t * np.cos(hdg)
                polygon_points.append([x, y, base_z + z])  # Add road elevation

        if len(polygon_points) >= 3:  # Valid polygon
            polygon = np.array(polygon_points)
            crosswalks[obj_id] = polygon

            # Update extent
            min_xyz[:] = np.minimum(min_xyz, polygon.min(axis=0))
            max_xyz[:] = np.maximum(max_xyz, polygon.max(axis=0))

    return min_xyz, max_xyz


def _parse_neighbor_lanes(root: ET.Element, lane_geoms: Dict[str, LaneGeom]) -> None:
    """Parse lane neighbor references from XODR lane links.

    Args:
        root: Root XML element
        lane_geoms: Dictionary of lane geometries to update
    """
    for road in root.findall("road"):
        road_id = road.attrib["id"]
        lanes_elem = road.find("lanes")
        if lanes_elem is None:
            continue

        for section_idx, lane_section in enumerate(lanes_elem.findall("laneSection")):
            for side in ["left", "right"]:
                side_elem = lane_section.find(side)
                if side_elem is None:
                    continue

                for lane in side_elem.findall("lane"):
                    lane_id = int(lane.attrib["id"])
                    unique_id = f"{road_id}_{lane_id}"

                    if unique_id not in lane_geoms:
                        continue

                    lg = lane_geoms[unique_id]

                    # Parse lane links for neighbor references
                    link_elem = lane.find("link")
                    if link_elem is None:
                        continue

                    # Left neighbor
                    left_elem = link_elem.find("left")
                    if left_elem is not None:
                        neighbor_id = left_elem.attrib.get("id")
                        if neighbor_id:
                            direction = left_elem.attrib.get("direction", "forward")
                            if direction == "forward":
                                lg.left_neighbor_forward.add(f"{road_id}_{neighbor_id}")
                            elif direction == "backward":
                                lg.left_neighbor_backward.add(
                                    f"{road_id}_{neighbor_id}"
                                )

                    # Right neighbor
                    right_elem = link_elem.find("right")
                    if right_elem is not None:
                        neighbor_id = right_elem.attrib.get("id")
                        if neighbor_id:
                            direction = right_elem.attrib.get("direction", "forward")
                            if direction == "forward":
                                lg.right_neighbor_forward.add(
                                    f"{road_id}_{neighbor_id}"
                                )
                            elif direction == "backward":
                                lg.right_neighbor_backward.add(
                                    f"{road_id}_{neighbor_id}"
                                )
