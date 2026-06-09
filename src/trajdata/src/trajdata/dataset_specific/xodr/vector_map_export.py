# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""VectorMap export functionality for XODR parsed data.

This module converts parsed OpenDRIVE data into trajdata VectorMap format.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np

from trajdata.maps import vec_map
from trajdata.maps.vec_map_elements import (
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadEdge,
    RoadLane,
    TrafficSign,
    WaitLine,
)

from .geo_transform import apply_transform
from .parser import parse_xodr

# Constants
# Maximum distance (meters) between interpolated polyline points
MAX_POLYLINE_POINT_DIST = 2.0


def populate_vector_map_from_xodr(
    vmap: vec_map.VectorMap,
    xodr_str: str,
    *,
    resolution: float = 0.5,
    t_xodr_enu_to_sim: Optional[np.ndarray] = None,
) -> None:
    """Populate a VectorMap with data parsed from an OpenDRIVE XML string.

    This function parses the XODR data and converts it into trajdata's
    VectorMap format, optionally applying a coordinate transformation.

    Args:
        vmap: Empty VectorMap instance to populate. map_id should be
              set by the caller.
        xodr_str: OpenDRIVE XML content as string
        resolution: Optional sampling resolution along road centerlines in meters.
                   Defaults to 0.5 meters.
        t_xodr_enu_to_sim: Optional 4x4 SE(3) matrix mapping XODR ENU coords
                            to sim space. If provided, every xyz point is
                            transformed before insertion.
    """
    # Parse the XODR data
    parsed = parse_xodr(xodr_str, resolution=resolution)

    # Create partial function for coordinate transformation
    transform_points = partial(apply_transform, transform_mat=t_xodr_enu_to_sim)

    # Convert lanes to VectorMap format
    _add_lanes_to_vmap(vmap, parsed, transform_points, MAX_POLYLINE_POINT_DIST)

    # Convert road edges to VectorMap format
    _add_road_edges_to_vmap(vmap, parsed, transform_points, MAX_POLYLINE_POINT_DIST)

    # Convert traffic signs to VectorMap format
    _add_traffic_signs_to_vmap(vmap, parsed, transform_points)

    # Convert wait lines to VectorMap format
    _add_wait_lines_to_vmap(vmap, parsed, transform_points)

    # Convert sidewalks to VectorMap format
    _add_sidewalks_to_vmap(vmap, parsed, transform_points, MAX_POLYLINE_POINT_DIST)

    # Convert crosswalks to VectorMap format
    _add_crosswalks_to_vmap(vmap, parsed, transform_points, MAX_POLYLINE_POINT_DIST)

    # Set the map extent, transforming if necessary
    vmap.extent = _transform_extent(parsed.extent, transform_points, t_xodr_enu_to_sim)


def _transform_extent(extent, transform_fn, t_xodr_enu_to_sim):
    """Transform a 3D bounding box extent if a transformation is provided.

    When coordinates are transformed (e.g., rotated), the axis-aligned bounding box
    changes. This function transforms all 8 corners of the original box and
    computes the new axis-aligned bounds.

    Args:
        extent: Original extent as [min_x, min_y, min_z, max_x, max_y, max_z]
        transform_fn: Function to transform coordinates
        t_xodr_enu_to_sim: Transformation matrix from XODR ENU to sim space
                            (if None, returns original extent)

    Returns:
        Transformed extent in same format as input
    """
    if t_xodr_enu_to_sim is None:
        return extent

    # Extract min and max corners
    min_corner = extent[:3]
    max_corner = extent[3:]

    # Create all 8 corners of the bounding box by taking all combinations
    # of min/max values for each dimension
    corners = np.array(
        [
            [x, y, z]
            for x in [min_corner[0], max_corner[0]]
            for y in [min_corner[1], max_corner[1]]
            for z in [min_corner[2], max_corner[2]]
        ]
    )

    # Transform all corners
    transformed_corners = transform_fn(corners)

    # Find new axis-aligned bounding box
    new_min = np.min(transformed_corners, axis=0)
    new_max = np.max(transformed_corners, axis=0)

    return np.concatenate([new_min, new_max])


def _add_lanes_to_vmap(vmap, parsed, transform_fn, max_dist):
    """Add lane geometries to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
        max_dist: Maximum distance between interpolated points
    """
    # Build a stable numeric ID mapping for compatibility with trajdata
    # We assign incremental integers starting from 0 in deterministic order
    # The IDs are stored as strings so that downstream code that casts to
    # int(float(id)) still succeeds, replicating the clipgt lane-id convention
    lane_id_map = {old_id: str(i) for i, old_id in enumerate(parsed.lanes.keys())}

    for lg in parsed.lanes.values():
        center_pl = Polyline(transform_fn(lg.center)).interpolate(max_dist=max_dist)
        left_pl = (
            Polyline(transform_fn(lg.left_edge)).interpolate(max_dist=max_dist)
            if lg.left_edge is not None
            else None
        )
        right_pl = (
            Polyline(transform_fn(lg.right_edge)).interpolate(max_dist=max_dist)
            if lg.right_edge is not None
            else None
        )

        # Remap connectivity sets to numeric ids
        # For VectorMap export, we use the legal adjacency (can_change_*) as this is
        # what's most relevant for autonomous driving path planning
        adj_left = {lane_id_map[x] for x in lg.can_change_left if x in lane_id_map}
        adj_right = {lane_id_map[x] for x in lg.can_change_right if x in lane_id_map}
        next_lanes = {lane_id_map[x] for x in lg.next_lanes if x in lane_id_map}
        prev_lanes = {lane_id_map[x] for x in lg.prev_lanes if x in lane_id_map}
        lane_elem = RoadLane(
            id=lane_id_map[lg.unique_id],
            center=center_pl,
            left_edge=left_pl,
            right_edge=right_pl,
            adj_lanes_left=adj_left,
            adj_lanes_right=adj_right,
            next_lanes=next_lanes,
            prev_lanes=prev_lanes,
            traffic_sign_ids=lg.traffic_sign_ids,
            wait_line_ids=lg.wait_line_ids,
        )
        vmap.add_map_element(lane_elem)


def _add_road_edges_to_vmap(vmap, parsed, transform_fn, max_dist):
    """Add road edge geometries to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
        max_dist: Maximum distance between interpolated points
    """
    for edge_id, xyz in parsed.road_edges.items():
        edge_elem = RoadEdge(
            id=edge_id,
            polyline=Polyline(transform_fn(xyz)).interpolate(max_dist=max_dist),
        )
        vmap.add_map_element(edge_elem)


def _add_traffic_signs_to_vmap(vmap, parsed, transform_fn):
    """Add traffic signs to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
    """
    for ts_id, (xyz, sign_type) in parsed.traffic_signs.items():
        # Transform single point (reshape to 2D for transform, then get first row)
        tf_xyz = transform_fn(xyz.reshape(1, -1))[0]
        ts_elem = TrafficSign(id=ts_id, position=tf_xyz, sign_type=sign_type)
        vmap.add_map_element(ts_elem)


def _add_wait_lines_to_vmap(vmap, parsed, transform_fn):
    """Add wait lines to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
    """
    for wl_id, (xyzs, wl_type) in parsed.wait_lines.items():
        wl_elem = WaitLine(
            id=wl_id,
            polyline=Polyline(transform_fn(xyzs)),
            wait_line_type=wl_type,  # "STOP" or "YIELD"
            is_implicit=False,
        )
        vmap.add_map_element(wl_elem)


def _add_sidewalks_to_vmap(vmap, parsed, transform_fn, max_dist):
    """Add sidewalk (PedWalkway) geometries to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
        max_dist: Maximum distance between interpolated points
    """
    for sidewalk_id, polygon_points in parsed.sidewalks.items():
        sidewalk_elem = PedWalkway(
            id=sidewalk_id,
            polygon=Polyline(transform_fn(polygon_points)).interpolate(
                max_dist=max_dist
            ),
        )
        vmap.add_map_element(sidewalk_elem)


def _add_crosswalks_to_vmap(vmap, parsed, transform_fn, max_dist):
    """Add crosswalk (PedCrosswalk) geometries to the VectorMap.

    Args:
        vmap: VectorMap to populate
        parsed: ParsedXodr object
        transform_fn: Function to transform coordinates
        max_dist: Maximum distance between interpolated points
    """
    for crosswalk_id, polygon_points in parsed.crosswalks.items():
        crosswalk_elem = PedCrosswalk(
            id=crosswalk_id,
            polygon=Polyline(transform_fn(polygon_points)).interpolate(
                max_dist=max_dist
            ),
        )
        vmap.add_map_element(crosswalk_elem)
