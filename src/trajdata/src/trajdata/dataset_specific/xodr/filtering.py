# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Road edge filtering and cleanup for XODR parsing.

This module contains sophisticated filtering logic to remove invalid
or problematic road edges, particularly around junctions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set

import numpy as np
from shapely.geometry import LineString, MultiPoint
from shapely.strtree import STRtree

from .datatypes import LaneGeom

# Configuration constants
# These values were empirically determined through experimentation
# and can be adjusted based on specific dataset characteristics

# Threshold for considering edge crossing (meters)
# Smaller values = stricter crossing detection, may miss valid edges
# Larger values = more permissive, may keep invalid crossing edges
_CROSS_EPS = 0.20

# Angular threshold for parallel edge detection (~10°)
# Smaller angles = only very parallel edges considered duplicates
# Larger angles = more aggressive duplicate removal
_PARALLEL_ANG_THRESH = math.radians(11.0)

# Distance threshold for duplicate edges (meters)
# Based on typical lane widths and junction geometries
# Smaller values = only very close edges considered duplicates
# Larger values = may remove valid nearby edges
_CENTROID_DIST_THRESH = 2.0


def filter_road_edges(
    road_edges: Dict[str, np.ndarray],
    lane_geoms: Dict[str, LaneGeom],
    junction_road_ids: Set[str],
) -> Dict[str, np.ndarray]:
    """Apply comprehensive filtering to road edges.

    This function removes:
    1. Edges that cross through lanes
    2. Unnamed junction edges
    3. Nearly-parallel duplicate edges
    4. Junction edges inside the convex hull of lanes

    Args:
        road_edges: Dictionary of road edge polylines
        lane_geoms: Dictionary of lane geometries
        junction_road_ids: Set of road IDs that are junctions

    Returns:
        Filtered dictionary of road edges
    """
    # Build spatial index for fast crossing detection
    lane_tree = _build_lane_spatial_index(lane_geoms)

    # Apply crossing filter
    filtered_edges = _filter_crossing_edges(road_edges, lane_tree, junction_road_ids)

    # Remove duplicate parallel edges
    filtered_edges = _remove_duplicate_edges(filtered_edges)

    # Remove interior junction edges using convex hull
    filtered_edges = _filter_interior_junction_edges(
        filtered_edges, lane_geoms, junction_road_ids
    )

    return filtered_edges


def _build_lane_spatial_index(lane_geoms: Dict[str, LaneGeom]) -> STRtree | None:
    """Build a spatial index of lane segments for fast queries.

    Down-samples lane centerlines to reduce memory usage while
    maintaining good junction representation.

    Args:
        lane_geoms: Dictionary of lane geometries

    Returns:
        STRtree spatial index or None if no valid lanes
    """
    lane_segments: List[LineString] = []

    for lg in lane_geoms.values():
        # Skip lanes with less than 2 points - can't form a line segment
        if lg.center.shape[0] < 2:
            continue

        # Down-sample to every 3rd point
        # This reduces memory usage and speeds up queries while still maintaining
        # sufficient geometric accuracy. With typical 0.5m sampling resolution,
        # this gives us segments every 1.5m which is fine for detecting road edge
        # crossings (junctions typically have curves with radius > 5m)
        coords2d = lg.center[::3, :2]

        # Create line segments between consecutive downsampled points
        for i in range(coords2d.shape[0] - 1):
            lane_segments.append(LineString([coords2d[i], coords2d[i + 1]]))

    return STRtree(lane_segments) if lane_segments else None


def _filter_crossing_edges(
    road_edges: Dict[str, np.ndarray],
    lane_tree: STRtree | None,
    junction_road_ids: Set[str],
) -> Dict[str, np.ndarray]:
    """Remove edges that cross through driveable lanes.

    Args:
        road_edges: Dictionary of road edge polylines
        lane_tree: Spatial index of lane segments
        junction_road_ids: Set of junction road IDs

    Returns:
        Filtered dictionary without crossing edges
    """
    if lane_tree is None:
        return road_edges

    filtered_edges: Dict[str, np.ndarray] = {}

    for edge_id, edge_xyz in road_edges.items():
        if edge_xyz is None or edge_xyz.shape[0] < 2:
            continue  # Skip degenerate edges

        road_id = edge_id.split("_")[0]
        edge_ls = LineString(edge_xyz[:, :2])

        # Query spatial index for potential intersections
        candidate_segs = lane_tree.query(edge_ls)

        keep_edge = True
        for seg in candidate_segs:
            # Handle different Shapely versions
            if isinstance(seg, (int, np.integer)):
                seg = lane_tree.geometries[seg]

            # Skip if identical bounds (parallel/overlapping)
            if seg.bounds == edge_ls.bounds:
                continue

            if edge_ls.crosses(seg):
                keep_edge = False
                break

            # Check for near-crossing (numeric precision issues)
            d = seg.distance(edge_ls)
            if d < _CROSS_EPS:
                keep_edge = False
                break

        if not keep_edge:
            continue  # Drop crossing edge

        # Additional junction-specific filtering
        if road_id in junction_road_ids:
            is_named_outer = ("_L" in edge_id) or ("_R" in edge_id)
            if not is_named_outer:
                continue  # Drop unnamed junction edges

        filtered_edges[edge_id] = edge_xyz

    return filtered_edges


def _remove_duplicate_edges(road_edges: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Remove nearly-parallel duplicate edges.

    When two edges are almost parallel and close together,
    keep the longer one and discard the shorter.

    Args:
        road_edges: Dictionary of road edge polylines

    Returns:
        Filtered dictionary without duplicates
    """
    edge_ids = list(road_edges.keys())
    to_remove: Set[str] = set()

    for i, eid_a in enumerate(edge_ids):
        if eid_a in to_remove:
            continue

        ls_a = LineString(road_edges[eid_a][:, :2])
        if ls_a.length < 0.5:
            continue  # Skip very short edges

        # Calculate bearing of edge A
        x0, y0 = ls_a.coords[0]
        x1, y1 = ls_a.coords[-1]
        ang_a = math.atan2(y1 - y0, x1 - x0)

        # Compare with remaining edges
        edge_idx = i + 1
        for eid_b in edge_ids[edge_idx:]:
            if eid_b in to_remove:
                continue

            ls_b = LineString(road_edges[eid_b][:, :2])
            if ls_b.length < 0.5:
                continue

            # Quick distance check
            if ls_a.centroid.distance(ls_b.centroid) > _CENTROID_DIST_THRESH:
                continue

            # Calculate bearing difference
            ang_b = math.atan2(
                ls_b.coords[-1][1] - ls_b.coords[0][1],
                ls_b.coords[-1][0] - ls_b.coords[0][0],
            )
            dang = abs((ang_a - ang_b + math.pi) % (2 * math.pi) - math.pi)

            # Check if edges are similar
            if (
                dang <= _PARALLEL_ANG_THRESH
                or ls_a.centroid.distance(ls_b.centroid) <= _CENTROID_DIST_THRESH * 0.5
            ):
                # Remove the shorter edge
                if ls_a.length >= ls_b.length:
                    to_remove.add(eid_b)
                else:
                    to_remove.add(eid_a)
                    # Continue checking - A might help identify other duplicates

    # Apply removals
    return {eid: xyz for eid, xyz in road_edges.items() if eid not in to_remove}


def _filter_interior_junction_edges(
    road_edges: Dict[str, np.ndarray],
    lane_geoms: Dict[str, LaneGeom],
    junction_road_ids: Set[str],
) -> Dict[str, np.ndarray]:
    """Remove junction edges inside the convex hull of lane centers.

    This catches residual interior connector edges that don't
    intersect lanes but aren't part of the outer boundary.

    Args:
        road_edges: Dictionary of road edge polylines
        lane_geoms: Dictionary of lane geometries
        junction_road_ids: Set of junction road IDs

    Returns:
        Filtered dictionary without interior junction edges
    """
    # Build convex hull of all lane center points
    lane_pts = []
    for lg in lane_geoms.values():
        # Sparse sample to reduce computation
        lane_pts.extend([tuple(pt[:2]) for pt in lg.center[::5]])

    if not lane_pts:
        return road_edges

    hull_poly = MultiPoint(lane_pts).convex_hull.buffer(0)
    if hull_poly.is_empty:
        return road_edges

    filtered_edges = {}
    for eid, xyz in road_edges.items():
        road_id = eid.split("_")[0]

        # Only filter junction edges
        if road_id not in junction_road_ids:
            filtered_edges[eid] = xyz
            continue

        # Preserve explicitly marked outer boundaries
        if "_L" in eid or "_R" in eid:
            filtered_edges[eid] = xyz
            continue

        # Check if edge is inside hull
        edge_ls = LineString(xyz[:, :2])
        if not hull_poly.contains(edge_ls):
            filtered_edges[eid] = xyz

    return filtered_edges


def get_junction_road_ids(root) -> Set[str]:
    """Extract IDs of roads that are junctions.

    Args:
        root: Root XML element of the XODR document

    Returns:
        Set of road IDs that have junction attribute != "-1"
    """
    junction_road_ids = set()
    for road in root.findall("road"):
        if road.get("junction", "-1") != "-1":
            junction_road_ids.add(road.get("id"))
    return junction_road_ids
