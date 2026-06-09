# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Data structures for XODR parsing.

This module contains the core data structures used throughout the XODR
parsing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Set, Tuple

import numpy as np


class MapElementType(IntEnum):
    """Types of map elements that can be parsed from XODR."""

    ROAD_LANE = 1
    PED_CROSSWALK = 3
    PED_WALKWAY = 4
    ROAD_EDGE = 5
    TRAFFIC_SIGN = 6
    WAIT_LINE = 7


@dataclass
class LaneGeom:
    """Geometry and metadata for a single lane along an OpenDRIVE road."""

    lane_id_xml: int  # id attribute from XML (negative for right side)
    unique_id: str  # roadId_laneId string, globally unique
    center: np.ndarray  # (N, 3) xyz
    left_edge: Optional[np.ndarray]  # (N, 3) xyz when present
    right_edge: Optional[np.ndarray]  # (N, 3) xyz when present
    headings: np.ndarray  # (N,) radians
    road_id: str
    lane_type: str  # Original lane type from XODR (driving, parking, sidewalk, etc.)
    is_driving: bool
    is_left: bool  # True if lane_id > 0 per OpenDRIVE convention
    direction: str = "standard"  # Lane direction: "standard", "reversed", or "both"

    # Connectivity placeholders (filled later by parse_xodr)
    next_lanes: Set[str] = field(default_factory=set)
    prev_lanes: Set[str] = field(default_factory=set)
    # Geometric adjacency (physically adjacent)
    adj_left: Set[str] = field(default_factory=set)
    adj_right: Set[str] = field(default_factory=set)

    # Legal adjacency from XODR (where lane changes are allowed)
    # Lanes we can legally change to on the left/right
    can_change_left: Set[str] = field(default_factory=set)
    can_change_right: Set[str] = field(default_factory=set)

    # Traffic infrastructure (filled later)
    traffic_sign_ids: Set[str] = field(default_factory=set)
    wait_line_ids: Set[str] = field(default_factory=set)

    # Neighbor lanes from XODR (forward/backward references)
    left_neighbor_forward: Set[str] = field(default_factory=set)  # From lane.link.left
    right_neighbor_forward: Set[str] = field(
        default_factory=set
    )  # From lane.link.right
    left_neighbor_backward: Set[str] = field(default_factory=set)
    right_neighbor_backward: Set[str] = field(default_factory=set)


@dataclass
class ParsedXodr:
    """Return object of parse_xodr holding lanes & auxiliary map data."""

    lanes: Dict[str, LaneGeom]
    extent: np.ndarray  # (6,) [min_x, min_y, min_z, max_x, max_y, max_z]
    road_edges: Dict[str, np.ndarray]  # Values: (N, 3) xyz polylines
    traffic_signs: Dict[
        str, Tuple[np.ndarray, str]
    ]  # Values: ((3,) xyz position, type_str)
    wait_lines: Dict[
        str, Tuple[np.ndarray, str]
    ]  # Values: ((N, 3) xyz polylines, wait_line_type)
    sidewalks: Dict[str, np.ndarray]  # Values: (N, 3) xyz polylines
    crosswalks: Dict[str, np.ndarray]  # Values: (N, 3) xyz polylines
