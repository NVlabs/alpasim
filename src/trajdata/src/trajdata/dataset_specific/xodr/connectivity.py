# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Lane connectivity and junction handling for XODR parsing.

This module handles road-to-road connectivity, lane adjacency,
and junction connections.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List

from .datatypes import LaneGeom


def build_lane_adjacency(road_to_lanes: Dict[str, List[LaneGeom]]) -> None:
    """Build physical adjacency relationships between lanes within each road.

    Sets the adj_left and adj_right attributes for each lane based on
    lane ID ordering within the same road.

    Args:
        road_to_lanes: Dictionary mapping road IDs to their lanes.
                      Modified in place to add adjacency info.
    """
    for lanes in road_to_lanes.values():
        # Sort lanes by ID for adjacency determination
        sorted_lanes = sorted(lanes, key=lambda lane: lane.lane_id_xml)

        # Create a map from lane ID to lane object for quick lookup
        id_to_lane = {lane.lane_id_xml: lane for lane in sorted_lanes}

        for lane in sorted_lanes:
            lane_id = lane.lane_id_xml

            # Check for left adjacent lane
            # Left = geometrically to the left = higher ID (for both positive and negative)
            # But never cross zero (lane 0 is reference line, not a real lane)
            left_id = lane_id + 1
            if left_id != 0 and left_id in id_to_lane:
                adj_lane = id_to_lane[left_id]
                if adj_lane.is_driving:  # Only connect to driveable lanes
                    lane.adj_left.add(adj_lane.unique_id)

            # Check for right adjacent lane
            # Right = geometrically to the right = lower ID (for both positive and negative)
            # But never cross zero (lane 0 is reference line, not a real lane)
            right_id = lane_id - 1
            if right_id != 0 and right_id in id_to_lane:
                adj_lane = id_to_lane[right_id]
                if adj_lane.is_driving:  # Only connect to driveable lanes
                    lane.adj_right.add(adj_lane.unique_id)

        # Populate legal lane change permissions from XODR neighbor references
        # IMPORTANT: Only allow lane changes that are BOTH legal AND physically possible
        for lane in lanes:
            # For most lanes, only forward neighbors are relevant
            # Intersect with geometric adjacency to ensure physical reachability
            lane.can_change_left.update(lane.left_neighbor_forward & lane.adj_left)
            lane.can_change_right.update(lane.right_neighbor_forward & lane.adj_right)

            # For lanes with direction="both", backward neighbors are also important
            # (vehicles can legally travel in either direction)
            if lane.direction == "both":
                # When traveling backward, left becomes right and vice versa
                lane.can_change_right.update(
                    lane.left_neighbor_backward & lane.adj_right
                )
                lane.can_change_left.update(
                    lane.right_neighbor_backward & lane.adj_left
                )


def extract_road_connections(root: ET.Element) -> Dict[str, Dict]:
    """Extract road-to-road connectivity from XODR link elements.

    Args:
        root: Root XML element of the XODR document

    Returns:
        Dictionary mapping road IDs to their connections:
        {road_id: {"successor": {"road_id": str, "contactPoint": str},
                   "predecessor": {"road_id": str, "contactPoint": str}}}
    """
    road_connections = {}

    for road in root.findall("road"):
        road_id = road.attrib["id"]
        link_elem = road.find("link")

        if link_elem is not None:
            connections = {}

            # Extract successor
            successor = link_elem.find("successor")
            if successor is not None and successor.attrib.get("elementType") == "road":
                connections["successor"] = {
                    "road_id": successor.attrib["elementId"],
                    "contactPoint": successor.attrib.get("contactPoint", "start"),
                }

            # Extract predecessor
            predecessor = link_elem.find("predecessor")
            if (
                predecessor is not None
                and predecessor.attrib.get("elementType") == "road"
            ):
                connections["predecessor"] = {
                    "road_id": predecessor.attrib["elementId"],
                    "contactPoint": predecessor.attrib.get("contactPoint", "end"),
                }

            if connections:
                road_connections[road_id] = connections

    return road_connections


def _add_lane_link_connection(
    link_elem: ET.Element,
    link_type: str,
    connected_road_id: str,
    current_lane_id: str,
    current_lane_geom: LaneGeom,
    lane_geoms: Dict[str, LaneGeom],
    to_next: bool,
) -> None:
    """Parse and add a single lane link connection (successor or predecessor).

    Handles directional logic for positive lane IDs (left lanes) which flow
    opposite to the road's reference direction.

    Args:
        link_elem: The <link> XML element containing successor/predecessor
        link_type: "successor" or "predecessor"
        connected_road_id: ID of the road containing the connected lane
        current_lane_id: Unique ID (roadId_laneId) of current lane
        current_lane_geom: LaneGeom object of current lane to update
        lane_geoms: Dictionary of all lane geometries
        to_next: True if connection should be added to next_lanes,
                False for prev_lanes

    """
    elem = link_elem.find(link_type)
    if elem is not None:
        connected_lane_id = elem.attrib.get("id")
        if connected_lane_id and connected_road_id is not None:
            connected_unique_id = f"{connected_road_id}_{connected_lane_id}"
            if connected_unique_id in lane_geoms:
                if to_next:
                    current_lane_geom.next_lanes.add(connected_unique_id)
                    lane_geoms[connected_unique_id].prev_lanes.add(current_lane_id)
                else:
                    current_lane_geom.prev_lanes.add(connected_unique_id)
                    lane_geoms[connected_unique_id].next_lanes.add(current_lane_id)


def parse_lane_successors_predecessors(
    root: ET.Element, lane_geoms: Dict[str, LaneGeom], road_connections: Dict[str, Dict]
) -> None:
    """Parse lane-level successor/predecessor connections from XODR.

    This handles the lane.link.successor and lane.link.predecessor elements
    which specify direct lane-to-lane connections across road boundaries.

    Args:
        root: Root XML element
        lane_geoms: Dictionary of lane geometries to update
        road_connections: Road-to-road connectivity info
    """
    for road in root.findall("road"):
        road_id = road.attrib["id"]
        lanes_elem = road.find("lanes")
        if lanes_elem is None:
            continue

        # Get the successor/predecessor road IDs for this road
        successor_road_id = None
        predecessor_road_id = None

        if road_id in road_connections:
            if "successor" in road_connections[road_id]:
                successor_road_id = road_connections[road_id]["successor"]["road_id"]
            if "predecessor" in road_connections[road_id]:
                predecessor_road_id = road_connections[road_id]["predecessor"][
                    "road_id"
                ]

        for lane_section in lanes_elem.findall("laneSection"):
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

                    # Parse lane links
                    link_elem = lane.find("link")
                    if link_elem is None:
                        continue

                    # Lane direction depends on lane ID sign:
                    # - Negative lanes (right): follow road direction
                    # - Positive lanes (left): opposite to road direction
                    # For positive lanes, we need to swap predecessor/successor meaning
                    should_invert = lane_id > 0

                    # Parse successor
                    # For positive lanes (should_invert=True): successor connects to prev
                    # For negative lanes (should_invert=False): successor connects to next
                    _add_lane_link_connection(
                        link_elem,
                        "successor",
                        successor_road_id,
                        unique_id,
                        lg,
                        lane_geoms,
                        to_next=not should_invert,
                    )

                    # Parse predecessor
                    # For positive lanes (should_invert=True): predecessor connects to next
                    # For negative lanes (should_invert=False): predecessor connects to prev
                    _add_lane_link_connection(
                        link_elem,
                        "predecessor",
                        predecessor_road_id,
                        unique_id,
                        lg,
                        lane_geoms,
                        to_next=should_invert,
                    )


def process_junction_connections(
    root: ET.Element,
    lane_geoms: Dict[str, LaneGeom],
) -> None:
    """Process explicit junction connections from XODR.

    Parses <junction><connection><laneLink> elements to establish
    lane-to-lane connectivity within junctions.

    Args:
        root: Root XML element
        lane_geoms: Dictionary of lane geometries to update
    """
    for junction in root.findall("junction"):
        for connection in junction.findall("connection"):
            incoming_road = connection.attrib.get("incomingRoad")
            connecting_road = connection.attrib.get("connectingRoad")
            contact_point = connection.attrib.get("contactPoint", "start")

            if not incoming_road or not connecting_road:
                continue

            # Process lane links
            for lane_link in connection.findall("laneLink"):
                from_lane_id = lane_link.attrib.get("from")
                to_lane_id = lane_link.attrib.get("to")

                if not from_lane_id or not to_lane_id:
                    continue

                # Construct unique lane IDs
                from_unique = f"{incoming_road}_{from_lane_id}"
                to_unique = f"{connecting_road}_{to_lane_id}"

                if from_unique not in lane_geoms or to_unique not in lane_geoms:
                    continue

                # Establish prev/next according to OpenDRIVE spec:
                # incomingRoad -> connectingRoad -> (other outgoing road)
                if contact_point == "start":
                    # connectingRoad starts at incoming road, so incoming -> connecting
                    lane_geoms[from_unique].next_lanes.add(to_unique)
                    lane_geoms[to_unique].prev_lanes.add(from_unique)
                else:  # 'end'
                    # connectingRoad ends at incoming road, so connecting -> incoming
                    lane_geoms[to_unique].next_lanes.add(from_unique)
                    lane_geoms[from_unique].prev_lanes.add(to_unique)
