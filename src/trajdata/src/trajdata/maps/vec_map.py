# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajdata.maps.map_kdtree import (
        MapElementKDTree,
        LaneCenterKDTree,
        RoadEdgeKDTree,
    )
    from trajdata.maps.map_strtree import MapElementSTRTree

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil
from typing import (
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from shapely.geometry import Polygon
from tqdm import tqdm

import trajdata.proto.vectorized_map_pb2 as map_proto
from trajdata.maps.map_kdtree import LaneCenterKDTree, RoadAreaKDTree, RoadEdgeKDTree
from trajdata.maps.map_strtree import MapElementSTRTree
from trajdata.maps.traffic_light_status import TrafficLightStatus
from trajdata.maps.vec_map_elements import (
    MapElement,
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadEdge,
    RoadLane,
    TrafficSign,
    WaitLine,
)
from trajdata.utils import map_utils


@dataclass(repr=False)
class VectorMap:
    map_id: str
    extent: Optional[np.ndarray] = (
        None  # extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    )
    elements: DefaultDict[MapElementType, Dict[str, MapElement]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    search_kdtrees: Optional[Dict[MapElementType, MapElementKDTree]] = None
    search_rtrees: Optional[Dict[MapElementType, MapElementSTRTree]] = None
    traffic_light_status: Optional[Dict[Tuple[str, int], TrafficLightStatus]] = None
    online_metadict: Optional[Dict[Tuple[str, int], Dict]] = None

    def __post_init__(self) -> None:
        self.env_name, self.map_name = self.map_id.split(":")

        self.lanes: Optional[List[RoadLane]] = None
        if MapElementType.ROAD_LANE in self.elements:
            self.lanes = list(self.elements[MapElementType.ROAD_LANE].values())
        self.road_edges: Optional[List[RoadEdge]] = None
        if MapElementType.ROAD_EDGE in self.elements:
            self.road_edges = list(self.elements[MapElementType.ROAD_EDGE].values())

    def add_map_element(self, map_elem: MapElement) -> None:
        self.elements[map_elem.elem_type][map_elem.id] = map_elem

    def compute_search_indices(self) -> None:
        # TODO(bivanovic@nvidia.com): merge tree dicts?
        self.search_kdtrees = {
            MapElementType.ROAD_LANE: LaneCenterKDTree(self),
        }
        if MapElementType.ROAD_EDGE in self.elements:
            self.search_kdtrees[MapElementType.ROAD_EDGE] = RoadEdgeKDTree(self)
        if MapElementType.ROAD_AREA in self.elements:
            self.search_kdtrees[MapElementType.ROAD_AREA] = RoadAreaKDTree(self)
        self.search_rtrees = {
            elem_type: MapElementSTRTree(self, elem_type)
            for elem_type in [
                MapElementType.ROAD_AREA,
                MapElementType.PED_CROSSWALK,
                MapElementType.PED_WALKWAY,
            ]
        }

    def iter_elems(self) -> Iterator[MapElement]:
        for elems_dict in self.elements.values():
            for elem in elems_dict.values():
                yield elem

    def get_road_lane(self, lane_id: str) -> RoadLane:
        return self.elements[MapElementType.ROAD_LANE][lane_id]

    def get_road_edge(self, lane_id: str) -> RoadEdge:
        return self.elements[MapElementType.ROAD_EDGE][lane_id]

    def __len__(self) -> int:
        return sum(len(elems_dict) for elems_dict in self.elements.values())

    def _write_road_lanes(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        road_lane: RoadLane
        for elem_id, road_lane in self.elements[MapElementType.ROAD_LANE].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_lane: map_proto.RoadLane = new_element.road_lane
            map_utils.populate_lane_polylines(new_lane, road_lane, shifted_origin)

            new_lane.entry_lanes.extend(
                [lane_id.encode() for lane_id in road_lane.prev_lanes]
            )
            new_lane.exit_lanes.extend(
                [lane_id.encode() for lane_id in road_lane.next_lanes]
            )

            new_lane.adjacent_lanes_left.extend(
                [lane_id.encode() for lane_id in road_lane.adj_lanes_left]
            )
            new_lane.adjacent_lanes_right.extend(
                [lane_id.encode() for lane_id in road_lane.adj_lanes_right]
            )

    def _write_road_areas(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        road_area: RoadArea
        for elem_id, road_area in self.elements[MapElementType.ROAD_AREA].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_area: map_proto.RoadArea = new_element.road_area
            map_utils.populate_polygon(
                new_area.exterior_polygon,
                road_area.exterior_polygon.xyz,
                shifted_origin,
            )

            hole: Polyline
            for hole in road_area.interior_holes:
                new_hole: map_proto.Polyline = new_area.interior_holes.add()
                map_utils.populate_polygon(
                    new_hole,
                    hole.xyz,
                    shifted_origin,
                )

    def _write_road_edges(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        road_edge: RoadEdge
        for elem_id, road_edge in self.elements[MapElementType.ROAD_EDGE].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_edge: map_proto.RoadEdge = new_element.road_edge
            map_utils.populate_road_edge_polylines(new_edge, road_edge, shifted_origin)

    def _write_traffic_signs(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        traffic_sign: TrafficSign
        for elem_id, traffic_sign in self.elements[MapElementType.TRAFFIC_SIGN].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_traffic_sign: map_proto.TrafficSign = new_element.traffic_sign
            shifted_position: np.ndarray = traffic_sign.position - shifted_origin
            new_traffic_sign.position.x = shifted_position[0]
            new_traffic_sign.position.y = shifted_position[1]
            new_traffic_sign.position.z = shifted_position[2]
            new_traffic_sign.sign_type = traffic_sign.sign_type.encode()

    def _write_wait_lines(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        wait_line: WaitLine
        for elem_id, wait_line in self.elements[MapElementType.WAIT_LINE].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_wait_line: map_proto.WaitLine = new_element.wait_line
            map_utils.populate_polygon(
                new_wait_line.polyline, wait_line.polyline.xyz, shifted_origin
            )
            new_wait_line.wait_line_type = wait_line.wait_line_type.encode()
            new_wait_line.is_implicit = wait_line.is_implicit

    def _write_ped_crosswalks(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        ped_crosswalk: PedCrosswalk
        for elem_id, ped_crosswalk in self.elements[
            MapElementType.PED_CROSSWALK
        ].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_crosswalk: map_proto.PedCrosswalk = new_element.ped_crosswalk
            map_utils.populate_polygon(
                new_crosswalk.polygon,
                ped_crosswalk.polygon.xyz,
                shifted_origin,
            )

    def _write_ped_walkways(
        self, vectorized_map: map_proto.VectorizedMap, shifted_origin: np.ndarray
    ) -> None:
        ped_walkway: PedWalkway
        for elem_id, ped_walkway in self.elements[MapElementType.PED_WALKWAY].items():
            new_element: map_proto.MapElement = vectorized_map.elements.add()
            new_element.id = elem_id.encode()

            new_walkway: map_proto.PedWalkway = new_element.ped_walkway
            map_utils.populate_polygon(
                new_walkway.polygon,
                ped_walkway.polygon.xyz,
                shifted_origin,
            )

    def to_proto(self) -> map_proto.VectorizedMap:
        output_map = map_proto.VectorizedMap()
        output_map.name = self.map_id

        (
            output_map.min_pt.x,
            output_map.min_pt.y,
            output_map.min_pt.z,
            output_map.max_pt.x,
            output_map.max_pt.y,
            output_map.max_pt.z,
        ) = self.extent

        shifted_origin: np.ndarray = self.extent[:3]
        (
            output_map.shifted_origin.x,
            output_map.shifted_origin.y,
            output_map.shifted_origin.z,
        ) = shifted_origin

        # Populating the elements in the vectorized map protobuf.
        self._write_road_lanes(output_map, shifted_origin)
        self._write_road_areas(output_map, shifted_origin)
        self._write_ped_crosswalks(output_map, shifted_origin)
        self._write_ped_walkways(output_map, shifted_origin)
        self._write_road_edges(output_map, shifted_origin)
        self._write_traffic_signs(output_map, shifted_origin)
        self._write_wait_lines(output_map, shifted_origin)

        return output_map

    @classmethod
    def from_proto(cls, vec_map: map_proto.VectorizedMap, **kwargs):
        # Options for which map elements to include.
        incl_road_lanes: bool = kwargs.get("incl_road_lanes", True)
        incl_road_areas: bool = kwargs.get("incl_road_areas", False)
        incl_ped_crosswalks: bool = kwargs.get("incl_ped_crosswalks", False)
        incl_ped_walkways: bool = kwargs.get("incl_ped_walkways", False)
        incl_road_edges: bool = kwargs.get("incl_road_edges", False)
        incl_traffic_signs: bool = kwargs.get("incl_traffic_signs", False)
        incl_wait_lines: bool = kwargs.get("incl_wait_lines", False)

        # Add any map offset in case the map origin was shifted for storage efficiency.
        shifted_origin: np.ndarray = np.array(
            [
                vec_map.shifted_origin.x,
                vec_map.shifted_origin.y,
                vec_map.shifted_origin.z,
                0.0,  # Some polylines also have heading so we're adding
                # this (zero) coordinate to account for that.
            ]
        )

        map_elem_dict: Dict[str, Dict[str, MapElement]] = defaultdict(dict)

        map_elem: MapElement
        for map_elem in vec_map.elements:
            elem_id: str = map_elem.id.decode()
            if incl_road_lanes and map_elem.HasField("road_lane"):
                road_lane_obj: map_proto.RoadLane = map_elem.road_lane

                center_pl: Polyline = Polyline(
                    map_utils.proto_to_np(road_lane_obj.center) + shifted_origin
                )

                # We do not care for the heading of the left and right edges
                # (only the center matters).
                left_pl: Optional[Polyline] = None
                if road_lane_obj.HasField("left_boundary"):
                    left_pl = Polyline(
                        map_utils.proto_to_np(
                            road_lane_obj.left_boundary, incl_heading=False
                        )
                        + shifted_origin[:3]
                    )

                right_pl: Optional[Polyline] = None
                if road_lane_obj.HasField("right_boundary"):
                    right_pl = Polyline(
                        map_utils.proto_to_np(
                            road_lane_obj.right_boundary, incl_heading=False
                        )
                        + shifted_origin[:3]
                    )

                traffic_sign_ids: Optional[Set[str]] = None
                if len(road_lane_obj.traffic_sign_ids) > 0:
                    traffic_sign_ids = set(
                        [iden.decode() for iden in road_lane_obj.traffic_sign_ids]
                    )

                wait_line_ids: Optional[Set[str]] = None
                if len(road_lane_obj.wait_line_ids) > 0:
                    wait_line_ids = set(
                        [iden.decode() for iden in road_lane_obj.wait_line_ids]
                    )

                adj_lanes_left: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.adjacent_lanes_left]
                )
                adj_lanes_right: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.adjacent_lanes_right]
                )

                next_lanes: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.exit_lanes]
                )
                prev_lanes: Set[str] = set(
                    [iden.decode() for iden in road_lane_obj.entry_lanes]
                )

                # Double-using the connectivity attributes for lane IDs now (will
                # replace them with Lane objects after all Lane objects have been created).
                curr_lane = RoadLane(
                    elem_id,
                    center_pl,
                    left_pl,
                    right_pl,
                    traffic_sign_ids,
                    wait_line_ids,
                    adj_lanes_left,
                    adj_lanes_right,
                    next_lanes,
                    prev_lanes,
                )
                map_elem_dict[MapElementType.ROAD_LANE][elem_id] = curr_lane

            elif incl_road_areas and map_elem.HasField("road_area"):
                road_area_obj: map_proto.RoadArea = map_elem.road_area

                exterior: Polyline = Polyline(
                    map_utils.proto_to_np(
                        road_area_obj.exterior_polygon, incl_heading=False
                    )
                    + shifted_origin[:3]
                )

                interior_holes: List[Polyline] = list()
                interior_hole: map_proto.Polyline
                for interior_hole in road_area_obj.interior_holes:
                    interior_holes.append(
                        Polyline(
                            map_utils.proto_to_np(interior_hole, incl_heading=False)
                            + shifted_origin[:3]
                        )
                    )

                curr_area = RoadArea(elem_id, exterior, interior_holes)
                map_elem_dict[MapElementType.ROAD_AREA][elem_id] = curr_area

            elif incl_ped_crosswalks and map_elem.HasField("ped_crosswalk"):
                ped_crosswalk_obj: map_proto.PedCrosswalk = map_elem.ped_crosswalk

                polygon_vertices: Polyline = Polyline(
                    map_utils.proto_to_np(ped_crosswalk_obj.polygon, incl_heading=False)
                    + shifted_origin[:3]
                )

                curr_area = PedCrosswalk(elem_id, polygon_vertices)
                map_elem_dict[MapElementType.PED_CROSSWALK][elem_id] = curr_area

            elif incl_ped_walkways and map_elem.HasField("ped_walkway"):
                ped_walkway_obj: map_proto.PedCrosswalk = map_elem.ped_walkway

                polygon_vertices: Polyline = Polyline(
                    map_utils.proto_to_np(ped_walkway_obj.polygon, incl_heading=False)
                    + shifted_origin[:3]
                )

                curr_area = PedWalkway(elem_id, polygon_vertices)
                map_elem_dict[MapElementType.PED_WALKWAY][elem_id] = curr_area

            elif incl_road_edges and map_elem.HasField("road_edge"):
                road_edge_obj: map_proto.RoadEdge = map_elem.road_edge

                polyline: Polyline = Polyline(
                    map_utils.proto_to_np(road_edge_obj.polyline) + shifted_origin
                )

                curr_edge = RoadEdge(elem_id, polyline)
                map_elem_dict[MapElementType.ROAD_EDGE][elem_id] = curr_edge

            elif incl_traffic_signs and map_elem.HasField("traffic_sign"):
                traffic_sign_obj: map_proto.TrafficSign = map_elem.traffic_sign

                position: np.ndarray = (
                    np.array(
                        [
                            traffic_sign_obj.position.x,
                            traffic_sign_obj.position.y,
                            traffic_sign_obj.position.z,
                        ]
                    )
                    + shifted_origin[:3]
                )

                curr_traffic_sign = TrafficSign(
                    elem_id, position, traffic_sign_obj.sign_type
                )
                map_elem_dict[MapElementType.TRAFFIC_SIGN][elem_id] = curr_traffic_sign

            elif incl_wait_lines and map_elem.HasField("wait_line"):
                wait_line_obj: map_proto.WaitLine = map_elem.wait_line

                polyline: Polyline = Polyline(
                    map_utils.proto_to_np(wait_line_obj.polyline, incl_heading=False)
                    + shifted_origin[:3]
                )

                curr_wait_line = WaitLine(
                    elem_id,
                    polyline,
                    wait_line_obj.wait_line_type,
                    wait_line_obj.is_implicit,
                )
                map_elem_dict[MapElementType.WAIT_LINE][elem_id] = curr_wait_line

        return cls(
            map_id=vec_map.name,
            extent=np.array(
                [
                    vec_map.min_pt.x,
                    vec_map.min_pt.y,
                    vec_map.min_pt.z,
                    vec_map.max_pt.x,
                    vec_map.max_pt.y,
                    vec_map.max_pt.z,
                ]
            ),
            elements=map_elem_dict,
            search_kdtrees=None,
            search_rtrees=None,
            traffic_light_status=None,
        )

    def associate_scene_data(
        self, traffic_light_status_dict: Dict[Tuple[str, int], TrafficLightStatus]
    ) -> None:
        """Associates vector map with scene-specific data like traffic light information"""
        self.traffic_light_status = traffic_light_status_dict

    def get_current_lane(
        self,
        xyzh: np.ndarray,
        max_dist: float = 2.0,
        max_heading_error: float = np.pi / 8,
    ) -> List[RoadLane]:
        """
        Args:
            xyzh (np.ndarray): 3d position and heading of agent in world coordinates

        Returns:
            List[RoadLane]: List of possible road lanes that agent could be on
        """
        assert (
            self.search_kdtrees is not None
        ), "Search KDTree not found, please rebuild cache."
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        return [
            self.lanes[idx]
            for idx in lane_kdtree.current_lane_inds(xyzh, max_dist, max_heading_error)
        ]

    def get_closest_lane(self, xyz: np.ndarray) -> RoadLane:
        assert (
            self.search_kdtrees is not None
        ), "Search KDTree not found, please rebuild cache."
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        return self.lanes[lane_kdtree.closest_polyline_ind(xyz)]

    def get_closest_road_edge(self, xyz: np.ndarray) -> RoadLane:
        assert (
            self.search_kdtrees is not None
        ), "Search KDTree not found, please rebuild cache."
        road_edge_kdtree: RoadEdgeKDTree = self.search_kdtrees[MapElementType.ROAD_EDGE]
        return self.road_edges[road_edge_kdtree.closest_polyline_ind(xyz)]

    def get_closest_unique_lanes(self, xyz_vec: np.ndarray) -> List[RoadLane]:
        assert (
            self.search_kdtrees is not None
        ), "Search KDTree not found, please rebuild cache."
        assert xyz_vec.ndim == 2  # xyz_vec is assumed to be (*, 3)
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        closest_inds = lane_kdtree.closest_polyline_ind(xyz_vec)
        unique_inds = np.unique(closest_inds)
        return [self.lanes[ind] for ind in unique_inds]

    def get_lanes_within(self, xyz: np.ndarray, dist: float) -> List[RoadLane]:
        assert (
            self.search_kdtrees is not None
        ), "Search KDTree not found, please rebuild cache."
        lane_kdtree: LaneCenterKDTree = self.search_kdtrees[MapElementType.ROAD_LANE]
        return [
            self.lanes[idx] for idx in lane_kdtree.polyline_inds_in_range(xyz, dist)
        ]

    def get_road_edges_within(self, xyz: np.ndarray, dist: float) -> List[RoadEdge]:
        assert (
            self.search_kdtrees is not None
        ), "Search KDTree not found, please rebuild cache."
        road_edge_kdtree: RoadEdgeKDTree = self.search_kdtrees[MapElementType.ROAD_EDGE]
        return [
            self.road_edges[idx]
            for idx in road_edge_kdtree.polyline_inds_in_range(xyz, dist)
        ]

    def get_closest_area(
        self, xy: np.ndarray, elem_type: MapElementType
    ) -> Union[RoadArea, PedCrosswalk, PedWalkway]:
        """
        Returns 2D MapElement closest to query point

        Args:
            xy (np.ndarray): query point
            elem_type (MapElementType): type of map element desired

        Returns:
            Union[RoadArea, PedCrosswalk, PedWalkway]: closest map elem of desired type to xy point
        """
        assert (
            self.search_rtrees is not None
        ), "Search RTree not found, please rebuild cache."
        elem_id = self.search_rtrees[elem_type].nearest_area(xy)
        return self.elements[elem_type][elem_id]

    def get_areas_within(
        self, xy: np.ndarray, elem_type: MapElementType, dist: float
    ) -> List[Union[RoadArea, PedCrosswalk, PedWalkway]]:
        """
        Returns all 2D MapElements within dist of query xy point

        Args:
            xy (np.ndarray): query point
            elem_type (MapElementType): type of map element desired
            dist (float): distance threshold

        Returns:
            List[Union[RoadArea, PedCrosswalk, PedWalkway]]: List of areas matching query
        """
        assert (
            self.search_rtrees is not None
        ), "Search RTree not found, please rebuild cache."
        ids = self.search_rtrees[elem_type].query_point(
            xy, predicate="dwithin", distance=dist
        )
        return [self.elements[elem_type][id] for id in ids]

    def get_road_areas_within(self, xyz: np.ndarray, dist: float) -> List[RoadArea]:
        road_area_kdtree: RoadAreaKDTree = self.search_kdtrees[MapElementType.ROAD_AREA]
        polyline_inds = road_area_kdtree.polyline_inds_in_range(xyz, dist)
        element_ids = set(
            [road_area_kdtree.metadata["map_elem_id"][ind] for ind in polyline_inds]
        )
        if MapElementType.ROAD_AREA not in self.elements:
            raise ValueError(
                "Road areas are not loaded. Use map_api.get_map(..., incl_road_areas=True)."
            )
        return [self.elements[MapElementType.ROAD_AREA][id] for id in element_ids]

    def get_road_area_polygon_2d(self, id: str) -> Polygon:
        if id not in self._road_area_polygons:
            road_area: RoadArea = self.elements[MapElementType.ROAD_AREA][id]
            road_area_polygon = Polygon(
                shell=[(pt[0], pt[1]) for pt in road_area.exterior_polygon.points],
                holes=[
                    [(pt[0], pt[1]) for pt in polyline.points]
                    for polyline in road_area.interior_holes
                ],
            )
            self._road_area_polygons[id] = road_area_polygon
        return self._road_area_polygons[id]

    def get_traffic_light_status(
        self, lane_id: str, scene_ts: int
    ) -> TrafficLightStatus:
        return (
            self.traffic_light_status.get(
                (lane_id, scene_ts), TrafficLightStatus.NO_DATA
            )
            if self.traffic_light_status is not None
            else TrafficLightStatus.NO_DATA
        )

    def has_stop_sign(self, lane_id: str) -> bool:
        traffic_sign_ids = self.get_road_lane(lane_id).traffic_sign_ids
        if traffic_sign_ids is not None:
            for sign_id in traffic_sign_ids:
                if (
                    self.elements[MapElementType.TRAFFIC_SIGN][sign_id].sign_type
                    == "TRAFFIC_SIGN_REGULATORY_R1_STOP"
                ):
                    return True
        return False

    def get_wait_lines(self, lane_id: str) -> List[WaitLine]:
        # Get waitlines for a lane
        wait_line_ids = self.get_road_lane(lane_id).wait_line_ids
        wait_lines: List[WaitLine] = []
        if wait_line_ids is not None:
            wait_lines: List[WaitLine] = [
                self.elements[MapElementType.WAIT_LINE][wait_line_id]
                for wait_line_id in wait_line_ids
            ]
        return wait_lines

    def get_traffic_signs(self, lane_id: str) -> List[TrafficSign]:
        # Get traffic signs for a lane
        traffic_sign_ids = self.get_road_lane(lane_id).traffic_sign_ids
        traffic_signs: List[TrafficSign] = []
        if traffic_sign_ids is not None:
            traffic_signs: List[TrafficSign] = [
                self.elements[MapElementType.TRAFFIC_SIGN][traffic_sign_id]
                for traffic_sign_id in traffic_sign_ids
            ]

        return traffic_signs

    def associate_traffic_sign_with_wait_line(
        self, lane_id: str
    ) -> List[Tuple[TrafficSign, WaitLine]]:
        """Associate traffic signs with wait lines for a lane using nearest distance.

        Args:
            lane_id (str): lane id

        Returns:
            List[Tuple[TrafficSign, WaitLine]]: List of (traffic_sign, wait_line) tuples that are associated.
        """

        # Get waitlines and traffic signs
        wait_lines: List[WaitLine] = self.get_wait_lines(lane_id)
        traffic_signs: List[TrafficSign] = self.get_traffic_signs(lane_id)

        # Associate traffic signs with wait lines
        traffic_sign_wait_line_associations: List[Tuple[TrafficSign, WaitLine]] = []
        for traffic_sign in traffic_signs:
            smallest_distance_to_wait_line = np.inf
            nearest_wait_line = None
            for wait_line in wait_lines:
                distance_to_wait_line = wait_line.polyline.distance_to_point(
                    traffic_sign.position[None, :]
                )
                if distance_to_wait_line < smallest_distance_to_wait_line:
                    smallest_distance_to_wait_line = distance_to_wait_line
                    nearest_wait_line = wait_line
            if nearest_wait_line is not None:
                traffic_sign_wait_line_associations.append(
                    (traffic_sign, nearest_wait_line)
                )
        return traffic_sign_wait_line_associations

    def is_stop_sign(self, traffic_sign: TrafficSign) -> bool:
        if traffic_sign.sign_type == "TRAFFIC_SIGN_REGULATORY_R1_STOP":
            return True
        return False

    def get_wait_line(self, lane_id: str) -> Optional[WaitLine]:
        wait_line_ids = self.get_road_lane(lane_id).wait_line_ids
        if len(wait_line_ids) == 0:
            return None
        return self.elements[MapElementType.WAIT_LINE][wait_line_ids.pop()]

    def get_online_metadict(self, lane_id: str, scene_ts: int = 0) -> Dict:
        return (
            self.online_metadict[(str(lane_id), scene_ts)]
            if self.online_metadict is not None
            else {}
        )

    def rasterize(
        self, resolution: float = 2, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Renders this vector map at the specified resolution.

        Args:
            resolution (float): The rasterized image's resolution in pixels per meter.

        Returns:
            np.ndarray: The rasterized RGB image.
        """
        raise NotImplementedError()

    @overload
    def visualize_lane_graph(
        self,
        origin_lane: RoadLane,
        num_hops: int,
        **kwargs,
    ) -> Axes: ...

    @overload
    def visualize_lane_graph(
        self, origin_lane: str, num_hops: int, **kwargs
    ) -> Axes: ...

    @overload
    def visualize_lane_graph(
        self, origin_lane: int, num_hops: int, **kwargs
    ) -> Axes: ...

    def visualize_lane_graph(
        self, origin_lane: Union[RoadLane, str, int], num_hops: int, **kwargs
    ) -> Axes:
        ax = kwargs.get("ax", None)
        if ax is None:
            fig, ax = plt.subplots()

        origin: str
        if isinstance(origin_lane, RoadLane):
            origin = origin_lane.id
        elif isinstance(origin_lane, str):
            origin = origin_lane
        elif isinstance(origin_lane, int):
            origin = self.lanes[origin_lane].id

        viridis = mpl.colormaps[kwargs.get("cmap", "rainbow")].resampled(num_hops + 1)

        already_seen: Set[str] = set()
        lanes_to_plot: List[Tuple[str, int]] = [(origin, 0)]

        if kwargs.get("legend", True):
            ax.scatter([], [], label=f"Lane Endpoints", color="k")
            ax.plot([], [], label=f"Origin Lane ({origin})", color=viridis(0))
            for h in range(1, num_hops + 1):
                ax.plot(
                    [],
                    [],
                    label=f"{h} Lane{'s' if h > 1 else ''} Away",
                    color=viridis(h),
                )

        raster_from_world = kwargs.get("raster_from_world", None)
        while len(lanes_to_plot) > 0:
            lane_id, curr_hops = lanes_to_plot.pop(0)
            already_seen.add(lane_id)
            lane: RoadLane = self.get_road_lane(lane_id)

            center: np.ndarray = lane.center.points[..., :2]
            first_pt_heading: float = lane.center.points[0, -1]
            mdpt: np.ndarray = lane.center.midpoint[..., :2]

            if raster_from_world is not None:
                center = map_utils.transform_points(center, raster_from_world)
                mdpt = map_utils.transform_points(mdpt[None, :], raster_from_world)[0]

            ax.plot(center[:, 0], center[:, 1], color=viridis(curr_hops))
            ax.scatter(center[[0, -1], 0], center[[0, -1], 1], color=viridis(curr_hops))
            ax.quiver(
                [center[0, 0]],
                [center[0, 1]],
                [np.cos(first_pt_heading)],
                [np.sin(first_pt_heading)],
                color=viridis(curr_hops),
            )
            ax.text(mdpt[0], mdpt[1], s=lane_id)

            if curr_hops < num_hops:
                lanes_to_plot += [
                    (l, curr_hops + 1)
                    for l in lane.reachable_lanes
                    if l not in already_seen
                ]

        if kwargs.get("legend", True):
            ax.legend(loc="best", frameon=True)

        return ax


def split_lane_segments(lane, n=None, max_len=None):
    if n is None:
        length = np.linalg.norm(lane.center.xy[1:] - lane.center.xy[:-1], axis=-1).sum()
        n = ceil(length / max_len)
    if n == 1:
        return [lane]
    idx = np.linspace(0, lane.center.xy.shape[0] - 1, n + 1).astype(int)
    left_idx = (
        np.linspace(0, lane.left_edge.xy.shape[0] - 1, n + 1).astype(int)
        if lane.left_edge is not None
        else None
    )
    right_idx = (
        np.linspace(0, lane.right_edge.xy.shape[0] - 1, n + 1).astype(int)
        if lane.right_edge is not None
        else None
    )

    split_lanes = list()
    for i in range(n):
        center = Polyline(points=lane.center.points[idx[i] : idx[i + 1] + 1])
        left_edge = (
            Polyline(lane.left_edge.points[left_idx[i] : left_idx[i + 1] + 1])
            if lane.left_edge is not None
            else None
        )
        right_edge = (
            Polyline(lane.right_edge.points[right_idx[i] : right_idx[i + 1] + 1])
            if lane.right_edge is not None
            else None
        )

        new_lane = RoadLane(
            center=center,
            left_edge=left_edge,
            right_edge=right_edge,
            adj_lanes_left=lane.adj_lanes_left,
            adj_lanes_right=lane.adj_lanes_right,
            next_lanes=lane.next_lanes if i == n - 1 else set(),
            prev_lanes=lane.prev_lanes if i == 0 else {split_lanes[i - 1].id},
            road_area_ids=lane.road_area_ids,
            elem_type=lane.elem_type,
            id=lane.id + f"_{i}",
        )
        if i > 0:
            split_lanes[i - 1].next_lanes = {new_lane.id}
        split_lanes.append(new_lane)
    return split_lanes
