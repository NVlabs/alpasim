# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterator

if TYPE_CHECKING:
    from trajdata.maps.vec_map import VectorMap

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from trajdata.maps.vec_map_elements import MapElement, MapElementType, Polyline
from trajdata.utils.arr_utils import angle_wrap


class MapElementKDTree:
    """
    Constructs a KDTree of MapElements and exposes fast lookup functions.

    Inheriting classes need to implement the _extract_points function that defines for a MapElement
    the coordinates we want to store in the KDTree.
    """

    def __init__(self, vector_map: VectorMap, verbose: bool = False) -> None:
        # Build kd-tree
        self.kdtree, self.polyline_inds, self.metadata = self._build_kdtree(
            vector_map, verbose
        )

    def _build_kdtree(self, vector_map: VectorMap, verbose: bool = False):
        polylines = []
        polyline_inds = []
        metadata = defaultdict(list)

        map_elem: MapElement
        for map_elem in tqdm(
            vector_map.iter_elems(),
            desc=f"Building K-D Trees",
            leave=False,
            total=len(vector_map),
            disable=not verbose,
        ):
            for result in self._extract_points_and_metadata(map_elem):
                points, extras = result
                polyline_inds.extend([len(polylines)] * points.shape[0])

                # Apply any map offsets to ensure we're in the same coordinate area as the
                # original world map.
                polylines.append(points)

                for k, v in extras.items():
                    metadata[k].append(v)
                metadata["map_elem_id"].append(np.array([map_elem.id]))

        points = np.concatenate(polylines, axis=0)
        polyline_inds = np.array(polyline_inds)
        metadata = {k: np.concatenate(v) for k, v in metadata.items()}

        kdtree = KDTree(points)
        return kdtree, polyline_inds, metadata

    def _extract_points_and_metadata(
        self, map_element: MapElement
    ) -> Iterator[Tuple[np.ndarray, dict[str, np.ndarray]]]:
        """Defines the coordinates we want to store in the KDTree for a MapElement.
        Args:
            map_element (MapElement): the MapElement to store in the KDTree.
        Returns:
            Optional[np.ndarray]: coordinates based on which we can search the KDTree, or None.
                If None, the MapElement will not be stored.
                Else, tuple of
                    np.ndarray: [B,d] set of B d-dimensional points to add,
                    Dict[str, np.ndarray] mapping names to meta-information about the points
        """
        raise NotImplementedError()

    def closest_point(self, query_points: np.ndarray) -> np.ndarray:
        """Find the closest KDTree points to (a batch of) query points.

        Args:
            query_points: np.ndarray of shape (..., data_dim).

        Return:
            np.ndarray of shape (..., data_dim), the KDTree points closest to query_point.
        """
        _, data_inds = self.kdtree.query(query_points, k=1)
        pts = self.kdtree.data[data_inds]
        return pts

    def closest_polyline_ind(self, query_points: np.ndarray) -> np.ndarray:
        """Find the index of the closest polyline(s) in self.polylines."""
        _, data_ind = self.kdtree.query(query_points, k=1)
        return self.polyline_inds[data_ind]

    def polyline_inds_in_range(self, point: np.ndarray, range: float) -> np.ndarray:
        """Find the index of polylines in self.polylines within 'range' distance to 'point'."""
        data_inds = self.kdtree.query_ball_point(point, range)
        return np.unique(self.polyline_inds[data_inds], axis=0)


class RoadEdgeKDTree(MapElementKDTree):
    """KDTree for lane center polylines."""

    def __init__(
        self, vector_map: VectorMap, max_segment_len: Optional[float] = None
    ) -> None:
        """
        Args:
            vec_map: the VectorizedMap object to build the KDTree for
            max_segment_len (float, optional): if specified, we will insert extra points into the KDTree
                such that all polyline segments are shorter then max_segment_len.
        """
        self.max_segment_len = max_segment_len
        super().__init__(vector_map)

    def _extract_points_and_metadata(
        self, map_element: MapElement
    ) -> Iterator[Tuple[np.ndarray, dict[str, np.ndarray]]]:
        if map_element.elem_type == MapElementType.ROAD_EDGE:
            pts: Polyline = map_element.polyline
            if self.max_segment_len is not None:
                pts = pts.interpolate(max_dist=self.max_segment_len)

            # We only want to store xyz in the kdtree, not heading.
            yield pts.xyz, {"heading": pts.h}


class LaneCenterKDTree(MapElementKDTree):
    """KDTree for lane center polylines."""

    def __init__(
        self, vector_map: VectorMap, max_segment_len: Optional[float] = None
    ) -> None:
        """
        Args:
            vec_map: the VectorizedMap object to build the KDTree for
            max_segment_len (float, optional): if specified, we will insert extra points into the KDTree
                such that all polyline segments are shorter then max_segment_len.
        """
        self.max_segment_len = max_segment_len
        super().__init__(vector_map)

    def _extract_points_and_metadata(
        self, map_element: MapElement
    ) -> Iterator[Tuple[np.ndarray, dict[str, np.ndarray]]]:
        if map_element.elem_type == MapElementType.ROAD_LANE:
            pts: Polyline = map_element.center
            if self.max_segment_len is not None:
                pts = pts.interpolate(max_dist=self.max_segment_len)

            # We only want to store xyz in the kdtree, not heading.
            yield pts.xyz, {"heading": pts.h}

    def current_lane_inds(
        self,
        xyzh: np.ndarray,
        distance_threshold: float,
        heading_threshold: float,
        sorted: bool = True,
        dist_weight: float = 1.0,
        heading_weight: float = 0.1,
    ) -> np.ndarray:
        """
        Args:
            xyzh (np.ndarray): [...,d]: (batch of) position and heading in world frame
            distance_threshold (Optional[float], optional). Defaults to None.
            heading_threshold (float, optional). Defaults to np.pi/8.

        Returns:
            np.ndarray: List of polyline inds that could be considered the current lane
                for the provided position and heading, ordered by heading similarity
        """
        query_point = xyzh[..., :3]  # query on xyz
        heading = xyzh[..., 3]
        data_inds = np.array(
            self.kdtree.query_ball_point(query_point, distance_threshold)
        )

        if len(data_inds) == 0:
            return []
        possible_points = self.kdtree.data[data_inds]
        possible_headings = self.metadata["heading"][data_inds]

        heading_errs = np.abs(angle_wrap(heading - possible_headings))
        dist_errs = np.linalg.norm(
            query_point[None, :] - possible_points, ord=2, axis=-1
        )

        under_thresh = heading_errs < heading_threshold
        lane_inds = self.polyline_inds[data_inds[under_thresh]]

        # we don't want to return duplicates of lanes
        unique_lane_inds = np.unique(lane_inds)

        if not sorted:
            return unique_lane_inds

        # if we are sorting results, evaluate cost:
        costs = (
            dist_weight * dist_errs[under_thresh]
            + heading_weight * heading_errs[under_thresh]
        )

        # cost for a lane is minimum over all possible points for that lane
        min_costs = [np.min(costs[lane_inds == ind]) for ind in unique_lane_inds]

        return unique_lane_inds[np.argsort(min_costs)]


class RoadAreaKDTree(MapElementKDTree):
    """KDTree for road area polygons.
    The polygons may have holes. We will simply store points along both the
    exterior_polygon and all interior_holes. Finding a nearest point in this KDTree will
    correspond to finding any
    """

    def __init__(
        self, vector_map: VectorMap, max_segment_len: Optional[float] = None
    ) -> None:
        """
        Args:
            vec_map: the VectorizedMap object to build the KDTree for
            max_segment_len (float, optional): if specified, we will insert extra points into the KDTree
                such that all polyline segments are shorter then max_segment_len.
        """
        self.max_segment_len = max_segment_len
        super().__init__(vector_map)

    def _extract_points_and_metadata(
        self, map_element: MapElement
    ) -> Iterator[Tuple[np.ndarray, dict[str, np.ndarray]]]:
        if map_element.elem_type == MapElementType.ROAD_AREA:
            # Exterior polygon
            pts: Polyline = map_element.exterior_polygon
            if self.max_segment_len is not None:
                pts = pts.interpolate(max_dist=self.max_segment_len)
            # We only want to store xyz in the kdtree, not heading.
            yield pts.xyz, {"exterior": np.array([True])}

            # Interior holes
            for pts in map_element.interior_holes:
                if self.max_segment_len is not None:
                    pts = pts.interpolate(max_dist=self.max_segment_len)
                # We only want to store xyz in the kdtree, not heading.
                yield pts.xyz, {"exterior": np.array([False])}
