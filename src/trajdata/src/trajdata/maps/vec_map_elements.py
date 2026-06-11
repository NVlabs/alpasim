# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Set, Union

import numpy as np

from trajdata.utils import map_utils
from trajdata.utils.arr_utils import angle_wrap


class MapElementType(IntEnum):
    ROAD_LANE = 1
    ROAD_AREA = 2
    PED_CROSSWALK = 3
    PED_WALKWAY = 4
    ROAD_EDGE = 5
    TRAFFIC_SIGN = 6
    WAIT_LINE = 7


@dataclass
class Polyline:
    points: np.ndarray

    def __post_init__(self) -> None:
        if self.points.shape[-1] < 2:
            raise ValueError(
                f"Polylines are expected to have 2 (xy), 3 (xyz), or 4 (xyzh) dimensions, but received {self.points.shape[-1]}."
            )

        if self.points.shape[-1] == 2:
            # If only xy are passed in, then append zero to the end for z.
            self.points = np.append(
                self.points, np.zeros_like(self.points[:, [0]]), axis=-1
            )

    @property
    def midpoint(self) -> np.ndarray:
        num_pts: int = self.points.shape[0]
        return self.points[num_pts // 2]

    @property
    def has_heading(self) -> bool:
        return self.points.shape[-1] == 4

    @property
    def xy(self) -> np.ndarray:
        return self.points[..., :2]

    @property
    def xyz(self) -> np.ndarray:
        return self.points[..., :3]

    @property
    def xyh(self) -> np.ndarray:
        if self.has_heading:
            return self.points[..., (0, 1, 3)]
        else:
            raise ValueError(
                f"This Polyline only has {self.points.shape[-1]} coordinates, expected 4."
            )

    @property
    def xyzh(self) -> np.ndarray:
        if self.has_heading:
            return self.points[..., :4]
        else:
            raise ValueError(
                f"This Polyline only has {self.points.shape[-1]} coordinates, expected 4."
            )

    @property
    def h(self) -> np.ndarray:
        return self.points[..., 3]

    def interpolate(
        self, num_pts: Optional[int] = None, max_dist: Optional[float] = None
    ) -> "Polyline":
        return Polyline(
            map_utils.interpolate(self.points, num_pts=num_pts, max_dist=max_dist)
        )

    def project_onto(
        self, xyz_or_xyzh: np.ndarray, return_index: bool = False
    ) -> Union[np.ndarray, List]:
        """Project the given points onto this Polyline.

        Args:
            xyzh (np.ndarray): Points to project, of shape (M, D)
            return_indices (bool): Return the index of starting point of the line segment
                on which the projected points lies on.

        Returns:
            np.ndarray: The projected points, of shape (M, D)
            np.ndarray: The index of previous polyline points if return_indices == True.

        Note:
            D = 4 if this Polyline has headings, otherwise D = 3
        """
        # xyzh is now (M, 1, 3), we do not use heading for projection.
        xyz = xyz_or_xyzh[:, np.newaxis, :3]

        # p0, p1 are (1, N, 3)
        p0: np.ndarray = self.points[np.newaxis, :-1, :3]
        p1: np.ndarray = self.points[np.newaxis, 1:, :3]

        # 1. Compute projections of each point to each line segment in a
        #    batched manner.
        line_seg_diffs: np.ndarray = p1 - p0
        point_seg_diffs: np.ndarray = xyz - p0

        dot_products: np.ndarray = (point_seg_diffs * line_seg_diffs).sum(
            axis=-1, keepdims=True
        )
        norms: np.ndarray = np.square(line_seg_diffs).sum(axis=-1, keepdims=True)

        # Clip ensures that the projected point stays within the line segment boundaries.
        projs: np.ndarray = (
            p0 + np.clip(dot_products / norms, a_min=0, a_max=1) * line_seg_diffs
        )

        # 2. Find the nearest projections to the original points.
        # We have nan values when two consecutive points are equal. This will never be
        # the closest projection point, so we replace nans with a large number.
        point_to_proj_dist = np.nan_to_num(
            np.linalg.norm(xyz - projs, axis=-1), nan=1e6
        )
        closest_proj_idxs: int = point_to_proj_dist.argmin(axis=-1)

        proj_points = projs[range(xyz.shape[0]), closest_proj_idxs]

        if self.has_heading:
            # Adding in the heading of the corresponding p0 point (which makes
            # sense as p0 to p1 is a line => same heading along it).
            proj_points = np.concatenate(
                [
                    proj_points,
                    np.expand_dims(self.points[closest_proj_idxs, -1], axis=-1),
                ],
                axis=-1,
            )

        if return_index:
            return proj_points, closest_proj_idxs
        else:
            return proj_points

    def distance_to_point(self, xyz: np.ndarray):
        assert xyz.ndim == 2
        xyz_proj = self.project_onto(xyz)
        return np.linalg.norm(xyz[..., :3] - xyz_proj[..., :3], axis=-1)

    def get_length(self):
        # TODO we could store cummulative distances to speed this up
        dists = np.linalg.norm(self.xyz[1:, :3] - self.xyz[:-1, :3], axis=-1)
        length = dists.sum()
        return length

    def get_length_from(self, start_ind: np.ndarray):
        # TODO we could store cummulative distances to speed this up
        assert start_ind.ndim == 1
        dists = np.linalg.norm(self.xyz[1:, :3] - self.xyz[:-1, :3], axis=-1)
        length_upto = np.cumsum(np.pad(dists, (1, 0)))
        length_from = length_upto[-1][None] - length_upto[start_ind]
        return length_from

    def traverse_along(
        self, dist: np.ndarray, start_ind: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Interpolated endpoint of traversing `dist` distance along polyline from a starting point.

        Returns nan if the end point is not inside the polyline.
        TODO we could store cummulative distances to speed this up

        Args:
            dist (np.ndarray): distances, any shape [...]
            start_ind (np.ndarray): index of point along polyline to calcualte distance from.
                Optional. Shape must match dist. [...]

        Returns:
            endpoint_xyzh (np.ndarray): points along polyline `dist` distance from the
                starting point. Nan if endpoint would require extrapolation. [..., 4]

        """
        assert self.has_heading

        # Add up distances from beginning of polyline
        segment_lens = np.linalg.norm(self.xyz[1:] - self.xyz[:-1], axis=-1)  # n-1
        cum_len = np.pad(np.cumsum(segment_lens, axis=0), (1, 0))  # n

        # Increase dist with the length of lane up to start_ind
        if start_ind is not None:
            assert start_ind.ndim == dist.ndim
            dist = dist + cum_len[start_ind]

        # Find the first index where cummulative length is larger or equal than `dist`
        inds = np.searchsorted(cum_len, dist, side="right")
        # Invalidate inds == 0 and inds == len(cum_len), which means endpoint is outside the polyline.
        invalid = np.logical_or(inds == 0, inds == len(cum_len))
        # Replace invalid indices so we can easily carry out computation below, and invalidate output eventually.
        inds[invalid] = 1

        # Remaining distance from last point
        remaining_dist = dist - cum_len[inds - 1]

        # Invalidate negative remaining dist (this should only happen when dist < 0)
        invalid = np.logical_or(invalid, remaining_dist < 0.0)

        # Interpolate between the previous and next points.
        segment_vect_xyz = self.xyz[inds] - self.xyz[inds - 1]
        segment_len = np.linalg.norm(segment_vect_xyz, axis=-1)
        assert (segment_len > 0.0).all(), "Polyline segment has zero length"

        proportion = remaining_dist / segment_len
        endpoint_xyz = segment_vect_xyz * proportion[..., np.newaxis] + self.xyz[inds]
        endpoint_h = angle_wrap(
            angle_wrap(self.h[inds] - self.h[inds - 1]) * proportion + self.h[inds - 1]
        )
        endpoint_xyzh = np.concatenate(
            (endpoint_xyz, endpoint_h[..., np.newaxis]), axis=-1
        )

        # Invalidate dummy output
        endpoint_xyzh[invalid] = np.nan

        return endpoint_xyzh

    def concatenate_with(self, other: "Polyline") -> "Polyline":
        return self.concatenate([self, other])

    @staticmethod
    def concatenate(polylines: List["Polyline"]) -> "Polyline":
        # Assumes no overlap between consecutive polylines, i.e. next lane starts after current lane ends.
        points = np.concatenate([polyline.points for polyline in polylines], axis=0)
        return Polyline(points)


@dataclass
class MapElement:
    id: str


@dataclass
class RoadLane(MapElement):
    center: Polyline
    left_edge: Optional[Polyline] = None
    right_edge: Optional[Polyline] = None
    traffic_sign_ids: Optional[Set[str]] = None
    wait_line_ids: Optional[Set[str]] = None
    adj_lanes_left: Set[str] = field(default_factory=lambda: set())
    adj_lanes_right: Set[str] = field(default_factory=lambda: set())
    next_lanes: Set[str] = field(default_factory=lambda: set())
    prev_lanes: Set[str] = field(default_factory=lambda: set())
    road_area_ids: Set[str] = field(default_factory=lambda: set())
    elem_type: MapElementType = MapElementType.ROAD_LANE

    def __post_init__(self) -> None:
        if not self.center.has_heading:
            self.center = Polyline(
                np.append(
                    self.center.xyz,
                    map_utils.get_polyline_headings(self.center.xyz),
                    axis=-1,
                )
            )

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def reachable_lanes(self) -> Set[str]:
        return self.adj_lanes_left | self.adj_lanes_right | self.next_lanes

    def combine_next(self, next_lane):
        assert next_lane.id in self.next_lanes
        self.next_lanes.remove(next_lane.id)
        self.next_lanes = self.next_lanes.union(next_lane.next_lanes)
        self.center = self.center.concatenate_with(next_lane.center)
        if self.left_edge is not None and next_lane.left_edge is not None:
            self.left_edge = self.left_edge.concatenate_with(next_lane.left_edge)
        if self.right_edge is not None and next_lane.right_edge is not None:
            self.right_edge = self.right_edge.concatenate_with(next_lane.right_edge)
        self.adj_lanes_right = self.adj_lanes_right.union(next_lane.adj_lanes_right)
        self.adj_lanes_left = self.adj_lanes_left.union(next_lane.adj_lanes_left)
        self.road_area_ids = self.road_area_ids.union(next_lane.road_area_ids)

    def combine_prev(self, prev_lane):
        assert prev_lane.id in self.prev_lanes
        self.prev_lanes.remove(prev_lane.id)
        self.prev_lanes = self.prev_lanes.union(prev_lane.prev_lanes)
        self.center = prev_lane.center.concatenate_with(self.center)
        if self.left_edge is not None and prev_lane.left_edge is not None:
            self.left_edge = prev_lane.left_edge.concatenate_with(self.left_edge)
        if self.right_edge is not None and prev_lane.right_edge is not None:
            self.right_edge = prev_lane.right_edge.concatenate_with(self.right_edge)
        self.adj_lanes_right = self.adj_lanes_right.union(prev_lane.adj_lanes_right)
        self.adj_lanes_left = self.adj_lanes_left.union(prev_lane.adj_lanes_left)
        self.road_area_ids = self.road_area_ids.union(prev_lane.road_area_ids)


@dataclass
class RoadArea(MapElement):
    exterior_polygon: Polyline
    interior_holes: List[Polyline] = field(default_factory=lambda: list())
    elem_type: MapElementType = MapElementType.ROAD_AREA


@dataclass
class PedCrosswalk(MapElement):
    polygon: Polyline
    elem_type: MapElementType = MapElementType.PED_CROSSWALK


@dataclass
class PedWalkway(MapElement):
    polygon: Polyline
    elem_type: MapElementType = MapElementType.PED_WALKWAY


@dataclass
class RoadEdge(MapElement):
    polyline: Polyline
    elem_type: MapElementType = MapElementType.ROAD_EDGE

    def __post_init__(self) -> None:
        if not self.polyline.has_heading:
            self.polyline = Polyline(
                np.append(
                    self.polyline.xyz,
                    map_utils.get_polyline_headings(self.polyline.xyz),
                    axis=-1,
                )
            )

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class TrafficSign(MapElement):
    position: np.ndarray
    sign_type: str
    elem_type: MapElementType = MapElementType.TRAFFIC_SIGN


@dataclass
class WaitLine(MapElement):
    polyline: Polyline
    wait_line_type: str
    is_implicit: bool
    elem_type: MapElementType = MapElementType.WAIT_LINE
