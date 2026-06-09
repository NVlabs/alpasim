# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajdata.maps import map_kdtree, vec_map, map_strtree
    from trajdata.maps.vec_map_elements import MapElementType

import enum
from pathlib import Path
from typing import Dict, Final, List, NamedTuple, Optional

import dill
import numpy as np
from scipy.stats import circmean

import trajdata.proto.vectorized_map_pb2 as map_proto
from trajdata.utils import arr_utils

NUM_DECIMALS: Final[int] = 5
COMPRESSION_SCALE: Final[float] = 10**NUM_DECIMALS


class LaneSegRelation(enum.IntEnum):
    """
    Categorical token describing the relationship between an agent and a Lane
    """

    NOTCONNECTED = 0
    NEXT = 1
    PREV = 2
    LEFT = 3
    RIGHT = 4


class LaneGraph(NamedTuple):
    lane_ids: List[str]
    lane_centerlines: np.ndarray
    lane_left_edges: np.ndarray
    lane_right_edges: np.ndarray
    lane_connectivity: np.ndarray


def decompress_values(data: np.ndarray) -> np.ndarray:
    # From https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/data/proto/road_network.proto#L446
    # The delta for the first point is just its coordinates tuple, i.e. it is a "delta" from
    # the origin. For subsequent points, this field stores the difference between the point's
    # coordinates and the previous point's coordinates. This is for representation efficiency.
    return np.cumsum(data, axis=0, dtype=float) / COMPRESSION_SCALE


def compress_values(data: np.ndarray) -> np.ndarray:
    return (np.diff(data, axis=0, prepend=0.0) * COMPRESSION_SCALE).astype(np.int64)


def get_polyline_headings(points: np.ndarray) -> np.ndarray:
    """Get approximate heading angles for points in a polyline.

    Args:
        points: XY points, np.ndarray of shape [N, 2]

    Returns:
        np.ndarray: approximate heading angles in radians, shape [N, 1]
    """
    if points.ndim < 2 and points.shape[-1] != 2 and points.shape[-2] <= 1:
        raise ValueError("Unexpected shape")

    vectors = points[..., 1:, :] - points[..., :-1, :]
    vec_headings = np.arctan2(vectors[..., 1], vectors[..., 0])  # -pi..pi

    # For internal points compute the mean heading of consecutive segments.
    # Need to use circular mean to average directions.
    # TODO(pkarkus) this would be more accurate if weighted with the distance to the neighbor
    if vec_headings.shape[-1] <= 1:
        # Handle special case because circmean unfortunately returns nan for such input.
        mean_consec_headings = np.zeros(
            list(vec_headings.shape[:-1]) + [0], dtype=vec_headings.dtype
        )
    else:
        mean_consec_headings = circmean(
            np.stack([vec_headings[..., :-1], vec_headings[..., 1:]], axis=-1),
            high=np.pi,
            low=-np.pi,
            axis=-1,
        )

    headings = np.concatenate(
        [
            vec_headings[..., :1],  # heading of first segment
            mean_consec_headings,  # mean heading of consecutive segments
            vec_headings[..., -1:],  # heading of last segment
        ],
        axis=-1,
    )
    return headings[..., np.newaxis]


def populate_lane_polylines(
    new_lane_proto: map_proto.RoadLane,
    road_lane_py: vec_map.RoadLane,
    origin: np.ndarray,
) -> None:
    """Fill a Lane object's polyline attributes.
    All points should be in world coordinates.

    Args:
        new_lane (Lane): _description_
        midlane_pts (np.ndarray): _description_
        left_pts (np.ndarray): _description_
        right_pts (np.ndarray): _description_
    """
    compressed_mid_pts: np.ndarray = compress_values(road_lane_py.center.xyz - origin)
    new_lane_proto.center.dx_mm.extend(compressed_mid_pts[:, 0].tolist())
    new_lane_proto.center.dy_mm.extend(compressed_mid_pts[:, 1].tolist())
    new_lane_proto.center.dz_mm.extend(compressed_mid_pts[:, 2].tolist())
    new_lane_proto.center.h_rad.extend(road_lane_py.center.h.tolist())

    if road_lane_py.left_edge is not None:
        compressed_left_pts: np.ndarray = compress_values(
            road_lane_py.left_edge.xyz - origin
        )
        new_lane_proto.left_boundary.dx_mm.extend(compressed_left_pts[:, 0].tolist())
        new_lane_proto.left_boundary.dy_mm.extend(compressed_left_pts[:, 1].tolist())
        new_lane_proto.left_boundary.dz_mm.extend(compressed_left_pts[:, 2].tolist())

    if road_lane_py.right_edge is not None:
        compressed_right_pts: np.ndarray = compress_values(
            road_lane_py.right_edge.xyz - origin
        )
        new_lane_proto.right_boundary.dx_mm.extend(compressed_right_pts[:, 0].tolist())
        new_lane_proto.right_boundary.dy_mm.extend(compressed_right_pts[:, 1].tolist())
        new_lane_proto.right_boundary.dz_mm.extend(compressed_right_pts[:, 2].tolist())
    if road_lane_py.traffic_sign_ids is not None:
        new_lane_proto.traffic_sign_ids.extend(
            [iden.encode() for iden in road_lane_py.traffic_sign_ids]
        )
    if road_lane_py.wait_line_ids is not None:
        new_lane_proto.wait_line_ids.extend(
            [iden.encode() for iden in road_lane_py.wait_line_ids]
        )


def populate_road_edge_polylines(
    new_road_edge_proto: map_proto.RoadEdge,
    road_edge_py: vec_map.RoadEdge,
    origin: np.ndarray,
) -> None:
    """Fill a Lane object's polyline attributes.
    All points should be in world coordinates.

    Args:
        new_road_edge_proto (RoadEdge): _description_
        road_edge_py (np.ndarray): _description_
        origin (np.ndarray): _description_
    """
    compressed_pts: np.ndarray = compress_values(road_edge_py.polyline.xyz - origin)
    new_road_edge_proto.polyline.dx_mm.extend(compressed_pts[:, 0].tolist())
    new_road_edge_proto.polyline.dy_mm.extend(compressed_pts[:, 1].tolist())
    new_road_edge_proto.polyline.dz_mm.extend(compressed_pts[:, 2].tolist())
    new_road_edge_proto.polyline.h_rad.extend(road_edge_py.polyline.h.tolist())


def populate_polygon(
    polygon_proto: map_proto.Polyline,
    polygon_pts: np.ndarray,
    origin: np.ndarray,
) -> None:
    """Fill an object's polygon.
    All points should be in world coordinates.

    Args:
        polygon_proto (Polyline): _description_
        polygon_pts (np.ndarray): _description_
    """
    compressed_pts: np.ndarray = compress_values(polygon_pts - origin)

    polygon_proto.dx_mm.extend(compressed_pts[:, 0].tolist())
    polygon_proto.dy_mm.extend(compressed_pts[:, 1].tolist())
    polygon_proto.dz_mm.extend(compressed_pts[:, 2].tolist())


def proto_to_np(polyline: map_proto.Polyline, incl_heading: bool = True) -> np.ndarray:
    dx: np.ndarray = np.asarray(polyline.dx_mm)
    dy: np.ndarray = np.asarray(polyline.dy_mm)

    if len(polyline.dz_mm) > 0:
        dz: np.ndarray = np.asarray(polyline.dz_mm)
        pts: np.ndarray = np.stack([dx, dy, dz], axis=1)
    else:
        # Default z is all zeros.
        pts: np.ndarray = np.stack([dx, dy, np.zeros_like(dx)], axis=1)

    ret_pts: np.ndarray = decompress_values(pts)

    if incl_heading and len(polyline.h_rad) > 0:
        headings: np.ndarray = np.asarray(polyline.h_rad)
        ret_pts = np.concatenate((ret_pts, headings[:, np.newaxis]), axis=1)
    elif incl_heading:
        raise ValueError(
            f"Polyline must have heading, but it does not (polyline.h_rad is empty)."
        )

    return ret_pts


def transform_points(points: np.ndarray, transf_mat: np.ndarray):
    n_dim = points.shape[-1]
    return points @ transf_mat[:n_dim, :n_dim] + transf_mat[:n_dim, -1]


def order_matches(pts: np.ndarray, ref: np.ndarray) -> bool:
    """Evaluate whether `pts0` is ordered the same as `ref`, based on the distance from
    `pts0`'s start and end points to `ref`'s start point.

    Args:
        pts0 (np.ndarray): The first array of points, of shape (N, D).
        pts1 (np.ndarray): The second array of points, of shape (M, D).

    Returns:
        bool: True if `pts0`'s first point is closest to `ref`'s first point,
        False if `pts0`'s endpoint is closer (e.g., they are flipped relative to each other).
    """
    return np.linalg.norm(pts[0] - ref[0]) <= np.linalg.norm(pts[-1] - ref[0])


def endpoints_intersect(left_edge: np.ndarray, right_edge: np.ndarray) -> bool:
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = left_edge[-1], right_edge[-1]
    C, D = right_edge[0], left_edge[0]
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def interpolate(
    pts: np.ndarray, num_pts: Optional[int] = None, max_dist: Optional[float] = None
) -> np.ndarray:
    """
    Interpolate points either based on cumulative distances from the first one (`num_pts`)
    or by adding extra points until neighboring points are within `max_dist` of each other.

    In particular, `num_pts` will interpolate using a variable step such that we always get
    the requested number of points.

    Args:
        pts (np.ndarray): XYZ(H) coords.
        num_pts (int, optional): Desired number of total points.
        max_dist (float, optional): Maximum distance between points of the polyline.

    Note:
        Only one of `num_pts` or `max_dist` can be specified.

    Returns:
        np.ndarray: The new interpolated coordinates.
    """
    if num_pts is not None and max_dist is not None:
        raise ValueError("Only one of num_pts or max_dist can be used!")

    if pts.ndim != 2:
        raise ValueError("pts is expected to be 2 dimensional")

    # 3 because XYZ (heading does not count as a positional distance).
    pos_dim: int = min(pts.shape[-1], 3)
    has_heading: bool = pts.shape[-1] == 4

    if num_pts is not None:
        assert num_pts > 1, f"num_pts must be at least 2, but got {num_pts}"

        if pts.shape[0] == num_pts:
            return pts

        cum_dist: np.ndarray = np.cumsum(
            np.linalg.norm(np.diff(pts[..., :pos_dim], axis=0), axis=-1)
        )
        cum_dist = np.insert(cum_dist, 0, 0)

        steps: np.ndarray = np.linspace(cum_dist[0], cum_dist[-1], num_pts)
        xyz_inter: np.ndarray = np.empty((num_pts, pts.shape[-1]), dtype=pts.dtype)
        for i in range(pos_dim):
            xyz_inter[:, i] = np.interp(steps, xp=cum_dist, fp=pts[:, i])

        if has_heading:
            # Heading, so make sure to unwrap, interpolate, and wrap it.
            xyz_inter[:, 3] = arr_utils.angle_wrap(
                np.interp(steps, xp=cum_dist, fp=np.unwrap(pts[:, 3]))
            )

        return xyz_inter

    elif max_dist is not None:
        unwrapped_pts: np.ndarray = pts
        if has_heading:
            unwrapped_pts[..., 3] = np.unwrap(unwrapped_pts[..., 3])

        segments = unwrapped_pts[..., 1:, :] - unwrapped_pts[..., :-1, :]
        seg_lens = np.linalg.norm(segments[..., :pos_dim], axis=-1)
        new_pts = [unwrapped_pts[..., 0:1, :]]
        for i in range(segments.shape[-2]):
            num_extra_points = seg_lens[..., i] // max_dist
            if num_extra_points > 0:
                step_vec = segments[..., i, :] / (num_extra_points + 1)
                new_pts.append(
                    unwrapped_pts[..., i, np.newaxis, :]
                    + step_vec[..., np.newaxis, :]
                    * np.arange(1, num_extra_points + 1)[:, np.newaxis]
                )

            new_pts.append(unwrapped_pts[..., i + 1 : i + 2, :])

        new_pts = np.concatenate(new_pts, axis=-2)
        if has_heading:
            new_pts[..., 3] = arr_utils.angle_wrap(new_pts[..., 3])

        return new_pts


def load_vector_map(vector_map_path: Path) -> map_proto.VectorizedMap:
    if not vector_map_path.exists():
        raise ValueError(f"{vector_map_path} does not exist!")

    vec_map = map_proto.VectorizedMap()

    # Saving the vectorized map data.
    with open(vector_map_path, "rb") as f:
        vec_map.ParseFromString(f.read())

    return vec_map


def load_kdtrees(
    kdtrees_path: Path,
) -> Dict[MapElementType, map_kdtree.MapElementKDTree]:
    if not kdtrees_path.exists():
        raise ValueError(f"{kdtrees_path} does not exist!")

    with open(kdtrees_path, "rb") as f:
        kdtrees: Dict[MapElementType, map_kdtree.MapElementKDTree] = dill.load(f)

    return kdtrees


def obtain_lane_graph(scene, map_api, vector_map_params) -> LaneGraph:
    map_name = f"{scene.env_name}:{scene.location}"
    num_pts = vector_map_params.get("num_lane_pts", 30)
    vec_map = map_api.get_map(
        map_name,
        None,
        **vector_map_params,
    )
    infer_lane_connectivity = vector_map_params.get("infer_lane_connectivity", False)
    lane_ids = [lane.id for lane in vec_map.lanes]
    lane_centerlines = np.stack(
        [
            lane.center.interpolate(num_pts).points[:, [0, 1, 3]]
            for lane in vec_map.lanes
        ],
        0,
    )
    lane_left_edges = np.stack(
        [
            (
                lane.left_edge.interpolate(num_pts).points[:, :2]
                if lane.left_edge is not None
                else np.full(
                    [num_pts, 2], dtype=lane_centerlines.dtype, fill_value=np.nan
                )
            )
            for lane in vec_map.lanes
        ],
        0,
    )
    lane_right_edges = np.stack(
        [
            (
                lane.right_edge.interpolate(num_pts).points[:, :2]
                if lane.right_edge is not None
                else np.full(
                    [num_pts, 2], dtype=lane_centerlines.dtype, fill_value=np.nan
                )
            )
            for lane in vec_map.lanes
        ],
        0,
    )
    # Lane connectivity
    rough_dis_map = lane_centerlines.mean()
    lane_adj = np.zeros((len(lane_ids), len(lane_ids)), dtype=np.int8)

    for i, lane in enumerate(vec_map.lanes):
        try:
            for adj_lane_id in lane.next_lanes:
                lane_adj[i, lane_ids.index(adj_lane_id)] = LaneSegRelation.NEXT.value

            for adj_lane_id in lane.prev_lanes:
                lane_adj[i, lane_ids.index(adj_lane_id)] = LaneSegRelation.PREV.value

            for adj_lane_id in lane.adj_lanes_left:
                lane_adj[i, lane_ids.index(adj_lane_id)] = LaneSegRelation.LEFT.value

            for adj_lane_id in lane.adj_lanes_right:
                lane_adj[i, lane_ids.index(adj_lane_id)] = LaneSegRelation.RIGHT.value
        except:
            pass
    if infer_lane_connectivity:
        LAT_THRESHOLD = 4.5
        H_THRESHOLD = 0.1
        PTS_THRESHOLD = 3
        lane_center_pt = lane_centerlines[:, :, :2].mean(1)
        rough_dis_map = np.linalg.norm(
            lane_center_pt[:, None] - lane_center_pt[None], axis=-1
        )
        topk = 10
        topk_idx = np.argsort(rough_dis_map, axis=1)[:, 1 : topk + 1]
        infered_lane_adj = np.zeros((len(lane_ids), len(lane_ids)), dtype=np.int8)
        for i, lane in enumerate(vec_map.lanes):
            centerline = lane_centerlines[i]
            relevant_lanes = lane_centerlines[topk_idx[i]]
            dx, dy, dh = arr_utils.batch_proj(
                centerline.repeat(topk, 0), np.tile(relevant_lanes, (num_pts, 1, 1))
            )
            min_idx = np.argmin(np.abs(dx), axis=1)
            min_dx = np.take_along_axis(dx, min_idx[:, None], axis=1).reshape(
                num_pts, topk
            )
            min_dy = np.take_along_axis(dy, min_idx[:, None], axis=1).reshape(
                num_pts, topk
            )
            dh = dh.reshape(num_pts, topk)
            dx = dx.reshape(num_pts, topk, num_pts)

            # if one of the waypoints has a negative and positive dx, then it is contained in between the start and end of the adjacent lane
            longi_adj = np.logical_and(dx.min(2) < 0, dx.max(2) > 0)

            lat_left_adj = np.logical_and(min_dy > 0, min_dy < LAT_THRESHOLD)
            lat_right_adj = np.logical_and(min_dy < 0, min_dy > -LAT_THRESHOLD)
            heading_adj = np.abs(dh) < H_THRESHOLD

            left_adj_flag = (longi_adj * lat_left_adj * heading_adj).sum(
                0
            ) > PTS_THRESHOLD
            right_adj_flag = (longi_adj * lat_right_adj * heading_adj).sum(
                0
            ) > PTS_THRESHOLD
            infered_lane_adj[i, topk_idx[i][left_adj_flag]] = LaneSegRelation.LEFT.value
            infered_lane_adj[i, topk_idx[i][right_adj_flag]] = (
                LaneSegRelation.RIGHT.value
            )

        lane_adj = lane_adj + (lane_adj == 0) * infered_lane_adj
    return LaneGraph(
        lane_ids, lane_centerlines, lane_left_edges, lane_right_edges, lane_adj
    )


def load_rtrees(
    rtrees_path: Path,
) -> Optional[Dict[MapElementType, map_strtree.MapElementSTRTree]]:
    if not rtrees_path.exists():
        warnings.warn(
            (
                "Trying to load cached RTree encoding 2D Map elements, "
                f"but {rtrees_path} does not exist. Earlier versions of "
                "trajdata did not build and cache this RTree. If area queries "
                "are needed, please rebuild the map cache (see "
                "examples/preprocess_maps.py for an example of how to do this). "
                "Otherwise, please ignore this warning."
            ),
            UserWarning,
        )
        return None

    with open(rtrees_path, "rb") as f:
        rtrees: Dict[MapElementType, map_strtree.MapElementSTRTree] = dill.load(f)

    return rtrees
