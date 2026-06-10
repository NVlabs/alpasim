# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from typing import Literal

import numpy as np
import shapely
import shapely.geometry
import shapely.ops
from alpasim_utils import geometry
from matplotlib import pyplot as plt
from shapely import plotting as shapely_plotting
from shapely.geometry.base import BaseGeometry

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.scorers.base import Scorer
from trajdata.maps import vec_map_elements


def _get_center_line_yaw_at_projection(
    center_line: shapely.LineString, point: shapely.Point, eps: float = 0.01
) -> float:
    distance_along_line = center_line.project(point)
    point_ahead_xy = np.array(
        center_line.interpolate(
            min(distance_along_line + eps, center_line.length)
        ).coords
    ).squeeze()
    point_behind_xy = np.array(
        center_line.interpolate(max(distance_along_line - eps, 0)).coords
    ).squeeze()
    delta_xy = point_ahead_xy - point_behind_xy
    return np.arctan2(delta_xy[1], delta_xy[0])


def _repair_polygonal_geometry(geom: BaseGeometry) -> BaseGeometry:
    """Return valid polygonal geometry for map lanes.

    Some lane boundaries produce slightly self-intersecting rings. GEOS
    operations such as unary_union can throw TopologyException on those
    polygons, breaking evaluation, so we repair if needed before further
    processing.
    """
    if geom.is_valid:
        return geom

    repaired = shapely.make_valid(geom)
    if repaired.geom_type == "GeometryCollection":
        polygonal_parts = [
            part
            for part in repaired.geoms
            if part.geom_type in {"Polygon", "MultiPolygon"}
        ]
        if polygonal_parts:
            repaired = shapely.ops.unary_union(polygonal_parts)
        else:
            repaired = geom.buffer(0)

    if not repaired.is_valid:
        repaired = repaired.buffer(0)

    return repaired


def _get_lane_polygon(
    lane: vec_map_elements.RoadLane, road_width_m: float = 3.7
) -> BaseGeometry:
    if lane.left_edge is not None:
        return _repair_polygonal_geometry(
            shapely.Polygon(
                np.concatenate(
                    [
                        lane.left_edge.points[..., :2],
                        np.flip(lane.right_edge.points[..., :2], axis=0),
                    ],
                    axis=0,
                )
            )
        )
    else:
        return _repair_polygonal_geometry(
            shapely.LineString(lane.center.points[..., :2]).buffer(road_width_m / 2)
        )


def _has_map_element_kdtree(
    simulation_result: SimulationResult,
    elem_type: vec_map_elements.MapElementType,
) -> bool:
    search_kdtrees = getattr(simulation_result.vec_map, "search_kdtrees", None)
    return search_kdtrees is not None and elem_type in search_kdtrees


def _road_areas_near_ego(
    simulation_result: SimulationResult,
    ego_xyzh: np.ndarray,
    query_dist_m: float = 15.0,
) -> list[vec_map_elements.RoadArea] | None:
    if not _has_map_element_kdtree(
        simulation_result, vec_map_elements.MapElementType.ROAD_AREA
    ):
        return None

    return simulation_result.vec_map.get_road_areas_within(
        ego_xyzh[..., :3], query_dist_m
    )


def _road_area_union(
    simulation_result: SimulationResult,
    road_areas: list[vec_map_elements.RoadArea],
) -> BaseGeometry:
    polygons = [
        simulation_result.vec_map.get_road_area_polygon_2d(area.id)
        for area in road_areas
    ]
    return _repair_polygonal_geometry(shapely.ops.unary_union(polygons))


def _is_offroad_using_road_area(
    simulation_result: SimulationResult,
    ego_xyzh: np.ndarray,
    ego_polygon: BaseGeometry,
) -> bool | None:
    road_areas = _road_areas_near_ego(simulation_result, ego_xyzh)
    if road_areas is None:
        return None
    if not road_areas:
        return True

    road_area_union = _road_area_union(simulation_result, road_areas)
    return not road_area_union.covers(ego_polygon)


def _compute_off_lane(
    simulation_result: SimulationResult,
    ts: int,
    ego_pose: geometry.Pose | None = None,
) -> dict[
    str,
    float | list[vec_map_elements.RoadLane] | list[shapely.LineString] | list[float],
]:
    if ego_pose is None:
        ego_pose = simulation_result.actor_trajectories["EGO"].interpolate_pose(ts)
    ego_xyzh = np.array([*ego_pose.vec3, ego_pose.yaw()])

    possible_current_lanes = simulation_result.vec_map.get_current_lane(
        ego_xyzh, max_dist=6, max_heading_error=2 * np.pi
    )

    ego_polygon = simulation_result.actor_polygons.get_polygon_for_agent_at_time(
        "EGO", ts
    )
    ego_yaw = simulation_result.actor_polygons.get_yaw_for_agent_at_time("EGO", ts)

    curr_lanes_containing_centroid = [
        lane
        for lane in possible_current_lanes
        if _get_lane_polygon(lane).contains(ego_polygon.centroid)
    ]

    curr_lanes_containing_polygon = [
        lane
        for lane in possible_current_lanes
        if _get_lane_polygon(lane).contains(ego_polygon)
    ]

    center_lines = [
        shapely.LineString(lane.center.points[..., :2])
        for lane in curr_lanes_containing_centroid
    ]

    center_line_yaws = [
        _get_center_line_yaw_at_projection(center_line, ego_polygon.centroid)
        for center_line in center_lines
    ]

    rel_ego_yaw_to_center_line = [
        center_line_yaw - ego_yaw for center_line_yaw in center_line_yaws
    ]

    return {
        "ego_xyzh": ego_xyzh,
        "ego_polygon": ego_polygon,
        "ego_yaw": ego_yaw,
        "possible_current_lanes": possible_current_lanes,
        "curr_lanes_containing_centroid": curr_lanes_containing_centroid,
        "center_lines": center_lines,
        "center_line_yaws": center_line_yaws,
        "rel_ego_yaw_to_center_line": rel_ego_yaw_to_center_line,
        "curr_lanes_containing_polygon": curr_lanes_containing_polygon,
    }


class OffRoadScorer(Scorer):
    """Road-based metrics.

    Adds the following metrics:
    * offroad: Whether the ego vehicle is off the road at any point in time.
        NOTE: Determining being offroad is not as straightforward. We use a
        simplified approach here and only either there is a lane that fully
        covers the ego-polygon, or if we're not touching the road-edge.
        That misses cases where we outside the road, but never touch the
        road-edge, e.g. because the vehicle already started off-road.
    * off_lane: Whether the ego vehicle is on a lane with opposite direction.
        NOTE: Not a very useful metric, lots of false positives, e.g. on
        intersections.

    """

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        offroad = []
        wrong_lane = []
        lane_boundary_dist = []

        ego_traj = simulation_result.actor_trajectories["EGO"]
        ego_poses = ego_traj.interpolate_poses_list(
            np.asarray(simulation_result.timestamps_us, dtype=np.uint64)
        )

        for ts, ego_pose in zip(
            simulation_result.timestamps_us, ego_poses, strict=True
        ):
            res = _compute_off_lane(simulation_result, ts, ego_pose=ego_pose)
            ego_polygon = res["ego_polygon"]
            possible_lanes = res["possible_current_lanes"]

            wrong_lane.append(
                # Heuristic value taken from Yulong's implementation
                any(np.abs(res["rel_ego_yaw_to_center_line"]) > np.pi * 2 / 3)
            )

            # Lane boundary distance: 0.0 when off-road, positive gap when on-road.
            lane_union = (
                shapely.ops.unary_union(
                    [_get_lane_polygon(lane) for lane in possible_lanes]
                )
                if possible_lanes
                else None
            )
            if lane_union is not None and lane_union.contains(ego_polygon):
                lane_boundary_dist.append(ego_polygon.distance(lane_union.boundary))
            else:
                lane_boundary_dist.append(0.0)

            # Check if we're on at least one lane fully containing the ego polygon
            if len(res["curr_lanes_containing_polygon"]) > 0:
                offroad.append(False)
                continue

            # When straddling two adjacent lanes (e.g. during a lane change
            # across a dashed line), no single lane fully contains the ego
            # polygon.  Check whether the union of all nearby lane polygons
            # covers it instead (see GitHub issue #61).
            if (
                lane_union is not None
                and len(possible_lanes) > 1
                and lane_union.contains(ego_polygon)
            ):
                offroad.append(False)
                continue

            if not _has_map_element_kdtree(
                simulation_result, vec_map_elements.MapElementType.ROAD_EDGE
            ):
                # The nuPlan dataset, for example, doesn't have road edges, but it does have road areas.
                road_area_offroad = _is_offroad_using_road_area(
                    simulation_result,
                    res["ego_xyzh"],
                    ego_polygon,
                )
                offroad.append(
                    False if road_area_offroad is None else road_area_offroad
                )
            else:
                # Check if we're too close to the road edge. This will still miss
                # offroad cases when we're far outside the road - but then either
                # we started offroad or we had to go offroad at some point.
                closest_road_edge_xy = simulation_result.vec_map.get_closest_road_edge(
                    xyz=res["ego_xyzh"][..., :3]
                ).polyline.xy

                distance = shapely.geometry.LineString(closest_road_edge_xy).distance(
                    ego_polygon
                )
                offroad.append(distance < 1e-3)
        return [
            MetricReturn(
                name="offroad",
                values=offroad,
                valid=[True] * len(offroad),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="wrong_lane",
                values=wrong_lane,
                valid=[True] * len(wrong_lane),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="min_distance_to_lane_boundary_m",
                values=lane_boundary_dist,
                valid=[True] * len(lane_boundary_dist),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MIN,
            ),
        ]

    def debug_plot(
        self,
        simulation_result: SimulationResult,
        ts: int,
        lanes_to_plot: Literal[
            "possible_current_lanes",
            "curr_lanes_containing_centroid",
            "curr_lanes_containing_polygon",
        ] = "possible_current_lanes",
    ) -> None:
        res = _compute_off_lane(simulation_result, ts)
        _, ax = plt.subplots()
        ax.set_aspect("equal")
        cm = plt.cm.tab10 if len(res[lanes_to_plot]) <= 10 else plt.cm.tab20
        for lane, color in zip(res[lanes_to_plot], cm.colors):
            # Generate a random color for each lane
            shapely_plotting.plot_polygon(_get_lane_polygon(lane), ax=ax, color=color)
        shapely_plotting.plot_polygon(res["ego_polygon"], ax=ax, color="black")

        # Draw arrow from ego centroid in direction of yaw
        arrow_length = 1.0
        dx = arrow_length * np.cos(res["ego_yaw"])
        dy = arrow_length * np.sin(res["ego_yaw"])
        ego_center = res["ego_polygon"].centroid
        ax.arrow(
            ego_center.x,
            ego_center.y,
            dx,
            dy,
            head_width=1,
            head_length=1,
            fc="k",
            ec="k",
        )
