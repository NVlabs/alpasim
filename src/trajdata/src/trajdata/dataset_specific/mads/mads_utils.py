# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
import os
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import (
    PedCrosswalk,
    Polyline,
    RoadEdge,
    RoadLane,
    TrafficSign,
    WaitLine,
)

MAX_POLYLINE_POINT_DIST = 2.0


def df_expand_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand json columns in a pandas DataFrame.
    """
    columns = df.keys()
    for key in columns:
        cap_key = (
            "".join([x.capitalize() for x in key.split("_")])
            if key not in ["key", "version"]
            else key
        )
        df = df.join(pd.json_normalize(df[key]).add_prefix(f"{cap_key}."))

    return df


def find_lane_polylines_parquet(
    df_lane: pd.DataFrame,
    df_lane_relation: pd.DataFrame,
    clip_id: str,
    df_wait_line: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    lanes_dict = {}
    for i in range(df_lane.shape[0]):
        lane_id = df_lane["key.map_id"][i]
        df_this_lane_relation = df_lane_relation[
            (df_lane_relation["key.clip_id"] == clip_id)
            & (df_lane_relation["Association.subjects"] == lane_id)
        ]
        prev_lane_df = df_this_lane_relation[
            (df_this_lane_relation["key.kind"] == "PREVIOUS_LANE")
        ]
        prev_lane_id = (
            list(prev_lane_df["Association.objects"].values[0])
            if prev_lane_df.shape[0] > 0
            else {}
        )

        next_lane_df = df_this_lane_relation[
            (df_this_lane_relation["key.kind"] == "NEXT_LANE")
        ]
        next_lane_id = (
            list(next_lane_df["Association.objects"].values[0])
            if next_lane_df.shape[0] > 0
            else {}
        )

        right_lane_df = df_this_lane_relation[
            (df_this_lane_relation["key.kind"] == "RIGHT_LANE")
        ]
        right_lane_id = (
            list(right_lane_df["Association.objects"].values[0])
            if right_lane_df.shape[0] > 0
            else {}
        )

        left_lane_df = df_this_lane_relation[
            (df_this_lane_relation["key.kind"] == "LEFT_LANE")
        ]
        left_lane_id = (
            list(left_lane_df["Association.objects"].values[0])
            if left_lane_df.shape[0] > 0
            else {}
        )

        traffic_sign_df = df_this_lane_relation[
            (df_this_lane_relation["key.kind"] == "SIGN_TO_LANE")
        ]
        traffic_sign_id = (
            list(traffic_sign_df["Association.objects"].values[0])
            if traffic_sign_df.shape[0] > 0
            else {}
        )

        wait_line_df = df_wait_line[
            (df_wait_line["key.clip_id"] == clip_id)
            & (df_wait_line["lane.map_id"] == lane_id)
        ]
        wait_line_id = (
            list(wait_line_df["key.map_id"].values) if wait_line_df.shape[0] > 0 else {}
        )

        if lane_id not in lanes_dict:
            lanes_dict[lane_id] = {}
        if "Lane.left_rail.x" in df_lane.keys():
            lanes_dict[lane_id]["left_rail"] = np.stack(
                [
                    df_lane["Lane.left_rail.x"][i],
                    df_lane["Lane.left_rail.y"][i],
                    df_lane["Lane.left_rail.z"][i],
                ],
                axis=1,
            )
            lanes_dict[lane_id]["right_rail"] = np.stack(
                [
                    df_lane["Lane.right_rail.x"][i],
                    df_lane["Lane.right_rail.y"][i],
                    df_lane["Lane.right_rail.z"][i],
                ],
                axis=1,
            )
        else:
            lanes_dict[lane_id]["left_rail"] = np.stack(
                [
                    [pt["x"] for pt in df_lane["Lane.left_rail"][i]],
                    [pt["y"] for pt in df_lane["Lane.left_rail"][i]],
                    [pt["z"] for pt in df_lane["Lane.left_rail"][i]],
                ],
                axis=1,
            )
            lanes_dict[lane_id]["right_rail"] = np.stack(
                [
                    [pt["x"] for pt in df_lane["Lane.right_rail"][i]],
                    [pt["y"] for pt in df_lane["Lane.right_rail"][i]],
                    [pt["z"] for pt in df_lane["Lane.right_rail"][i]],
                ],
                axis=1,
            )

        lanes_dict[lane_id]["next_lane"] = next_lane_id
        lanes_dict[lane_id]["prev_lane"] = prev_lane_id
        lanes_dict[lane_id]["left_lane"] = left_lane_id
        lanes_dict[lane_id]["right_lane"] = right_lane_id
        lanes_dict[lane_id]["traffic_sign"] = traffic_sign_id
        lanes_dict[lane_id]["wait_line"] = wait_line_id

    return lanes_dict


def find_road_edges_parquet(df_road_edge: pd.DataFrame, clip_id: str) -> Dict[str, Any]:
    road_edges_dict = {}
    for i in range(df_road_edge.shape[0]):
        road_edge_id = df_road_edge["key.map_id"][i]
        if road_edge_id not in road_edges_dict:
            if "RoadBoundary.location.x" in df_road_edge.keys():
                road_edges_dict[road_edge_id] = np.stack(
                    [
                        df_road_edge["RoadBoundary.location.x"][i],
                        df_road_edge["RoadBoundary.location.y"][i],
                        df_road_edge["RoadBoundary.location.z"][i],
                    ],
                    axis=1,
                )
            else:
                road_edges_dict[road_edge_id] = np.stack(
                    [
                        [pt["x"] for pt in df_road_edge["RoadBoundary.location"][i]],
                        [pt["y"] for pt in df_road_edge["RoadBoundary.location"][i]],
                        [pt["z"] for pt in df_road_edge["RoadBoundary.location"][i]],
                    ],
                    axis=1,
                )
    return road_edges_dict


def find_traffic_signs_parquet(
    df_traffic_sign: pd.DataFrame, clip_id: str
) -> Dict[str, Any]:
    """
    return:
        traffic_signs_dict: {
            traffic_sign_id: {
                position: np.array([x, y, z]), position of the traffic sign
                type: str, traffic sign type, e.g., stop sign
                }
        }
    """
    # TODO: including all traffic signs here which might be not linked by the lane
    traffic_signs_dict = {}
    for i in range(df_traffic_sign.shape[0]):
        traffic_sign_id = df_traffic_sign["key.map_id"][i]
        if traffic_sign_id not in traffic_signs_dict:
            traffic_signs_dict[traffic_sign_id] = {}
        traffic_signs_dict[traffic_sign_id]["position"] = np.array(
            [
                df_traffic_sign["TrafficSign.center.x"][i],
                df_traffic_sign["TrafficSign.center.y"][i],
                df_traffic_sign["TrafficSign.center.z"][i],
            ]
        )
        traffic_signs_dict[traffic_sign_id]["type"] = df_traffic_sign[
            "TrafficSign.category"
        ][i]
    return traffic_signs_dict


def find_wait_lines_parquet(df_wait_line: pd.DataFrame, clip_id: str) -> Dict[str, Any]:
    """
    return:
        wait_lines_dict: {
            wait_line_id: {
                location: np.array([[x1, y1, z1], [x2, y2, z2], ...]), location of the wait line
                category: str, wait line category, e.g., Yield/Stop
                implicit: bool, unclear what this means, need to check with clipgt team
                }
        }
    """
    wait_lines_dict = {}
    for i in range(df_wait_line.shape[0]):
        wait_line_id = df_wait_line["key.map_id"][i]
        if wait_line_id not in wait_lines_dict:
            wait_lines_dict[wait_line_id] = {}
        if "WaitLine.location.x" in df_wait_line.keys():
            wait_lines_dict[wait_line_id]["location"] = np.stack(
                [
                    df_wait_line["WaitLine.location.x"][i],
                    df_wait_line["WaitLine.location.y"][i],
                    df_wait_line["WaitLine.location.z"][i],
                ],
                axis=1,
            )
        else:
            wait_lines_dict[wait_line_id]["location"] = np.stack(
                [
                    [pt["x"] for pt in df_wait_line["WaitLine.location"][i]],
                    [pt["y"] for pt in df_wait_line["WaitLine.location"][i]],
                    [pt["z"] for pt in df_wait_line["WaitLine.location"][i]],
                ],
                axis=1,
            )
        wait_lines_dict[wait_line_id]["category"] = df_wait_line["WaitLine.category"][
            i
        ]  # Yield/Stop
        wait_lines_dict[wait_line_id]["implicit"] = df_wait_line[
            "WaitLine.is_implicit"
        ][
            i
        ]  # TODO: need to clarify what this specifically refers to
    return wait_lines_dict


def interpolate_points(list1, list2):
    # Determine which list is shorter
    if len(list1) > len(list2):
        longer, shorter = list1, list2
    else:
        longer, shorter = list2, list1

    # Extract x, y, z coordinates
    x, y, z = zip(*shorter)
    x_long, y_long, z_long = zip(*longer)

    # Create an array of indices for interpolation
    interp_indices = np.linspace(0, len(shorter) - 1, num=len(longer))

    # Interpolate x, y, z
    x_interp = interp1d(range(len(x)), x, kind="linear")(interp_indices)
    y_interp = interp1d(range(len(y)), y, kind="linear")(interp_indices)
    z_interp = interp1d(range(len(z)), z, kind="linear")(interp_indices)

    # Combine interpolated coordinates
    interpolated_points = list(zip(x_interp, y_interp, z_interp))

    # Return the original longer list and the new interpolated list
    return longer, interpolated_points


def populate_vector_map(vector_map: VectorMap, map_root) -> None:
    # populate vector map from mads parquet files
    # map label schema: https://docs.nvda.ai/ndas/avdnn/latest/reference/maglev/data/clip/reference/schemas/labels.html?#map-derived-labels

    maximum_bound: np.ndarray = np.full((3,), np.nan)
    minimum_bound: np.ndarray = np.full((3,), np.nan)
    df_lane = df_expand_json(pd.read_parquet(os.path.join(map_root, "lane.parquet")))
    df_lane_relation = df_expand_json(
        pd.read_parquet(os.path.join(map_root, "association.parquet"))
    )
    df_road_edge = df_expand_json(
        pd.read_parquet(os.path.join(map_root, "road_boundary.parquet"))
    )
    df_meta = df_expand_json(pd.read_parquet(os.path.join(map_root, "clip.parquet")))
    df_traffic_sign = df_expand_json(
        pd.read_parquet(os.path.join(map_root, "traffic_sign.parquet"))
    )
    # TODO: invalid traffic light data from mads
    # df_traffic_light = pd.read_parquet(os.path.join(map_root, "traffic_light.parquet"))
    df_wait_line = df_expand_json(
        pd.read_parquet(os.path.join(map_root, "wait_line.parquet"))
    )
    # wait_line id is formatted as {wait_line_id}-{lane_id}
    df_wait_line["lane.map_id"] = df_wait_line["key.map_id"].map(
        lambda x: x.split("-")[1]
    )
    clip_id = df_meta["key.clip_id"][0]
    all_lanes_dict = find_lane_polylines_parquet(
        df_lane, df_lane_relation, clip_id, df_wait_line
    )
    all_road_edges_dict = find_road_edges_parquet(df_road_edge, clip_id)
    all_traffic_signs_dict = find_traffic_signs_parquet(df_traffic_sign, clip_id)
    all_wait_lines_dict = find_wait_lines_parquet(df_wait_line, clip_id)
    del df_lane, df_lane_relation, df_meta, df_road_edge

    if not all_lanes_dict:
        print("No valid data available in map file")
        return

    for lane_id, lane_info_dict in tqdm(
        all_lanes_dict.items(), desc="Creating Vectorized Map"
    ):
        left_polyline = np.array(lane_info_dict["left_rail"])
        right_polyline = np.array(lane_info_dict["right_rail"])

        midlane_pts: np.ndarray = (left_polyline + right_polyline) / 2

        # Computing the maximum and minimum map coordinates.
        maximum_bound = np.fmax(maximum_bound, left_polyline.max(axis=0))
        minimum_bound = np.fmin(minimum_bound, left_polyline.min(axis=0))

        maximum_bound = np.fmax(maximum_bound, right_polyline.max(axis=0))
        minimum_bound = np.fmin(minimum_bound, right_polyline.min(axis=0))

        maximum_bound = np.fmax(maximum_bound, midlane_pts.max(axis=0))
        minimum_bound = np.fmin(minimum_bound, midlane_pts.min(axis=0))

        new_lane = RoadLane(
            id=lane_id,
            center=Polyline(midlane_pts).interpolate(max_dist=MAX_POLYLINE_POINT_DIST),
            left_edge=Polyline(left_polyline).interpolate(
                max_dist=MAX_POLYLINE_POINT_DIST
            ),
            right_edge=Polyline(right_polyline).interpolate(
                max_dist=MAX_POLYLINE_POINT_DIST
            ),
            next_lanes=lane_info_dict["next_lane"],
            adj_lanes_left=lane_info_dict["left_lane"],
            adj_lanes_right=lane_info_dict["right_lane"],
            prev_lanes=lane_info_dict["prev_lane"],
            traffic_sign_ids=lane_info_dict["traffic_sign"],
            wait_line_ids=lane_info_dict["wait_line"],
        )
        vector_map.add_map_element(new_lane)

    for road_edge_id, road_edge_pts in tqdm(
        all_road_edges_dict.items(), desc="Creating Road Edges"
    ):
        maximum_bound = np.fmax(maximum_bound, road_edge_pts.max(axis=0))
        minimum_bound = np.fmin(minimum_bound, road_edge_pts.min(axis=0))
        new_road_edge = RoadEdge(
            id=road_edge_id,
            polyline=Polyline(road_edge_pts).interpolate(
                max_dist=MAX_POLYLINE_POINT_DIST
            ),
        )

        vector_map.add_map_element(new_road_edge)

    for traffic_sign_id, traffic_sign_info_dict in tqdm(
        all_traffic_signs_dict.items(), desc="Creating Traffic Signs"
    ):
        # TODO better error handling for invalid traffic sign data
        if traffic_sign_id is None:
            continue
        new_traffic_sign = TrafficSign(
            id=traffic_sign_id,
            position=traffic_sign_info_dict["position"],
            sign_type=traffic_sign_info_dict["type"],
        )
        vector_map.add_map_element(new_traffic_sign)

    for wait_line_id, wait_line_info_dict in tqdm(
        all_wait_lines_dict.items(), desc="Creating Wait Lines"
    ):
        # TODO better error handling for invalid wait line data
        if wait_line_id is None:
            continue
        new_wait_line = WaitLine(
            id=wait_line_id,
            polyline=Polyline(
                wait_line_info_dict["location"]
            ),  # yulongc: do we need interpolate here?
            wait_line_type=wait_line_info_dict["category"],
            is_implicit=wait_line_info_dict["implicit"],
        )
        vector_map.add_map_element(new_wait_line)

    # Setting the map bounds.
    # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vector_map.extent = np.concatenate((minimum_bound, maximum_bound))
