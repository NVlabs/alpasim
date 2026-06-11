# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
from pathlib import Path
from subprocess import check_call, check_output
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from intervaltree import Interval, IntervalTree
from tqdm import tqdm
from waymo_open_dataset.protos import map_pb2 as waymo_map_pb2
from waymo_open_dataset.protos import scenario_pb2

from trajdata.maps import TrafficLightStatus, VectorMap
from trajdata.maps.vec_map import split_lane_segments
from trajdata.maps.vec_map_elements import PedCrosswalk, Polyline, RoadEdge, RoadLane

WAYMO_DT: Final[float] = 0.1
WAYMO_DATASET_NAMES = [
    "testing",
    "testing_interactive",
    "training",
    "training_20s",
    "validation",
    "validation_interactive",
]

TRAIN_SCENE_LENGTH = 91
VAL_SCENE_LENGTH = 91
TEST_SCENE_LENGTH = 11
TRAIN_20S_SCENE_LENGTH = 201

GREEN = [
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_GO,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_GO,
]
RED = [
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_CAUTION,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_ARROW_STOP,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_STOP,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_CAUTION,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_STOP,
    waymo_map_pb2.TrafficSignalLaneState.State.LANE_STATE_FLASHING_CAUTION,
]

from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)


class WaymoScenarios:
    def __init__(
        self,
        dataset_name: str,
        source_dir: Path,
        download: bool = False,
        split: bool = False,
    ):
        if dataset_name not in WAYMO_DATASET_NAMES:
            raise RuntimeError(
                "Wrong dataset name. Please choose name from "
                + str(WAYMO_DATASET_NAMES)
            )

        self.name = dataset_name
        self.source_dir = source_dir
        if dataset_name in ["training"]:
            self.scene_length = TRAIN_SCENE_LENGTH
        elif dataset_name in ["validation", "validation_interactive"]:
            self.scene_length = VAL_SCENE_LENGTH
        elif dataset_name in ["testing", "testing_interactive"]:
            self.scene_length = TEST_SCENE_LENGTH
        elif dataset_name in ["training_20s"]:
            self.scene_length = TRAIN_20S_SCENE_LENGTH

        if download:
            self.download_dataset()

        split_path = self.source_dir / (self.name + "_splitted")
        if split or not split_path.is_dir():
            self.split_scenarios()
        else:
            self.num_scenarios = len(os.listdir(split_path))

    def download_dataset(self) -> None:
        # check_call("snap install google-cloud-sdk --classic".split())
        gsutil = check_output(["which", "gsutil"])
        download_cmd = (
            str(gsutil.decode("utf-8"))
            + "-m cp -r gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/"
            + str(self.name)
            + " "
            + str(self.source_dir)
        ).split()
        check_call(download_cmd)

    def split_scenarios(
        self, num_parallel_reads: int = 20, verbose: bool = True
    ) -> None:
        source_it: Path = (self.source_dir / self.name).glob("*")
        file_names: List[str] = [str(file_name) for file_name in source_it]
        if verbose:
            print("Loading tfrecord files...")
        dataset = tf.data.TFRecordDataset(
            file_names, compression_type="", num_parallel_reads=num_parallel_reads
        )

        if verbose:
            print("Splitting tfrecords...")

        splitted_dir: Path = self.source_dir / f"{self.name}_splitted"
        if not splitted_dir.exists():
            splitted_dir.mkdir(parents=True)

        scenario_num: int = 0
        for data in tqdm(dataset):
            file_name: Path = (
                splitted_dir / f"{self.name}_splitted_{scenario_num}.tfrecords"
            )
            with tf.io.TFRecordWriter(str(file_name)) as file_writer:
                file_writer.write(data.numpy())

            scenario_num += 1

        self.num_scenarios = scenario_num
        if verbose:
            print(
                str(self.num_scenarios)
                + " scenarios from "
                + str(len(file_names))
                + " file(s) have been split into "
                + str(self.num_scenarios)
                + " files."
            )

    def get_filename(self, data_idx):
        return (
            self.source_dir
            / f"{self.name}_splitted"
            / f"{self.name}_splitted_{data_idx}.tfrecords"
        )


def extract_vectorized(
    map_features: List[waymo_map_pb2.MapFeature],
    map_name: str,
    verbose: bool = False,
    max_lane_length: Optional[float] = None,
) -> VectorMap:
    vec_map = VectorMap(map_id=map_name)

    max_pt = np.array([np.nan, np.nan, np.nan])
    min_pt = np.array([np.nan, np.nan, np.nan])

    boundaries: Dict[int, Polyline] = {}
    for map_feature in tqdm(
        map_features, desc="Extracting road boundaries", disable=not verbose
    ):
        if map_feature.WhichOneof("feature_data") == "road_line":
            boundaries[map_feature.id] = Polyline(
                np.array([(pt.x, pt.y, pt.z) for pt in map_feature.road_line.polyline])
            )
        elif map_feature.WhichOneof("feature_data") == "road_edge":
            boundaries[map_feature.id] = Polyline(
                np.array([(pt.x, pt.y, pt.z) for pt in map_feature.road_edge.polyline])
            )

    lane_id_remap_dict = {}
    for map_feature in tqdm(
        map_features, desc="Extracting map elements", disable=not verbose
    ):
        if map_feature.WhichOneof("feature_data") == "lane":
            if len(map_feature.lane.polyline) == 1:
                # TODO: Why does Waymo have single-point polylines that
                # aren't interpolating between others??
                continue

            road_lanes, modified_lane_ids = translate_lane(
                map_feature, boundaries, max_lane_length=max_lane_length
            )
            if modified_lane_ids:
                lane_id_remap_dict.update(modified_lane_ids)

            for road_lane in road_lanes:
                vec_map.add_map_element(road_lane)

                max_pt = np.fmax(max_pt, road_lane.center.xyz.max(axis=0))
                min_pt = np.fmin(min_pt, road_lane.center.xyz.min(axis=0))

                if road_lane.left_edge:
                    max_pt = np.fmax(max_pt, road_lane.left_edge.xyz.max(axis=0))
                    min_pt = np.fmin(min_pt, road_lane.left_edge.xyz.min(axis=0))

                if road_lane.right_edge:
                    max_pt = np.fmax(max_pt, road_lane.right_edge.xyz.max(axis=0))
                    min_pt = np.fmin(min_pt, road_lane.right_edge.xyz.min(axis=0))

        elif map_feature.WhichOneof("feature_data") == "crosswalk":
            crosswalk = PedCrosswalk(
                id=str(map_feature.id),
                polygon=Polyline(
                    np.array(
                        [(pt.x, pt.y, pt.z) for pt in map_feature.crosswalk.polygon]
                    )
                ),
            )
            vec_map.add_map_element(crosswalk)

            max_pt = np.fmax(max_pt, crosswalk.polygon.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, crosswalk.polygon.xyz.min(axis=0))

        elif map_feature.WhichOneof("feature_data") == "road_edge":
            if len(map_feature.road_edge.polyline) == 1:
                continue
            road_edge = RoadEdge(
                id=str(map_feature.id),
                polyline=Polyline(
                    np.array(
                        [(pt.x, pt.y, pt.z) for pt in map_feature.road_edge.polyline]
                    )
                ),
            )
            vec_map.add_map_element(road_edge)
            max_pt = np.fmax(max_pt, road_edge.polyline.xyz.max(axis=0))
            min_pt = np.fmin(min_pt, road_edge.polyline.xyz.min(axis=0))

        else:
            continue

    for elem in vec_map.iter_elems():
        if not isinstance(elem, RoadLane):
            continue

        to_remove = set()
        to_add = set()
        for l_id in elem.adj_lanes_left:
            if l_id in lane_id_remap_dict:
                # Remove the original lanes, replace them with our chunked versions.
                to_remove.add(l_id)
                to_add.update(lane_id_remap_dict[l_id])

        elem.adj_lanes_left -= to_remove
        elem.adj_lanes_left |= to_add

        to_remove = set()
        to_add = set()
        for l_id in elem.adj_lanes_right:
            if l_id in lane_id_remap_dict:
                # Remove the original lanes, replace them with our chunked versions.
                to_remove.add(l_id)
                to_add.update(lane_id_remap_dict[l_id])

        elem.adj_lanes_right -= to_remove
        elem.adj_lanes_right |= to_add

        to_remove = set()
        to_add = set()
        for l_id in elem.prev_lanes:
            if l_id in lane_id_remap_dict:
                # Remove the original prev lanes, replace them with
                # the tail of our equivalent chunked version.
                to_remove.add(l_id)
                to_add.add(lane_id_remap_dict[l_id][-1])

        elem.prev_lanes -= to_remove
        elem.prev_lanes |= to_add

        to_remove = set()
        to_add = set()
        for l_id in elem.next_lanes:
            if l_id in lane_id_remap_dict:
                # Remove the original prev lanes, replace them with
                # the first of our equivalent chunked version.
                to_remove.add(l_id)
                to_add.add(lane_id_remap_dict[l_id][0])

        elem.next_lanes -= to_remove
        elem.next_lanes |= to_add

    # Setting the map bounds.
    # vec_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vec_map.extent = np.concatenate((min_pt, max_pt))

    return vec_map


def translate_agent_type(agent_type):
    if agent_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
        return AgentType.VEHICLE
    elif agent_type == scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN:
        return AgentType.PEDESTRIAN
    elif agent_type == scenario_pb2.Track.ObjectType.TYPE_CYCLIST:
        return AgentType.BICYCLE
    elif agent_type == scenario_pb2.Track.ObjectType.OTHER:
        return AgentType.UNKNOWN
    return -1


def is_full_boundary(lane_boundaries, num_lane_indices: int) -> bool:
    """Returns True if a given boundary is connected (there are no gaps)
    and every lane center index has a corresponding boundary point.

    Returns:
        bool
    """
    covers_all: bool = lane_boundaries[0].lane_start_index == 0 and lane_boundaries[
        0
    ].lane_end_index == (num_lane_indices - 1)
    for idx in range(1, len(lane_boundaries)):
        if (
            lane_boundaries[idx].lane_start_index
            != lane_boundaries[idx - 1].lane_end_index + 1
        ):
            covers_all = False
            break

    return covers_all


def _merge_interval_data(data1: str, data2: str) -> str:
    if data1 == data2 == "none":
        return "none"

    if data1 == "none" and data2 != "none":
        return data2

    if data1 != "none" and data2 == "none":
        return data1

    if data1 != "none" and data2 != "none":
        return "both"


def split_lane_into_chunks(
    lane: waymo_map_pb2.LaneCenter, boundaries: Dict[int, Polyline]
) -> List[Tuple[Polyline, Optional[Polyline], Optional[Polyline]]]:
    boundary_intervals = IntervalTree.from_tuples(
        [
            (b.lane_start_index, b.lane_end_index + 1, "left")
            for b in lane.left_boundaries
        ]
        + [
            (b.lane_start_index, b.lane_end_index + 1, "right")
            for b in lane.right_boundaries
        ]
        + [(0, len(lane.polyline), "none")]
    )

    boundary_intervals.split_overlaps()

    boundary_intervals.merge_equals(data_reducer=_merge_interval_data)
    intervals: List[Interval] = sorted(boundary_intervals)

    if len(intervals) > 1:
        merged_intervals: List[Interval] = [intervals.pop(0)]
        while intervals:
            last_interval: Interval = merged_intervals[-1]
            curr_interval: Interval = intervals.pop(0)

            if last_interval.end != curr_interval.begin:
                raise ValueError("Non-consecutive intervals in merging!")

            if last_interval.data == curr_interval.data:
                # Simple merging of same-data neighbors.
                merged_intervals[-1] = Interval(
                    last_interval.begin,
                    curr_interval.end,
                    last_interval.data,
                )
            elif (
                last_interval.end - last_interval.begin == 1
                or curr_interval.end - curr_interval.begin == 1
            ):
                # Trying to remove 1-length chunks by merging them with neighbors.
                data_to_keep: str = (
                    curr_interval.data
                    if curr_interval.end - curr_interval.begin
                    > last_interval.end - last_interval.begin
                    else last_interval.data
                )

                merged_intervals[-1] = Interval(
                    last_interval.begin,
                    curr_interval.end,
                    data_to_keep,
                )
            else:
                merged_intervals.append(curr_interval)

        intervals = merged_intervals

    left_boundary_tree = IntervalTree.from_tuples(
        [
            (b.lane_start_index, b.lane_end_index + 1, b.boundary_feature_id)
            for b in lane.left_boundaries
        ]
    )
    right_boundary_tree = IntervalTree.from_tuples(
        [
            (b.lane_start_index, b.lane_end_index + 1, b.boundary_feature_id)
            for b in lane.right_boundaries
        ]
    )
    lane_chunk_data: List[Tuple[Polyline, Optional[Polyline], Optional[Polyline]]] = []
    for interval in intervals:
        center_chunk = Polyline(
            np.array(
                [
                    (point.x, point.y, point.z)
                    for point in lane.polyline[interval.begin : interval.end]
                ]
            )
        )
        if interval.data == "none":
            lane_chunk_data.append((center_chunk, None, None))
        elif interval.data == "left":
            left_chunk = subselect_boundary(
                boundaries, center_chunk, interval, left_boundary_tree
            )
            lane_chunk_data.append((center_chunk, left_chunk, None))
        elif interval.data == "right":
            right_chunk = subselect_boundary(
                boundaries, center_chunk, interval, right_boundary_tree
            )
            lane_chunk_data.append((center_chunk, None, right_chunk))
        elif interval.data == "both":
            left_chunk = subselect_boundary(
                boundaries, center_chunk, interval, left_boundary_tree
            )
            right_chunk = subselect_boundary(
                boundaries, center_chunk, interval, right_boundary_tree
            )
            lane_chunk_data.append((center_chunk, left_chunk, right_chunk))
        else:
            raise ValueError()

    return lane_chunk_data


def subselect_boundary(
    boundaries: Dict[int, Polyline],
    lane_center: Polyline,
    chunk_interval: Interval,
    boundary_tree: IntervalTree,
) -> Polyline:
    relevant_boundaries: List[Interval] = sorted(
        boundary_tree[chunk_interval.begin : chunk_interval.end]
    )

    if (
        len(relevant_boundaries) == 1
        and relevant_boundaries[0].begin == chunk_interval.begin
        and relevant_boundaries[0].end == chunk_interval.end
    ):
        # Return immediately for an exact match.
        return boundaries[relevant_boundaries[0].data]

    polyline_pts: List[Polyline] = []
    for boundary_interval in relevant_boundaries:
        # Below we are trying to find relevant boundary regions to the current lane chunk center
        # by projecting the boundary onto the lane and seeing where it stops following the center.
        # After that point, the projections will cease to change
        # (they will typically all be the last point of the center line).
        boundary = boundaries[boundary_interval.data]

        if boundary.points.shape[0] == 1:
            polyline_pts.append(boundary.points)
            continue

        proj = lane_center.project_onto(boundary.points)
        local_diffs = np.diff(proj, axis=0, append=proj[[-1]] - proj[[-2]])

        nonzero_mask = (local_diffs != 0.0).any(axis=1)
        nonzero_idxs = np.nonzero(nonzero_mask)[0]
        marker_idx = np.nonzero(np.ediff1d(nonzero_idxs, to_begin=[2]) > 1)[0]

        # TODO(bivanovic): Only taking the first group. Adding 1 to the
        # first ends because it otherwise ignores the first element of
        # the repeated value group.
        start = np.minimum.reduceat(nonzero_idxs, marker_idx)[0]
        end = np.maximum.reduceat(nonzero_idxs, marker_idx)[0] + 1

        # TODO(bivanovic): This may or may not end up being a problem, but
        # polyline_pts[0][-1] and polyline_pts[1][0] can be exactly identical.
        polyline_pts.append(boundary.points[start : end + 1])

    return Polyline(points=np.concatenate(polyline_pts, axis=0))


def translate_lane(
    map_feature: waymo_map_pb2.MapFeature,
    boundaries: Dict[int, Polyline],
    max_lane_length: Optional[float] = None,
) -> Tuple[RoadLane, Optional[Dict[int, List[bytes]]]]:
    lane: waymo_map_pb2.LaneCenter = map_feature.lane

    if lane.left_boundaries or lane.right_boundaries:
        # Waymo lane boundaries are... complicated. See
        # https://github.com/waymo-research/waymo-open-dataset/issues/389
        # for more information. For now, we split lanes into chunks which
        # have consistent lane boundaries (either both left and right,
        # one of them, or none).
        lane_chunks = split_lane_into_chunks(lane, boundaries)
        road_lanes: List[RoadLane] = []
        new_ids: List[bytes] = []

        for idx, (lane_center, left_edge, right_edge) in enumerate(lane_chunks):

            road_lane = RoadLane(
                id=(
                    f"{map_feature.id}_{idx}"
                    if len(lane_chunks) > 1
                    else str(map_feature.id)
                ),
                center=lane_center,
                left_edge=left_edge,
                right_edge=right_edge,
            )
            if max_lane_length is not None:
                split_lanes = split_lane_segments(road_lane, max_len=max_lane_length)
                if idx == 0:
                    split_lanes[0].prev_lanes.update(
                        [str(eid) for eid in lane.entry_lanes]
                    )
                else:
                    split_lanes[0].prev_lanes.add(f"{map_feature.id}_{idx-1}")

                if idx == len(lane_chunks) - 1:
                    split_lanes[-1].next_lanes.update(
                        [str(eid) for eid in lane.exit_lanes]
                    )
                else:
                    split_lanes[-1].next_lanes.add(f"{map_feature.id}_{idx+1}")

                for split_lane in split_lanes:
                    # We'll take care of reassigning these IDs to the chunked versions later.
                    for neighbor in lane.left_neighbors:
                        split_lane.adj_lanes_left.add(str(neighbor.feature_id))

                    for neighbor in lane.right_neighbors:
                        split_lane.adj_lanes_right.add(str(neighbor.feature_id))
                    new_ids.append(split_lane.id)
                    road_lanes.append(split_lane)

            else:
                new_ids.append(road_lane.id)

                if idx == 0:
                    road_lane.prev_lanes.update([str(eid) for eid in lane.entry_lanes])
                else:
                    road_lane.prev_lanes.add(f"{map_feature.id}_{idx-1}")

                if idx == len(lane_chunks) - 1:
                    road_lane.next_lanes.update([str(eid) for eid in lane.exit_lanes])
                else:
                    road_lane.next_lanes.add(f"{map_feature.id}_{idx+1}")

                # We'll take care of reassigning these IDs to the chunked versions later.
                for neighbor in lane.left_neighbors:
                    road_lane.adj_lanes_left.add(str(neighbor.feature_id))

                for neighbor in lane.right_neighbors:
                    road_lane.adj_lanes_right.add(str(neighbor.feature_id))

                road_lanes.append(road_lane)

        if len(road_lanes) > 1:
            return road_lanes, {str(map_feature.id): new_ids}
        else:
            return road_lanes, None

    else:
        road_lane = RoadLane(
            id=str(map_feature.id),
            center=Polyline(np.array([(pt.x, pt.y, pt.z) for pt in lane.polyline])),
        )

        road_lane.prev_lanes.update([str(eid) for eid in lane.entry_lanes])
        road_lane.next_lanes.update([str(eid) for eid in lane.exit_lanes])

        for neighbor in lane.left_neighbors:
            road_lane.adj_lanes_left.add(str(neighbor.feature_id))

        for neighbor in lane.right_neighbors:
            road_lane.adj_lanes_right.add(str(neighbor.feature_id))

        return [road_lane], None


def extract_traffic_lights(
    dynamic_states: List[scenario_pb2.DynamicMapState],
) -> Dict[Tuple[str, int], TrafficLightStatus]:
    ret: Dict[Tuple[str, int], TrafficLightStatus] = {}
    for i, dynamic_state in enumerate(dynamic_states):
        for lane_state in dynamic_state.lane_states:
            ret[(str(lane_state.lane), i)] = translate_traffic_state(lane_state.state)

    return ret


def translate_traffic_state(
    state: waymo_map_pb2.TrafficSignalLaneState.State,
) -> TrafficLightStatus:
    # TODO(bivanovic): The traffic light type doesn't align between waymo and trajdata,
    # since trajdata's TrafficLightStatus does not include a yellow light yet.
    # For now, we set caution = red.
    if state in GREEN:
        return TrafficLightStatus.GREEN
    if state in RED:
        return TrafficLightStatus.RED
    return TrafficLightStatus.UNKNOWN


def interpolate_array(data: List) -> np.array:
    return pd.DataFrame(data).interpolate(limit_area="inside").to_numpy()


def sort_df_by_distance_to_ego_vehicle(
    df: pd.DataFrame,
    timestep: int,
    ego_name: str = "ego",
    keep_distance_column: bool = False,
):
    """Sort df by distance to ego vehicle at timestep."""
    ego_xy = tuple(df.loc["ego", timestep][["x", "y"]])

    def distance_to_ego_xy(x, y):
        return np.linalg.norm([x, y] - np.array(ego_xy))

    all_xy_positions = np.stack([df["x"], df["y"]], axis=-1)

    df["dist_to_ego"] = np.linalg.norm(all_xy_positions - np.array(ego_xy), axis=-1)
    only_t0 = (
        df.groupby(level="agent_id")
        .apply(
            lambda x: (
                x[x.index.get_level_values("scene_ts") == timestep]
                if timestep in x.index.get_level_values("scene_ts")
                else x.iloc[[0]]
            )
        )
        .droplevel(0)
    )
    only_t0_sorted = only_t0.sort_values(by="dist_to_ego")
    sorted_ids = list(only_t0_sorted.index.get_level_values("agent_id"))
    sorted_df = df.reindex(labels=sorted_ids, level=0)
    df.drop(columns="dist_to_ego", inplace=True)
    if not keep_distance_column:
        sorted_df.drop(columns="dist_to_ego", inplace=True)
    return sorted_df
