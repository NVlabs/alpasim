# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Dict, Final, List

import l5kit.data.proto.road_network_pb2 as l5_pb2
import numpy as np
import pandas as pd
from l5kit.data import ChunkedDataset
from l5kit.data.map_api import InterpolationMethod, MapAPI
from l5kit.geometry import rotation33_as_yaw
from tqdm import tqdm

from trajdata.data_structures import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    Scene,
    VariableExtent,
)
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import PedCrosswalk, Polyline, RoadLane
from trajdata.utils import map_utils

LYFT_DT: Final[float] = 0.1


def agg_ego_data(lyft_obj: ChunkedDataset, scene: Scene) -> Agent:
    scene_frame_start = scene.data_access_info[0]
    scene_frame_end = scene.data_access_info[1]

    ego_translations = lyft_obj.frames[scene_frame_start:scene_frame_end][
        "ego_translation"
    ][:, :3]

    # Doing this prepending so that the first velocity isn't zero (rather it's just the first actual velocity duplicated)
    prepend_pos = ego_translations[0, :2] - (
        ego_translations[1, :2] - ego_translations[0, :2]
    )
    ego_velocities = (
        np.diff(
            ego_translations[:, :2], axis=0, prepend=np.expand_dims(prepend_pos, axis=0)
        )
        / LYFT_DT
    )

    # Doing this prepending so that the first acceleration isn't zero (rather it's just the first actual acceleration duplicated)
    prepend_vel = ego_velocities[0] - (ego_velocities[1] - ego_velocities[0])
    ego_accelerations = (
        np.diff(ego_velocities, axis=0, prepend=np.expand_dims(prepend_vel, axis=0))
        / LYFT_DT
    )

    ego_rotations = lyft_obj.frames[scene_frame_start:scene_frame_end]["ego_rotation"]
    ego_yaws = np.array(
        [rotation33_as_yaw(ego_rotations[i]) for i in range(scene.length_timesteps)]
    )

    ego_extents = FixedExtent(length=4.869, width=1.852, height=1.476).get_extents(
        scene_frame_start, scene_frame_end - 1
    )
    extent_cols: List[str] = ["length", "width", "height"]

    ego_data_np = np.concatenate(
        [
            ego_translations,
            ego_velocities,
            ego_accelerations,
            np.expand_dims(ego_yaws, axis=1),
            ego_extents,
        ],
        axis=1,
    )
    ego_data_df = pd.DataFrame(
        ego_data_np,
        columns=["x", "y", "z", "vx", "vy", "ax", "ay", "heading"] + extent_cols,
        index=pd.MultiIndex.from_tuples(
            [("ego", idx) for idx in range(ego_data_np.shape[0])],
            names=["agent_id", "scene_ts"],
        ),
    )

    ego_metadata = AgentMetadata(
        name="ego",
        agent_type=AgentType.VEHICLE,
        first_timestep=0,
        last_timestep=ego_data_np.shape[0] - 1,
        extent=VariableExtent(),
    )
    return Agent(
        metadata=ego_metadata,
        data=ego_data_df,
    )


def lyft_type_to_unified_type(lyft_type: int) -> AgentType:
    # TODO(bivanovic): Currently not handling TRAM or ANIMAL.
    if lyft_type in [0, 1, 2, 16]:
        return AgentType.UNKNOWN
    elif lyft_type in [3, 4, 6, 7, 8, 9]:
        return AgentType.VEHICLE
    elif lyft_type in [10, 12]:
        return AgentType.BICYCLE
    elif lyft_type in [11, 13]:
        return AgentType.MOTORCYCLE
    elif lyft_type == 14:
        return AgentType.PEDESTRIAN


def populate_vector_map(vector_map: VectorMap, mapAPI: MapAPI) -> None:
    maximum_bound: np.ndarray = np.full((3,), np.nan)
    minimum_bound: np.ndarray = np.full((3,), np.nan)
    for l5_element in tqdm(mapAPI.elements, desc="Creating Vectorized Map"):
        if mapAPI.is_lane(l5_element):
            l5_element_id: str = mapAPI.id_as_str(l5_element.id)
            l5_lane: l5_pb2.Lane = l5_element.element.lane

            lane_dict = mapAPI.get_lane_coords(l5_element_id)
            left_pts = lane_dict["xyz_left"]
            right_pts = lane_dict["xyz_right"]

            # Ensuring the left and right bounds have the same numbers of points.
            if len(left_pts) < len(right_pts):
                left_pts = mapAPI.interpolate(
                    left_pts, len(right_pts), InterpolationMethod.INTER_ENSURE_LEN
                )
            elif len(right_pts) < len(left_pts):
                right_pts = mapAPI.interpolate(
                    right_pts, len(left_pts), InterpolationMethod.INTER_ENSURE_LEN
                )

            midlane_pts: np.ndarray = (left_pts + right_pts) / 2

            # Computing the maximum and minimum map coordinates.
            maximum_bound = np.fmax(maximum_bound, left_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, left_pts.min(axis=0))

            maximum_bound = np.fmax(maximum_bound, right_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, right_pts.min(axis=0))

            maximum_bound = np.fmax(maximum_bound, midlane_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, midlane_pts.min(axis=0))

            # Adding the element to the map.
            new_lane = RoadLane(
                id=l5_element_id,
                center=Polyline(midlane_pts),
                left_edge=Polyline(left_pts),
                right_edge=Polyline(right_pts),
            )

            new_lane.next_lanes.update(
                [mapAPI.id_as_str(gid) for gid in l5_lane.lanes_ahead]
            )

            left_lane_change_id: str = mapAPI.id_as_str(
                l5_lane.adjacent_lane_change_left
            )
            if left_lane_change_id:
                new_lane.adj_lanes_left.add(left_lane_change_id)

            right_lane_change_id: str = mapAPI.id_as_str(
                l5_lane.adjacent_lane_change_right
            )
            if right_lane_change_id:
                new_lane.adj_lanes_right.add(right_lane_change_id)

            vector_map.add_map_element(new_lane)

        if mapAPI.is_crosswalk(l5_element):
            l5_element_id: str = mapAPI.id_as_str(l5_element.id)
            crosswalk_pts: np.ndarray = mapAPI.get_crosswalk_coords(l5_element_id)[
                "xyz"
            ]

            # Computing the maximum and minimum map coordinates.
            maximum_bound = np.fmax(maximum_bound, crosswalk_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, crosswalk_pts.min(axis=0))

            vector_map.add_map_element(
                PedCrosswalk(id=l5_element_id, polygon=Polyline(crosswalk_pts))
            )

    # Setting the map bounds.
    # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vector_map.extent = np.concatenate((minimum_bound, maximum_bound))
