# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import dataclasses
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from av2.datasets.motion_forecasting.data_schema import (
    ArgoverseScenario,
    ObjectType,
    Track,
)
from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)
from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    _ESTIMATED_CYCLIST_LENGTH_M,
    _ESTIMATED_CYCLIST_WIDTH_M,
    _ESTIMATED_VEHICLE_LENGTH_M,
    _ESTIMATED_VEHICLE_WIDTH_M,
)
from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap

from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import PedCrosswalk, Polyline, RoadArea, RoadLane

AV2_SPLITS = ("train", "val", "test")
DELIM = "_"
T_Split = Literal["train", "val", "test"]


# {ObjectType: (AgentType, length, width, height)}.
# Uses av2 constants where possible.
OBJECT_TYPE_DATA: Dict[str, Tuple[AgentType, float, float, float]] = {
    ObjectType.VEHICLE: (
        AgentType.VEHICLE,
        _ESTIMATED_VEHICLE_LENGTH_M,
        _ESTIMATED_VEHICLE_WIDTH_M,
        2,
    ),
    ObjectType.PEDESTRIAN: (AgentType.PEDESTRIAN, 0.7, 0.7, 2),
    ObjectType.MOTORCYCLIST: (
        AgentType.MOTORCYCLE,
        _ESTIMATED_CYCLIST_LENGTH_M,
        _ESTIMATED_CYCLIST_WIDTH_M,
        2,
    ),
    ObjectType.CYCLIST: (
        AgentType.BICYCLE,
        _ESTIMATED_CYCLIST_LENGTH_M,
        _ESTIMATED_CYCLIST_WIDTH_M,
        2,
    ),
    ObjectType.BUS: (AgentType.VEHICLE, 9, 3, 4),
}


@dataclasses.dataclass
class Av2ScenarioIds:
    train: list[str]
    val: list[str]
    test: list[str]

    @staticmethod
    def create(dataset_path: Path) -> "Av2ScenarioIds":
        train = os.listdir(dataset_path / "train")
        val = os.listdir(dataset_path / "val")
        test = os.listdir(dataset_path / "test")
        return Av2ScenarioIds(train=train, val=val, test=test)

    @property
    def scene_split_map(self) -> Dict[str, T_Split]:
        """Compute a map of {scenario_name: split}."""
        return {
            _pack_av2_scenario_name(split, scenario_id): split
            for split, scenario_ids in dataclasses.asdict(self).items()
            for scenario_id in scenario_ids
        }


def scenario_name_to_split(scenario_name: str) -> T_Split:
    split, _ = _unpack_av2_scenario_name(scenario_name)
    return split


def _pack_av2_scenario_name(split: T_Split, scenario_id: str) -> str:
    return split + DELIM + scenario_id


def _unpack_av2_scenario_name(scenario_name: str) -> Tuple[T_Split, str]:
    return tuple(scenario_name.split(DELIM, maxsplit=1))


def _scenario_df_filename(scenario_id: str) -> str:
    return f"scenario_{scenario_id}.parquet"


def _scenario_map_filename(scenario_id: str) -> str:
    return f"log_map_archive_{scenario_id}.json"


class Av2Object:
    """Object for interfacing with Av2 data on disk."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.scenario_ids = Av2ScenarioIds.create(dataset_path)

    @property
    def scenario_names(self) -> list[str]:
        return list(self.scenario_ids.scene_split_map)

    def _parse_scenario_name(self, scenario_name: str) -> Tuple[Path, str]:
        split, scenario_id = _unpack_av2_scenario_name(scenario_name)
        del scenario_name

        scenario_dir = self.dataset_path / split / scenario_id
        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario path {scenario_dir} not found")
        return scenario_dir, scenario_id

    def load_scenario(self, scenario_name: str) -> ArgoverseScenario:
        scenario_dir, scenario_id = self._parse_scenario_name(scenario_name)
        return load_argoverse_scenario_parquet(
            scenario_dir / _scenario_df_filename(scenario_id)
        )

    def load_map(self, scenario_name: str) -> ArgoverseStaticMap:
        scenario_dir, scenario_id = self._parse_scenario_name(scenario_name)
        return ArgoverseStaticMap.from_json(
            scenario_dir / _scenario_map_filename(scenario_id)
        )


def av2_map_to_vector_map(map_id: str, av2_map: ArgoverseStaticMap) -> VectorMap:
    vector_map = VectorMap(map_id)

    extents: Optional[Tuple[np.ndarray, np.ndarray]] = None

    for lane_segment in av2_map.vector_lane_segments.values():
        lane_max = np.maximum(
            lane_segment.left_lane_boundary.xyz.max(0),
            lane_segment.right_lane_boundary.xyz.max(0),
        )
        lane_min = np.minimum(
            lane_segment.left_lane_boundary.xyz.min(0),
            lane_segment.right_lane_boundary.xyz.min(0),
        )

        if extents is None:
            extents = (lane_min, lane_max)
        else:
            extents = (
                np.minimum(lane_min, extents[0]),
                np.maximum(lane_max, extents[1]),
            )

        center, _ = compute_midpoint_line(
            lane_segment.left_lane_boundary.xyz, lane_segment.right_lane_boundary.xyz
        )
        vector_map.add_map_element(
            RoadLane(
                id=_road_lane_id(lane_segment.id),
                center=Polyline(center),
                left_edge=Polyline(lane_segment.left_lane_boundary.xyz),
                right_edge=Polyline(lane_segment.right_lane_boundary.xyz),
                adj_lanes_left=_adj_lanes_set(lane_segment.left_neighbor_id),
                adj_lanes_right=_adj_lanes_set(lane_segment.right_neighbor_id),
                next_lanes={_road_lane_id(i) for i in lane_segment.successors},
                prev_lanes={_road_lane_id(i) for i in lane_segment.predecessors},
            )
        )

    for drivavble_area in av2_map.vector_drivable_areas.values():
        assert extents is not None
        extents = (
            np.minimum(drivavble_area.xyz.min(0), extents[0]),
            np.maximum(drivavble_area.xyz.max(0), extents[1]),
        )

        vector_map.add_map_element(
            RoadArea(
                id=_road_area_id(drivavble_area.id),
                exterior_polygon=Polyline(drivavble_area.xyz),
            )
        )

    for ped_crossing in av2_map.vector_pedestrian_crossings.values():
        assert extents is not None
        extents = (
            np.minimum(ped_crossing.polygon.min(0), extents[0]),
            np.maximum(ped_crossing.polygon.max(0), extents[1]),
        )
        vector_map.add_map_element(
            PedCrosswalk(
                id=_ped_crosswalk_id(ped_crossing.id),
                polygon=Polyline(ped_crossing.polygon),
            )
        )

    # extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vector_map.extent = np.concatenate(extents)

    return vector_map


def _adj_lanes_set(neighbor_id: Optional[int]) -> set[str]:
    if neighbor_id is None:
        return set()
    return {_road_lane_id(neighbor_id)}


def _road_lane_id(lane_segment_id: int) -> str:
    return f"RoadLane{lane_segment_id}"


def _road_area_id(drivable_area_id: int) -> str:
    return f"RoadArea{drivable_area_id}"


def _ped_crosswalk_id(ped_crossing_id: int) -> str:
    return f"PedCrosswalk{ped_crossing_id}"


def get_track_metadata(track: Track) -> Optional[AgentMetadata]:
    agent_data = OBJECT_TYPE_DATA.get(track.object_type)
    if agent_data is None:
        return None

    agent_type, length, width, height = agent_data

    timesteps = [_to_int(state.timestep) for state in track.object_states]
    if not timesteps:
        return None

    # Av2 uses the name "AV" for the robot. Trajdata expects the name "ego" for the robot.
    name = track.track_id
    if name == "AV":
        name = "ego"

    return AgentMetadata(
        name=name,
        agent_type=agent_type,
        first_timestep=min(timesteps),
        last_timestep=max(timesteps),
        extent=FixedExtent(length=length, width=width, height=height),
    )


def _to_int(x: float) -> int:
    """Safe convert floats like 42.0 to 42."""
    y = int(x)
    assert x == y
    return y
