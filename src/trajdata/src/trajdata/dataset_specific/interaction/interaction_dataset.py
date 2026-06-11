# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import lanelet2
import numpy as np
import pandas as pd
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import InteractionRecord
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import Polyline, RoadLane
from trajdata.utils import arr_utils

# SDD was captured at 10 frames per second.
INTERACTION_DT: Final[float] = 0.1

# For training, 1 second of history is used to predict 3 seconds into the future.
# For testing, only 1 second of observations are provided.
INTERACTION_TRAIN_LENGTH: Final[int] = 40
INTERACTION_TEST_LENGTH: Final[int] = 10
INTERACTION_NUM_FILES: Final[int] = 56
INTERACTION_LOCATIONS: Final[Tuple[str, str, str, str]] = (
    "usa",
    "china",
    "germany",
    "bulgaria",
)

INTERACTION_DETAILED_LOCATIONS: Final[Tuple[str, ...]] = (
    "CHN_Merging_ZS0",
    "CHN_Merging_ZS2",
    "CHN_Roundabout_LN",
    "DEU_Merging_MT",
    "DEU_Roundabout_OF",
    "Intersection_CM",
    "LaneChange_ET0",
    "LaneChange_ET1",
    "Merging_TR0",
    "Merging_TR1",
    "Roundabout_RW",
    "USA_Intersection_EP0",
    "USA_Intersection_EP1",
    "USA_Intersection_GL",
    "USA_Intersection_MA",
    "USA_Roundabout_EP",
    "USA_Roundabout_FT",
    "USA_Roundabout_SR",
)


def get_split(scene_name: str, no_case: bool = False) -> str:
    if no_case:
        case_id_str = ""
    else:
        case_id_str = f"_{int(scene_name.split('_')[-1])}"
    if scene_name.endswith(f"test_condition{case_id_str}"):
        return "test_condition"
    else:
        if no_case:
            return scene_name.split("_")[-1]
        else:
            return scene_name.split("_")[-2]


def get_location(scene_name: str) -> Tuple[str, str]:
    if scene_name.startswith("DR_DEU"):
        country = "germany"
    elif scene_name.startswith("DR_CHN"):
        country = "china"
    elif scene_name.startswith("DR_USA"):
        country = "usa"
    else:
        country = "bulgaria"

    if country != "bulgaria":
        detailed_loc = "_".join(scene_name.split("_")[1:4])
    else:
        detailed_loc = "_".join(scene_name.split("_")[1:3])

    return country, detailed_loc


def interaction_type_to_unified_type(label: str) -> AgentType:
    if label == "car":
        return AgentType.VEHICLE
    elif label == "pedestrian/bicycle":
        return AgentType.PEDESTRIAN
    raise


def get_last_line(file_path: Path) -> str:
    with open(file_path, "rb") as file:
        # Go to the end of the file before the last break-line
        file.seek(-2, os.SEEK_END)

        # Keep reading backward until you find the next break-line
        while file.read(1) != b"\n":
            file.seek(-2, os.SEEK_CUR)

        return file.readline().decode()


class InteractionDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # INTERACTION dataset possibilities are the Cartesian product of these.
        dataset_parts = [
            ("train", "val", "test", "test_conditional"),
            INTERACTION_LOCATIONS,
        ]

        if env_name not in {"interaction_multi", "interaction_single"}:
            raise ValueError(
                f"{env_name} not found in INTERACTION dataset. Options are {'interaction_multi', 'interaction_single'}"
            )

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=INTERACTION_DT,
            parts=dataset_parts,
            # No need since we'll have it in the scene name (and the scene names
            # are not unique between the two test types).
            scene_split_map=None,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=INTERACTION_DETAILED_LOCATIONS,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        data_dir_path = Path(self.metadata.data_dir)

        # Just storing the filepath and scene length (number of frames).
        # One could load the entire dataset here, but there's no need since some
        # of these are large in size and we can parallel process it later easily.
        self.dataset_obj: Dict[str, Tuple[Path, int, np.ndarray]] = dict()
        for scene_path in tqdm(
            data_dir_path.glob("**/*.csv"),
            disable=not verbose,
            total=INTERACTION_NUM_FILES,
        ):
            scene_name = scene_path.stem

            scene_split: str = ""
            if scene_name.endswith("obs"):
                scene_split = f"_{scene_path.parent.stem[:-len('-multi-agent')-1]}"

            num_scenarios = int(float(get_last_line(scene_path).split(",")[0]))

            self.dataset_obj[f"{scene_name}{scene_split}"] = (
                scene_path,
                (
                    INTERACTION_TRAIN_LENGTH
                    if len(scene_split) == 0
                    else INTERACTION_TEST_LENGTH
                ),
                num_scenarios,
            )

        if verbose:
            print(
                f"The first ~60 iterations might be slow, don't worry the following ones will be fast.",
                flush=True,
            )

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[InteractionRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        idx: int = 0
        for scene_name, (_, scene_length, num_scenarios) in self.dataset_obj.items():
            scene_split: str = get_split(scene_name, no_case=True)
            country, _ = get_location(scene_name)

            for scenario_num in range(num_scenarios):
                scene_name_with_num = f"{scene_name}_{scenario_num}"

                # Saving all scene records for later caching.
                all_scenes_list.append(
                    InteractionRecord(scene_name_with_num, scene_length, idx)
                )

                if (
                    country in scene_tag
                    and scene_split in scene_tag
                    and scene_desc_contains is None
                ):
                    scene_metadata = SceneMetadata(
                        env_name=self.metadata.name,
                        name=scene_name_with_num,
                        dt=self.metadata.dt,
                        raw_data_idx=idx,
                    )
                    scenes_list.append(scene_metadata)

                idx += 1

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[InteractionRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length, data_idx = scene_record

            scene_split: str = get_split(scene_name)
            country, scene_location = get_location(scene_name)

            if (
                country in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info

        scene_length = (
            INTERACTION_TRAIN_LENGTH
            if scene_name.split("_")[-2] in {"train", "val"}
            else INTERACTION_TEST_LENGTH
        )
        scene_split: str = get_split(scene_name)
        _, scene_location = get_location(scene_name)

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            None,  # No data access info necessary for the INTERACTION dataset.
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_name_parts: List[str] = scene.name.split("_")
        base_scene_name: str = "_".join(scene_name_parts[:-1])
        orig_scenario_num = int(scene_name_parts[-1])

        scene_metadata_path = EnvCache.scene_metadata_path(
            cache_path, scene.env_name, scene.name, scene.dt
        )
        if scene_metadata_path.exists():
            # Try repeatedly to open the file because it might still be
            # being created in another process.
            while True:
                try:
                    already_done_scene = EnvCache.load(scene_metadata_path)
                    break
                except:
                    time.sleep(1)

            # Already processed, so we can immediately return our cached results.
            return (
                already_done_scene.agents,
                already_done_scene.agent_presence,
            )

        scene_filepath, _, num_scenarios = self.dataset_obj[base_scene_name]

        data_df: pd.DataFrame = pd.read_csv(
            scene_filepath, index_col=False, dtype={"case_id": int}
        )

        # The first frame and case IDs of INTERACTION data is always "1".
        data_df["frame_id"] -= 1
        data_df["case_id"] -= 1

        # Ensuring case_ids are kept within track_ids.
        data_df["track_id"] = (
            data_df["case_id"].astype(str) + "_" + data_df["track_id"].astype(str)
        )

        # Don't need these columns anymore.
        data_df.drop(
            columns=["timestamp_ms"],
            inplace=True,
        )

        # Add in zero for z value
        data_df["z"] = np.zeros_like(data_df["x"])

        # Renaming columns to match our usual names.
        data_df.rename(
            columns={
                "frame_id": "scene_ts",
                "psi_rad": "heading",
                "track_id": "agent_id",
            },
            inplace=True,
        )

        # Ensuring data is sorted by agent ID and scene timestep.
        data_df.set_index(["agent_id", "scene_ts"], inplace=True)
        data_df.sort_index(inplace=True)
        data_df.reset_index(level=1, inplace=True)

        agent_ids: np.ndarray = data_df.index.get_level_values(0).to_numpy()

        ### Calculating agent classes
        agent_class: Dict[int, str] = (
            data_df.groupby("agent_id")["agent_type"].first().to_dict()
        )

        ### Calculating agent extents
        agent_length: Dict[int, float] = (
            data_df.groupby("agent_id")["length"].first().to_dict()
        )

        agent_width: Dict[int, float] = (
            data_df.groupby("agent_id")["width"].first().to_dict()
        )

        # This is likely to be very noisy... Unfortunately, ETH/UCY only
        # provide center of mass data.
        non_car_mask = data_df["agent_type"] != "car"
        data_df.loc[non_car_mask, "heading"] = np.arctan2(
            data_df.loc[non_car_mask, "vy"], data_df.loc[non_car_mask, "vx"]
        )

        del data_df["agent_type"]

        ### Calculating agent accelerations
        data_df[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(data_df[["vx", "vy"]].to_numpy(), agent_ids)
            / INTERACTION_DT
        )

        agent_list: Dict[int, List[AgentMetadata]] = defaultdict(list)
        agent_presence: Dict[int, List[List[AgentMetadata]]] = dict()
        for agent_id, frames in data_df.groupby("agent_id")["scene_ts"]:
            case_id = int(agent_id.split("_")[0])
            start_frame: int = frames.iat[0].item()
            last_frame: int = frames.iat[-1].item()

            agent_type: AgentType = interaction_type_to_unified_type(
                agent_class[agent_id]
            )

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_frame,
                last_timestep=last_frame,
                # These values are as ballpark as it gets...
                # The vehicle height here is just taking 6 feet.
                extent=(
                    FixedExtent(0.75, 0.75, 1.5)
                    if agent_type != AgentType.VEHICLE
                    else FixedExtent(
                        agent_length[agent_id], agent_width[agent_id], 1.83
                    )
                ),
            )

            if case_id not in agent_presence:
                agent_presence[case_id] = [[] for _ in range(scene.length_timesteps)]

            agent_list[case_id].append(agent_metadata)
            for frame in frames:
                agent_presence[case_id][frame].append(agent_metadata)

        # Changing the agent_id dtype to str
        data_df.reset_index(inplace=True)
        data_df["agent_id"] = data_df["agent_id"].astype(str)
        data_df.set_index(["agent_id", "scene_ts"], inplace=True)

        for case_id, case_df in data_df.groupby("case_id"):
            case_scene = Scene(
                env_metadata=scene.env_metadata,
                name=base_scene_name + f"_{case_id}",
                location=scene.location,
                data_split=scene.data_split,
                length_timesteps=scene.length_timesteps,
                raw_data_idx=scene.raw_data_idx,
                data_access_info=scene.data_access_info,
                description=scene.description,
                agents=agent_list[case_id],
                agent_presence=agent_presence[case_id],
            )
            cache_class.save_agent_data(
                case_df.loc[
                    :,
                    [
                        "x",
                        "y",
                        "z",
                        "vx",
                        "vy",
                        "ax",
                        "ay",
                        "heading",
                    ],
                ],
                cache_path,
                case_scene,
            )
            EnvCache.save_scene_with_path(cache_path, case_scene)

        return agent_list[orig_scenario_num], agent_presence[orig_scenario_num]

    def cache_map(
        self,
        map_path: Path,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        vector_map = VectorMap(
            map_id=f"{self.name}:{'_'.join(map_path.stem.split('_')[1:])}"
        )

        map_projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0, 0.0))

        laneletmap = lanelet2.io.load(str(map_path), map_projector)
        traffic_rules = lanelet2.traffic_rules.create(
            # TODO(bivanovic): lanelet2 has only implemented Germany so far...
            # Thankfully all countries here drive on the right-hand side like
            # Germany, so maybe we can get away with it.
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        lane_graph = lanelet2.routing.RoutingGraph(laneletmap, traffic_rules)

        maximum_bound: np.ndarray = np.full((3,), np.nan)
        minimum_bound: np.ndarray = np.full((3,), np.nan)

        for lanelet in tqdm(
            laneletmap.laneletLayer, desc="Creating Vectorized Map", leave=False
        ):
            left_pts: np.ndarray = np.array(
                [(p.x, p.y, p.z) for p in lanelet.leftBound]
            )
            right_pts: np.ndarray = np.array(
                [(p.x, p.y, p.z) for p in lanelet.rightBound]
            )
            center_pts: np.ndarray = np.array(
                [(p.x, p.y, p.z) for p in lanelet.centerline]
            )

            # Adding the element to the map.
            new_lane = RoadLane(
                id=str(lanelet.id),
                center=Polyline(center_pts),
                left_edge=Polyline(left_pts),
                right_edge=Polyline(right_pts),
            )

            new_lane.next_lanes.update(
                [str(l.id) for l in lane_graph.following(lanelet)]
            )

            new_lane.prev_lanes.update(
                [str(l.id) for l in lane_graph.previous(lanelet)]
            )

            left_lane_change = lane_graph.left(lanelet)
            if left_lane_change:
                new_lane.adj_lanes_left.add(str(left_lane_change.id))

            right_lane_change = lane_graph.right(lanelet)
            if right_lane_change:
                new_lane.adj_lanes_right.add(str(right_lane_change.id))

            vector_map.add_map_element(new_lane)

            # Computing the maximum and minimum map coordinates.
            maximum_bound = np.fmax(maximum_bound, left_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, left_pts.min(axis=0))

            maximum_bound = np.fmax(maximum_bound, right_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, right_pts.min(axis=0))

            maximum_bound = np.fmax(maximum_bound, center_pts.max(axis=0))
            minimum_bound = np.fmin(minimum_bound, center_pts.min(axis=0))

        # Setting the map bounds.
        # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
        vector_map.extent = np.concatenate((minimum_bound, maximum_bound))

        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        data_dir_path = Path(self.metadata.data_dir)
        file_paths = list(data_dir_path.glob("**/*.osm"))
        for map_path in tqdm(
            file_paths,
            desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
            position=0,
        ):
            self.cache_map(map_path, cache_path, map_cache_class, map_params)
