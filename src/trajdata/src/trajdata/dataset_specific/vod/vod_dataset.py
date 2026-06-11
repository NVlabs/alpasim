# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from tqdm import tqdm
from vod.map_expansion.map_api import VODMap, locations
from vod.utils.splits import create_splits_scenes
from vod.vod import VOD

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import (
    Agent,
    AgentMetadata,
    AgentType,
    FixedExtent,
    VariableExtent,
)
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import VODSceneRecord
from trajdata.dataset_specific.vod import vod_utils
from trajdata.maps import VectorMap


class VODDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # See https://github.com/tudelft-iv/view-of-delft-prediction-devkit/blob/main/src/vod/utils/splits.py
        # for full details on how the splits are obtained below.
        all_scene_splits: Dict[str, List[str]] = create_splits_scenes()

        train_scenes: List[str] = deepcopy(all_scene_splits["train"])

        if env_name == "vod_trainval":
            vod_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["train", "train_val", "val"]
            }

            # VoD possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("train", "train_val", "val"),
                ("delft",),
            ]
        elif env_name == "vod_test":
            vod_scene_splits: Dict[str, List[str]] = {
                k: all_scene_splits[k] for k in ["test"]
            }

            # VoD possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("test",),
                ("delft",),
            ]

        # elif env_name == "vod_mini":
        #     vod_scene_splits: Dict[str, List[str]] = {
        #         k: all_scene_splits[k] for k in ["mini_train", "mini_val"]
        #     }

        #     # VoD possibilities are the Cartesian product of these
        #     dataset_parts: List[Tuple[str, ...]] = [
        #         ("mini_train", "mini_val"),
        #         ("delft",),
        #     ]
        else:
            raise ValueError(f"Unknown VoD environment name: {env_name}")

        # Inverting the dict from above, associating every scene with its data split.
        vod_scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in vod_scene_splits.items() for v_elem in v
        }

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=vod_utils.VOD_DT,
            parts=dataset_parts,
            scene_split_map=vod_scene_split_map,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=tuple(locations),
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        if self.name == "vod_trainval":
            version_str = "v1.0-trainval"
        elif self.name == "vod_test":
            version_str = "v1.0-test"
        # elif self.name == "vod_mini":
        #     version_str = "v1.0-mini"

        self.dataset_obj = VOD(version=version_str, dataroot=self.metadata.data_dir)

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[VODSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, scene_record in enumerate(self.dataset_obj.scene):
            scene_token: str = scene_record["token"]
            scene_name: str = scene_record["name"]
            scene_desc: str = scene_record["description"].lower()
            scene_location: str = self.dataset_obj.get(
                "log", scene_record["log_token"]
            )["location"]
            scene_split: str = self.metadata.scene_split_map[scene_token]
            scene_length: int = scene_record["nbr_samples"]

            # Saving all scene records for later caching.
            all_scenes_list.append(
                VODSceneRecord(
                    scene_token,
                    scene_name,
                    scene_location,
                    scene_length,
                    scene_desc,
                    idx,
                )
            )

            if scene_location in scene_tag and scene_split in scene_tag:
                if scene_desc_contains is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_contains
                ):
                    continue

                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[VODSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            (
                scene_token,
                scene_name,
                scene_location,
                scene_length,
                scene_desc,
                data_idx,
            ) = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_token]

            if scene_location.split("-")[0] in scene_tag and scene_split in scene_tag:
                if scene_desc_contains is not None and not any(
                    desc_query in scene_desc for desc_query in scene_desc_contains
                ):
                    continue

                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                    scene_desc,
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, _, _, data_idx = scene_info

        scene_record = self.dataset_obj.scene[data_idx]
        scene_token: str = scene_record["token"]
        scene_name: str = scene_record["name"]
        scene_desc: str = scene_record["description"].lower()
        scene_location: str = self.dataset_obj.get("log", scene_record["log_token"])[
            "location"
        ]
        scene_split: str = self.metadata.scene_split_map[scene_token]
        scene_length: int = scene_record["nbr_samples"]

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            scene_record,
            scene_desc,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene.length_timesteps - 1,
            extent=FixedExtent(length=4.084, width=1.730, height=1.562),
        )

        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene.length_timesteps)
        ]

        agent_data_list: List[pd.DataFrame] = list()
        existing_agents: Dict[str, AgentMetadata] = dict()

        all_frames: List[Dict[str, Union[str, int]]] = list(
            vod_utils.frame_iterator(self.dataset_obj, scene)
        )
        frame_idx_dict: Dict[str, int] = {
            frame_dict["token"]: idx for idx, frame_dict in enumerate(all_frames)
        }
        for frame_idx, frame_info in enumerate(all_frames):
            for agent_info in vod_utils.agent_iterator(self.dataset_obj, frame_info):
                if agent_info["instance_token"] in existing_agents:
                    continue

                if agent_info["category_name"] == "vehicle.ego":
                    # Do not double-count the ego vehicle
                    continue

                if not agent_info["next"]:
                    # There are some agents with only a single detection to them, we don't care about these.
                    continue

                agent: Agent = vod_utils.agg_agent_data(
                    self.dataset_obj, agent_info, frame_idx, frame_idx_dict
                )

                for scene_ts in range(
                    agent.metadata.first_timestep, agent.metadata.last_timestep + 1
                ):
                    agent_presence[scene_ts].append(agent.metadata)

                existing_agents[agent.name] = agent.metadata

                agent_data_list.append(agent.data)

        ego_agent: Agent = vod_utils.agg_ego_data(self.dataset_obj, scene)
        agent_data_list.append(ego_agent.data)

        agent_list: List[AgentMetadata] = [ego_agent_info] + list(
            existing_agents.values()
        )

        cache_class.save_agent_data(pd.concat(agent_data_list), cache_path, scene)

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        vod_map: VODMap = VODMap(dataroot=self.metadata.data_dir, map_name=map_name)

        vector_map = VectorMap(map_id=f"{self.name}:{map_name}")
        vod_utils.populate_vector_map(vector_map, vod_map)

        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Stores rasterized maps to disk for later retrieval.

        Below are the map origins (south western corner, in [lat, lon]) for each of
        the 4 maps in VoD:
            delft:       []

        The dimensions of the maps are as follows ([width, height] in meters). They
        can also be found in vod_utils.py
            delft:       []
        The rasterized semantic maps published with VoD v1.0 have a scale of 10px/m,
        hence the above numbers are the image dimensions divided by 10.

        VoD uses the same WGS 84 Web Mercator (EPSG:3857) projection as Google Maps/Earth.
        """
        for map_name in tqdm(
            locations,
            desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
            position=0,
        ):
            self.cache_map(map_name, cache_path, map_cache_class, map_params)
