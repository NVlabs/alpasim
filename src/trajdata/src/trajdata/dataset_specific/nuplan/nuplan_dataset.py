# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from tqdm import tqdm

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
from trajdata.dataset_specific.nuplan import nuplan_utils
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import NuPlanSceneRecord
from trajdata.maps.vec_map import VectorMap
from trajdata.utils import arr_utils


class NuplanDataset(RawDataset):
    def __init__(
        self,
        env_name: str,
        data_dir: str,
        parallelizable: bool = True,
        has_maps: bool = True,
        central_tokens_config: Optional[List[Dict[str, Any]]] = None,
        num_timesteps_before: Optional[int] = None,
        num_timesteps_after: Optional[int] = None,
    ) -> None:
        """
        Args:
            env_name: Dataset name
            data_dir: Data directory path
            parallelizable: Whether dataset is parallelizable
            has_maps: Whether dataset has maps
            central_tokens_config: Optional central tokens configuration
            num_timesteps_before: Number of timesteps before central token
            num_timesteps_after: Number of timesteps after central token

        Note:
            The cache key is based solely on env_name. If you change
            central_tokens_config, num_timesteps_before, or num_timesteps_after
            after an initial cache build, pass rebuild_cache=True to
            UnifiedDataset to avoid stale cached data being silently reused.
        """
        super().__init__(env_name, data_dir, parallelizable, has_maps)
        self._central_tokens_config = central_tokens_config
        self._num_timesteps_before = (
            num_timesteps_before if num_timesteps_before is not None else 30
        )
        self._num_timesteps_after = (
            num_timesteps_after if num_timesteps_after is not None else 80
        )

    @property
    def use_central_tokens(self) -> bool:
        return bool(self._central_tokens_config)

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        all_log_splits: Dict[str, List[str]] = nuplan_utils.create_splits_logs()

        nup_log_splits: Dict[str, List[str]]
        if env_name == "nuplan_mini":
            nup_log_splits = {
                k: all_log_splits.get(k[5:], [])
                for k in ["mini_train", "mini_val", "mini_test"]
            }

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                ("mini_train", "mini_val", "mini_test"),
                nuplan_utils.NUPLAN_LOCATIONS,
            ]
        elif env_name.startswith("nuplan"):
            split_str = env_name.split("_")[-1]
            nup_log_splits = {split_str: all_log_splits.get(split_str, [])}

            # nuScenes possibilities are the Cartesian product of these
            dataset_parts: List[Tuple[str, ...]] = [
                (split_str,),
                nuplan_utils.NUPLAN_LOCATIONS,
            ]
        else:
            raise ValueError(f"Unknown nuPlan environment name: {env_name}")

        # Inverting the dict from above, associating every log with its data split.
        nup_log_split_map: Dict[str, str] = {
            v_elem: k for k, v in nup_log_splits.items() for v_elem in v
        }

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=nuplan_utils.NUPLAN_DT,
            parts=dataset_parts,
            scene_split_map=nup_log_split_map,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=nuplan_utils.NUPLAN_LOCATIONS,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        if self.name == "nuplan_mini":
            subfolder = "mini"
        elif self.name == "nuplan_test":
            subfolder = "test"
        elif self.name.startswith("nuplan"):
            subfolder = "trainval"

        # Create NuPlanObject with central token configuration
        self.dataset_obj = nuplan_utils.NuPlanObject(
            self.metadata.data_dir,
            subfolder,
            central_tokens_config=self._central_tokens_config,
            num_timesteps_before=self._num_timesteps_before,
            num_timesteps_after=self._num_timesteps_after,
        )

    def _get_log_name_from_scene_name(self, scene_name: str) -> str:
        """Extract originating log name from scene name.

        Supports both formats:
        - Old format: logfile=token
        - New format: logfile-token
        """
        if "=" in scene_name:
            return scene_name.split("=")[0]
        return scene_name.rsplit("-", 1)[0]

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[NuPlanSceneRecord] = list()

        default_split = "mini_train" if "mini" in self.metadata.name else "train"

        scenes_list: List[SceneMetadata] = list()
        for idx, scene_record in enumerate(self.dataset_obj.scenes):
            scene_name: str = scene_record["name"]
            originating_log: str = self._get_log_name_from_scene_name(scene_name)
            # scene_desc: str = scene_record["description"].lower()
            scene_location: str = scene_record["location"]
            scene_split: str = self.metadata.scene_split_map.get(
                originating_log, default_split
            )
            scene_length: int = scene_record["num_timesteps"]

            if scene_length == 1:
                # nuPlan has scenes with only a single frame of data which we
                # can't do much with in terms of prediction/planning/etc. As a
                # result, we skip it.
                # As an example, nuplan_mini scene e958b276c7a65197
                # from log 2021.06.14.19.22.11_veh-38_01480_01860.
                continue

            # Saving all scene records for later caching.
            all_scenes_list.append(
                NuPlanSceneRecord(
                    scene_name,
                    scene_location,
                    scene_length,
                    scene_split,
                    # scene_desc,
                    idx,
                )
            )

            if (
                scene_location in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                # if scene_desc_contains is not None and not any(
                #     desc_query in scene_desc for desc_query in scene_desc_contains
                # ):
                #     continue

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
        all_scenes_list: List[NuPlanSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                scene_split,
                # scene_desc,
                data_idx,
            ) = scene_record

            if (
                scene_location in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                # if scene_desc_contains is not None and not any(
                #     desc_query in scene_desc for desc_query in scene_desc_contains
                # ):
                #     continue

                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    scene_location,
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                    # scene_desc,
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, _, _, data_idx = scene_info
        default_split = "mini_train" if "mini" in self.metadata.name else "train"

        scene_record: Dict[str, str] = self.dataset_obj.scenes[data_idx]

        scene_name: str = scene_record["name"]
        originating_log: str = self._get_log_name_from_scene_name(scene_name)
        # scene_desc: str = scene_record["description"].lower()
        scene_location: str = scene_record["location"]
        scene_split: str = self.metadata.scene_split_map.get(
            originating_log, default_split
        )
        scene_length: int = scene_record["num_timesteps"]

        data_access_info = dict(scene_record)

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            data_access_info,
            # scene_desc,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        # instantiate VectorMap from map_api if necessary
        log_filename = self._get_log_name_from_scene_name(scene.name)
        self.dataset_obj.open_db(log_filename + ".db")

        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene.length_timesteps - 1,
            # From https://github.com/motional/nuplan-devkit/blob/761cdbd52d699560629c79ba1b10b29c18ebc068/nuplan/common/actor_state/vehicle_parameters.py#L125
            extent=FixedExtent(length=4.049 + 1.127, width=1.1485 * 2.0, height=1.777),
        )

        agent_list: List[AgentMetadata] = [ego_agent_info]
        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene.length_timesteps)
        ]

        # Automatically select the correct method based on scene type
        all_frames: pd.DataFrame = self.dataset_obj.get_scene_frames(scene)

        ego_df = (
            all_frames[
                ["ego_x", "ego_y", "ego_z", "ego_vx", "ego_vy", "ego_ax", "ego_ay"]
            ]
            .rename(columns=lambda name: name[4:])
            .reset_index(drop=True)
        )
        ego_df["heading"] = arr_utils.quaternion_to_yaw(
            all_frames[["ego_qw", "ego_qx", "ego_qy", "ego_qz"]].values
        )
        ego_df["scene_ts"] = np.arange(len(ego_df))
        ego_df["agent_id"] = "ego"

        lpc_tokens: List[bytearray] = all_frames.index.tolist()
        agents_df: pd.DataFrame = self.dataset_obj.get_detected_agents(lpc_tokens)
        tls_df: pd.DataFrame = self.dataset_obj.get_traffic_light_status(lpc_tokens)

        # Extract sensor calibration information.
        log_filename = self._get_log_name_from_scene_name(scene.name)
        sensor_calib = self.dataset_obj.get_sensor_calibration(log_filename)

        # Store calibration information into Scene.data_access_info.
        if scene.data_access_info is None:
            scene.data_access_info = {}
        elif not isinstance(scene.data_access_info, dict):
            raise TypeError(
                f"scene.data_access_info must be a dict, got "
                f"{type(scene.data_access_info).__name__!r} for scene {scene.name!r}"
            )

        scene.data_access_info["sensor_calibration"] = sensor_calib

        self.dataset_obj.close_db()

        agents_df["scene_ts"] = agents_df["lidar_pc_token"].map(
            {lpc_token: scene_ts for scene_ts, lpc_token in enumerate(lpc_tokens)}
        )
        agents_df["agent_id"] = agents_df["track_token"].apply(lambda x: x.hex())

        # Recording agent metadata for later.
        agent_metadata_dict: Dict[str, Dict[str, Any]] = dict()
        for agent_id, agent_data in agents_df.groupby("agent_id").first().iterrows():
            if agent_id not in agent_metadata_dict:
                agent_metadata_dict[agent_id] = {
                    "type": nuplan_utils.nuplan_type_to_unified_type(
                        agent_data["category_name"]
                    ),
                    "width": agent_data["width"],
                    "length": agent_data["length"],
                    "height": agent_data["height"],
                }

        agents_df = agents_df.drop(
            columns=[
                "lidar_pc_token",
                "track_token",
                "category_name",
                "width",
                "length",
                "height",
            ],
        ).rename(columns={"yaw": "heading"})

        # Sorting the agents' combined DataFrame here.
        agents_df.set_index(["agent_id", "scene_ts"], inplace=True)
        agents_df.sort_index(inplace=True)
        agents_df.reset_index(level=1, inplace=True)

        one_detection_agents: List[str] = list()
        for agent_id in agent_metadata_dict:
            agent_metadata_entry = agent_metadata_dict[agent_id]

            agent_specific_df = agents_df.loc[agent_id]
            if len(agent_specific_df.shape) <= 1 or agent_specific_df.shape[0] <= 1:
                # Removing agents that are only observed once.
                one_detection_agents.append(agent_id)
                continue

            first_timestep: int = agent_specific_df.iat[0, 0].item()
            last_timestep: int = agent_specific_df.iat[-1, 0].item()
            agent_info: AgentMetadata = AgentMetadata(
                name=agent_id,
                agent_type=agent_metadata_entry["type"],
                first_timestep=first_timestep,
                last_timestep=last_timestep,
                extent=FixedExtent(
                    length=agent_metadata_entry["length"],
                    width=agent_metadata_entry["width"],
                    height=agent_metadata_entry["height"],
                ),
            )

            agent_list.append(agent_info)
            for timestep in range(
                agent_info.first_timestep, agent_info.last_timestep + 1
            ):
                agent_presence[timestep].append(agent_info)

        # Removing agents with only one detection.
        agents_df.drop(index=one_detection_agents, inplace=True)

        ### Calculating agent accelerations
        agent_ids: np.ndarray = agents_df.index.get_level_values(0).to_numpy()
        if len(agent_ids) > 0:
            agents_df[["ax", "ay"]] = (
                arr_utils.agent_aware_diff(
                    agents_df[["vx", "vy"]].to_numpy(), agent_ids
                )
                / nuplan_utils.NUPLAN_DT
            )
        else:
            agents_df[["ax", "ay"]] = agents_df[["vx", "vy"]]

        # for agent_id, frames in agents_df.groupby("agent_id")["scene_ts"]:
        #     if frames.shape[0] <= 1:
        #         raise ValueError("nuPlan can have one-detection agents :(")

        #     start_frame: int = frames.iat[0].item()
        #     last_frame: int = frames.iat[-1].item()

        #     if frames.shape[0] < last_frame - start_frame + 1:
        #         raise ValueError("nuPlan indeed can have missing frames :(")

        overall_agents_df = pd.concat([ego_df, agents_df.reset_index()]).set_index(
            ["agent_id", "scene_ts"]
        )
        cache_class.save_agent_data(overall_agents_df, cache_path, scene)

        # similar process to clean up and traffic light data
        tls_df["scene_ts"] = tls_df["lidar_pc_token"].map(
            {lpc_token: scene_ts for scene_ts, lpc_token in enumerate(lpc_tokens)}
        )
        tls_df = tls_df.drop(columns=["lidar_pc_token"]).set_index(
            ["lane_id", "scene_ts"]
        )

        cache_class.save_traffic_light_data(tls_df, cache_path, scene)

        return agent_list, agent_presence

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Stores rasterized maps to disk for later retrieval.
        """
        for map_name in tqdm(
            nuplan_utils.NUPLAN_LOCATIONS,
            desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
            position=0,
        ):
            cache_map(
                map_root=str(self.metadata.data_dir.parent / "maps"),
                env_name=self.name,
                map_name=map_name,
                cache_path=cache_path,
                map_cache_class=map_cache_class,
                map_params=map_params,
            )


def cache_map(
    map_root: str,
    env_name: str,
    map_name: str,
    cache_path: Path,
    map_cache_class: Type[SceneCache],
    map_params: Dict[str, Any],
) -> None:
    from nuplan.common.maps.nuplan_map import map_factory

    nuplan_map: NuPlanMap = map_factory.get_maps_api(
        map_root=map_root,
        map_version=nuplan_utils.NUPLAN_MAP_VERSION,
        map_name=nuplan_utils.NUPLAN_FULL_MAP_NAME_DICT[map_name],
    )

    # Loading all layer geometries.
    nuplan_map.initialize_all_layers()

    # This df has the normal lane_connectors with additional boundary information,
    # which we want to use, however the default index is not the lane_connector_fid,
    # although it is a 1:1 mapping so we instead create another index with the
    # lane_connector_fids as the key and the resulting integer indices as the value.
    lane_connector_fids: pd.Series = nuplan_map._vector_map[
        "gen_lane_connectors_scaled_width_polygons"
    ]["lane_connector_fid"]
    lane_connector_idxs: pd.Series = pd.Series(
        index=lane_connector_fids, data=range(len(lane_connector_fids))
    )

    vector_map = VectorMap(map_id=f"{env_name}:{map_name}")
    nuplan_utils.populate_vector_map(
        vector_map,
        nuplan_map,
        lane_connector_idxs,
        max_lane_length=map_params.get("max_lane_length", None),
    )

    map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)
