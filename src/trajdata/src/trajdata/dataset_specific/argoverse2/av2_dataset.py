# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import pandas as pd
import tqdm
from av2.datasets.motion_forecasting.constants import (
    AV2_SCENARIO_OBS_TIMESTEPS,
    AV2_SCENARIO_STEP_HZ,
    AV2_SCENARIO_TOTAL_TIMESTEPS,
)

from trajdata.caching.env_cache import EnvCache
from trajdata.caching.scene_cache import SceneCache
from trajdata.data_structures import AgentMetadata, EnvMetadata, Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.argoverse2.av2_utils import (
    AV2_SPLITS,
    Av2Object,
    Av2ScenarioIds,
    av2_map_to_vector_map,
    get_track_metadata,
    scenario_name_to_split,
)
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import Argoverse2Record
from trajdata.utils import arr_utils

AV2_MOTION_FORECASTING = "av2_motion_forecasting"
AV2_DT = 1 / AV2_SCENARIO_STEP_HZ


class Av2Dataset(RawDataset):

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name != AV2_MOTION_FORECASTING:
            raise ValueError(f"Unknown Argoverse 2 env name: {env_name}")

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=AV2_DT,
            parts=[AV2_SPLITS],
            scene_split_map=None,
            map_locations=None,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)
        self.dataset_obj = Av2Object(self.metadata.data_dir)

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Union[List[str], None],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        """Compute SceneMetadata for all samples from self.dataset_obj.

        Also saves records to env_cache for later reuse.
        """
        if scene_desc_contains:
            raise ValueError("Argoverse dataset does not support scene descriptions.")

        record_list = []
        metadata_list = []

        for idx, scenario_name in enumerate(self.dataset_obj.scenario_names):
            record_list.append(Argoverse2Record(scenario_name, idx))
            metadata_list.append(
                SceneMetadata(
                    env_name=self.metadata.name,
                    name=scenario_name,
                    dt=AV2_DT,
                    raw_data_idx=idx,
                )
            )

        self.cache_all_scenes_list(env_cache, record_list)
        return metadata_list

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Union[List[str], None],
        env_cache: EnvCache,
    ) -> List[Scene]:
        """Computes Scene data for all samples by reading data from env_cache."""
        if scene_desc_contains:
            raise ValueError("Argoverse dataset does not support scene descriptions.")

        splits = [s for s in AV2_SPLITS if s in scene_tag]
        record_list: List[Argoverse2Record] = env_cache.load_env_scenes_list(self.name)
        assert len(splits) == 1, f"Expected 1 split in scene tag, but got {splits}"

        return [
            self._create_scene(record.name, record.data_idx)
            for record in record_list
            if splits[0] in record.name
        ]

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        return self._create_scene(scene_info.name, scene_info.raw_data_idx)

    def _create_scene(self, scenario_name: str, data_idx: int) -> Scene:
        data_split = scenario_name_to_split(scenario_name)
        return Scene(
            env_metadata=self.metadata,
            name=scenario_name,
            location=scenario_name,
            data_split=data_split,
            length_timesteps=(
                AV2_SCENARIO_OBS_TIMESTEPS
                if data_split == "test"
                else AV2_SCENARIO_TOTAL_TIMESTEPS
            ),
            raw_data_idx=data_idx,
            data_access_info=None,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        """
        Get frame-level information from source dataset, caching it
        to cache_path.

        Always called after cache_maps, can load map if needed
        to associate map information to positions.
        """
        scenario = self.dataset_obj.load_scenario(scene.name)

        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [[] for _ in scenario.timestamps_ns]

        df_records = []

        for track in scenario.tracks:
            track_metadata = get_track_metadata(track)
            if track_metadata is None:
                continue

            agent_list.append(track_metadata)

            for object_state in track.object_states:
                agent_presence[int(object_state.timestep)].append(track_metadata)

                df_records.append(
                    {
                        "agent_id": track_metadata.name,
                        "scene_ts": object_state.timestep,
                        "x": object_state.position[0],
                        "y": object_state.position[1],
                        "z": 0.0,
                        "vx": object_state.velocity[0],
                        "vy": object_state.velocity[1],
                        "heading": object_state.heading,
                    }
                )

        df = pd.DataFrame.from_records(df_records)
        df.set_index(["agent_id", "scene_ts"], inplace=True)
        df.sort_index(inplace=True)

        df[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(
                df[["vx", "vy"]].to_numpy(), df.index.get_level_values(0)
            )
            / AV2_DT
        )
        cache_class.save_agent_data(df, cache_path, scene)

        return agent_list, agent_presence

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Get static, scene-level info from the source dataset, caching it
        to cache_path. (Primarily this is info needed to construct VectorMap)

        Resolution is in pixels per meter.
        """
        for scenario_name in tqdm.tqdm(
            self.dataset_obj.scenario_names,
            desc=f"{self.name} cache maps",
            dynamic_ncols=True,
        ):
            av2_map = self.dataset_obj.load_map(scenario_name)
            vector_map = av2_map_to_vector_map(f"{self.name}:{scenario_name}", av2_map)
            map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)
