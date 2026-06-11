# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import defaultdict
from functools import partial
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from l5kit.configs.config import load_metadata
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.map_api import MapAPI

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import (
    AgentMetadata,
    EnvMetadata,
    Scene,
    SceneMetadata,
    SceneTag,
)
from trajdata.data_structures.agent import Agent, AgentType, VariableExtent
from trajdata.dataset_specific.lyft import lyft_utils
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import LyftSceneRecord
from trajdata.maps import VectorMap
from trajdata.utils import arr_utils


def const_lambda(const_val: Any) -> Any:
    return const_val


def get_mode_val(series: pd.Series) -> float:
    return series.mode().iloc[0].item()


class LyftDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        if env_name == "lyft_sample":
            dataset_parts: List[Tuple[str, ...]] = [
                ("mini_train", "mini_val"),
                ("palo_alto",),
            ]
            # Using seeded randomness to assign 80% of scenes to "mini_train" and 20% to "mini_val"
            rng = Random(0)
            scene_split = ["mini_train"] * 80 + ["mini_val"] * 20
            rng.shuffle(scene_split)

            scene_split_map = {
                f"scene-{idx:04d}": scene_split[idx] for idx in range(len(scene_split))
            }
        elif env_name == "lyft_train":
            dataset_parts: List[Tuple[str, ...]] = [
                ("train",),
                ("palo_alto",),
            ]

            scene_split_map = defaultdict(partial(const_lambda, const_val="train"))
        elif env_name == "lyft_train_full":
            dataset_parts: List[Tuple[str, ...]] = [
                ("train",),
                ("palo_alto",),
            ]

            scene_split_map = defaultdict(partial(const_lambda, const_val="train"))
        elif env_name == "lyft_val":
            dataset_parts: List[Tuple[str, ...]] = [
                ("val",),
                ("palo_alto",),
            ]

            scene_split_map = defaultdict(partial(const_lambda, const_val="val"))
        else:
            raise ValueError(f"Unknown Lyft environment name: {env_name}")

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=lyft_utils.LYFT_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=("palo_alto",),
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj = ChunkedDataset(str(self.metadata.data_dir)).open()

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[LyftSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        all_scene_frames = self.dataset_obj.scenes["frame_index_interval"]
        for idx in range(all_scene_frames.shape[0]):
            scene_name: str = f"scene-{idx:04d}"
            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = (
                all_scene_frames[idx, 1] - all_scene_frames[idx, 0]
            ).item()  # Doing .item() otherwise it'll be a numpy.int64

            # Saving all scene records for later caching.
            all_scenes_list.append(LyftSceneRecord(scene_name, scene_length, idx))

            if (
                "palo_alto" in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=scene_name,
                    dt=self.metadata.dt,
                    raw_data_idx=idx,
                )
                scenes_list.append(scene_metadata)

        self.cache_all_scenes_list(env_cache, all_scenes_list)
        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, _, _, data_idx = scene_info
        scene_frames: np.ndarray = self.dataset_obj.scenes["frame_index_interval"][
            data_idx
        ]
        scene_name: str = f"scene-{data_idx:04d}"
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = (
            scene_frames[1] - scene_frames[0]
        ).item()  # Doing .item() otherwise it'll be a numpy.int64

        return Scene(
            self.metadata,
            scene_name,
            "palo_alto",
            scene_split,
            scene_length,
            data_idx,
            scene_frames,
        )

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[LyftSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length, data_idx = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if (
                "palo_alto" in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    "palo_alto",
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    # @profile
    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        ego_agent_info: AgentMetadata = AgentMetadata(
            name="ego",
            agent_type=AgentType.VEHICLE,
            first_timestep=0,
            last_timestep=scene.length_timesteps - 1,
            extent=VariableExtent(),
        )

        agent_list: List[AgentMetadata] = [ego_agent_info]
        agent_presence: List[List[AgentMetadata]] = [
            [ego_agent_info] for _ in range(scene.length_timesteps)
        ]

        ego_agent: Agent = lyft_utils.agg_ego_data(self.dataset_obj, scene)

        scene_frame_start = scene.data_access_info[0]
        scene_frame_end = scene.data_access_info[1]

        agent_indices = self.dataset_obj.frames[scene_frame_start:scene_frame_end][
            "agent_index_interval"
        ]
        agent_start_idx = agent_indices[0, 0]
        agent_end_idx = agent_indices[-1, 1]

        lyft_agents = self.dataset_obj.agents[agent_start_idx:agent_end_idx]
        agent_ids = lyft_agents["track_id"]

        # This is so we later know what is the first scene timestep that an agent appears in the scene.
        num_agents_per_ts = agent_indices[:, 1] - agent_indices[:, 0]
        agent_frame_ids = np.repeat(
            np.arange(scene.length_timesteps), num_agents_per_ts
        )

        agent_translations = lyft_agents["centroid"]
        agent_velocities = lyft_agents["velocity"]
        agent_yaws = lyft_agents["yaw"]

        agent_probs = lyft_agents["label_probabilities"]
        agent_ml_class = np.argmax(agent_probs, axis=1).astype(int)

        agent_sizes = lyft_agents["extent"]

        traj_cols = ["x", "y", "vx", "vy", "heading"]
        class_cols = ["class_id"]
        extent_cols = ["length", "width", "height"]

        all_agent_data = np.concatenate(
            [
                agent_translations,
                agent_velocities,
                np.expand_dims(agent_yaws, axis=1),
                np.expand_dims(agent_ml_class, axis=1),
                agent_sizes,
            ],
            axis=1,
        )
        all_agent_data_df = pd.DataFrame(
            all_agent_data,
            columns=traj_cols + class_cols + extent_cols,
            index=[agent_ids, agent_frame_ids],
        )
        all_agent_data_df.index.names = ["agent_id", "scene_ts"]
        all_agent_data_df.sort_index(inplace=True)
        all_agent_data_df.reset_index(level=1, inplace=True)

        # Add in z data based on ego data
        all_agent_data_df = all_agent_data_df.join(
            ego_agent.data.xs("ego", level=0)["z"], on="scene_ts"
        )

        ### Calculating agent classes
        agent_class: Dict[int, float] = (
            all_agent_data_df.groupby("agent_id")["class_id"]
            .agg(get_mode_val)
            .to_dict()
        )

        ### Calculating agent accelerations
        all_agent_data_df[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(
                all_agent_data_df[["vx", "vy"]].to_numpy(), agent_ids
            )
            / lyft_utils.LYFT_DT
        )

        agents_to_remove: List[int] = list()
        for agent_id, frames in all_agent_data_df.groupby("agent_id")["scene_ts"]:
            if frames.shape[0] <= 1:
                # There are some agents with only a single detection to them, we don't care about these.
                agents_to_remove.append(agent_id)
                continue

            start_frame: int = frames.iat[0].item()
            last_frame: int = frames.iat[-1].item()

            if frames.shape[0] < last_frame - start_frame + 1:
                # Fun fact: this is never hit which means Lyft has no missing
                # timesteps (which could be caused by, e.g., occlusion).
                raise ValueError("Lyft indeed can have missing frames :(")

            agent_type: AgentType = lyft_utils.lyft_type_to_unified_type(
                int(agent_class[agent_id])
            )

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_frame,
                last_timestep=last_frame,
                extent=VariableExtent(),
            )

            agent_list.append(agent_metadata)
            for frame in frames:
                agent_presence[frame].append(agent_metadata)

        # For now only saving non-prob columns since Lyft is effectively one-hot (see https://arxiv.org/abs/2104.12446).
        final_cols = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "ax",
            "ay",
            "heading",
        ] + extent_cols

        # Removing agents with only one detection.
        all_agent_data_df.drop(index=agents_to_remove, inplace=True)

        # Changing the agent_id dtype to str
        all_agent_data_df.reset_index(inplace=True)
        all_agent_data_df["agent_id"] = all_agent_data_df["agent_id"].astype(str)
        all_agent_data_df.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            pd.concat([ego_agent.data, all_agent_data_df.loc[:, final_cols]]),
            cache_path,
            scene,
        )

        return agent_list, agent_presence

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        resolution: float = map_params["px_per_m"]
        map_name: str = "palo_alto"
        print(f"Caching {map_name} Map at {resolution:.2f} px/m...", flush=True)

        # We have to do this .parent.parent stuff because the data_dir for lyft is scenes/*.zarr
        dm = LocalDataManager((self.metadata.data_dir.parent.parent).resolve())

        dataset_meta = load_metadata(dm.require("meta.json"))
        semantic_map_filepath = dm.require("semantic_map/semantic_map.pb")
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        mapAPI = MapAPI(semantic_map_filepath, world_to_ecef)

        vector_map = VectorMap(map_id=f"{self.name}:{map_name}")
        lyft_utils.populate_vector_map(vector_map, mapAPI)

        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)
