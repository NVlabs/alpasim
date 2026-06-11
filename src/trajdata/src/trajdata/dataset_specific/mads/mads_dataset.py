# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import glob
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.mads import mads_utils
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import MadsSceneRecord
from trajdata.maps import VectorMap
from trajdata.utils import arr_utils

MADS_DT: Final[float] = 0.1
EGO_LENGTH = 5.2993629
EGO_WIDTH = 2.11311007
EGO_HEIGHT = 1.34435794

# Minimum frames for an agent to be considered
MIN_FRAMES = 10

USE_CUBIC_INTERPOLATION = True

agents_to_remove: List[str] = list()


def mads_type_to_unified_type(mads_type: str) -> AgentType:
    if mads_type.startswith("person"):
        return AgentType.PEDESTRIAN
    elif mads_type == "automobile":
        return AgentType.VEHICLE
    elif mads_type == "other_vehicle":
        return AgentType.VEHICLE
    elif mads_type.startswith("cycle"):
        return AgentType.BICYCLE
    elif mads_type.startswith("motocycle"):
        return AgentType.MOTORCYCLE
    else:
        return AgentType.UNKNOWN


class MADSDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # Create scene splits
        dataset_parts: List[Tuple[str, ...]] = [("mini_train", "mini_val")]
        self.data_dir = data_dir

        # List to hold file names without extensions
        file_names = []
        expanded_data_dir = str(Path(data_dir).expanduser())
        clip_dir = dict()
        clip_duration = dict()
        for subdir in os.listdir(expanded_data_dir):
            # Construct the full path using the expanded data_dir

            # Sometimes the folder structure is .../clipgt_id/clipgt_id/files
            clip_file = glob.glob(
                os.path.join(expanded_data_dir, subdir) + "/**/clip.parquet",
                recursive=True,
            )

            if len(clip_file) == 0:
                continue
            clip_file = clip_file[0]
            df_meta = pd.read_parquet(clip_file)
            df_meta = mads_utils.df_expand_json(df_meta)
            session_id = df_meta["key.session_id"][0]
            clip_id = df_meta["key.clip_id"][0]
            clip_dir[clip_id] = str(Path(clip_file).parent.absolute())
            t0 = df_meta["key.time_range.start_micros"][0]
            tf = df_meta["key.time_range.end_micros"][0]
            clip_duration[clip_id] = (tf - t0) / 1e6
        self.clip_duration = clip_duration

        # Match scene with corresponding map. Each scene has a matching map.

        clip_items = list(clip_dir.items())
        random.Random(42).shuffle(clip_items)
        self.clip_dir = dict(clip_items)
        clip_ids = list(clip_dir.keys())

        split_index = int(0.8 * len(clip_dir))  # 80% for training, 20% for validation
        train_clips = [clip_id for clip_id, _ in clip_items[:split_index]]
        test_clips = [clip_id for clip_id, _ in clip_items[split_index:]]

        scene_split_map = {}
        for clip_id in train_clips:
            scene_split_map[clip_id] = "mini_train"
        for clip_id in test_clips:
            scene_split_map[clip_id] = "mini_val"
        # scene_0: mini_train
        # scene_1: mini_val
        # scene_2: mini_train
        # scene_3: mini_train
        # ... and so on
        env_metadata = EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=MADS_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
            # The location names should match the map names used in
            # the unified data cache.
            map_locations=tuple(clip_ids),
        )

        return env_metadata

    def load_dataset_obj(self, verbose: bool = False) -> None:
        pass

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[MadsSceneRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, (clip_id, dir) in enumerate(self.clip_dir.items()):
            scene_location = "clipGT"
            if clip_id not in self.metadata.scene_split_map:
                raise ValueError(f"Scene {clip_id} not in scene_split_map")

            scene_split: str = self.metadata.scene_split_map[clip_id]
            scene_length: int = int(self.clip_duration[clip_id] / MADS_DT)

            if scene_length > 1:
                all_scenes_list.append(
                    MadsSceneRecord(
                        clip_id, scene_location, scene_length, scene_split, idx
                    )
                )

            if (scene_split in scene_tag) and scene_desc_contains is None:
                scene_metadata = SceneMetadata(
                    env_name=self.metadata.name,
                    name=clip_id,
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
        all_scenes_list: List[MadsSceneRecord] = env_cache.load_env_scenes_list(
            self.name
        )

        scenes_list: List[SceneMetadata] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_location, scene_length, scene_split, data_idx = (
                scene_record
            )

            if scene_split in scene_tag and scene_desc_contains is None:
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
        # Type hinting for scene_info is not working properly in python 3.10
        # _, scene_name, _, data_idx = scene_info
        scene_name = scene_info.name
        data_idx = scene_info.raw_data_idx

        scene_location = scene_info.name
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = int(self.clip_duration[scene_name] / MADS_DT) + 1

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            None,
        )

    def _get_df_from_path(self, scene_path, scene_name):
        dynamic_df = pd.read_parquet(os.path.join(scene_path, "obstacle.parquet"))
        dynamic_df = mads_utils.df_expand_json(dynamic_df)

        if len(dynamic_df["key.clip_id"].unique()) != 1:
            raise ValueError(
                f"Expected only one clip_id, but got {dynamic_df['key.clip_id'].unique()}"
            )

        # Filter for clip_id
        dynamic_df = dynamic_df[(dynamic_df["key.clip_id"] == scene_name)]

        manual_label_df = dynamic_df[
            dynamic_df["key.label_class_id"] == "seq3d:obstacles:v1"
        ].copy()
        manual_label_df["Obstacle.trackline_id"] = manual_label_df[
            "Obstacle.trackline_id"
        ].map(lambda x: x.split(":")[-1])

        manual_label_df["agent_id"] = "agent_gt_" + manual_label_df[
            "Obstacle.trackline_id"
        ].astype(float).astype(int).astype(str)

        auto_label_df = dynamic_df[
            dynamic_df["key.label_class_id"] == "scene:obstacles:autolabels:v1"
        ]
        available_gt_labels = manual_label_df["Obstacle.trackline_id"].unique()
        auto_label_df = auto_label_df[
            ~auto_label_df["Obstacle.trackline_id"].isin(available_gt_labels)
        ]
        auto_label_df["agent_id"] = "agent_auto_" + auto_label_df[
            "Obstacle.trackline_id"
        ].astype(float).astype(int).astype(str)

        nr_auto_labels = len(auto_label_df["Obstacle.trackline_id"].unique())
        nr_gt_labels = len(available_gt_labels)
        nr_obstacles = nr_auto_labels + nr_gt_labels
        fraction = nr_gt_labels / nr_obstacles if nr_obstacles > 0 else 0
        print(f"{fraction * 100:.2f}% of {nr_obstacles} labels are manual.")

        manual_label_df["source"] = "manual"
        auto_label_df["source"] = "autolabel"
        dynamic_df = pd.concat([manual_label_df, auto_label_df])

        dynamic_df.drop(columns=["Obstacle.trackline_id"], inplace=True)

        ego_df = pd.read_parquet(os.path.join(scene_path, "egomotion_estimate.parquet"))
        ego_df = mads_utils.df_expand_json(ego_df)

        assert ego_df["EgomotionEstimate.name"].unique().size == 1
        ego_df = ego_df.assign(
            length=EGO_LENGTH,
            width=EGO_WIDTH,
            height=EGO_HEIGHT,
            type="automobile",
            agent_id="ego",
            source="manual",
        )
        ego_df["key.label_class_id"] = ego_df["EgomotionEstimate.name"].iat[0]

        assert ego_df["EgomotionEstimate.name"].unique().size == 1
        ego_df = ego_df.drop(columns=["EgomotionEstimate.name"])

        ego_df.rename(
            columns={
                "EgomotionEstimate.location.x": "x",
                "EgomotionEstimate.location.y": "y",
                "EgomotionEstimate.location.z": "z",
                "EgomotionEstimate.orientation.x": "qx",
                "EgomotionEstimate.orientation.y": "qy",
                "EgomotionEstimate.orientation.z": "qz",
                "EgomotionEstimate.orientation.w": "qw",
            },
            inplace=True,
        )
        dynamic_df.rename(
            columns={
                "Obstacle.center.x": "x",
                "Obstacle.center.y": "y",
                "Obstacle.center.z": "z",
                "Obstacle.orientation.x": "qx",
                "Obstacle.orientation.y": "qy",
                "Obstacle.orientation.z": "qz",
                "Obstacle.orientation.w": "qw",
                "Obstacle.size.x": "length",
                "Obstacle.size.y": "width",
                "Obstacle.size.z": "height",
                "Obstacle.category": "type",
            },
            inplace=True,
        )

        t0 = ego_df["key.timestamp_micros"].iat[0]
        tf = ego_df["key.timestamp_micros"].iat[-1]

        if not (dynamic_df["key.timestamp_micros"] < tf).all():
            raise ValueError("Some dynamic data is after the last ego data.")

        dynamic_df = pd.concat([ego_df, dynamic_df])

        # Only select relevant dynamic data
        dynamic_df = dynamic_df[(dynamic_df["key.timestamp_micros"] <= tf)]

        dynamic_df["rel_time_seconds"] = (dynamic_df["key.timestamp_micros"] - t0) / 1e6

        interpolated_dfs = []
        for group_name, group_df in dynamic_df.groupby(
            ["key.clip_id", "key.label_class_id", "agent_id"]
        ):
            group_df = group_df.sort_values(by=["rel_time_seconds"])

            duplicated = group_df.duplicated(subset=["rel_time_seconds"])

            if duplicated.sum() > 0:
                print(f"Duplicated timestamps found for agent: {group_name}")
                group_df = group_df[~duplicated]

            min_time = group_df["rel_time_seconds"].min()
            min_step = int(np.ceil(min_time / MADS_DT))
            max_time = group_df["rel_time_seconds"].max()
            max_step = int(np.floor(max_time / MADS_DT))
            target_steps = np.arange(min_step, max_step + 1)
            target_times = target_steps * MADS_DT

            if max_step - min_step + 1 < MIN_FRAMES:
                continue

            def _interp(col_name):
                x = group_df["rel_time_seconds"]
                y = group_df[col_name]
                if USE_CUBIC_INTERPOLATION:
                    return CubicSpline(x, y)(target_times)
                return np.interp(target_times, x, y)

            if not group_df["type"].unique().size == 1:
                print(
                    "Multiple types encountered for agent: "
                    f"{group_name}, {group_df['type'].unique()}"
                )

            # [N, 4]
            quats_tensor = np.stack(
                [
                    group_df["qx"],
                    group_df["qy"],
                    group_df["qz"],
                    group_df["qw"],
                ],
                axis=1,
            )
            # Takes in scalar-last quaternion (x, y, z, w)
            r = R.from_quat(quats_tensor)
            slerp = Slerp(group_df["rel_time_seconds"], r)
            interp_r = slerp(target_times)
            headings = interp_r.as_euler("zyx", degrees=False)[:, 0]

            df = pd.DataFrame(
                {
                    "key.clip_id": group_name[0],
                    "key.label_class_id": group_name[1],
                    "agent_id": group_name[2],
                    "scene_ts": target_steps,
                    "rel_time_seconds": target_times,
                    "x": _interp("x"),
                    "y": _interp("y"),
                    "z": _interp("z"),
                    "heading": headings,
                    # We interpolate this as this might change!
                    # In particular, I found this to change for manual labels.
                    "length": _interp("length"),
                    "width": _interp("width"),
                    "height": _interp("height"),
                    "type": group_df["type"].iat[0],
                    "source": group_df["source"].iat[0],
                }
            )

            df["vx"] = df["x"].diff() / MADS_DT
            df["vy"] = df["y"].diff() / MADS_DT

            # Calculate ego accelerations 'ax' and 'ay'
            df["ax"] = df["vx"].diff() / MADS_DT
            df["ay"] = df["vy"].diff() / MADS_DT

            # Replace infinity with nan for later nan handling
            df["ax"] = df["ax"].replace([np.inf, -np.inf], np.nan)
            df["ay"] = df["ay"].replace([np.inf, -np.inf], np.nan)

            # The first row of ax and ay is NaN, fill in values where NaN exists
            df["vx"] = df["vx"].bfill().ffill()
            df["vy"] = df["vy"].bfill().ffill()
            df["ax"] = df["ax"].bfill().ffill()
            df["ay"] = df["ay"].bfill().ffill()

            interpolated_dfs.append(df)
        interpolated_df = pd.concat(interpolated_dfs).reset_index(drop=True)
        assert interpolated_df.duplicated(subset=["scene_ts", "agent_id"]).sum() == 0

        T = (tf - t0) / (1e6 * MADS_DT)
        interpolated_df = interpolated_df[
            (interpolated_df["scene_ts"] >= 0) & (interpolated_df["scene_ts"] <= T)
        ]

        # Sort by distance to ego
        ego_start = interpolated_df.query("agent_id == 'ego' and scene_ts == 0")
        ego_x = ego_start["x"].iat[0]
        ego_y = ego_start["y"].iat[0]

        unique_distances = set()

        def get_group_distance_to_ego(group_df):
            min_ts = group_df["scene_ts"].min()
            agent_start = group_df.query(f"scene_ts == {min_ts}")
            agent_x = agent_start["x"].iat[0]
            agent_y = agent_start["y"].iat[0]
            distance_to_ego = np.sqrt((ego_x - agent_x) ** 2 + (ego_y - agent_y) ** 2)
            assert distance_to_ego not in unique_distances
            unique_distances.add(distance_to_ego)
            group_df["distance_to_ego"] = distance_to_ego
            return group_df

        sorted_df = interpolated_df.groupby("agent_id").apply(
            get_group_distance_to_ego, include_groups=False
        )
        sorted_df = (
            sorted_df.sort_values(by=["distance_to_ego", "agent_id", "scene_ts"])
            .reset_index()
            .drop(columns=["level_1"])
        )

        # Filter out agents that are too close to each other
        # Strategy: For simplicity and speed we only compare the xy locations of agents
        # when they are first seen. This might miss cases when the agent moves
        # and the 'ghost' object appears later.
        # We start by adding all agents with gt labels. Then, we iterate over the rest
        # of the agents and either:
        # - accept them and add their first seen location to `first_states`
        # - reject them and add them to `agents_to_remove`
        def get_row_first_seen(df):
            return (
                df.groupby("agent_id", sort=False)
                .apply(lambda gdf: gdf.iloc[0], include_groups=False)
                .reset_index()
            )

        first_states = get_row_first_seen(sorted_df.query("source == 'manual'"))

        agents_to_remove = set()
        # Add agents one by one if they don't overlap
        for agent_id, group_df in sorted_df.query("source != 'manual'").groupby(
            "agent_id", sort=False
        ):
            current_agent_first_state = group_df.iloc[0]
            first_states["distance_to_current_agent"] = np.sqrt(
                (first_states["x"] - current_agent_first_state.x) ** 2
                + (first_states["y"] - current_agent_first_state.y) ** 2
            )
            closest_state = first_states.sort_values(
                by=["distance_to_current_agent"]
            ).iloc[0]
            # Only conservative filtering based on dimensions:
            if (
                min(closest_state.width, closest_state.height)
                < closest_state.distance_to_current_agent
            ):
                agents_to_remove.add(agent_id)
            else:
                first_states = pd.concat([first_states, get_row_first_seen(group_df)])

        sorted_df = sorted_df[~sorted_df["agent_id"].isin(agents_to_remove)]
        return sorted_df

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        sorted_df = self._get_df_from_path(self.clip_dir[scene.name], scene.name)

        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        agents_to_remove = []
        for agent_id, frames in sorted_df.groupby("agent_id", sort=False)[
            ["scene_ts", "type", "length", "width", "height"]
        ]:
            all_frame_ids = frames["scene_ts"]

            start_frame: int = all_frame_ids.iat[0]
            last_frame: int = all_frame_ids.iat[-1]

            agent_length = (
                frames["length"].iloc[0][0]
                if isinstance(frames["length"].iloc[0], list)
                else frames["length"].iloc[0]
            )
            agent_width = (
                frames["width"].iloc[0][0]
                if isinstance(frames["width"].iloc[0], list)
                else frames["width"].iloc[0]
            )
            agent_height = (
                frames["height"].iloc[0][0]
                if isinstance(frames["height"].iloc[0], list)
                else frames["height"].iloc[0]
            )

            agent_metadata = AgentMetadata(
                name=agent_id,
                agent_type=mads_type_to_unified_type(frames["type"].iloc[0]),
                first_timestep=start_frame,
                last_timestep=last_frame,
                extent=FixedExtent(
                    length=agent_length, width=agent_width, height=agent_height
                ),
            )

            agent_list.append(agent_metadata)
            for frame in range(
                agent_metadata.first_timestep, agent_metadata.last_timestep
            ):
                agent_presence[frame].append(agent_metadata)

        sorted_df = sorted_df[~sorted_df["agent_id"].isin(agents_to_remove)]
        sorted_df.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            sorted_df,
            cache_path,
            scene,
        )

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        Stores rasterized maps to disk for later retrieval.
        """
        resolution: float = map_params["px_per_m"]
        print(f"Caching {map_name} Map at {resolution:.2f} px/m...", flush=True)

        vector_map = VectorMap(map_id=f"{self.name}:{map_name}")
        mads_utils.populate_vector_map(vector_map, self.clip_dir[map_name])

        map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)

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
            self.clip_dir.keys(),
            desc=f"Caching {self.name} Maps at {map_params['px_per_m']:.2f} px/m",
            position=0,
        ):
            self.cache_map(map_name, cache_path, map_cache_class, map_params)
