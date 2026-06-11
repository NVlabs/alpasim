# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from random import Random
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import SDDPedsRecord
from trajdata.utils import arr_utils

from .estimated_homography import SDD_HOMOGRAPHY_SCALES

# SDD was captured at 30 frames per second.
SDDPEDS_DT: Final[float] = 1.0 / 30.0


# There are 60 scenes in total.
SDDPEDS_SCENE_COUNTS: Final[Dict[str, int]] = {
    "bookstore": 7,
    "coupa": 4,
    "deathCircle": 5,
    "gates": 9,
    "hyang": 15,
    "little": 4,
    "nexus": 12,
    "quad": 4,
}


def sdd_type_to_unified_type(label: str) -> AgentType:
    if label == "Pedestrian":
        return AgentType.PEDESTRIAN
    elif label == "Biker":
        return AgentType.BICYCLE
    elif label in {"Cart", "Car", "Bus"}:
        return AgentType.VEHICLE
    elif label == "Skater":
        return AgentType.UNKNOWN


class SDDPedsDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        # Using seeded randomness to assign 42 scenes (70% of all scenes) to "train",
        # 9 (15%) to "val", and 9 (15%) to "test".
        rng = Random(0)
        scene_split = ["train"] * 42 + ["val"] * 9 + ["test"] * 9
        rng.shuffle(scene_split)

        scene_list: List[str] = []
        for scene_name, video_count in SDDPEDS_SCENE_COUNTS.items():
            scene_list += [f"{scene_name}_{idx}" for idx in range(video_count)]

        scene_split_map: Dict[str, str] = {
            scene_list[idx]: scene_split[idx] for idx in range(len(scene_split))
        }

        # SDD possibilities are the Cartesian product of these,
        dataset_parts: List[Tuple[str, ...]] = [
            ("train", "val", "test"),
            ("stanford",),
        ]

        env_metadata = EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=SDDPEDS_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )
        return env_metadata

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        # Just storing the filepath and scene length (number of frames).
        # One could load the entire dataset here, but there's no need
        # since it's ~500 MB in size and we can parallel process it later easily.
        self.dataset_obj: Dict[str, Tuple[Path, int]] = dict()
        for scene_name, video_count in SDDPEDS_SCENE_COUNTS.items():
            for video_num in range(video_count):
                data_filepath: Path = (
                    Path(self.metadata.data_dir)
                    / scene_name
                    / f"video{video_num}"
                    / "annotations.txt"
                )

                csv_columns = [
                    "agent_id",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "frame_id",
                    "lost",
                    "occluded",
                    "generated",
                    "label",
                ]
                data = pd.read_csv(
                    data_filepath,
                    sep=" ",
                    index_col=False,
                    header=None,
                    names=csv_columns,
                    usecols=["frame_id", "generated"],
                    dtype={"frame_id": int, "generated": bool},
                )
                # Ignoring generated frames in the count here since
                # we will remove them later (we'll do our own interpolation).
                data = data[~data["generated"]]
                data["frame_id"] -= data["frame_id"].min()

                self.dataset_obj[f"{scene_name}_{video_num}"] = (
                    data_filepath,
                    data["frame_id"].max().item() + 1,
                )

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[SDDPedsRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, (scene_name, (scene_filepath, scene_length)) in enumerate(
            self.dataset_obj.items()
        ):
            if scene_name not in self.metadata.scene_split_map:
                raise ValueError()

            scene_split: str = self.metadata.scene_split_map[scene_name]

            # Saving all scene records for later caching.
            all_scenes_list.append(SDDPedsRecord(scene_name, scene_length, idx))

            if (
                "stanford" in scene_tag
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

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        all_scenes_list: List[SDDPedsRecord] = env_cache.load_env_scenes_list(self.name)

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            scene_name, scene_length, data_idx = scene_record
            scene_split: str = self.metadata.scene_split_map[scene_name]

            if (
                "stanford" in scene_tag
                and scene_split in scene_tag
                and scene_desc_contains is None
            ):
                scene_metadata = Scene(
                    self.metadata,
                    scene_name,
                    "stanford",
                    scene_split,
                    scene_length,
                    data_idx,
                    None,  # This isn't used if everything is already cached.
                )
                scenes_list.append(scene_metadata)

        return scenes_list

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        _, scene_name, _, data_idx = scene_info

        _, scene_length = self.dataset_obj[scene_name]
        scene_split: str = self.metadata.scene_split_map[scene_name]

        return Scene(
            self.metadata,
            scene_name,
            "stanford",
            scene_split,
            scene_length,
            data_idx,
            None,  # No data access info necessary for the ETH/UCY datasets.
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_filepath, _ = self.dataset_obj[scene.name]

        csv_columns = [
            "agent_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "frame_id",
            "lost",
            "occluded",
            "generated",
            "label",
        ]
        data_df: pd.DataFrame = pd.read_csv(
            scene_filepath,
            sep=" ",
            index_col=False,
            header=None,
            names=csv_columns,
            dtype={"generated": bool},
        )

        # Setting generated frames to NaN, we'll do our own interpolation later.
        data_df.loc[data_df["generated"], ["x_min", "y_min"]] = np.nan
        data_df["frame_id"] -= data_df["frame_id"].min()

        scale: float = SDD_HOMOGRAPHY_SCALES[scene.name]["scale"]
        data_df["x"] = scale * (data_df["x_min"] + data_df["x_max"]) / 2.0
        data_df["y"] = scale * (data_df["y_min"] + data_df["y_max"]) / 2.0

        # Don't need these columns anymore.
        data_df.drop(
            columns=[
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "lost",
                "occluded",
                "generated",
            ],
            inplace=True,
        )

        # Renaming columns to match our usual names.
        data_df.rename(
            columns={"frame_id": "scene_ts", "label": "agent_type"},
            inplace=True,
        )

        # Ensuring data is sorted by agent ID and scene timestep.
        data_df.set_index(["agent_id", "scene_ts"], inplace=True)
        data_df.sort_index(inplace=True)

        # Re-interpolating because the original SDD interpolation yielded discrete position steps,
        # which is not very natural. Also, the data is already sorted by agent and time so
        # we can safely do this without worrying about contaminating position data across agents.
        data_df.interpolate(
            method="linear", axis="index", inplace=True, limit_area="inside"
        )

        data_df.reset_index(level=1, inplace=True)

        agent_ids: np.ndarray = data_df.index.get_level_values(0).to_numpy()

        # Add in zero for z value
        data_df["z"] = np.zeros_like(data_df["x"])

        ### Calculating agent classes
        agent_class: Dict[int, str] = (
            data_df.groupby("agent_id")["agent_type"].first().to_dict()
        )

        ### Calculating agent velocities
        data_df[["vx", "vy"]] = (
            arr_utils.agent_aware_diff(data_df[["x", "y"]].to_numpy(), agent_ids)
            / SDDPEDS_DT
        )

        ### Calculating agent accelerations
        data_df[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(data_df[["vx", "vy"]].to_numpy(), agent_ids)
            / SDDPEDS_DT
        )

        # This is likely to be very noisy... Unfortunately, SDD only
        # provides center of mass data.
        data_df["heading"] = np.arctan2(data_df["vy"], data_df["vx"])

        agent_list: List[AgentMetadata] = list()
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        for agent_id, frames in data_df.groupby("agent_id")["scene_ts"]:
            start_frame: int = frames.iat[0].item()
            last_frame: int = frames.iat[-1].item()

            agent_type: AgentType = sdd_type_to_unified_type(agent_class[agent_id])

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=agent_type,
                first_timestep=start_frame,
                last_timestep=last_frame,
                # These values are as ballpark as it gets... It's not super reliable to use
                # the pixel extents in the annotations since they are all always axis-aligned.
                extent=FixedExtent(0.75, 0.75, 1.5),
            )

            agent_list.append(agent_metadata)
            for frame in frames:
                agent_presence[frame].append(agent_metadata)

        # Changing the agent_id dtype to str
        data_df.reset_index(inplace=True)
        data_df["agent_id"] = data_df["agent_id"].astype(str)
        data_df.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            data_df,
            cache_path,
            scene,
        )

        return agent_list, agent_presence

    def cache_map(
        self,
        map_name: str,
        layer_names: List[str],
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        resolution: float,
    ) -> None:
        """
        No maps in this dataset!
        """
        pass

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
    ) -> None:
        """
        No maps in this dataset!
        """
        pass
