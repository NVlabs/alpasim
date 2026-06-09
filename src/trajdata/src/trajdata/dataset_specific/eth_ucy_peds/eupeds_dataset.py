# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.data_structures.environment import EnvMetadata
from trajdata.data_structures.scene_metadata import Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import EUPedsRecord
from trajdata.utils import arr_utils

TRAIN_SCENES: Final[List[str]] = [
    "biwi_eth",
    "biwi_hotel",
    "crowds_zara01",
    "crowds_zara02",
    "crowds_zara03",
    "students001",
    "students003",
    "uni_examples",
]
TRAINVAL_FRAME_SPLITS: Final[Dict[str, int]] = {
    # Note: These split indices are applied after dividing the
    # frame number by 10 (they are only annotated every 10 frames) and
    # subtracting the frame_id of the first frame (it is not always 0, for
    # biwi_eth it is 78 and for crowds_zara02 it is 1, hence the subtractions here).
    "biwi_eth": 1024 - 78,
    "biwi_hotel": 1440,
    "crowds_zara01": 711,
    "crowds_zara02": 842 - 1,
    "crowds_zara03": 603,
    "students001": 355,  # => train: [start, 355), val: [355, end)
    "students003": 432,
    "uni_examples": 594,
}
TEST_SCENES: Final[Dict[str, List[str]]] = {
    "eupeds_eth": ["biwi_eth"],
    "eupeds_hotel": ["biwi_hotel"],
    "eupeds_univ": ["students001", "students003"],
    "eupeds_zara1": ["crowds_zara01"],
    "eupeds_zara2": ["crowds_zara02"],
}
EUPEDS_DT: Final[float] = 0.4


def get_location(scene_name: str) -> str:
    if "eth" in scene_name or "hotel" in scene_name:
        return "zurich"
    else:
        return "cyprus"


class EUPedsDataset(RawDataset):
    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        scene_splits: Dict[str, List[str]] = {
            "train": [sn + "_train" for sn in TEST_SCENES[env_name]],
            "val": [sn + "_val" for sn in TEST_SCENES[env_name]],
            "train_loo": [
                sn + "_train" for sn in TRAIN_SCENES if sn not in TEST_SCENES[env_name]
            ],
            "val_loo": [
                sn + "_val" for sn in TRAIN_SCENES if sn not in TEST_SCENES[env_name]
            ],
            "test_loo": TEST_SCENES[env_name],
        }

        # ETH/UCY possibilities are the Cartesian product of these,
        # but note that some may not exist, such as ("eth", "train", "cyprus").
        # "*_loo" = Leave One Out (this is how the ETH/UCY dataset
        # is most commonly used).
        dataset_parts: List[Tuple[str, ...]] = [
            ("train", "val"),
            ("zurich", "cyprus"),
        ]
        dataset_parts_loo: List[str] = ["train_loo", "val_loo", "test_loo"]

        # Inverting the dict from above, associating every scene with its data split.
        scene_split_map: Dict[str, str] = {
            v_elem: k for k, v in scene_splits.items() for v_elem in v
        }

        env_metadata = EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=EUPEDS_DT,
            parts=dataset_parts,
            scene_split_map=scene_split_map,
        )
        env_metadata.scene_tags += [
            SceneTag(
                (
                    env_name,
                    loo_part,
                )
            )
            for loo_part in dataset_parts_loo
        ]
        return env_metadata

    def load_dataset_obj(self, verbose: bool = False) -> None:
        if verbose:
            print(f"Loading {self.name} dataset...", flush=True)

        self.dataset_obj: Dict[str, pd.DataFrame] = dict()
        for scene_name in TRAIN_SCENES:
            data_filepath: Path = Path(self.metadata.data_dir) / (scene_name + ".txt")

            data = pd.read_csv(data_filepath, sep="\t", index_col=False, header=None)
            data.columns = ["frame_id", "track_id", "pos_x", "pos_y"]
            data["frame_id"] = pd.to_numeric(data["frame_id"], downcast="integer")
            data["frame_id"] = (data["frame_id"] - data["frame_id"].min()) // 10

            self.dataset_obj[scene_name] = data
            self.dataset_obj[scene_name + "_train"] = data[
                data["frame_id"] < TRAINVAL_FRAME_SPLITS[scene_name]
            ]

            # Creating a copy because we have to fix the frame_id values (to ensure they start from 0).
            self.dataset_obj[scene_name + "_val"] = data[
                data["frame_id"] >= TRAINVAL_FRAME_SPLITS[scene_name]
            ].copy()
            self.dataset_obj[scene_name + "_val"]["frame_id"] -= TRAINVAL_FRAME_SPLITS[
                scene_name
            ]

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        all_scenes_list: List[EUPedsRecord] = list()

        scenes_list: List[SceneMetadata] = list()
        for idx, (scene_name, scene_df) in enumerate(self.dataset_obj.items()):
            scene_location: str = get_location(scene_name)

            if scene_name not in self.metadata.scene_split_map:
                # This happens when the scene is "test_loo" for another eupeds dataset.
                # For example, we don't care about "biwi_hotel" (which would be
                # the test scene for "eupeds_hotel") when getting scenes for "eupeds_eth".
                continue

            scene_split: str = self.metadata.scene_split_map[scene_name]
            scene_length: int = scene_df["frame_id"].max().item() + 1

            # Saving all scene records for later caching.
            all_scenes_list.append(
                EUPedsRecord(scene_name, scene_location, scene_length, scene_split, idx)
            )

            if (
                (scene_location in scene_tag or "loo" in scene_split)
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
        all_scenes_list: List[EUPedsRecord] = env_cache.load_env_scenes_list(self.name)

        scenes_list: List[Scene] = list()
        for scene_record in all_scenes_list:
            (
                scene_name,
                scene_location,
                scene_length,
                scene_split,
                data_idx,
            ) = scene_record

            if (
                (scene_location in scene_tag or "loo" in scene_split)
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

        scene_data: pd.DataFrame = self.dataset_obj[scene_name]
        scene_location: str = get_location(scene_name)
        scene_split: str = self.metadata.scene_split_map[scene_name]
        scene_length: int = scene_data["frame_id"].max().item() + 1

        return Scene(
            self.metadata,
            scene_name,
            scene_location,
            scene_split,
            scene_length,
            data_idx,
            None,  # No data access info necessary for the ETH/UCY datasets.
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        scene_data: pd.DataFrame = self.dataset_obj[scene.name].copy()
        scene_data.rename(
            columns={
                "frame_id": "scene_ts",
                "track_id": "agent_id",
                "pos_x": "x",
                "pos_y": "y",
            },
            inplace=True,
        )

        scene_data["agent_id"] = pd.to_numeric(
            scene_data["agent_id"], downcast="integer"
        )

        scene_data.set_index(["agent_id", "scene_ts"], inplace=True)
        scene_data.sort_index(inplace=True)
        scene_data.reset_index(level=1, inplace=True)

        agent_ids: np.ndarray = scene_data.index.get_level_values(0).to_numpy()

        # Add in zero for z value
        scene_data["z"] = np.zeros_like(scene_data["x"])

        ### Calculating agent velocities
        scene_data[["vx", "vy"]] = (
            arr_utils.agent_aware_diff(scene_data[["x", "y"]].to_numpy(), agent_ids)
            / EUPEDS_DT
        )

        ### Calculating agent accelerations
        scene_data[["ax", "ay"]] = (
            arr_utils.agent_aware_diff(scene_data[["vx", "vy"]].to_numpy(), agent_ids)
            / EUPEDS_DT
        )

        # This is likely to be very noisy... Unfortunately, ETH/UCY only
        # provide center of mass data.
        scene_data["heading"] = np.arctan2(scene_data["vy"], scene_data["vx"])

        agent_list: List[AgentMetadata] = list()
        agent_presence: List[List[AgentMetadata]] = [
            [] for _ in range(scene.length_timesteps)
        ]
        agents_to_remove: List[int] = list()
        for agent_id, frames in scene_data.groupby("agent_id")["scene_ts"]:
            if frames.shape[0] <= 1:
                # There are some agents with only a single detection to them, we don't care about these.
                agents_to_remove.append(agent_id)
                continue

            start_frame: int = frames.iat[0].item()
            last_frame: int = frames.iat[-1].item()

            if frames.shape[0] < last_frame - start_frame + 1:
                raise ValueError("ETH/UCY indeed can have missing frames :(")

            agent_metadata = AgentMetadata(
                name=str(agent_id),
                agent_type=AgentType.PEDESTRIAN,
                first_timestep=start_frame,
                last_timestep=last_frame,
                # These values are as ballpark as it gets...
                extent=FixedExtent(0.75, 0.75, 1.5),
            )

            agent_list.append(agent_metadata)
            for frame in frames:
                agent_presence[frame].append(agent_metadata)

        # Removing agents with only one detection.
        scene_data.drop(index=agents_to_remove, inplace=True)

        # Changing the agent_id dtype to str
        scene_data.reset_index(inplace=True)
        scene_data["agent_id"] = scene_data["agent_id"].astype(str)
        scene_data.set_index(["agent_id", "scene_ts"], inplace=True)

        cache_class.save_agent_data(
            scene_data,
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
