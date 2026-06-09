# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from torch.utils.data import Dataset

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import Scene, SceneMetadata
from trajdata.utils import agent_utils, env_utils


def scene_paths_collate_fn(filled_scenes: List) -> List:
    return filled_scenes


class ParallelDatasetPreprocessor(Dataset):
    def __init__(
        self,
        scene_info_list: List[SceneMetadata],
        envs_dir_dict: Dict[str, str],
        env_cache_path: str,
        desired_dt: Optional[float],
        cache_class: Type[SceneCache],
        rebuild_cache: bool,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env_cache_path = np.array(env_cache_path).astype(np.bytes_)
        self.desired_dt = desired_dt
        self.cache_class = cache_class
        self.rebuild_cache = rebuild_cache
        self.dataset_kwargs = dataset_kwargs or {}

        env_names: List[str] = list(envs_dir_dict.keys())
        scene_idxs_names: List[Tuple[int, str]] = [
            (idx, scene_info.name) for idx, scene_info in enumerate(scene_info_list)
        ]
        scene_name_idxs, scene_names = zip(*scene_idxs_names)

        self.scene_idxs = np.array(
            [scene_info.raw_data_idx for scene_info in scene_info_list], dtype=int
        )
        self.env_name_idxs = np.array(
            [env_names.index(scene_info.env_name) for scene_info in scene_info_list],
            dtype=int,
        )

        self.scene_name_idxs = np.array(scene_name_idxs, dtype=int)
        self.env_names_arr = np.array(env_names).astype(np.bytes_)
        self.scene_names_arr = np.array(scene_names).astype(np.bytes_)
        self.data_dir_arr = np.array(list(envs_dir_dict.values())).astype(np.bytes_)

        self.data_len: int = len(scene_info_list)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> str:
        env_cache_path: Path = Path(str(self.env_cache_path, encoding="utf-8"))
        env_cache: EnvCache = EnvCache(env_cache_path)

        env_idx: int = self.env_name_idxs[idx]
        scene_idx: int = self.scene_name_idxs[idx]

        env_name: str = str(self.env_names_arr[env_idx], encoding="utf-8")

        # Extract dataset-specific kwargs (empty dict if not specified)
        # Note: For nuplan datasets, config_dir has been converted to central_tokens_config
        # in the main process to avoid repeatedly loading YAML files in each worker
        specific_kwargs = self.dataset_kwargs.get(env_name, {})

        raw_dataset = env_utils.get_raw_dataset(
            env_name,
            str(self.data_dir_arr[env_idx], encoding="utf-8"),
            **specific_kwargs,
        )

        scene_name: str = str(self.scene_names_arr[scene_idx], encoding="utf-8")

        scene_info = SceneMetadata(
            env_name, scene_name, raw_dataset.metadata.dt, self.scene_idxs[idx]
        )

        # Leaving verbose False here so that we don't spam
        # stdout with loading messages.
        raw_dataset.load_dataset_obj(verbose=False)
        scene: Scene = agent_utils.get_agent_data(
            scene_info,
            raw_dataset,
            env_cache,
            self.rebuild_cache,
            self.cache_class,
            self.desired_dt,
        )
        raw_dataset.del_dataset_obj()

        if scene is None:
            # This provides an escape hatch in case there's a reason we
            # don't want to add a scene to the list of scenes. As an example,
            # nuPlan has a scene with only a single frame of data which we
            # can't do much with in terms of prediction/planning/etc.
            return None

        scene_path: Path = EnvCache.scene_metadata_path(
            env_cache.path, scene.env_name, scene.name, scene.dt
        )
        return str(scene_path)
