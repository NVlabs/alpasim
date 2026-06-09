# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Type, Union

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import (
    AgentMetadata,
    EnvMetadata,
    Scene,
    SceneMetadata,
    SceneTag,
)


class RawDataset:
    def __init__(
        self, env_name: str, data_dir: str, parallelizable: bool, has_maps: bool
    ) -> None:
        metadata = self.compute_metadata(env_name, data_dir)

        self.metadata = metadata
        self.name = metadata.name
        self.scene_tags = metadata.scene_tags
        self.dataset_obj = None

        self.parallelizable = parallelizable
        self.has_maps = has_maps

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        raise NotImplementedError()

    def get_matching_scene_tags(self, query: Set[str]) -> List[SceneTag]:
        return [scene_tag for scene_tag in self.scene_tags if scene_tag.contains(query)]

    def load_dataset_obj(self, verbose: bool = False) -> None:
        raise NotImplementedError()

    def del_dataset_obj(self) -> None:
        del self.dataset_obj
        self.dataset_obj = None

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[Scene]:
        raise NotImplementedError()

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        raise NotImplementedError()

    def cache_all_scenes_list(
        self, env_cache: EnvCache, all_scenes_list: List[NamedTuple]
    ) -> None:
        env_cache.save_env_scenes_list(self.name, all_scenes_list)

    def get_matching_scenes(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Optional[List[str]],
        env_cache: EnvCache,
        rebuild_cache: bool,
    ) -> Union[List[Scene], List[SceneMetadata]]:
        if self.dataset_obj is None and not rebuild_cache:
            return self._get_matching_scenes_from_cache(
                scene_tag, scene_desc_contains, env_cache
            )
        else:
            return self._get_matching_scenes_from_obj(
                scene_tag, scene_desc_contains, env_cache
            )

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        raise NotImplementedError()

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        """
        Get frame-level information from source dataset, caching it
        to cache_path.

        Always called after cache_maps, can load map if needed
        to associate map information to positions.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
