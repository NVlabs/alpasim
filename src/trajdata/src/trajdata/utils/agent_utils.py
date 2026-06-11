# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Optional, Type

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import Scene, SceneMetadata
from trajdata.dataset_specific import RawDataset
from trajdata.utils import scene_utils


def get_agent_data(
    scene_info: SceneMetadata,
    raw_dataset: RawDataset,
    env_cache: EnvCache,
    rebuild_cache: bool,
    cache_class: Type[SceneCache],
    desired_dt: Optional[float] = None,
) -> Scene:
    if not rebuild_cache and env_cache.scene_is_cached(
        scene_info.env_name, scene_info.name, scene_info.dt
    ):
        scene: Scene = env_cache.load_scene(
            scene_info.env_name, scene_info.name, scene_info.dt
        )

        # If the original data is already cached...
        if scene_utils.enforce_desired_dt(scene_info, desired_dt, dry_run=True):
            # If the original data is already cached,
            # but this scene's dt doesn't match what we desire:
            # First, interpolate and save the data.
            # Then, return the interpolated scene.

            # Interpolating the scene metadata and caching it.
            scene_utils.enforce_desired_dt(scene, desired_dt)
            env_cache.save_scene(scene)

            # Interpolating the agent data and caching it.
            # The core point of doing this here rather than in Line 45 and below
            # is that we do not need to access the raw dataset object, we can
            # leverage the already cached data.
            scene_cache: SceneCache = cache_class(env_cache.path, scene)
            scene_cache.write_cache_to_disk()

        # Once this scene's dt matches what we desire: Return it.
        return scene

    # Obtaining and caching the original scene data.
    scene: Scene = raw_dataset.get_scene(scene_info)
    agent_list, agent_presence = raw_dataset.get_agent_info(
        scene, env_cache.path, cache_class
    )
    if agent_list is None and agent_presence is None:
        raise ValueError(f"Scene {scene_info.name} contains no agents!")

    scene.update_agent_info(agent_list, agent_presence)
    env_cache.save_scene(scene)

    if scene_utils.enforce_desired_dt(scene, desired_dt, dry_run=True):
        # In case the user specified a desired_dt that's different from the scene's
        # native dt, we will perform the interpolation here and cache the result for
        # later reuse.

        # Interpolating the scene metadata and caching it.
        scene_utils.enforce_desired_dt(scene, desired_dt)
        env_cache.save_scene(scene)

        # Interpolating the agent data and caching it.
        scene_cache: SceneCache = cache_class(env_cache.path, scene)
        scene_cache.write_cache_to_disk()

    return scene
