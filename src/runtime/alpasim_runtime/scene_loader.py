# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""SceneLoader: Manages scene data loading and caching."""

from __future__ import annotations

import logging

from alpasim_runtime.config import SimulatorConfig
from alpasim_runtime.daemon.exceptions import UnknownSceneError
from alpasim_utils.scene_data_source import SceneDataSource
from alpasim_utils.trajdata_data_source import TrajdataDataSource
from omegaconf import OmegaConf
from trajdata.dataset import UnifiedDataset

logger = logging.getLogger(__name__)


class SceneLoader:
    """Manages scene data loading and caching.

    Encapsulates UnifiedDataset, scene ID to index mapping, and lazy loading
    of SceneDataSource objects. Provides a clean interface for on-demand scene
    data access with automatic caching.

    Attributes:
        _dataset: UnifiedDataset for accessing trajdata scenes
        _scene_id_to_idx: Mapping from scene IDs to dataset indices
        _config: Simulator configuration for scene parameters
        _cache: Cache of loaded SceneDataSource objects
        _asset_base_path_map: Mapping from dataset name to asset_base_path
    """

    def __init__(
        self,
        dataset: UnifiedDataset,
        scene_id_to_idx: dict[str, int],
        config: SimulatorConfig,
    ):
        """Initialize SceneLoader with dataset and configuration.

        Args:
            dataset: UnifiedDataset for scene access
            scene_id_to_idx: Mapping from scene ID to dataset index
            config: Simulator configuration
        """
        self._dataset = dataset
        self._scene_id_to_idx = scene_id_to_idx
        self._config = config
        self._cache: dict[str, SceneDataSource] = {}

        # Build dataset_name -> asset_base_path mapping
        self._asset_base_path_map: dict[str, str] = {}
        data_source_config = OmegaConf.to_object(config.user.data_source)

        for dataset_name, source in data_source_config.sources.items():
            asset_base_path = source.extra_params.get("asset_base_path")
            if asset_base_path is not None:
                self._asset_base_path_map[dataset_name] = asset_base_path
                logger.info(
                    f"Registered asset_base_path for {dataset_name}: {asset_base_path}"
                )

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        """Get or create a data source for the given scene_id.

        Implements lazy loading with caching. On first access, creates a
        TrajdataDataSource from the UnifiedDataset scene. Subsequent accesses
        return the cached instance.

        Args:
            scene_id: Scene identifier to load

        Returns:
            SceneDataSource for the requested scene

        Raises:
            UnknownSceneError: If scene_id is not found in the dataset
            RuntimeError: If scene loading fails
        """
        # Check cache first
        if scene_id in self._cache:
            return self._cache[scene_id]

        # Validate scene exists
        scene_idx = self._scene_id_to_idx.get(scene_id)
        if scene_idx is None:
            raise UnknownSceneError(scene_id)

        try:
            # Load scene from dataset
            scene = self._dataset.get_scene(scene_idx)
            if scene is None:
                raise UnknownSceneError(scene_id)

            # Get asset_base_path for this scene's dataset
            # Use scene.env_name to lookup the correct asset_base_path
            asset_base_path = self._asset_base_path_map.get(scene.env_name)

            # Get map_api for lazy loading (lightweight, can be passed to workers)
            map_api = getattr(self._dataset, "_map_api", None)

            # Create scene_cache (pre-create to avoid pickle errors)
            scene_cache = self._dataset.cache_class(
                self._dataset.cache_path, scene, self._dataset.augmentations
            )
            scene_cache.set_obs_format(self._dataset.obs_format)

            # Create TrajdataDataSource
            data_source = TrajdataDataSource.from_trajdata_scene(
                scene=scene,
                dataset=None,  # Don't pass dataset to avoid pickle errors
                map_api=map_api,  # Pass map_api
                scene_cache=scene_cache,
                smooth_trajectories=self._config.user.smooth_trajectories,
                asset_base_path=asset_base_path,
            )

            # Cache for future use
            self._cache[scene_id] = data_source
            logger.debug(f"Loaded data source for scene {scene_id}")
            return data_source

        except Exception as e:
            logger.error(f"Failed to load scene {scene_id}: {e}")
            raise RuntimeError(f"Scene loading failed for {scene_id}") from e

    @property
    def num_scenes(self) -> int:
        """Return total number of scenes available in the dataset."""
        return self._dataset.num_scenes()

    @property
    def num_cached(self) -> int:
        """Return number of scenes currently cached."""
        return len(self._cache)
