# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Scene loading registry for artifact-backed and trajdata-backed scenes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from alpasim_runtime.config import (
    SceneProviderConfig,
    TrajdataProviderConfig,
    UsdzProviderConfig,
    UserSimulatorConfig,
)
from alpasim_runtime.errors import UnknownSceneError
from alpasim_runtime.worker.artifact_cache import make_artifact_loader
from alpasim_utils.artifact import Artifact
from alpasim_utils.scene_data_source import SceneDataSource
from alpasim_utils.trajdata_data_source import TrajdataDataSource
from omegaconf import OmegaConf
from trajdata.dataset import UnifiedDataset

logger = logging.getLogger(__name__)


def build_trajdata_params(
    *,
    desired_data: list[str],
    data_dirs: dict[str, str],
    cache_location: str,
    incl_vector_map: bool,
    rebuild_cache: bool,
    rebuild_maps: bool,
    num_workers: int,
    desired_dt: float,
    dataset_kwargs: dict[str, dict] | None = None,
) -> dict:
    """Build UnifiedDataset kwargs from normalized trajdata config values."""
    params = {
        "desired_data": desired_data,
        "data_dirs": data_dirs,
        "cache_location": cache_location,
        "incl_vector_map": incl_vector_map,
        "rebuild_cache": rebuild_cache,
        "rebuild_maps": rebuild_maps,
        "num_workers": num_workers,
        "desired_dt": desired_dt,
    }
    if dataset_kwargs:
        params["dataset_kwargs"] = dataset_kwargs
    return params


def trajdata_provider_config_to_params(
    trajdata_provider_config: TrajdataProviderConfig,
) -> dict:
    """Convert TrajdataProviderConfig into UnifiedDataset kwargs."""
    if trajdata_provider_config.dataset is None:
        raise ValueError("scene_provider.trajdata.dataset must be configured")
    if not trajdata_provider_config.dataset.name:
        raise ValueError("scene_provider.trajdata.dataset.name is required")
    if trajdata_provider_config.dataset.data_dir is None:
        raise ValueError("scene_provider.trajdata.dataset.data_dir is required")

    dataset_name = trajdata_provider_config.dataset.name
    dataset_kwargs = None
    if trajdata_provider_config.dataset.extra_params:
        dataset_kwargs = {
            dataset_name: trajdata_provider_config.dataset.extra_params,
        }

    return build_trajdata_params(
        desired_data=[dataset_name],
        data_dirs={dataset_name: trajdata_provider_config.dataset.data_dir},
        cache_location=trajdata_provider_config.cache_location,
        incl_vector_map=trajdata_provider_config.load_vector_map,
        rebuild_cache=trajdata_provider_config.rebuild_cache,
        rebuild_maps=trajdata_provider_config.rebuild_maps,
        num_workers=trajdata_provider_config.num_workers,
        desired_dt=trajdata_provider_config.desired_dt,
        dataset_kwargs=dataset_kwargs,
    )


class SceneProvider(Protocol):
    """Provider interface for resolving scene IDs to SceneDataSource instances."""

    @property
    def scene_ids(self) -> set[str]:
        """Return the scene IDs owned by this provider."""
        ...

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        """Load a scene data source for the given scene ID."""
        ...


class ArtifactSceneProvider:
    """Scene provider for direct USDZ artifact loading."""

    def __init__(
        self,
        artifact_paths: dict[str, str],
        *,
        smooth_trajectories: bool,
        max_cache_size: int | None = None,
    ) -> None:
        self._artifact_paths = dict(artifact_paths)
        self._load_artifact = make_artifact_loader(
            smooth_trajectories=smooth_trajectories,
            max_cache_size=max_cache_size,
        )

    @property
    def scene_ids(self) -> set[str]:
        return set(self._artifact_paths)

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        if scene_id not in self._artifact_paths:
            raise UnknownSceneError(scene_id)
        return self._load_artifact(scene_id, self._artifact_paths[scene_id])


class TrajdataSceneProvider:
    """Scene provider that materializes data sources from trajdata scenes."""

    def __init__(
        self,
        *,
        dataset: UnifiedDataset,
        scene_id_to_idx: dict[str, int],
        user_config: UserSimulatorConfig,
    ) -> None:
        self._dataset = dataset
        self._scene_id_to_idx = dict(scene_id_to_idx)
        self._user_config = user_config
        self._asset_base_path_map = self._build_asset_base_path_map(user_config)
        self._cache: dict[str, SceneDataSource] = {}

    @staticmethod
    def _build_asset_base_path_map(user_config: UserSimulatorConfig) -> dict[str, str]:
        asset_base_path_map: dict[str, str] = {}
        trajdata_provider = user_config.scene_provider.trajdata
        if trajdata_provider is None:
            return asset_base_path_map

        dataset = trajdata_provider.dataset
        if dataset is None or dataset.name is None:
            return asset_base_path_map

        asset_base_path = dataset.extra_params.get("asset_base_path")
        if asset_base_path is not None:
            asset_base_path_map[dataset.name] = asset_base_path
            logger.info(
                "Registered asset_base_path for %s: %s",
                dataset.name,
                asset_base_path,
            )
        return asset_base_path_map

    @property
    def scene_ids(self) -> set[str]:
        return set(self._scene_id_to_idx)

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        cached = self._cache.get(scene_id)
        if cached is not None:
            return cached

        scene_idx = self._scene_id_to_idx.get(scene_id)
        if scene_idx is None:
            raise UnknownSceneError(scene_id)

        try:
            scene = self._dataset.get_scene(scene_idx)
            if scene is None:
                raise UnknownSceneError(scene_id)

            asset_base_path = self._asset_base_path_map.get(scene.env_name)
            map_api = getattr(self._dataset, "_map_api", None)

            scene_cache = self._dataset.cache_class(
                self._dataset.cache_path, scene, self._dataset.augmentations
            )
            scene_cache.set_obs_format(self._dataset.obs_format)

            data_source = TrajdataDataSource.from_trajdata_scene(
                scene=scene,
                dataset=None,
                map_api=map_api,
                scene_cache=scene_cache,
                smooth_trajectories=self._user_config.smooth_trajectories,
                asset_base_path=asset_base_path,
            )
            self._cache[scene_id] = data_source
            return data_source
        except UnknownSceneError:
            raise
        except Exception as exc:
            logger.error("Failed to load scene %s: %s", scene_id, exc)
            raise RuntimeError(f"Scene loading failed for {scene_id}") from exc


class SceneLoader:
    """Scene loader over a single backend-specific provider."""

    def __init__(self, provider: SceneProvider) -> None:
        self._provider = provider
        self._scene_ids = set(provider.scene_ids)

    def has_scene(self, scene_id: str) -> bool:
        return scene_id in self._scene_ids

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        if scene_id not in self._scene_ids:
            raise UnknownSceneError(scene_id)

        data_source = self._provider.get_data_source(scene_id)
        logger.debug(
            "Loaded data source for scene %s via %s",
            scene_id,
            type(self._provider).__name__,
        )
        return data_source

    @property
    def num_scenes(self) -> int:
        return len(self._scene_ids)


def build_scene_loader(user_config: UserSimulatorConfig) -> SceneLoader:
    """Build a SceneLoader from user config.

    The configured backend owns its own cache policy so workers can build a
    long-lived loader locally and reuse scene-local state across jobs.
    """
    scene_provider_config = OmegaConf.to_object(user_config.scene_provider)
    provider = _build_scene_provider(
        user_config=user_config,
        scene_provider_config=scene_provider_config,
    )
    loader = SceneLoader(provider)
    logger.info(
        "Registered %d scenes via %s",
        loader.num_scenes,
        type(provider).__name__,
    )
    return loader


def _build_scene_provider(
    *,
    user_config: UserSimulatorConfig,
    scene_provider_config: SceneProviderConfig,
) -> ArtifactSceneProvider | TrajdataSceneProvider:
    kind = scene_provider_config.kind
    if kind == "usdz":
        if scene_provider_config.usdz is None:
            raise ValueError("scene_provider.usdz must be configured when kind='usdz'")
        return _build_artifact_scene_provider(
            user_config=user_config,
            usdz_provider_config=scene_provider_config.usdz,
        )
    if kind == "trajdata":
        if scene_provider_config.trajdata is None:
            raise ValueError(
                "scene_provider.trajdata must be configured when kind='trajdata'"
            )
        return _build_trajdata_scene_provider(
            user_config=user_config,
            trajdata_provider_config=scene_provider_config.trajdata,
        )
    raise ValueError(f"Unsupported scene_provider.kind: {kind!r}")


def _build_artifact_scene_provider(
    *,
    user_config: UserSimulatorConfig,
    usdz_provider_config: UsdzProviderConfig,
) -> ArtifactSceneProvider:
    if usdz_provider_config.data_dir is None:
        raise ValueError("scene_provider.usdz.data_dir is required")

    path = Path(usdz_provider_config.data_dir)
    glob_query = str(path) if path.suffix == ".usdz" else str(path / "**/*.usdz")
    discovered = Artifact.discover_from_glob(
        glob_query,
        smooth_trajectories=False,
    )
    logger.info(
        "Discovered %d USDZ scenes from %s",
        len(discovered),
        glob_query,
    )

    artifact_paths: dict[str, str] = {}
    for scene_id, artifact in discovered.items():
        existing = artifact_paths.get(scene_id)
        if existing is not None:
            raise ValueError(
                f"Duplicate scene_id {scene_id!r} discovered from USDZ sources "
                f"{existing!r} and {artifact.source!r}"
            )
        artifact_paths[scene_id] = artifact.source

    return ArtifactSceneProvider(
        artifact_paths,
        smooth_trajectories=user_config.smooth_trajectories,
        max_cache_size=usdz_provider_config.artifact_cache_size,
    )


def _build_trajdata_scene_provider(
    *,
    user_config: UserSimulatorConfig,
    trajdata_provider_config: TrajdataProviderConfig,
) -> TrajdataSceneProvider:
    logger.info("Creating UnifiedDataset from trajdata-backed source")
    trajdata_params = trajdata_provider_config_to_params(trajdata_provider_config)
    dataset = UnifiedDataset(**trajdata_params)
    logger.info(
        "Created UnifiedDataset with %d scenes for desired_data=%s",
        dataset.num_scenes(),
        trajdata_params["desired_data"],
    )

    scene_id_to_idx: dict[str, int] = {}
    for idx in range(dataset.num_scenes()):
        try:
            scene = dataset.get_scene(idx)
            scene_id_to_idx[scene.name] = idx
        except Exception as exc:
            logger.warning("Failed to get scene at index %d: %s", idx, exc)

    logger.info("Built scene_id mapping for %d trajdata scenes", len(scene_id_to_idx))
    return TrajdataSceneProvider(
        dataset=dataset,
        scene_id_to_idx=scene_id_to_idx,
        user_config=user_config,
    )
