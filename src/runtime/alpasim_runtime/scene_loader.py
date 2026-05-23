# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Scene loading registry for artifact-backed and trajdata-backed scenes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from alpasim_utils.scene_metadata import Metadata
from alpasim_utils.trajdata_data_source import TrajdataDataSource
from omegaconf import OmegaConf
from trajdata.dataset import UnifiedDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneInfo:
    """Catalog entry for one scene known to the runtime."""

    scene_id: str
    provider_kind: str
    metadata: Metadata


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


def scene_names_from_dataset_index(
    dataset: UnifiedDataset,
) -> dict[str, int] | None:
    """Resolve scene names from trajdata's ``_scene_index`` without loading.

    trajdata stores scene cache paths as
    ``{cache}/{env_name}/{scene_name}/scene_metadata_dt{dt:.2f}.dill``, so the
    scene name is just ``path.parent.name``. Parsing the path is O(N) string
    work instead of O(N) ``dill.load`` calls, which is 4–5 orders of magnitude
    faster on datasets like NuPlan with 10k+ scenes.

    Returns ``None`` if the index doesn't exist or the layout doesn't match the
    expected structure, so callers can fall back to ``dataset.get_scene``.

    Note: ``_scene_index`` is a trajdata-internal attribute. If trajdata changes
    its cache layout, this returns ``None`` and we degrade to the slow path.
    """
    scene_index = getattr(dataset, "_scene_index", None)
    if not scene_index:
        return None

    result: dict[str, int] = {}
    for idx, raw_path in enumerate(scene_index):
        path = Path(raw_path)
        scene_name = path.parent.name
        if not scene_name or scene_name == path.anchor:
            return None
        result[scene_name] = idx
    return result


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
    def provider_kind(self) -> str:
        """Return the provider backend kind."""
        ...

    @property
    def scene_ids(self) -> set[str]:
        """Return the scene IDs owned by this provider."""
        ...

    @property
    def scene_infos(self) -> list[SceneInfo]:
        """Return lightweight catalog entries for scenes owned by this provider."""
        ...

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        """Load a scene data source for the given scene ID."""
        ...


class ArtifactSceneProvider:
    """Scene provider for direct USDZ artifact loading."""

    def __init__(
        self,
        artifact_paths: dict[str, str],
        scene_infos: list[SceneInfo],
        *,
        smooth_trajectories: bool,
        max_cache_size: int | None = None,
    ) -> None:
        self._artifact_paths = dict(artifact_paths)
        self._scene_infos = list(scene_infos)
        self._load_artifact = make_artifact_loader(
            smooth_trajectories=smooth_trajectories,
            max_cache_size=max_cache_size,
        )

    @property
    def provider_kind(self) -> str:
        return "usdz"

    @property
    def scene_ids(self) -> set[str]:
        return set(self._artifact_paths)

    @property
    def scene_infos(self) -> list[SceneInfo]:
        return list(self._scene_infos)

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        if scene_id not in self._artifact_paths:
            raise UnknownSceneError(scene_id)
        return self._load_artifact(scene_id, self._artifact_paths[scene_id])


class TrajdataSceneProvider:
    """Scene provider that loads scenes from trajdata's UnifiedDataset."""

    def __init__(
        self,
        dataset: UnifiedDataset,
        scene_id_to_idx: dict[str, int],
        *,
        smooth_trajectories: bool,
        asset_base_path_map: dict[str, str] | None = None,
    ) -> None:
        self._dataset = dataset
        self._scene_id_to_idx = scene_id_to_idx
        self._smooth_trajectories = smooth_trajectories
        self._asset_base_path_map = asset_base_path_map or {}
        self._cache: dict[str, SceneDataSource] = {}

    @property
    def scene_ids(self) -> set[str]:
        return set(self._scene_id_to_idx)

    def get_data_source(self, scene_id: str) -> SceneDataSource:
        if scene_id in self._cache:
            return self._cache[scene_id]

        scene_idx = self._scene_id_to_idx.get(scene_id)
        if scene_idx is None:
            raise UnknownSceneError(scene_id)

        scene = self._dataset.get_scene(scene_idx)
        if scene is None:
            raise UnknownSceneError(scene_id)

        asset_base_path = self._asset_base_path_map.get(scene.env_name)
        map_api = getattr(self._dataset, "_map_api", None)
        vector_map_params = getattr(self._dataset, "vector_map_params", None)

        scene_cache = self._dataset.cache_class(
            self._dataset.cache_path, scene, self._dataset.augmentations
        )
        scene_cache.set_obs_format(TrajdataDataSource.OBS_FORMAT)

        data_source = TrajdataDataSource.from_trajdata_scene(
            scene=scene,
            map_api=map_api,
            vector_map_params=vector_map_params,
            scene_cache=scene_cache,
            smooth_trajectories=self._smooth_trajectories,
            asset_base_path=asset_base_path,
        )

        self._cache[scene_id] = data_source
        return data_source


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

    @property
    def scene_infos(self) -> list[SceneInfo]:
        return self._provider.scene_infos


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
) -> SceneProvider:
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
    scene_infos: list[SceneInfo] = []
    for scene_id, artifact in discovered.items():
        existing = artifact_paths.get(scene_id)
        if existing is not None:
            raise ValueError(
                f"Duplicate scene_id {scene_id!r} discovered from USDZ sources "
                f"{existing!r} and {artifact.source!r}"
            )
        artifact_paths[scene_id] = artifact.source
        scene_infos.append(
            SceneInfo(
                scene_id=scene_id,
                provider_kind="usdz",
                metadata=artifact.metadata,
            )
        )

    return ArtifactSceneProvider(
        artifact_paths,
        sorted(scene_infos, key=lambda scene_info: scene_info.scene_id),
        smooth_trajectories=user_config.smooth_trajectories,
        max_cache_size=usdz_provider_config.artifact_cache_size,
    )


def _build_trajdata_scene_provider(
    *,
    user_config: UserSimulatorConfig,
    trajdata_provider_config: TrajdataProviderConfig,
) -> TrajdataSceneProvider:
    params = trajdata_provider_config_to_params(trajdata_provider_config)

    logger.info("Creating UnifiedDataset with desired_data=%s", params["desired_data"])
    dataset = UnifiedDataset(**params)
    logger.info("Created UnifiedDataset with %d scenes", dataset.num_scenes())

    scene_id_to_idx = scene_names_from_dataset_index(dataset)
    if scene_id_to_idx is None:
        logger.warning(
            "trajdata _scene_index unavailable; falling back to per-scene "
            "deserialization. This may be slow for large datasets."
        )
        scene_id_to_idx = {}
        for idx in range(dataset.num_scenes()):
            try:
                scene = dataset.get_scene(idx)
                scene_id_to_idx[scene.name] = idx
            except Exception as e:
                logger.warning("Failed to get scene at index %d: %s", idx, e)
                continue
    logger.info("Built scene_id mapping for %d scenes", len(scene_id_to_idx))

    asset_base_path_map: dict[str, str] = {}
    if (
        trajdata_provider_config.dataset is not None
        and trajdata_provider_config.dataset.extra_params
    ):
        asset_base_path = trajdata_provider_config.dataset.extra_params.get(
            "asset_base_path"
        )
        if asset_base_path is not None and trajdata_provider_config.dataset.name:
            asset_base_path_map[trajdata_provider_config.dataset.name] = asset_base_path

    return TrajdataSceneProvider(
        dataset=dataset,
        scene_id_to_idx=scene_id_to_idx,
        smooth_trajectories=user_config.smooth_trajectories,
        asset_base_path_map=asset_base_path_map,
    )
