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
from omegaconf import OmegaConf

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

    @classmethod
    def from_path(
        cls,
        data_path: str | Path,
        *,
        smooth_trajectories: bool,
        max_cache_size: int | None = None,
    ) -> ArtifactSceneProvider:
        """Build an artifact-backed provider from a USDZ file or directory."""
        path = Path(data_path)
        glob_query = str(path) if path.suffix == ".usdz" else str(path / "**/*.usdz")
        discovered = Artifact.discover_from_glob(
            glob_query,
            smooth_trajectories=smooth_trajectories,
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

        return cls(
            artifact_paths,
            sorted(scene_infos, key=lambda scene_info: scene_info.scene_id),
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


# TODO(mwatson, caojun): Add a TrajdataSceneProvider that loads scenes from a UnifiedDataset


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
    def scene_ids(self) -> set[str]:
        return set(self._scene_ids)

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
        raise NotImplementedError("trajdata scene provider is not yet implemented")
    raise ValueError(f"Unsupported scene_provider.kind: {kind!r}")


def _build_artifact_scene_provider(
    *,
    user_config: UserSimulatorConfig,
    usdz_provider_config: UsdzProviderConfig,
) -> ArtifactSceneProvider:
    if usdz_provider_config.data_dir is None:
        raise ValueError("scene_provider.usdz.data_dir is required")

    return ArtifactSceneProvider.from_path(
        usdz_provider_config.data_dir,
        smooth_trajectories=user_config.smooth_trajectories,
        max_cache_size=usdz_provider_config.artifact_cache_size,
    )
