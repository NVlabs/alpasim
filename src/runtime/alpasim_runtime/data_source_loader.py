# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Runtime data-source loading for artifact-backed and trajdata-backed data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from alpasim_runtime.config import (
    DataSourceProviderConfig,
    TrajdataProviderConfig,
    UsdzProviderConfig,
    UserSimulatorConfig,
)
from alpasim_runtime.errors import UnknownClipError
from alpasim_runtime.worker.artifact_cache import make_artifact_loader
from alpasim_utils.artifact import Artifact
from alpasim_utils.scene_data_source import SceneDataSource
from alpasim_utils.scene_metadata import Metadata
from alpasim_utils.types import ClipDataSource, DataSource
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSourceInfo:
    """Catalog entry for one data source known to the runtime."""

    source: DataSource
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
        raise ValueError("data_source_provider.trajdata.dataset must be configured")
    if not trajdata_provider_config.dataset.name:
        raise ValueError("data_source_provider.trajdata.dataset.name is required")
    if trajdata_provider_config.dataset.data_dir is None:
        raise ValueError("data_source_provider.trajdata.dataset.data_dir is required")

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


class DataSourceProvider(Protocol):
    """Provider interface for resolving descriptors to loaded data sources."""

    @property
    def data_source_infos(self) -> list[DataSourceInfo]:
        """Return lightweight catalog entries owned by this provider."""
        ...

    def get_data_source(self, source_id: str) -> SceneDataSource:
        """Load the data source for the given source ID."""
        ...


class ArtifactDataSourceProvider:
    """Data-source provider for direct USDZ artifact loading."""

    def __init__(
        self,
        artifact_paths: dict[str, str],
        data_source_infos: list[DataSourceInfo],
        *,
        smooth_trajectories: bool,
        max_cache_size: int | None = None,
    ) -> None:
        self._artifact_paths = dict(artifact_paths)
        self._data_source_infos = list(data_source_infos)
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
    ) -> ArtifactDataSourceProvider:
        """Build an artifact-backed provider from a USDZ file or directory."""
        path = Path(data_path)
        glob_query = str(path) if path.suffix == ".usdz" else str(path / "**/*.usdz")
        discovered = Artifact.discover_from_glob(
            glob_query,
            smooth_trajectories=smooth_trajectories,
        )
        logger.info(
            "Discovered %d USDZ clips from %s",
            len(discovered),
            glob_query,
        )

        artifact_paths: dict[str, str] = {}
        data_source_infos: list[DataSourceInfo] = []
        for clip_id, artifact in discovered.items():
            existing = artifact_paths.get(clip_id)
            if existing is not None:
                raise ValueError(
                    f"Duplicate clip_id {clip_id!r} discovered from USDZ sources "
                    f"{existing!r} and {artifact.source!r}"
                )
            artifact_paths[clip_id] = artifact.source
            data_source_infos.append(
                DataSourceInfo(
                    source=ClipDataSource(
                        clip_id=clip_id,
                        usdz_uuid=str(artifact.metadata.uuid),
                    ),
                    metadata=artifact.metadata,
                )
            )

        return cls(
            artifact_paths,
            sorted(
                data_source_infos,
                key=lambda data_source_info: data_source_info.source.source_id,
            ),
            smooth_trajectories=smooth_trajectories,
            max_cache_size=max_cache_size,
        )

    @property
    def data_source_infos(self) -> list[DataSourceInfo]:
        return list(self._data_source_infos)

    def get_data_source(self, source_id: str) -> SceneDataSource:
        if source_id not in self._artifact_paths:
            raise UnknownClipError(source_id)
        return self._load_artifact(self._artifact_paths[source_id])


# TODO(mwatson, caojun): Add a TrajdataClipProvider that loads scenes from a UnifiedDataset


class DataSourceLoader:
    """Loader over a single backend-specific data-source provider."""

    def __init__(self, provider: DataSourceProvider) -> None:
        self._provider = provider
        self._source_ids = {
            info.source.source_id for info in provider.data_source_infos
        }

    def has_data_source(self, source_id: str) -> bool:
        return source_id in self._source_ids

    def get_data_source(self, source_id: str) -> SceneDataSource:
        if source_id not in self._source_ids:
            raise UnknownClipError(source_id)

        data_source = self._provider.get_data_source(source_id)
        logger.debug(
            "Loaded data source %s via %s",
            source_id,
            type(self._provider).__name__,
        )
        return data_source

    @property
    def source_ids(self) -> set[str]:
        return set(self._source_ids)

    @property
    def data_source_infos(self) -> list[DataSourceInfo]:
        return self._provider.data_source_infos


def build_data_source_loader(user_config: UserSimulatorConfig) -> DataSourceLoader:
    """Build a DataSourceLoader from user config.

    The configured backend owns its own cache policy so workers can build a
    long-lived loader locally and reuse clip-local state across jobs.
    """
    data_source_provider_config = OmegaConf.to_object(user_config.data_source_provider)
    provider = _build_data_source_provider(
        user_config=user_config,
        data_source_provider_config=data_source_provider_config,
    )
    loader = DataSourceLoader(provider)
    logger.info(
        "Registered %d data sources via %s",
        len(loader.source_ids),
        type(provider).__name__,
    )
    return loader


def _build_data_source_provider(
    *,
    user_config: UserSimulatorConfig,
    data_source_provider_config: DataSourceProviderConfig,
) -> DataSourceProvider:
    kind = data_source_provider_config.kind
    if kind == "usdz":
        if data_source_provider_config.usdz is None:
            raise ValueError(
                "data_source_provider.usdz must be configured when kind='usdz'"
            )
        return _build_artifact_data_source_provider(
            user_config=user_config,
            usdz_provider_config=data_source_provider_config.usdz,
        )
    if kind == "trajdata":
        raise NotImplementedError("trajdata clip provider is not yet implemented")
    raise ValueError(f"Unsupported data_source_provider.kind: {kind!r}")


def _build_artifact_data_source_provider(
    *,
    user_config: UserSimulatorConfig,
    usdz_provider_config: UsdzProviderConfig,
) -> ArtifactDataSourceProvider:
    if usdz_provider_config.data_dir is None:
        raise ValueError("data_source_provider.usdz.data_dir is required")

    return ArtifactDataSourceProvider.from_path(
        usdz_provider_config.data_dir,
        smooth_trajectories=user_config.smooth_trajectories,
        max_cache_size=usdz_provider_config.artifact_cache_size,
    )
