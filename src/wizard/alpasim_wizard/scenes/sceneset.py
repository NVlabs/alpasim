# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Polars-based scene management for querying and downloading scene artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass

import polars as pl  # type: ignore[import-not-found]
from alpasim_wizard.s3_api import S3Connection, S3Path
from alpasim_wizard.scenes.csv_utils import ArtifactRepository
from alpasim_wizard.schema import ScenesConfig
from filelock import FileLock
from tqdm.asyncio import tqdm  # type: ignore[import-untyped]
from typing_extensions import ClassVar, Self

logger = logging.getLogger("alpasim_wizard")


@dataclass
class SceneIdAndUuid:
    scene_id: str
    uuid: str

    @staticmethod
    def list_from_df(df: pl.DataFrame) -> list[SceneIdAndUuid]:
        if "scene_id" not in df.columns or "uuid" not in df.columns:
            raise ValueError(
                f"DataFrame must have columns 'scene_id' and 'uuid'. Got {df.columns}."
            )
        return [
            SceneIdAndUuid(row["scene_id"], row["uuid"])
            for row in df.iter_rows(named=True)
        ]


class USDZQueryError(Exception):
    """Raised when a USDZ query fails."""


def _deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the most recently modified artifact per scene_id."""
    return df.sort("last_modified", descending=True).unique(
        subset=["scene_id"], keep="first"
    )


@dataclass
class USDZManager:
    """Manager for querying and downloading USDZ scene artifacts."""

    sim_scenes: pl.DataFrame
    sim_suites: pl.DataFrame
    s3: S3Connection
    cache_dir: str

    ALL_USDZ_DIR_NAME: ClassVar[str] = "all-usdzs"
    SCENESETS_DIR_NAME: ClassVar[str] = "scenesets"

    @property
    def scenesets_dir(self) -> str:
        return os.path.join(self.cache_dir, self.SCENESETS_DIR_NAME)

    @property
    def all_usdzs_dir(self) -> str:
        return os.path.join(self.cache_dir, self.ALL_USDZ_DIR_NAME)

    @classmethod
    def from_cfg(cls, cfg: ScenesConfig) -> Self:
        """Create a USDZManager from a ScenesConfig."""
        sim_scenes = pl.read_csv(cfg.scenes_csv)
        sim_suites = pl.read_csv(cfg.suites_csv)

        # Ensure directories exist
        cache_dir = cfg.scene_cache
        if not os.path.isdir(cache_dir):
            raise ValueError(f"Cache directory {cache_dir} does not exist.")

        manager = cls(
            sim_scenes=sim_scenes,
            sim_suites=sim_suites,
            s3=S3Connection.from_env_vars(),
            cache_dir=cache_dir,
        )

        if not os.path.isdir(manager.scenesets_dir):
            logger.warning(f"{manager.scenesets_dir=} doesn't exist. Creating it.")
            os.makedirs(manager.scenesets_dir, exist_ok=True)

        if not os.path.isdir(manager.all_usdzs_dir):
            logger.warning(f"{manager.all_usdzs_dir=} doesn't exist. Creating it.")
            os.makedirs(manager.all_usdzs_dir, exist_ok=True)

        return manager

    def query_by_scene_ids(
        self, scene_ids: list[str], nre_versions: list[str]
    ) -> list[SceneIdAndUuid]:
        """Query scenes by scene IDs and compatible NRE versions."""
        if len(scene_ids) == 0:
            return []

        if len(nre_versions) == 0:
            raise ValueError("At least one nre_version must be provided.")

        df = self.sim_scenes.filter(
            pl.col("scene_id").is_in(scene_ids)
            & pl.col("nre_version_string").is_in(nre_versions)
        ).select(["scene_id", "uuid", "last_modified", "nre_version_string"])

        found = set(df["scene_id"].to_list()) if df.height > 0 else set()
        missing = set(scene_ids) - found
        if missing:
            raise USDZQueryError(
                f"Failed to find scenes for {missing} compatible with {nre_versions=}."
            )

        deduplicated = _deduplicate(df)
        logger.info(
            f"Scenes: \n{deduplicated.select(['scene_id', 'nre_version_string'])}"
        )

        return SceneIdAndUuid.list_from_df(deduplicated)

    def query_by_suite_id(
        self, test_suite_id: str, nre_versions: list[str]
    ) -> list[SceneIdAndUuid]:
        """Query scenes by test suite ID and compatible NRE versions."""
        if len(nre_versions) == 0:
            raise ValueError("At least one nre_version must be provided.")

        # Filter suites first
        suite_scenes = self.sim_suites.filter(pl.col("test_suite_id") == test_suite_id)

        # Left join with scenes filtered by nre_version
        scenes_filtered = self.sim_scenes.filter(
            pl.col("nre_version_string").is_in(nre_versions)
        )

        df = suite_scenes.join(
            scenes_filtered,
            on="scene_id",
            how="left",
        ).select(["uuid", "scene_id", "nre_version_string", "last_modified"])

        if df.height == 0:
            raise USDZQueryError(
                f"Failed to find any scenes for {test_suite_id=} with {nre_versions=}."
            )

        if df["uuid"].null_count() > 0:
            missing = df.filter(pl.col("uuid").is_null())["scene_id"].to_list()
            raise USDZQueryError(
                f"Failed to find some scenes for scene suite {test_suite_id} with {nre_versions=}. "
                f"Missing: {missing}."
                "A sceneset is expected to contain a valid artifact for each scene_id."
            )

        deduplicated = _deduplicate(df)
        logger.info(
            f"Scenes: \n{deduplicated.select(['scene_id', 'nre_version_string'])}"
        )

        return SceneIdAndUuid.list_from_df(deduplicated)

    def get_paths(self, uuids: list[str]) -> dict[str, str]:
        """Get artifact paths for given UUIDs."""
        if not uuids:
            return {}

        df = self.sim_scenes.filter(pl.col("uuid").is_in(uuids)).select(
            ["uuid", "path"]
        )
        return dict(zip(df["uuid"].to_list(), df["path"].to_list()))

    def get_artifact_info(
        self, uuids: list[str]
    ) -> dict[str, tuple[str, ArtifactRepository]]:
        """Get artifact paths and repositories for given UUIDs.

        Args:
            uuids: List of UUIDs to look up.

        Returns:
            Dict mapping uuid to (path, artifact_repository) tuple.
        """
        if not uuids:
            return {}

        # Check if artifact_repository column exists (for backwards compatibility)
        if "artifact_repository" in self.sim_scenes.columns:
            df = self.sim_scenes.filter(pl.col("uuid").is_in(uuids)).select(
                ["uuid", "path", "artifact_repository"]
            )
            result = {}
            for row in df.iter_rows(named=True):
                repo_str = row["artifact_repository"]
                # Handle potential whitespace in CSV values
                if repo_str:
                    repo_str = repo_str.strip()
                try:
                    repo = ArtifactRepository(repo_str)
                except ValueError:
                    # Default to swiftstack for unknown/missing values
                    logger.warning(
                        f"Unknown artifact_repository '{repo_str}' for uuid {row['uuid']}, "
                        "defaulting to swiftstack"
                    )
                    repo = ArtifactRepository.SWIFTSTACK
                result[row["uuid"]] = (row["path"], repo)
            return result
        else:
            # Backwards compatibility: assume all are SwiftStack
            df = self.sim_scenes.filter(pl.col("uuid").is_in(uuids)).select(
                ["uuid", "path"]
            )
            return {
                row["uuid"]: (row["path"], ArtifactRepository.SWIFTSTACK)
                for row in df.iter_rows(named=True)
            }

    async def _download_artifacts(self, uuids: list[str]) -> None:
        """Download artifacts for given UUIDs.

        Supports downloading from multiple artifact repositories:
        - swiftstack: Downloads via S3 API
        - huggingface: Not yet implemented (raises NotImplementedError)
        """
        artifact_info = self.get_artifact_info(uuids)

        # Group by repository type for better logging
        swiftstack_uuids = []
        huggingface_uuids = []
        for uuid, (path, repo) in artifact_info.items():
            if repo == ArtifactRepository.SWIFTSTACK:
                swiftstack_uuids.append(uuid)
            elif repo == ArtifactRepository.HUGGINGFACE:
                huggingface_uuids.append(uuid)

        # Handle HuggingFace (not yet implemented)
        if huggingface_uuids:
            raise NotImplementedError(
                f"HuggingFace artifact downloads are not yet implemented. "
                f"Found {len(huggingface_uuids)} scenes requiring HuggingFace: "
                f"{huggingface_uuids[:5]}{'...' if len(huggingface_uuids) > 5 else ''}"
            )

        # Download SwiftStack artifacts
        if swiftstack_uuids:
            tasks = []
            for uuid in swiftstack_uuids:
                path, _ = artifact_info[uuid]
                cache_path = os.path.join(self.all_usdzs_dir, f"{uuid}.usdz")
                s3_path = S3Path.from_swiftstack(path)
                tasks.append(self.s3.maybe_download_object(s3_path, cache_path))

            logger.info(
                "Downloading %d artifacts from SwiftStack. Downloads are parallel and skip "
                "existing files so the progress bar might not be accurate.",
                len(swiftstack_uuids),
            )
            await tqdm.gather(*tasks)

    def create_sceneset_directory(self, uuids: list[str]) -> str | None:
        """Download artifacts and create symlinked sceneset directory."""
        if not uuids:
            return None

        asyncio.get_event_loop().run_until_complete(self._download_artifacts(uuids))

        # Create sceneset directory with symlinks
        uuids_str = ", ".join(
            [f"'{uuid}'" for uuid in sorted(uuids)]
        )  # sort to make the cache directory deterministic
        sceneset_md5 = hashlib.md5(uuids_str.encode()).hexdigest()
        sceneset_dir = os.path.join(self.scenesets_dir, sceneset_md5)

        with FileLock(f"{sceneset_dir}.lock", mode=0o666):
            os.makedirs(sceneset_dir, exist_ok=True)

            for uuid in uuids:
                # relative path so it doesn't become invalidated when we mount the entire cache
                src_path = f"../../{self.ALL_USDZ_DIR_NAME}/{uuid}.usdz"
                dest_path = os.path.join(sceneset_dir, f"{uuid}.usdz")

                if not os.path.exists(dest_path):
                    os.symlink(src_path, dest_path)
                elif os.readlink(dest_path) != src_path:
                    raise RuntimeError(
                        f"Corrupt sceneset cache? Expected symlink at {dest_path} to point to {src_path}, "
                        f"but it points to {os.readlink(dest_path)}."
                    )

        logger.info(f"Created sceneset directory at {sceneset_dir}")
        return sceneset_dir
