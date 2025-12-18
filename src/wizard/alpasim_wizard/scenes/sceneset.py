# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Polars-based scene management for querying and downloading scene artifacts."""

from __future__ import annotations

import asyncio
import glob
import hashlib
import logging
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import polars as pl  # type: ignore[import-not-found]
import yaml
from alpasim_wizard.s3_api import S3Connection, S3Path
from alpasim_wizard.scenes.csv_utils import HUGGINGFACE_REPO, ArtifactRepository
from alpasim_wizard.schema import ScenesConfig
from filelock import FileLock
from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
from tqdm.asyncio import tqdm  # type: ignore[import-untyped]
from typing_extensions import ClassVar, Self

LOCAL_SUITE_ID = "local"

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


# TODO(mwatson): unify with car2sim.py logic wrt metadata extraction
def scan_local_usdz_directory(usdz_dir: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Scan a local directory for USDZ files and create sim_scenes/sim_suites DataFrames.

    This function reads metadata from each USDZ file to populate the DataFrames.

    Args:
        usdz_dir: Path to directory containing *.usdz files.

    Returns:
        Tuple of (sim_scenes DataFrame, sim_suites DataFrame).

    Raises:
        ValueError: If the directory doesn't exist or contains no USDZ files.
    """
    if not os.path.isdir(usdz_dir):
        raise ValueError(f"Local USDZ directory does not exist: {usdz_dir}")

    usdz_files = glob.glob(os.path.join(usdz_dir, "**/*.usdz"), recursive=True)
    if not usdz_files:
        raise ValueError(f"No *.usdz files found in directory: {usdz_dir}")

    logger.info(f"Scanning {len(usdz_files)} USDZ files in {usdz_dir}")

    scene_rows = []
    suite_rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for usdz_file in usdz_files:
        try:
            with zipfile.ZipFile(usdz_file, "r") as usdz_zip:
                with usdz_zip.open("metadata.yaml") as manifest_file:
                    data = yaml.safe_load(manifest_file)

                    # Extract metadata - similar to car2sim.py logic
                    uuid = data["uuid"]
                    scene_id = data.get("scene_id")
                    nre_version_string = data.get("version_string", "unknown")

                    scene_rows.append(
                        {
                            "uuid": uuid,
                            "scene_id": scene_id,
                            "nre_version_string": nre_version_string,
                            "path": os.path.abspath(usdz_file),
                            "last_modified": now,
                            "artifact_repository": "local",
                        }
                    )

                    suite_rows.append(
                        {
                            "test_suite_id": str(ArtifactRepository.LOCAL),
                            "scene_id": scene_id,
                        }
                    )

        except (zipfile.BadZipFile, KeyError, yaml.YAMLError) as e:
            logger.warning(f"Failed to read metadata from {usdz_file}: {e}")
            continue

    if not scene_rows:
        raise ValueError(
            f"No valid USDZ files with metadata found in directory: {usdz_dir}"
        )

    sim_scenes = pl.DataFrame(scene_rows)
    sim_suites = pl.DataFrame(suite_rows)

    sim_scenes.write_csv(os.path.join(usdz_dir, "sim_scenes.csv"))
    sim_suites.write_csv(os.path.join(usdz_dir, "sim_suites.csv"))

    logger.info(
        f"Found {len(scene_rows)} scenes in local directory. "
        f"Test suite '{str(ArtifactRepository.LOCAL)}' created with all scenes."
    )
    logger.info(f"Generated sim_scenes.csv and sim_suites.csv in {usdz_dir}")

    return sim_scenes, sim_suites


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
        """Create a USDZManager from a ScenesConfig.

        If cfg.local_usdz_dir is set, scans that directory for USDZ files
        instead of reading from CSV files. A "local" test suite is created
        automatically containing all discovered scenes.
        """
        # Handle local USDZ directory mode
        if cfg.local_usdz_dir is not None:
            sim_scenes, sim_suites = scan_local_usdz_directory(cfg.local_usdz_dir)
            # Use the local_usdz_dir as the cache directory for scenesets
            cache_dir = cfg.local_usdz_dir
        else:
            sim_scenes = pl.read_csv(cfg.scenes_csv)
            sim_suites = pl.read_csv(cfg.suites_csv)
            cache_dir = cfg.scene_cache

        # Ensure directories exist
        if not os.path.isdir(cache_dir):
            raise ValueError(f"Cache directory {cache_dir} does not exist.")

        manager = cls(
            sim_scenes=sim_scenes,
            sim_suites=sim_suites,
            s3=S3Connection.from_env_vars(),
            cache_dir=cache_dir,
        )

        if cfg.local_usdz_dir is None:
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
        - huggingface: Downloads from HuggingFace Hub
        - local: No download needed (files already on local filesystem)
        """
        artifact_info = self.get_artifact_info(uuids)

        # Group by repository type for better logging
        swiftstack_uuids = []
        huggingface_uuids = []
        local_uuids = []
        for uuid, (path, repo) in artifact_info.items():
            if repo == ArtifactRepository.SWIFTSTACK:
                swiftstack_uuids.append(uuid)
            elif repo == ArtifactRepository.HUGGINGFACE:
                huggingface_uuids.append(uuid)
            elif repo == ArtifactRepository.LOCAL:
                local_uuids.append(uuid)

        # Local artifacts - no download needed, but verify files exist
        if local_uuids:
            logger.info(
                f"Using {len(local_uuids)} local artifacts (no download needed)"
            )
            for uuid in local_uuids:
                path, _ = artifact_info[uuid]
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Local artifact not found: {path} (uuid={uuid})"
                    )

        # Handle HuggingFace artifacts
        if huggingface_uuids:
            self._download_huggingface_artifacts(huggingface_uuids, artifact_info)

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

    def _download_single_huggingface_artifact(
        self, uuid: str, hf_filepath: str, tmpdir: str
    ) -> None:
        """Download and validate a single HuggingFace artifact.

        Args:
            uuid: The expected UUID of the artifact.
            hf_filepath: The file path within the HuggingFace repository.
            tmpdir: Temporary directory for downloading.
        """
        logger.info(
            f"Downloading HuggingFace artifact for uuid {uuid} from {hf_filepath}"
        )
        downloaded_usdz = hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            repo_type="dataset",
            local_dir=tmpdir,
            filename=hf_filepath,
        )

        # sanity check that the uuid matches what we expect
        with zipfile.ZipFile(downloaded_usdz, "r") as usdz_zip:
            with usdz_zip.open("metadata.yaml") as manifest_file:
                data = yaml.safe_load(manifest_file)
                actual_uuid = data.get("uuid", None)
                if actual_uuid != uuid:
                    raise RuntimeError(
                        f"Downloaded HuggingFace artifact {hf_filepath} "
                        f"has unexpected uuid {actual_uuid}, expected {uuid}."
                    )
                else:
                    os.rename(
                        downloaded_usdz,
                        os.path.join(self.all_usdzs_dir, f"{uuid}.usdz"),
                    )

    def _download_huggingface_artifacts(
        self, uuids: list[str], artifact_info: dict[str, tuple[str, ArtifactRepository]]
    ) -> None:
        """
        Download the required HuggingFace artifacts, rename them based on their metadata uuids,
        and store them in the all_usdzs_dir.

        Downloads are parallelized with a maximum of 5 concurrent downloads.
        """
        missing_uuid_to_filepath = {}
        for uuid in uuids:
            cache_path = os.path.join(self.all_usdzs_dir, f"{uuid}.usdz")
            if not os.path.exists(cache_path):
                missing_uuid_to_filepath[uuid] = artifact_info[uuid][0]
        if missing_uuid_to_filepath:
            logger.info(
                f"Missing {len(missing_uuid_to_filepath)} required HuggingFace artifacts"
            )
            max_workers = min(5, len(missing_uuid_to_filepath))
            with tempfile.TemporaryDirectory(dir=self.scenesets_dir) as tmpdir:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._download_single_huggingface_artifact,
                            uuid,
                            hf_filepath,
                            tmpdir,
                        ): uuid
                        for uuid, hf_filepath in missing_uuid_to_filepath.items()
                    }
                    for future in as_completed(futures):
                        uuid = futures[future]
                        # Re-raise any exceptions from the download
                        future.result()

    def create_sceneset_directory(self, uuids: list[str]) -> str:
        """
        Download artifacts and create symlinked sceneset directory.
        Note: this should not be used for local USDZ directory configurations.
        """
        if not uuids:
            # Might be encountered if doing a build-only run of the wizard
            logger.warning("No scene ids provided--a sceneset dir is not created.")
            return self.scenesets_dir

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
