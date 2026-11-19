# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Polars-based scene management for querying and downloading scene artifacts."""

from __future__ import annotations

import asyncio
import glob
import hashlib
import logging
import os
import posixpath
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import polars as pl
import yaml
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene
from alpasim_wizard.s3_api import S3Connection, S3Path, S3SeekableReader
from alpasim_wizard.scenes.csv_utils import HUGGINGFACE_REPO, ArtifactRepository
from alpasim_wizard.scenes.scene_suite import resolve_scene_suites
from alpasim_wizard.schema import ScenesConfig, SceneSuiteConfig
from filelock import FileLock
from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
from tqdm.asyncio import tqdm  # type: ignore[import-untyped]
from typing_extensions import ClassVar, Self

logger = logging.getLogger("alpasim_wizard")


@dataclass(frozen=True)
class USDZTimingMetadata:
    """Identity and recording bounds needed to validate explicit scene timing."""

    uuid: str
    clip_id: str
    start_us: int
    end_us: int

    @property
    def duration_us(self) -> int:
        """Return the recording duration in microseconds."""
        return self.end_us - self.start_us


def _parse_usdz_timing_metadata(
    raw_metadata: bytes,
    *,
    source: str,
    expected_uuid: str,
    expected_clip_id: str,
) -> USDZTimingMetadata:
    """Parse and validate the identity and time range from ``metadata.yaml``.

    Raises:
        ValueError: The payload is malformed, has non-integer timing bounds,
            has a non-positive duration, or does not match the expected clip.
    """
    try:
        data = yaml.safe_load(raw_metadata)
        if not isinstance(data, dict):
            raise TypeError("metadata root is not a mapping")
        time_range = data["time_range"]
        if not isinstance(time_range, dict):
            raise TypeError("time_range is not a mapping")
        start_us = time_range["start"]
        end_us = time_range["end"]
        if not isinstance(start_us, int) or isinstance(start_us, bool):
            raise TypeError("time_range.start is not an integer")
        if not isinstance(end_us, int) or isinstance(end_us, bool):
            raise TypeError("time_range.end is not an integer")
        metadata = USDZTimingMetadata(
            uuid=str(data["uuid"]),
            clip_id=str(data["scene_id"]),
            start_us=start_us,
            end_us=end_us,
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid USDZ metadata from {source!r}: {exc}") from exc

    if metadata.uuid != expected_uuid:
        raise ValueError(
            f"USDZ metadata from {source!r} has uuid={metadata.uuid!r}, "
            f"expected {expected_uuid!r}"
        )
    if metadata.clip_id != expected_clip_id:
        raise ValueError(
            f"USDZ metadata from {source!r} has scene_id={metadata.clip_id!r}, "
            f"expected {expected_clip_id!r}"
        )
    if metadata.end_us <= metadata.start_us:
        raise ValueError(
            f"USDZ metadata from {source!r} has an invalid time range "
            f"[{metadata.start_us}, {metadata.end_us}]"
        )
    return metadata


def _load_and_merge_csvs(
    csv_paths: list[str],
    dedup_key: str | list[str],
) -> pl.DataFrame:
    """Load one or more CSVs, concatenate, and verify no duplicates across files.

    Args:
        csv_paths: Paths to CSV files to load and merge.
        dedup_key: Column name (str) or column names (list) that must be
            unique across all files.
    """
    frames = []
    for path in csv_paths:
        logger.info("Loading clip catalog: %s", path)
        # Force all columns to String to avoid type-inference mismatches
        # (e.g. hf_revision "25.07" inferred as Float64 in one file, String in another)
        frames.append(pl.read_csv(path, infer_schema_length=0))

    if len(frames) == 1:
        return frames[0]

    merged = pl.concat(frames, how="diagonal")

    # Check for duplicates across files
    key = [dedup_key] if isinstance(dedup_key, str) else dedup_key
    duplicate_count = merged.height - merged.unique(subset=key).height
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate rows (by {key}) across "
            f"clip catalog files: {csv_paths}. Each artifact must appear in "
            f"exactly one catalog file."
        )

    return merged


class USDZQueryError(Exception):
    """Raised when a USDZ query fails."""


def _deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    """Keep the most recently modified artifact per clip_id."""
    return df.sort("last_modified", descending=True).unique(
        subset=["clip_id"], keep="first"
    )


def _warn_duplicate_clips(df: pl.DataFrame) -> None:
    """Log a warning if any clip_id has artifacts for multiple NRE versions.

    Since NRE is fully backwards-compatible, duplicates are resolved by
    ``_deduplicate`` (newest wins). This warning gives operators visibility
    so the scene database can be cleaned up over time.
    """
    dupes = (
        df.group_by("clip_id")
        .agg(pl.col("nre_version_string").unique().alias("nre_versions"))
        .filter(pl.col("nre_versions").list.len() > 1)
    )
    if dupes.height > 0:
        details = "\n".join(
            f"  {row['clip_id']}: {sorted(row['nre_versions'])}"
            for row in dupes.iter_rows(named=True)
        )
        logger.warning(
            f"Found {dupes.height} clip(s) with artifacts for multiple NRE versions. "
            "The newest artifact per clip will be used. "
            "Consider cleaning up the clip catalog to remove stale entries.\n"
            f"{details}"
        )


# TODO(mwatson): unify with car2sim.py logic wrt metadata extraction
def scan_local_usdz_directory(usdz_dir: str) -> pl.DataFrame:
    """Scan a local directory for USDZ files and create a clip catalog.

    This function reads metadata from each USDZ file to populate the DataFrames.

    Args:
        usdz_dir: Path to directory containing *.usdz files.

    Returns:
        A clip-catalog-compatible DataFrame.

    Raises:
        ValueError: If the directory doesn't exist or contains no USDZ files.
    """
    if not os.path.isdir(usdz_dir):
        raise ValueError(f"Local USDZ directory does not exist: {usdz_dir}")

    usdz_files = glob.glob(os.path.join(usdz_dir, "**/*.usdz"), recursive=True)
    if not usdz_files:
        raise ValueError(f"No *.usdz files found in directory: {usdz_dir}")

    logger.info(f"Scanning {len(usdz_files)} USDZ files in {usdz_dir}")

    clip_rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for usdz_file in usdz_files:
        try:
            with zipfile.ZipFile(usdz_file, "r") as usdz_zip:
                with usdz_zip.open("metadata.yaml") as manifest_file:
                    data = yaml.safe_load(manifest_file)

                    # Extract metadata - similar to car2sim.py logic
                    uuid = data["uuid"]
                    # USDZ metadata uses the NRE field name ``scene_id`` for
                    # the source clip identifier.
                    clip_id = data.get("scene_id")
                    nre_version_string = data.get("version_string", "unknown")

                    clip_rows.append(
                        {
                            "uuid": uuid,
                            "clip_id": clip_id,
                            "nre_version_string": nre_version_string,
                            "path": os.path.abspath(usdz_file),
                            "last_modified": now,
                            "artifact_repository": "local",
                            "hf_revision": "",
                        }
                    )

        except (zipfile.BadZipFile, KeyError, yaml.YAMLError) as e:
            logger.warning(f"Failed to read metadata from {usdz_file}: {e}")
            continue

    if not clip_rows:
        raise ValueError(
            f"No valid USDZ files with metadata found in directory: {usdz_dir}"
        )

    logger.info("Found %d clips in local directory", len(clip_rows))
    return pl.DataFrame(clip_rows)


@dataclass
class USDZManager:
    """Manager for querying and downloading pinned USDZ clip artifacts."""

    clip_catalog: pl.DataFrame
    cache_dir: str
    _s3: S3Connection | None = None

    ALL_USDZ_DIR_NAME: ClassVar[str] = "all-usdzs"
    SCENESETS_DIR_NAME: ClassVar[str] = "scenesets"
    METADATA_DIR_NAME: ClassVar[str] = "metadata-yaml"
    LOCAL_METADATA_WORKERS: ClassVar[int] = 8
    REMOTE_METADATA_WORKERS: ClassVar[int] = 32

    @property
    def s3(self) -> S3Connection:
        """Lazily create S3 connection only when needed for Swiftstack downloads."""
        if self._s3 is None:
            self._s3 = S3Connection.from_env_vars(
                max_pool_connections=self.REMOTE_METADATA_WORKERS
            )
        return self._s3

    @property
    def scenesets_dir(self) -> str:
        return os.path.join(self.cache_dir, self.SCENESETS_DIR_NAME)

    @property
    def all_usdzs_dir(self) -> str:
        return os.path.join(self.cache_dir, self.ALL_USDZ_DIR_NAME)

    @property
    def metadata_dir(self) -> str:
        """Return the directory containing validated metadata sidecars."""
        return os.path.join(self.cache_dir, self.METADATA_DIR_NAME)

    @classmethod
    def from_cfg(cls, cfg: ScenesConfig) -> Self:
        """Create a USDZManager from a ScenesConfig.

        If cfg.local_usdz_dir is set, scans that directory for USDZ files
        instead of reading from CSV files. The local USDZ directory is not used
        as a writable scene database/cache.
        """
        # Handle local USDZ directory mode
        if cfg.local_usdz_dir is not None:
            clip_catalog = scan_local_usdz_directory(cfg.local_usdz_dir)
            # The generated local scene catalog is intentionally RAM-only. Use
            # scene_cache for any writable scratch state so local_usdz_dir can
            # be mounted read-only.
            cache_dir = cfg.scene_cache
        else:
            clip_catalog = _load_and_merge_csvs(cfg.clip_catalog_csvs, dedup_key="uuid")
            cache_dir = cfg.scene_cache

        # Ensure directories exist
        if not os.path.isdir(cache_dir):
            raise ValueError(f"Cache directory {cache_dir} does not exist.")

        manager = cls(
            clip_catalog=clip_catalog,
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

    def query_by_clip_ids(self, clip_ids: list[str]) -> list[Scene]:
        """Resolve direct clip selection to AUTO scenes."""
        if len(clip_ids) == 0:
            return []

        df = self.clip_catalog.filter(pl.col("clip_id").is_in(clip_ids)).select(
            ["clip_id", "uuid", "last_modified", "nre_version_string"]
        )

        found = set(df["clip_id"].to_list()) if df.height > 0 else set()
        missing = set(clip_ids) - found
        if missing:
            raise USDZQueryError(f"Failed to find clips for {missing}.")

        _warn_duplicate_clips(df)
        deduplicated = _deduplicate(df)
        logger.info(
            f"Clips: \n{deduplicated.select(['clip_id', 'nre_version_string'])}"
        )

        return [
            Scene(
                source=ClipDataSource(
                    clip_id=row["clip_id"],
                    usdz_uuid=row["uuid"],
                ),
                policy_start_us=T0_AUTO,
            )
            for row in deduplicated.iter_rows(named=True)
        ]

    def query_by_scene_suites(self, configs: list[SceneSuiteConfig]) -> list[Scene]:
        """Resolve suite pins while preserving authored order."""
        return resolve_scene_suites(configs, self.clip_catalog)

    def get_paths(self, uuids: list[str]) -> dict[str, str]:
        """Get artifact paths for given UUIDs."""
        if not uuids:
            return {}

        df = self.clip_catalog.filter(pl.col("uuid").is_in(uuids)).select(
            ["uuid", "path"]
        )
        return dict(zip(df["uuid"].to_list(), df["path"].to_list()))

    def external_artifact_mounts(
        self,
        uuids: list[str],
        container_cache_dir: str,
    ) -> list[tuple[str, str]]:
        """Return mounts needed to resolve cached USDZ symlinks in a container.

        The scene cache is mounted at ``container_cache_dir``. A symlink in
        ``all-usdzs`` can point outside that mount, so the target directory must
        also be mounted at the location the symlink resolves to in the
        container. Regular files and symlinks whose targets remain inside the
        scene cache need no additional mount.
        """
        container_cache = PurePosixPath(container_cache_dir)
        if not container_cache.is_absolute():
            raise ValueError(
                f"Container scene-cache path must be absolute: {container_cache_dir}"
            )

        cache_root = Path(self.cache_dir).resolve()
        host_dir_by_container_dir: dict[str, str] = {}
        for uuid in sorted(set(uuids)):
            artifact_link = Path(self.all_usdzs_dir) / f"{uuid}.usdz"
            if not artifact_link.is_symlink():
                continue

            container_link = (
                container_cache / self.ALL_USDZ_DIR_NAME / artifact_link.name
            )
            current_host = artifact_link
            current_container = container_link
            seen_host_links: set[Path] = set()
            while current_host.is_symlink():
                if current_host in seen_host_links:
                    raise RuntimeError(
                        f"Cached USDZ symlink cycle: {artifact_link} (uuid={uuid})"
                    )
                seen_host_links.add(current_host)

                link_target = PurePosixPath(os.readlink(current_host))
                if link_target.is_absolute():
                    container_target = PurePosixPath(
                        posixpath.normpath(str(link_target))
                    )
                else:
                    container_target = PurePosixPath(
                        posixpath.normpath(str(current_container.parent / link_target))
                    )

                if container_target.is_relative_to(container_cache):
                    relative_target = container_target.relative_to(container_cache)
                    current_host = cache_root.joinpath(*relative_target.parts)
                    current_container = container_target
                    continue

                try:
                    host_target = current_host.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"Cached USDZ symlink is broken: {artifact_link} (uuid={uuid})"
                    ) from exc
                if host_target.name != container_target.name:
                    raise ValueError(
                        "USDZ symlink target changes filename after resolution; "
                        f"cannot expose it with a directory mount: {artifact_link} -> "
                        f"{host_target}"
                    )

                host_dir = str(host_target.parent)
                container_dir = str(container_target.parent)
                covered_by_parent = False
                covered_children: list[str] = []
                for (
                    previous_container_dir,
                    previous_host_dir,
                ) in host_dir_by_container_dir.items():
                    if previous_container_dir == container_dir:
                        if previous_host_dir != host_dir:
                            raise ValueError(
                                "Conflicting USDZ symlink mounts for container "
                                f"directory {container_dir}: {previous_host_dir} and "
                                f"{host_dir}"
                            )
                        covered_by_parent = True
                        break
                    previous_container_path = PurePosixPath(previous_container_dir)
                    previous_host_path = Path(previous_host_dir)
                    if container_target.parent.is_relative_to(previous_container_path):
                        relative_path = container_target.parent.relative_to(
                            previous_container_path
                        )
                        if previous_host_path.joinpath(*relative_path.parts) == Path(
                            host_dir
                        ):
                            covered_by_parent = True
                            continue
                        raise ValueError(
                            "Conflicting overlapping USDZ symlink mounts: "
                            f"{previous_host_dir}:{previous_container_dir} and "
                            f"{host_dir}:{container_dir}"
                        )
                    if previous_container_path.is_relative_to(container_target.parent):
                        relative_path = previous_container_path.relative_to(
                            container_target.parent
                        )
                        if Path(host_dir).joinpath(*relative_path.parts) == (
                            previous_host_path
                        ):
                            covered_children.append(previous_container_dir)
                            continue
                        raise ValueError(
                            "Conflicting overlapping USDZ symlink mounts: "
                            f"{previous_host_dir}:{previous_container_dir} and "
                            f"{host_dir}:{container_dir}"
                        )
                if not covered_by_parent:
                    for covered_child in covered_children:
                        del host_dir_by_container_dir[covered_child]
                    host_dir_by_container_dir[container_dir] = host_dir
                break
            else:
                if not current_host.exists():
                    raise FileNotFoundError(
                        f"Cached USDZ symlink is broken: {artifact_link} (uuid={uuid})"
                    )

        return sorted(
            (host_dir, container_dir)
            for container_dir, host_dir in host_dir_by_container_dir.items()
        )

    def get_artifact_info(
        self,
        uuids: list[str],
        *,
        validate_backend: bool = False,
    ) -> dict[str, tuple[str, ArtifactRepository, str | None]]:
        """Get artifact paths, repositories, and HF revisions for given UUIDs.

        Args:
            uuids: List of UUIDs to look up.
            validate_backend: Reject missing or malformed backend fields rather
                than applying the legacy SwiftStack fallback.

        Returns:
            Dict mapping uuid to (path, artifact_repository, hf_revision) tuple.
            hf_revision is None for non-HuggingFace artifacts.
        """
        if not uuids:
            return {}

        has_hf_revision = "hf_revision" in self.clip_catalog.columns
        if validate_backend and "artifact_repository" not in self.clip_catalog.columns:
            raise ValueError(
                "Clip catalog is missing the required artifact_repository column"
            )

        # Check if artifact_repository column exists (for backwards compatibility)
        if "artifact_repository" in self.clip_catalog.columns:
            select_cols = ["uuid", "path", "artifact_repository"]
            if has_hf_revision:
                select_cols.append("hf_revision")

            df = self.clip_catalog.filter(pl.col("uuid").is_in(uuids)).select(
                select_cols
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
                    if validate_backend:
                        raise ValueError(
                            f"Invalid artifact_repository {repo_str!r} "
                            f"for uuid {row['uuid']!r}"
                        ) from None
                    # Default to swiftstack for unknown/missing values
                    logger.warning(
                        f"Unknown artifact_repository '{repo_str}' for uuid {row['uuid']}, "
                        "defaulting to swiftstack"
                    )
                    repo = ArtifactRepository.SWIFTSTACK

                revision = row.get("hf_revision") if has_hf_revision else None
                if revision is not None:
                    revision = str(revision).strip() if str(revision).strip() else None
                if (
                    validate_backend
                    and repo == ArtifactRepository.HUGGINGFACE
                    and revision is None
                ):
                    raise ValueError(
                        f"HuggingFace artifact for uuid {row['uuid']!r} "
                        "is missing hf_revision"
                    )

                result[row["uuid"]] = (row["path"], repo, revision)
            return result
        else:
            # Backwards compatibility: assume all are SwiftStack
            df = self.clip_catalog.filter(pl.col("uuid").is_in(uuids)).select(
                ["uuid", "path"]
            )
            return {
                row["uuid"]: (row["path"], ArtifactRepository.SWIFTSTACK, None)
                for row in df.iter_rows(named=True)
            }

    @staticmethod
    def _read_metadata_from_usdz(path: str) -> bytes:
        """Read only ``metadata.yaml`` from a local USDZ archive."""
        with (
            zipfile.ZipFile(path, "r") as usdz_zip,
            usdz_zip.open("metadata.yaml") as manifest_file,
        ):
            return manifest_file.read()

    def _read_metadata_from_swiftstack(self, path: str) -> bytes:
        """Range-read only ``metadata.yaml`` from a SwiftStack USDZ archive."""
        reader = S3SeekableReader(S3Path.from_swiftstack(path), self.s3.client)
        with (
            zipfile.ZipFile(reader, "r") as usdz_zip,
            usdz_zip.open("metadata.yaml") as manifest_file,
        ):
            return manifest_file.read()

    def _metadata_cache_path(self, uuid: str) -> Path:
        """Return the metadata-sidecar cache path for one USDZ UUID."""
        return Path(self.metadata_dir) / f"{uuid}.yaml"

    def _load_cached_metadata(
        self,
        source: ClipDataSource,
        path: Path,
    ) -> USDZTimingMetadata:
        """Load a metadata sidecar and revalidate its expected identity."""
        return _parse_usdz_timing_metadata(
            path.read_bytes(),
            source=str(path),
            expected_uuid=source.usdz_uuid,
            expected_clip_id=source.clip_id,
        )

    def _load_usdz_metadata(
        self,
        source: ClipDataSource,
        path: str,
    ) -> USDZTimingMetadata:
        """Load timing metadata from a complete local USDZ artifact."""
        return _parse_usdz_timing_metadata(
            self._read_metadata_from_usdz(path),
            source=path,
            expected_uuid=source.usdz_uuid,
            expected_clip_id=source.clip_id,
        )

    def _load_swiftstack_metadata(
        self,
        source: ClipDataSource,
        path: str,
    ) -> USDZTimingMetadata:
        """Range-read, validate, and atomically cache SwiftStack metadata."""
        cache_path = self._metadata_cache_path(source.usdz_uuid)
        os.makedirs(self.metadata_dir, exist_ok=True)
        with FileLock(f"{cache_path}.lock", mode=0o666):
            if cache_path.exists():
                return self._load_cached_metadata(source, cache_path)

            raw_metadata = self._read_metadata_from_swiftstack(path)
            metadata = _parse_usdz_timing_metadata(
                raw_metadata,
                source=path,
                expected_uuid=source.usdz_uuid,
                expected_clip_id=source.clip_id,
            )
            with tempfile.NamedTemporaryFile(
                dir=self.metadata_dir,
                prefix=f".{source.usdz_uuid}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_file.write(raw_metadata)
                temporary_path = temporary_file.name
            try:
                os.replace(temporary_path, cache_path)
            except OSError:
                os.unlink(temporary_path)
                raise
            return metadata

    def load_timing_metadata(
        self,
        sources: list[ClipDataSource],
    ) -> dict[str, USDZTimingMetadata]:
        """Load the minimum metadata needed for explicit-T0 validation.

        Sources are deduplicated only after confirming that each UUID maps to a
        single clip identity. Selected catalog rows are validated before cache
        lookup so warm and cold runs enforce the same backend contract.

        Returns:
            Validated timing metadata keyed by USDZ UUID.

        Raises:
            ValueError: Sources conflict, catalog backend fields are invalid,
                or metadata identity/timing is malformed.
            USDZQueryError: A requested UUID is absent from the clip catalog.
        """
        source_by_uuid: dict[str, ClipDataSource] = {}
        for source in sources:
            previous = source_by_uuid.setdefault(source.usdz_uuid, source)
            if previous != source:
                raise ValueError(
                    f"USDZ uuid {source.usdz_uuid!r} maps to both "
                    f"{previous.clip_id!r} and {source.clip_id!r}"
                )

        if not source_by_uuid:
            return {}

        artifact_info = self.get_artifact_info(
            list(source_by_uuid), validate_backend=True
        )
        missing = set(source_by_uuid) - set(artifact_info)
        if missing:
            raise USDZQueryError(f"Failed to find artifacts for UUIDs {missing}.")

        local_jobs: list[tuple[ClipDataSource, Path, bool]] = []
        remote_jobs: list[tuple[ClipDataSource, str]] = []
        huggingface_uuids: list[str] = []
        for uuid, source in source_by_uuid.items():
            path, repository, _revision = artifact_info[uuid]
            metadata_cache_path = self._metadata_cache_path(uuid)
            artifact_cache_path = Path(self.all_usdzs_dir) / f"{uuid}.usdz"
            if metadata_cache_path.exists():
                local_jobs.append((source, metadata_cache_path, True))
            elif artifact_cache_path.exists():
                local_jobs.append((source, artifact_cache_path, False))
            elif repository == ArtifactRepository.LOCAL:
                local_jobs.append((source, Path(path), False))
            elif repository == ArtifactRepository.SWIFTSTACK:
                remote_jobs.append((source, path))
            elif repository == ArtifactRepository.HUGGINGFACE:
                huggingface_uuids.append(uuid)

        if huggingface_uuids:
            logger.info(
                "Explicit-T0 validation requires metadata from %d uncached "
                "HuggingFace artifacts; preserving the existing full-download path.",
                len(huggingface_uuids),
            )
            self._download_huggingface_artifacts(huggingface_uuids, artifact_info)
            local_jobs.extend(
                (
                    source_by_uuid[uuid],
                    Path(self.all_usdzs_dir) / f"{uuid}.usdz",
                    False,
                )
                for uuid in huggingface_uuids
            )

        metadata_by_uuid: dict[str, USDZTimingMetadata] = {}

        def load_local(
            job: tuple[ClipDataSource, Path, bool],
        ) -> tuple[str, USDZTimingMetadata]:
            source, path, is_metadata_cache = job
            metadata = (
                self._load_cached_metadata(source, path)
                if is_metadata_cache
                else self._load_usdz_metadata(source, str(path))
            )
            return source.usdz_uuid, metadata

        if local_jobs:
            workers = min(self.LOCAL_METADATA_WORKERS, len(local_jobs))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                metadata_by_uuid.update(executor.map(load_local, local_jobs))

        def load_remote(
            job: tuple[ClipDataSource, str],
        ) -> tuple[str, USDZTimingMetadata]:
            source, path = job
            return source.usdz_uuid, self._load_swiftstack_metadata(source, path)

        if remote_jobs:
            logger.info(
                "Reading metadata only for %d uncached SwiftStack artifacts "
                "with %d workers.",
                len(remote_jobs),
                min(self.REMOTE_METADATA_WORKERS, len(remote_jobs)),
            )
            workers = min(self.REMOTE_METADATA_WORKERS, len(remote_jobs))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                metadata_by_uuid.update(executor.map(load_remote, remote_jobs))

        return metadata_by_uuid

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
        for uuid, (path, repo, _rev) in artifact_info.items():
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
                path, _, _ = artifact_info[uuid]
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
                path, _, _ = artifact_info[uuid]
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
        self, uuid: str, hf_filepath: str, tmpdir: str, revision: str | None = None
    ) -> None:
        """Download and validate a single HuggingFace artifact.

        Args:
            uuid: The expected UUID of the artifact.
            hf_filepath: The file path within the HuggingFace repository.
            tmpdir: Temporary directory for downloading.
            revision: Optional HuggingFace revision (branch, tag, or commit hash).
        """
        logger.info(
            f"Downloading HuggingFace artifact for uuid {uuid} from {hf_filepath}"
            f" (revision={revision})"
        )
        downloaded_usdz = hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            repo_type="dataset",
            local_dir=tmpdir,
            filename=hf_filepath,
            revision=revision,
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
        self,
        uuids: list[str],
        artifact_info: dict[str, tuple[str, ArtifactRepository, str | None]],
    ) -> None:
        """
        Download the required HuggingFace artifacts, rename them based on their metadata uuids,
        and store them in the all_usdzs_dir.

        Downloads are parallelized with a maximum of 5 concurrent downloads.
        """
        missing_uuid_info: dict[str, tuple[str, str | None]] = (
            {}
        )  # uuid -> (filepath, revision)
        for uuid in uuids:
            cache_path = os.path.join(self.all_usdzs_dir, f"{uuid}.usdz")
            if not os.path.exists(cache_path):
                path, _repo, revision = artifact_info[uuid]
                missing_uuid_info[uuid] = (path, revision)
        if missing_uuid_info:
            logger.info(
                f"Missing {len(missing_uuid_info)} required HuggingFace artifacts"
            )
            max_workers = min(5, len(missing_uuid_info))
            with tempfile.TemporaryDirectory(dir=self.scenesets_dir) as tmpdir:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._download_single_huggingface_artifact,
                            uuid,
                            hf_filepath,
                            tmpdir,
                            revision,
                        ): uuid
                        for uuid, (hf_filepath, revision) in missing_uuid_info.items()
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


def resolve_scene_selection(
    config: ScenesConfig,
    manager: USDZManager,
) -> list[Scene]:
    """Resolve one configured selector into complete scenes."""
    suite_configs = config.suite_configs
    clip_ids = config.clip_ids
    use_local_clips = config.local_usdz_dir is not None

    if suite_configs is not None and clip_ids is not None:
        raise ValueError(
            "Set either scenes.clip_ids or scenes.suite_configs, not both."
        )
    if use_local_clips and suite_configs is not None:
        raise ValueError(
            "scenes.local_usdz_dir cannot be combined with scenes.suite_configs"
        )
    if use_local_clips and clip_ids is None:
        clip_ids = manager.clip_catalog["clip_id"].to_list()
        logger.info(
            "Using all %d clips from local_usdz_dir=%s",
            len(clip_ids),
            config.local_usdz_dir,
        )

    if suite_configs is not None:
        scenes = manager.query_by_scene_suites(list(suite_configs))
    elif clip_ids is not None:
        scenes = manager.query_by_clip_ids(clip_ids)
        scenes.sort(key=lambda scene: scene.source.source_id)
    else:
        raise ValueError(
            "Set one scene selector: scenes.clip_ids, scenes.suite_configs, or "
            "scenes.local_usdz_dir"
        )

    excluded = set(config.exclude_clip_ids)
    selected = {scene.source.source_id for scene in scenes}
    missing = excluded - selected
    if missing:
        raise USDZQueryError(
            f"Cannot exclude clips that are not selected: {sorted(missing)}."
        )
    if excluded:
        scenes = [scene for scene in scenes if scene.source.source_id not in excluded]
        logger.info("Excluded %d clips: %s", len(excluded), sorted(excluded))
        if not scenes:
            raise USDZQueryError("Clip exclusions removed every selected clip.")
    if config.limit_to_first_n > 0:
        scenes = scenes[: config.limit_to_first_n]
    return scenes


def resolve_scene_groups(
    config: ScenesConfig,
    manager: USDZManager,
) -> tuple[list[Scene], dict[str, list[Scene]]]:
    """Return ``(runtime_scenes, scenes_by_group)`` for the configured selection.

    ``runtime_scenes`` is the scene-key-deduplicated union in group traversal
    order. ``scenes_by_group`` preserves each group's resolved scenes, including
    scenes shared by multiple groups. Without groups, the second value is empty.
    """
    if not config.suite_groups:
        return resolve_scene_selection(config, manager), {}

    groups = {
        name: manager.query_by_scene_suites(list(suite_configs))
        for name, suite_configs in config.suite_groups.items()
    }
    scenes_by_key: dict[str, Scene] = {}
    for scenes in groups.values():
        for scene in scenes:
            previous = scenes_by_key.get(scene.scene_key)
            if previous is not None and previous != scene:
                raise ValueError(
                    f"Scene key {scene.scene_key!r} resolves to conflicting scenes: "
                    f"{previous!r} and {scene!r}"
                )
            scenes_by_key.setdefault(scene.scene_key, scene)
    return list(scenes_by_key.values()), groups
