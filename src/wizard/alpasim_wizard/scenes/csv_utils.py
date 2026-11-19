# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Shared utilities for CSV operations on clip-catalog and scene-suite data."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from enum import Enum

import polars as pl

logger = logging.getLogger(__name__)


HUGGINGFACE_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"


class ArtifactRepository(str, Enum):
    """Supported artifact repositories for USDZ scene files."""

    SWIFTSTACK = "swiftstack"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"  # Files on local filesystem

    def __str__(self) -> str:
        return self.value


# Expected columns
CLIP_CATALOG_COLUMNS = [
    "uuid",
    "clip_id",
    "nre_version_string",
    "path",
    "last_modified",
    "artifact_repository",
    "hf_revision",
]
SCENE_SUITE_COLUMNS = ["clip_id", "uuid", "event_t0_us"]

# Validation patterns
# Matches ISO-format timestamps: "YYYY-MM-DD HH:MM:SS" (e.g., "2024-12-05 14:30:00")
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
# Matches alphanumeric identifiers with hyphens/underscores (e.g., "abc-123_def")
UUID_PATTERN = re.compile(r"^[\w-]+$")
# Matches clip IDs prefixed with "clipgt-" (e.g., "clipgt-abc-123")
CLIP_ID_PATTERN = re.compile(r"^clipgt-[\w-]+$")

# Valid artifact repository values (stripped/lowercased for comparison)
VALID_ARTIFACT_REPOSITORIES = {repo.value for repo in ArtifactRepository}


def normalize_scene_suite_catalog(
    frame: pl.DataFrame,
    source: str,
) -> pl.DataFrame:
    """Apply the existing suite validation to the canonical CSV schema."""
    missing = _missing_columns(frame.columns, SCENE_SUITE_COLUMNS)
    if missing:
        raise ValueError(f"Scene-suite CSV missing columns: {missing}")

    normalized = frame.select(SCENE_SUITE_COLUMNS).with_columns(
        *(
            pl.col(column).cast(pl.Utf8).str.strip_chars()
            for column in SCENE_SUITE_COLUMNS
        )
    )

    duplicates = normalized.filter(pl.struct(SCENE_SUITE_COLUMNS).is_duplicated())
    if duplicates.height:
        raise ValueError(
            "Duplicate scene-suite rows: "
            f"{duplicates.select(SCENE_SUITE_COLUMNS).head(3).rows()}"
        )

    empty = normalized.filter(
        pl.any_horizontal(
            [
                pl.col(column).is_null() | (pl.col(column) == "")
                for column in SCENE_SUITE_COLUMNS
            ]
        )
    )
    if empty.height:
        raise ValueError(
            f"Empty values in scene-suite CSV {source!r}: {empty.height} rows"
        )
    return normalized


def _missing_columns(columns: Sequence[str], required_columns: list[str]) -> list[str]:
    """Required columns absent from ``columns``, in contract order."""
    return [c for c in required_columns if c not in columns]


def check_catalog_headers(*catalogs: tuple[str | None, list[str]]) -> None:
    """Raise if any given catalog's header lacks its required columns.

    A path that is None or not on disk is skipped: there is no header to check.
    """
    for csv_path, required_columns in catalogs:
        if csv_path is None or not os.path.exists(csv_path):
            continue
        header = pl.read_csv(csv_path, n_rows=0).columns
        missing = _missing_columns(header, required_columns)
        if missing:
            raise ValueError(
                f"Existing CSV '{csv_path}' is missing required columns: {missing}"
            )


def align_to_existing(
    existing: pl.DataFrame, new_rows: pl.DataFrame, required_columns: list[str]
) -> pl.DataFrame:
    """Shape new rows to the columns the CSV on disk actually has.

    The required columns are the contract; a catalog may carry more, and its own header is the
    target both frames must agree on. Columns the target has but the caller omitted are filled
    with null; columns the caller supplies that the target lacks are dropped, so a merge can never
    widen the schema by accident. The result is cast to the target's dtypes so it can be
    concatenated with it.

    Raises ValueError when either frame is missing a required column: the caller's rows would be
    incomplete, and a target header narrower than the contract would silently drop values.
    """
    missing = _missing_columns(new_rows.columns, required_columns)
    if missing:
        raise ValueError(f"New rows are missing required columns: {missing}")

    if not existing.width:
        target = required_columns
    else:
        missing = _missing_columns(existing.columns, required_columns)
        if missing:
            raise ValueError(f"Existing CSV is missing required columns: {missing}")
        target = existing.columns

    aligned = new_rows
    for column in target:
        if column not in aligned.columns:
            aligned = aligned.with_columns(pl.lit(None).alias(column))
    aligned = aligned.select(target)

    if existing.width:
        aligned = aligned.cast({c: existing.schema[c] for c in target})
    return aligned


def load_or_create_csv(
    path: str, columns: list[str], create_if_missing: bool = False
) -> pl.DataFrame:
    """Load existing CSV or create empty DataFrame with columns."""
    if os.path.exists(path):
        # Force all columns to String, as sceneset.py does: inference would read a catalog of
        # numeric-looking hf_revision pins as Float64 and rewrite "26.10" as 26.1 on merge.
        return pl.read_csv(path, infer_schema_length=0)

    if not create_if_missing:
        raise FileNotFoundError(
            f"CSV file '{path}' does not exist. Use --create-file to create it."
        )

    return pl.DataFrame(schema={col: pl.Utf8 for col in columns})


def merge_clip_catalog_csv(
    csv_path: str,
    new_rows: pl.DataFrame,
    create_if_missing: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Merge new clip rows into an existing catalog, deduplicating by UUID.

    Args:
        csv_path: Path to the clip catalog CSV file.
        new_rows: DataFrame with new clip rows to merge.
        create_if_missing: If True, create the CSV if it doesn't exist.
        dry_run: If True, compute what would be added without writing.

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    if dry_run and not os.path.exists(csv_path):
        return (new_rows.height, 0)

    existing = load_or_create_csv(
        csv_path,
        CLIP_CATALOG_COLUMNS,
        create_if_missing=(create_if_missing or dry_run),
    )
    new_rows = align_to_existing(existing, new_rows, CLIP_CATALOG_COLUMNS)
    existing_uuids = set(existing["uuid"].to_list()) if existing.height > 0 else set()
    new_only = new_rows.filter(~pl.col("uuid").is_in(existing_uuids))
    duplicates = new_rows.height - new_only.height

    if dry_run or new_only.height == 0:
        return (new_only.height, duplicates)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    combined = pl.concat([existing, new_only])
    combined.write_csv(csv_path)

    return (new_only.height, duplicates)


def merge_scene_suite_csv(
    csv_path: str,
    new_rows: pl.DataFrame,
    create_if_missing: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Merge new rows into an existing scene-suite CSV.

    Args:
        csv_path: Path to the scene-suite CSV file.
        new_rows: DataFrame with new scene-suite rows to merge.
        create_if_missing: If True, create the CSV if it doesn't exist.
        dry_run: If True, compute what would be added without writing.

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    new_rows = normalize_scene_suite_catalog(
        new_rows,
        source="new scene-suite rows",
    )

    if dry_run and not os.path.exists(csv_path):
        return (new_rows.height, 0)

    existing = load_or_create_csv(
        csv_path, SCENE_SUITE_COLUMNS, create_if_missing=(create_if_missing or dry_run)
    )

    new_rows = align_to_existing(existing, new_rows, SCENE_SUITE_COLUMNS)

    if existing.height > 0:
        existing_rows = set(
            existing.select(SCENE_SUITE_COLUMNS)
            .with_columns(
                *(
                    pl.col(column).cast(pl.Utf8).str.strip_chars()
                    for column in SCENE_SUITE_COLUMNS
                )
            )
            .iter_rows()
        )
    else:
        existing_rows = set()

    new_only = new_rows.filter(
        pl.struct(SCENE_SUITE_COLUMNS).map_elements(
            lambda row: (
                tuple(row[column] for column in SCENE_SUITE_COLUMNS)
                not in existing_rows
            ),
            return_dtype=pl.Boolean,
        )
    )
    duplicates = new_rows.height - new_only.height

    if dry_run or new_only.height == 0:
        return (new_only.height, duplicates)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    combined = pl.concat([existing, new_only])
    combined.write_csv(csv_path)

    return (new_only.height, duplicates)


class CSVValidationError(Exception):
    """Raised when CSV validation fails."""


def validate_csvs(clip_catalog_csv: str, scene_suite_csv: str | None = None) -> None:
    """
    Validate CSV files for correctness and consistency.

    Checks performed:

    Clip catalog CSV:
        - Required columns present (uuid, clip_id, nre_version_string, path,
          last_modified, artifact_repository)
        - No duplicate UUIDs
        - UUID format valid (alphanumeric with hyphens/underscores)
        - clip_id format valid (must start with "clipgt-")
        - last_modified format valid (ISO format: "YYYY-MM-DD HH:MM:SS")
        - artifact_repository valid (one of: swiftstack, huggingface)
        - No empty/null values in required fields (uuid, clip_id, nre_version_string, path)

    Scene-suite CSV (if provided):
        - Required columns present (clip_id, uuid, event_t0_us)
        - No duplicate authored rows
        - No empty/null values in any column

    Cross-file (if scene_suite_csv provided):
        - All (clip_id, uuid) pairs referenced in the scene suite exist in the clip catalog

    Raises:
        CSVValidationError: If any validation check fails.

    """
    errors: list[str] = []

    # Load CSVs
    clips_df = pl.read_csv(clip_catalog_csv)
    scene_suite_df = (
        pl.read_csv(scene_suite_csv, infer_schema_length=0) if scene_suite_csv else None
    )

    # --- Clip catalog validation ---

    # Check required columns
    missing_cols = set(CLIP_CATALOG_COLUMNS) - set(clips_df.columns)
    if missing_cols:
        errors.append(f"Clip catalog CSV missing columns: {missing_cols}")

    if not missing_cols:
        # Check for duplicate UUIDs
        dup_uuids = clips_df.filter(pl.col("uuid").is_duplicated())
        if dup_uuids.height > 0:
            unique_dups = dup_uuids["uuid"].unique()
            dup_list = unique_dups.to_list()[:5]
            suffix = "..." if len(unique_dups) > 5 else ""
            errors.append(f"Duplicate UUIDs in clip catalog CSV: {dup_list}{suffix}")

        # Check UUID format
        invalid_uuids = clips_df.filter(
            ~pl.col("uuid").cast(pl.Utf8).str.contains(r"^[\w-]+$")
        )
        if invalid_uuids.height > 0:
            bad = invalid_uuids["uuid"].to_list()[:3]
            errors.append(f"Invalid UUID format: {bad}")

        # Check clip_id format
        invalid_clip_ids = clips_df.filter(
            ~pl.col("clip_id").cast(pl.Utf8).str.contains(r"^clipgt-[\w-]+$")
        )
        if invalid_clip_ids.height > 0:
            bad = invalid_clip_ids["clip_id"].to_list()[:3]
            errors.append(f"Invalid clip_id format (expected 'clipgt-...'): {bad}")

        # Check last_modified format (ISO: YYYY-MM-DD HH:MM:SS)
        invalid_timestamps = clips_df.filter(
            ~pl.col("last_modified")
            .cast(pl.Utf8)
            .str.contains(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        )
        if invalid_timestamps.height > 0:
            bad = invalid_timestamps["last_modified"].to_list()[:3]
            errors.append(
                f"Invalid last_modified format (expected 'YYYY-MM-DD HH:MM:SS'): {bad}"
            )

        # Check for empty/null values in required fields
        for col in ["uuid", "clip_id", "nre_version_string", "path"]:
            nulls = clips_df.filter(
                pl.col(col).is_null()
                | (pl.col(col).cast(pl.Utf8).str.strip_chars() == "")
            )
            if nulls.height > 0:
                errors.append(
                    f"Empty values in clip catalog CSV column '{col}': {nulls.height} rows"
                )

        # Check artifact_repository values are valid
        if "artifact_repository" in clips_df.columns:
            # Normalize values (strip whitespace) before comparison
            normalized_repos = clips_df.select(
                pl.col("artifact_repository").cast(pl.Utf8).str.strip_chars()
            )
            invalid_repos = clips_df.filter(
                ~normalized_repos["artifact_repository"].is_in(
                    list(VALID_ARTIFACT_REPOSITORIES)
                )
            )
            if invalid_repos.height > 0:
                bad = invalid_repos["artifact_repository"].unique().to_list()[:5]
                errors.append(
                    f"Invalid artifact_repository values (expected one of {list(VALID_ARTIFACT_REPOSITORIES)}): {bad}"
                )

        # Check hf_revision is set for huggingface artifacts
        if (
            "hf_revision" in clips_df.columns
            and "artifact_repository" in clips_df.columns
        ):
            hf_rows = clips_df.filter(
                pl.col("artifact_repository").cast(pl.Utf8).str.strip_chars()
                == "huggingface"
            )
            missing_revision = hf_rows.filter(
                pl.col("hf_revision").is_null()
                | (pl.col("hf_revision").cast(pl.Utf8).str.strip_chars() == "")
            )
            if missing_revision.height > 0:
                bad_uuids = missing_revision["uuid"].to_list()[:5]
                errors.append(
                    f"HuggingFace artifacts missing hf_revision: {bad_uuids}"
                    f"{'...' if missing_revision.height > 5 else ''}"
                )

    # --- Scene-suite CSV validation (if provided) ---

    if scene_suite_df is not None:
        try:
            scene_suite_df = normalize_scene_suite_catalog(
                scene_suite_df,
                source=scene_suite_csv or "Scene-suite CSV",
            )
        except ValueError as exc:
            errors.append(str(exc))

        # --- Cross-file validation ---

        if (
            scene_suite_df.height > 0
            and {"clip_id", "uuid"}.issubset(scene_suite_df.columns)
            and {"clip_id", "uuid"}.issubset(clips_df.columns)
        ):
            missing_in_clips = (
                scene_suite_df.select(["clip_id", "uuid"])
                .unique()
                .join(
                    clips_df.select(["clip_id", "uuid"]).unique(),
                    on=["clip_id", "uuid"],
                    how="anti",
                )
                .sort(["clip_id", "uuid"])
            )
            if missing_in_clips.height > 0:
                missing_list = missing_in_clips.head(5).rows()
                errors.append(
                    "Scene suite references (clip_id, uuid) pairs not in clip catalog CSV: "
                    f"{missing_list}"
                    f"{'...' if missing_in_clips.height > 5 else ''}"
                )

    # Raise if any errors
    if errors:
        raise CSVValidationError(
            f"CSV validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info("CSV validation passed")
