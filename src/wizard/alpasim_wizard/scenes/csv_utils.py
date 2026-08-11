# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Shared utilities for CSV operations on scene/suite data."""

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
SCENES_COLUMNS = [
    "uuid",
    "scene_id",
    "nre_version_string",
    "path",
    "last_modified",
    "artifact_repository",
    "hf_revision",
]
SUITES_COLUMNS = ["test_suite_id", "scene_id", "uuid"]

# Validation patterns
# Matches ISO-format timestamps: "YYYY-MM-DD HH:MM:SS" (e.g., "2024-12-05 14:30:00")
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
# Matches alphanumeric identifiers with hyphens/underscores (e.g., "abc-123_def")
UUID_PATTERN = re.compile(r"^[\w-]+$")
# Matches scene IDs prefixed with "clipgt-" (e.g., "clipgt-abc-123")
SCENE_ID_PATTERN = re.compile(r"^clipgt-[\w-]+$")

# Valid artifact repository values (stripped/lowercased for comparison)
VALID_ARTIFACT_REPOSITORIES = {repo.value for repo in ArtifactRepository}


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


def merge_scenes_csv(
    csv_path: str,
    new_rows: pl.DataFrame,
    create_if_missing: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Merge new scene rows into existing CSV, deduplicating by uuid.

    Args:
        csv_path: Path to the scenes CSV file.
        new_rows: DataFrame with new scene rows to merge.
        create_if_missing: If True, create the CSV if it doesn't exist.
        dry_run: If True, compute what would be added without writing.

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    if dry_run and not os.path.exists(csv_path):
        return (new_rows.height, 0)

    existing = load_or_create_csv(
        csv_path, SCENES_COLUMNS, create_if_missing=(create_if_missing or dry_run)
    )
    new_rows = align_to_existing(existing, new_rows, SCENES_COLUMNS)
    existing_uuids = set(existing["uuid"].to_list()) if existing.height > 0 else set()
    new_only = new_rows.filter(~pl.col("uuid").is_in(existing_uuids))
    duplicates = new_rows.height - new_only.height

    if dry_run or new_only.height == 0:
        return (new_only.height, duplicates)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    combined = pl.concat([existing, new_only])
    combined.write_csv(csv_path)

    return (new_only.height, duplicates)


def merge_suites_csv(
    csv_path: str,
    new_rows: pl.DataFrame,
    create_if_missing: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Merge new suite rows into existing CSV, deduplicating by (test_suite_id, uuid).

    Args:
        csv_path: Path to the suites CSV file.
        new_rows: DataFrame with new suite rows to merge.
        create_if_missing: If True, create the CSV if it doesn't exist.
        dry_run: If True, compute what would be added without writing.

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    if dry_run and not os.path.exists(csv_path):
        return (new_rows.height, 0)

    existing = load_or_create_csv(
        csv_path, SUITES_COLUMNS, create_if_missing=(create_if_missing or dry_run)
    )

    new_rows = align_to_existing(existing, new_rows, SUITES_COLUMNS)

    if existing.height > 0:
        existing_pairs = set(
            zip(
                existing["test_suite_id"].to_list(),
                existing["uuid"].to_list(),
            )
        )
    else:
        existing_pairs = set()

    new_only = new_rows.filter(
        pl.struct(["test_suite_id", "uuid"]).map_elements(
            lambda x: (x["test_suite_id"], x["uuid"]) not in existing_pairs,
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


def validate_csvs(scenes_csv: str, suites_csv: str | None = None) -> None:
    """
    Validate CSV files for correctness and consistency.

    Checks performed:

    Scenes CSV:
        - Required columns present (uuid, scene_id, nre_version_string, path,
          last_modified, artifact_repository)
        - No duplicate UUIDs
        - UUID format valid (alphanumeric with hyphens/underscores)
        - scene_id format valid (must start with "clipgt-")
        - last_modified format valid (ISO format: "YYYY-MM-DD HH:MM:SS")
        - artifact_repository valid (one of: swiftstack, huggingface)
        - No empty/null values in required fields (uuid, scene_id, nre_version_string, path)

    Suites CSV (if provided):
        - Required columns present (test_suite_id, scene_id, uuid)
        - No duplicate (test_suite_id, uuid) pairs
        - No empty/null values in any column

    Cross-file (if suites_csv provided):
        - All (scene_id, uuid) pairs referenced in suites exist in scenes

    Raises:
        CSVValidationError: If any validation check fails.

    """
    errors: list[str] = []

    # Load CSVs
    scenes_df = pl.read_csv(scenes_csv)
    suites_df = pl.read_csv(suites_csv) if suites_csv else None

    # --- Scenes CSV validation ---

    # Check required columns
    missing_cols = set(SCENES_COLUMNS) - set(scenes_df.columns)
    if missing_cols:
        errors.append(f"Scenes CSV missing columns: {missing_cols}")

    if not missing_cols:
        # Check for duplicate UUIDs
        dup_uuids = scenes_df.filter(pl.col("uuid").is_duplicated())
        if dup_uuids.height > 0:
            unique_dups = dup_uuids["uuid"].unique()
            dup_list = unique_dups.to_list()[:5]
            suffix = "..." if len(unique_dups) > 5 else ""
            errors.append(f"Duplicate UUIDs in scenes CSV: {dup_list}{suffix}")

        # Check UUID format
        invalid_uuids = scenes_df.filter(
            ~pl.col("uuid").cast(pl.Utf8).str.contains(r"^[\w-]+$")
        )
        if invalid_uuids.height > 0:
            bad = invalid_uuids["uuid"].to_list()[:3]
            errors.append(f"Invalid UUID format: {bad}")

        # Check scene_id format
        invalid_scene_ids = scenes_df.filter(
            ~pl.col("scene_id").cast(pl.Utf8).str.contains(r"^clipgt-[\w-]+$")
        )
        if invalid_scene_ids.height > 0:
            bad = invalid_scene_ids["scene_id"].to_list()[:3]
            errors.append(f"Invalid scene_id format (expected 'clipgt-...'): {bad}")

        # Check last_modified format (ISO: YYYY-MM-DD HH:MM:SS)
        invalid_timestamps = scenes_df.filter(
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
        for col in ["uuid", "scene_id", "nre_version_string", "path"]:
            nulls = scenes_df.filter(
                pl.col(col).is_null()
                | (pl.col(col).cast(pl.Utf8).str.strip_chars() == "")
            )
            if nulls.height > 0:
                errors.append(
                    f"Empty values in scenes CSV column '{col}': {nulls.height} rows"
                )

        # Check artifact_repository values are valid
        if "artifact_repository" in scenes_df.columns:
            # Normalize values (strip whitespace) before comparison
            normalized_repos = scenes_df.select(
                pl.col("artifact_repository").cast(pl.Utf8).str.strip_chars()
            )
            invalid_repos = scenes_df.filter(
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
            "hf_revision" in scenes_df.columns
            and "artifact_repository" in scenes_df.columns
        ):
            hf_rows = scenes_df.filter(
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

    # --- Suites CSV validation (if provided) ---

    if suites_df is not None:
        # Check required columns
        missing_cols = set(SUITES_COLUMNS) - set(suites_df.columns)
        if missing_cols:
            errors.append(f"Suites CSV missing columns: {missing_cols}")

        if suites_df.height > 0 and not missing_cols:
            # Check for duplicate (test_suite_id, uuid) pairs
            dup_pairs = suites_df.filter(
                pl.struct(["test_suite_id", "uuid"]).is_duplicated()
            )
            if dup_pairs.height > 0:
                dup_list = list(
                    zip(
                        dup_pairs["test_suite_id"].to_list()[:3],
                        dup_pairs["uuid"].to_list()[:3],
                    )
                )
                errors.append(f"Duplicate (test_suite_id, uuid) pairs: {dup_list}")

            # Check for empty values
            for col in SUITES_COLUMNS:
                nulls = suites_df.filter(
                    pl.col(col).is_null()
                    | (pl.col(col).cast(pl.Utf8).str.strip_chars() == "")
                )
                if nulls.height > 0:
                    errors.append(
                        f"Empty values in suites CSV column '{col}': {nulls.height} rows"
                    )

        # --- Cross-file validation ---

        if (
            suites_df.height > 0
            and {"scene_id", "uuid"}.issubset(suites_df.columns)
            and {"scene_id", "uuid"}.issubset(scenes_df.columns)
        ):
            missing_in_scenes = (
                suites_df.select(["scene_id", "uuid"])
                .unique()
                .join(
                    scenes_df.select(["scene_id", "uuid"]).unique(),
                    on=["scene_id", "uuid"],
                    how="anti",
                )
                .sort(["scene_id", "uuid"])
            )
            if missing_in_scenes.height > 0:
                missing_list = missing_in_scenes.head(5).rows()
                errors.append(
                    "Suites reference (scene_id, uuid) pairs not in scenes CSV: "
                    f"{missing_list}"
                    f"{'...' if missing_in_scenes.height > 5 else ''}"
                )

    # Raise if any errors
    if errors:
        raise CSVValidationError(
            f"CSV validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info("CSV validation passed")
