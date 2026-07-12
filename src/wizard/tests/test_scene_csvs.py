# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""CI test to validate scene and suite CSV files in the repository."""

from pathlib import Path

import polars as pl
import pytest
from alpasim_wizard.scenes.csv_utils import (
    CSVValidationError,
    merge_suites_csv,
    validate_csvs,
)


def get_repo_root() -> Path:
    """Find the repository root by looking for the data/scenes directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data" / "scenes").exists():
            return parent
    raise RuntimeError(
        "Could not find repository root (looking for data/scenes directory)"
    )


REPO_ROOT = get_repo_root()
SCENES_CSV = REPO_ROOT / "data" / "scenes" / "sim_scenes.csv"
SUITES_CSV = REPO_ROOT / "data" / "scenes" / "sim_suites.csv"
LEGACY_SCENES_CSV = REPO_ROOT / "data" / "scenes" / "sim_scenes_2505.csv"
LEGACY_SUITES_CSV = REPO_ROOT / "data" / "scenes" / "sim_suites_2505.csv"


@pytest.mark.parametrize(
    ("scenes_csv", "suites_csv"),
    [
        (SCENES_CSV, SUITES_CSV),
        (LEGACY_SCENES_CSV, LEGACY_SUITES_CSV),
    ],
)
def test_scene_csvs_are_valid(scenes_csv: Path, suites_csv: Path):
    """
    Validate that the repository's scene and suite CSV files are well-formed.

    This test runs in CI to catch:
    - Duplicate entries
    - Missing required columns
    - Invalid formats (UUIDs, timestamps, scene_ids)
    - Suite artifact pairs that do not exist in the scenes file
    """
    try:
        validate_csvs(str(scenes_csv), str(suites_csv))
    except CSVValidationError as e:
        pytest.fail(f"Scene CSV validation failed:\n{e}")


@pytest.mark.parametrize(
    ("suite_id", "hf_revision"),
    [("public_2601", "26.01"), ("public_2604", "26.04")],
)
def test_public_suite_pins_its_release(suite_id: str, hf_revision: str):
    """A versioned public suite selects artifacts from that release."""
    scenes = pl.read_csv(SCENES_CSV, infer_schema_length=0)
    suite = pl.read_csv(SUITES_CSV, infer_schema_length=0).filter(
        pl.col("test_suite_id") == suite_id
    )

    selected = suite.join(scenes, on=["scene_id", "uuid"], how="inner")

    assert selected.height == suite.height
    assert selected["hf_revision"].unique().to_list() == [hf_revision]


def test_validate_csvs_catches_duplicate_uuids(tmp_path):
    """Verify validation catches duplicate UUIDs."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "dup-uuid,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
        "dup-uuid,clipgt-bbb,0.2.220,path/b,2025-01-01 00:00:00,swiftstack,\n"  # duplicate!
    )
    suites.write_text("test_suite_id,scene_id,uuid\n")

    with pytest.raises(CSVValidationError, match="Duplicate UUIDs"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_orphaned_suite_references(tmp_path):
    """Verify validation catches suite entries referencing non-existent artifacts."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
    )
    suites.write_text(
        "test_suite_id,scene_id,uuid\n" "my-suite,clipgt-missing,uuid-missing\n"
    )

    with pytest.raises(CSVValidationError, match="pairs not in scenes CSV"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_mismatched_scene_and_uuid(tmp_path):
    """Verify a suite UUID must belong to the scene ID beside it."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
        "uuid-2,clipgt-bbb,0.2.220,path/b,2025-01-01 00:00:00,swiftstack,\n"
    )
    suites.write_text("test_suite_id,scene_id,uuid\n" "my-suite,clipgt-aaa,uuid-2\n")

    with pytest.raises(CSVValidationError, match="pairs not in scenes CSV"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_invalid_timestamp_format(tmp_path):
    """Verify validation catches non-ISO timestamp formats."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,01/15/2025 10:30:00,swiftstack,\n"  # wrong format!
    )
    suites.write_text("test_suite_id,scene_id,uuid\n")

    with pytest.raises(CSVValidationError, match="Invalid last_modified format"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_invalid_scene_id_format(tmp_path):
    """Verify validation catches invalid scene_id formats."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,invalid-scene-id,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"  # missing clipgt- prefix
    )
    suites.write_text("test_suite_id,scene_id,uuid\n")

    with pytest.raises(CSVValidationError, match="Invalid scene_id format"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_missing_columns(tmp_path):
    """Verify validation catches missing required columns."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id\n"  # missing nre_version_string, path, last_modified, artifact_repository
        "uuid-1,clipgt-aaa\n"
    )
    suites.write_text("test_suite_id,scene_id,uuid\n")

    with pytest.raises(CSVValidationError, match="missing columns"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_requires_suite_uuid(tmp_path):
    """Verify suite rows must pin an artifact UUID."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
    )
    suites.write_text("test_suite_id,scene_id\nmy-suite,clipgt-aaa\n")

    with pytest.raises(CSVValidationError, match="Suites CSV missing columns"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_duplicate_suite_entries(tmp_path):
    """Verify validation catches duplicate (test_suite_id, uuid) pairs."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
    )
    suites.write_text(
        "test_suite_id,scene_id,uuid\n"
        "my-suite,clipgt-aaa,uuid-1\n"
        "my-suite,clipgt-aaa,uuid-1\n"
    )

    with pytest.raises(CSVValidationError, match="Duplicate"):
        validate_csvs(str(scenes), str(suites))


def test_merge_suites_csv_uses_uuid_as_artifact_identity(tmp_path):
    """A suite can contain two artifacts for one scene, but not one UUID twice."""
    suites = tmp_path / "suites.csv"
    suites.write_text("test_suite_id,scene_id,uuid\n" "my-suite,clipgt-aaa,uuid-1\n")
    new_rows = pl.DataFrame(
        [
            {
                "test_suite_id": "my-suite",
                "scene_id": "clipgt-aaa",
                "uuid": "uuid-1",
            },
            {
                "test_suite_id": "my-suite",
                "scene_id": "clipgt-aaa",
                "uuid": "uuid-2",
            },
        ]
    )

    added, duplicates = merge_suites_csv(str(suites), new_rows)

    assert (added, duplicates) == (1, 1)
    assert pl.read_csv(suites)["uuid"].to_list() == ["uuid-1", "uuid-2"]


def test_validate_csvs_catches_invalid_artifact_repository(tmp_path):
    """Verify validation catches invalid artifact_repository values."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,invalid_repo,\n"  # invalid!
    )
    suites.write_text("test_suite_id,scene_id,uuid\n")

    with pytest.raises(CSVValidationError, match="Invalid artifact_repository"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_missing_hf_revision(tmp_path):
    """Verify validation catches huggingface rows without hf_revision."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa-bbb-ccc,0.2.220-abc123,path/to/file.usdz,2025-01-01 00:00:00,huggingface,\n"
    )
    suites.write_text("test_suite_id,scene_id,uuid\n")

    with pytest.raises(CSVValidationError, match="hf_revision"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_passes_for_valid_data(tmp_path):
    """Verify validation passes for correctly formatted CSVs."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa-bbb-ccc,0.2.220-abc123,alpasim/path/to/file.usdz,2025-01-01 00:00:00,swiftstack,\n"
        "uuid-2,clipgt-ddd-eee-fff,0.2.220-abc123,alpasim/path/to/file2.usdz,2025-01-02 12:30:45,huggingface,v1\n"
    )
    suites.write_text(
        "test_suite_id,scene_id,uuid\n"
        "my-suite,clipgt-aaa-bbb-ccc,uuid-1\n"
        "my-suite,clipgt-ddd-eee-fff,uuid-2\n"
        "another-suite,clipgt-aaa-bbb-ccc,uuid-1\n"
    )

    # Should not raise
    validate_csvs(str(scenes), str(suites))
