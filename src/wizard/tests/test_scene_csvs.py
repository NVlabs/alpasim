# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""CI test to validate scene and suite CSV files in the repository."""

from pathlib import Path

import pytest
from alpasim_wizard.scenes.csv_utils import CSVValidationError, validate_csvs


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


def test_scene_csvs_are_valid():
    """
    Validate that the repository's scene and suite CSV files are well-formed.

    This test runs in CI to catch:
    - Duplicate entries
    - Missing required columns
    - Invalid formats (UUIDs, timestamps, scene_ids)
    - Orphaned suite references (scene_ids not in scenes file)
    """
    try:
        validate_csvs(str(SCENES_CSV), str(SUITES_CSV))
    except CSVValidationError as e:
        pytest.fail(f"Scene CSV validation failed:\n{e}")


def test_validate_csvs_catches_duplicate_uuids(tmp_path):
    """Verify validation catches duplicate UUIDs."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "dup-uuid,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack\n"
        "dup-uuid,clipgt-bbb,0.2.220,path/b,2025-01-01 00:00:00,swiftstack\n"  # duplicate!
    )
    suites.write_text("test_suite_id,scene_id\n")

    with pytest.raises(CSVValidationError, match="Duplicate UUIDs"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_orphaned_suite_references(tmp_path):
    """Verify validation catches suite entries referencing non-existent scenes."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "uuid-1,clipgt-exists,0.2.220,path/a,2025-01-01 00:00:00,swiftstack\n"
    )
    suites.write_text(
        "test_suite_id,scene_id\n"
        "my-suite,clipgt-exists\n"
        "my-suite,clipgt-missing\n"  # not in scenes!
    )

    with pytest.raises(CSVValidationError, match="scene_ids not in scenes CSV"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_invalid_timestamp_format(tmp_path):
    """Verify validation catches non-ISO timestamp formats."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,01/15/2025 10:30:00,swiftstack\n"  # wrong format!
    )
    suites.write_text("test_suite_id,scene_id\n")

    with pytest.raises(CSVValidationError, match="Invalid last_modified format"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_invalid_scene_id_format(tmp_path):
    """Verify validation catches invalid scene_id formats."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "uuid-1,invalid-scene-id,0.2.220,path/a,2025-01-01 00:00:00,swiftstack\n"  # missing clipgt- prefix
    )
    suites.write_text("test_suite_id,scene_id\n")

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
    suites.write_text("test_suite_id,scene_id\n")

    with pytest.raises(CSVValidationError, match="missing columns"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_duplicate_suite_entries(tmp_path):
    """Verify validation catches duplicate (test_suite_id, scene_id) pairs."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack\n"
    )
    suites.write_text(
        "test_suite_id,scene_id\n"
        "my-suite,clipgt-aaa\n"
        "my-suite,clipgt-aaa\n"  # duplicate!
    )

    with pytest.raises(CSVValidationError, match="Duplicate"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_catches_invalid_artifact_repository(tmp_path):
    """Verify validation catches invalid artifact_repository values."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,invalid_repo\n"  # invalid!
    )
    suites.write_text("test_suite_id,scene_id\n")

    with pytest.raises(CSVValidationError, match="Invalid artifact_repository"):
        validate_csvs(str(scenes), str(suites))


def test_validate_csvs_passes_for_valid_data(tmp_path):
    """Verify validation passes for correctly formatted CSVs."""
    scenes = tmp_path / "scenes.csv"
    suites = tmp_path / "suites.csv"

    scenes.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "uuid-1,clipgt-aaa-bbb-ccc,0.2.220-abc123,alpasim/path/to/file.usdz,2025-01-01 00:00:00,swiftstack\n"
        "uuid-2,clipgt-ddd-eee-fff,0.2.220-abc123,alpasim/path/to/file2.usdz,2025-01-02 12:30:45,huggingface\n"
    )
    suites.write_text(
        "test_suite_id,scene_id\n"
        "my-suite,clipgt-aaa-bbb-ccc\n"
        "my-suite,clipgt-ddd-eee-fff\n"
        "another-suite,clipgt-aaa-bbb-ccc\n"
    )

    # Should not raise
    validate_csvs(str(scenes), str(suites))
