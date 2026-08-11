# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""CI test to validate scene and suite CSV files in the repository."""

from pathlib import Path

import polars as pl
import pytest
from alpasim_wizard.scenes.csv_utils import (
    SCENES_COLUMNS,
    SUITES_COLUMNS,
    CSVValidationError,
    check_catalog_headers,
    merge_scenes_csv,
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
    ("suite_id", "hf_revision", "expected_count"),
    [
        ("public_2601", "26.01", 913),
        ("public_2601_video_model", "26.01", 729),
        ("public_2604", "26.04", 1606),
    ],
)
def test_public_suite_pins_its_release(
    suite_id: str, hf_revision: str, expected_count: int
):
    """A versioned public suite selects artifacts from that release."""
    scenes = pl.read_csv(SCENES_CSV, infer_schema_length=0)
    suite = pl.read_csv(SUITES_CSV, infer_schema_length=0).filter(
        pl.col("test_suite_id") == suite_id
    )

    selected = suite.join(scenes, on=["scene_id", "uuid"], how="inner")

    assert selected.height == suite.height
    assert suite.height == expected_count
    assert suite["scene_id"].n_unique() == expected_count
    assert suite["uuid"].n_unique() == expected_count
    assert selected["hf_revision"].unique().to_list() == [hf_revision]
    paths = selected["path"]
    assert paths.is_not_null().all()
    assert paths.str.starts_with(f"sample_set/{hf_revision}_release/").all()


def test_video_model_suite_is_a_subset_of_public_2601():
    """Video-model scenes must also pass the public 26.01 validity filter."""
    suites = pl.read_csv(SUITES_CSV, infer_schema_length=0)
    public = suites.filter(pl.col("test_suite_id") == "public_2601").select(
        ["scene_id", "uuid"]
    )
    video_model = suites.filter(
        pl.col("test_suite_id") == "public_2601_video_model"
    ).select(["scene_id", "uuid"])

    assert video_model.join(public, on=["scene_id", "uuid"], how="anti").is_empty()


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


def test_merge_scenes_csv_appends_to_a_wider_catalog(tmp_path):
    """A catalog may carry more columns than SCENES_COLUMNS; merging must still work.

    The internal scenes CSV has fifteen columns against the seven in SCENES_COLUMNS. Selecting the
    constant rather than the file's own header made the two frames different widths, so nothing
    could be appended to that catalog at all.
    """
    scenes_csv = tmp_path / "sim_scenes.csv"
    extra = ["source_artifact_uuid", "session_id", "map_type"]
    header = SCENES_COLUMNS + extra
    existing = pl.DataFrame(
        [
            {
                **{c: "" for c in extra},
                "uuid": "11111111-1111-4111-8111-111111111111",
                "scene_id": "clipgt-11111111-1111-4111-8111-111111111111",
                "nre_version_string": "26.4.96-91b06fb8",
                "path": "alpasim/artifacts/NRE/run/a.usdz",
                "last_modified": "2026-07-31 12:00:00",
                "artifact_repository": "swiftstack",
                "hf_revision": "",
            }
        ]
    ).select(header)
    existing.write_csv(scenes_csv)

    new_rows = pl.DataFrame(
        [
            {
                "uuid": "22222222-2222-4222-8222-222222222222",
                "scene_id": "clipgt-22222222-2222-4222-8222-222222222222",
                "nre_version_string": "26.4.96-91b06fb8",
                "path": "alpasim/artifacts/NRE/run/b.usdz",
                "last_modified": "2026-07-31 12:00:00",
                "artifact_repository": "swiftstack",
                "hf_revision": "",
            }
        ]
    )

    added, duplicates = merge_scenes_csv(str(scenes_csv), new_rows)

    assert (added, duplicates) == (1, 0)
    result = pl.read_csv(scenes_csv)
    assert result.columns == header, "merge must not change the catalog's schema"
    assert result.height == 2


def test_merge_scenes_csv_rejects_rows_missing_required_columns(tmp_path):
    """A caller that omits a required column gets a clear error, not a silent null."""
    scenes_csv = tmp_path / "sim_scenes.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in SCENES_COLUMNS}).write_csv(scenes_csv)

    with pytest.raises(ValueError, match="missing required columns"):
        merge_scenes_csv(
            str(scenes_csv),
            pl.DataFrame([{"uuid": "abc", "scene_id": "clipgt-abc"}]),
        )


def test_merge_scenes_csv_rejects_rows_missing_the_dedupe_key(tmp_path):
    """Omitting uuid must hit the same contract, not polars' own column lookup.

    The dedupe filter reads uuid, so validation has to run before it or the caller gets a
    ColumnNotFoundError naming one column instead of a ValueError naming all of them.
    """
    scenes_csv = tmp_path / "sim_scenes.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in SCENES_COLUMNS}).write_csv(scenes_csv)

    with pytest.raises(ValueError, match="missing required columns"):
        merge_scenes_csv(
            str(scenes_csv),
            pl.DataFrame([{"scene_id": "clipgt-abc", "path": "alpasim/a.usdz"}]),
        )


def test_merge_suites_csv_rejects_rows_missing_the_dedupe_key(tmp_path):
    """The suites dedupe key gets the same treatment as the scenes one."""
    suites_csv = tmp_path / "sim_suites.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in SUITES_COLUMNS}).write_csv(suites_csv)

    with pytest.raises(ValueError, match="missing required columns"):
        merge_suites_csv(
            str(suites_csv),
            pl.DataFrame([{"uuid": "abc", "scene_id": "clipgt-abc"}]),
        )


def test_merge_scenes_csv_rejects_a_catalog_missing_required_columns(tmp_path):
    """A malformed header must fail before the write, not silently drop the values it lacks.

    Taking the file's own header as the target means a catalog narrower than the contract would
    otherwise absorb valid rows and discard the columns it happens to be missing.
    """
    scenes_csv = tmp_path / "sim_scenes.csv"
    header = [c for c in SCENES_COLUMNS if c not in ("scene_id", "hf_revision")]
    pl.DataFrame(schema={c: pl.Utf8 for c in header}).write_csv(scenes_csv)
    before = scenes_csv.read_text()

    new_rows = pl.DataFrame(
        [
            {
                "uuid": "22222222-2222-4222-8222-222222222222",
                "scene_id": "clipgt-22222222-2222-4222-8222-222222222222",
                "nre_version_string": "26.4.96-91b06fb8",
                "path": "alpasim/artifacts/NRE/run/b.usdz",
                "last_modified": "2026-07-31 12:00:00",
                "artifact_repository": "swiftstack",
                "hf_revision": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="Existing CSV is missing required columns"):
        merge_scenes_csv(str(scenes_csv), new_rows)

    assert scenes_csv.read_text() == before, "a rejected merge must not touch the file"


def test_merge_scenes_csv_preserves_numeric_looking_revision_pins(tmp_path):
    """Revision pins are strings; inferring them from the catalog's contents corrupts them.

    A catalog whose hf_revision values all look numeric reads back as Float64 unless the reader
    forces strings, which turns the existing "26.10" into 26.1 and the incoming "v1" into null.
    """
    scenes_csv = tmp_path / "sim_scenes.csv"

    def row(n: str, revision: str) -> dict[str, str]:
        return {
            "uuid": n * 8,
            "scene_id": f"clipgt-{n * 8}",
            "nre_version_string": "26.4.96-91b06fb8",
            "path": f"alpasim/artifacts/NRE/run/{n}.usdz",
            "last_modified": "2026-07-31 12:00:00",
            "artifact_repository": "huggingface",
            "hf_revision": revision,
        }

    pl.DataFrame([row("1", "26.10"), row("2", "25.05")]).write_csv(scenes_csv)

    added, duplicates = merge_scenes_csv(
        str(scenes_csv), pl.DataFrame([row("3", "v1")])
    )

    assert (added, duplicates) == (1, 0)
    result = pl.read_csv(scenes_csv, infer_schema_length=0)
    assert result["hf_revision"].to_list() == ["26.10", "25.05", "v1"]


def test_check_catalog_headers_rejects_the_pair_before_the_first_write(tmp_path):
    """A bad suites header must stop the scenes merge too, or the pair drifts apart.

    Callers write the scenes catalog first, so a suites header caught only when its own merge
    runs would leave the scenes file already updated and no matching suite rows.
    """
    scenes_csv = tmp_path / "sim_scenes.csv"
    suites_csv = tmp_path / "sim_suites.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in SCENES_COLUMNS}).write_csv(scenes_csv)
    pl.DataFrame(schema={c: pl.Utf8 for c in SUITES_COLUMNS if c != "uuid"}).write_csv(
        suites_csv
    )

    with pytest.raises(ValueError, match=r"sim_suites\.csv.*missing required columns"):
        check_catalog_headers(
            (str(scenes_csv), SCENES_COLUMNS), (str(suites_csv), SUITES_COLUMNS)
        )


def test_check_catalog_headers_ignores_files_that_do_not_exist(tmp_path):
    """Creating a catalog is the merge's job; a missing path is not a malformed one."""
    check_catalog_headers((str(tmp_path / "absent.csv"), SCENES_COLUMNS))


def test_check_catalog_headers_ignores_a_none_path(tmp_path):
    """--suites-csv is optional, so a scenes-only run passes None and must not crash."""
    scenes_csv = tmp_path / "sim_scenes.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in SCENES_COLUMNS}).write_csv(scenes_csv)

    check_catalog_headers((str(scenes_csv), SCENES_COLUMNS), (None, SUITES_COLUMNS))
