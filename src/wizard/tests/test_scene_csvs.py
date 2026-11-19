# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""CI test to validate clip-catalog and scene-suite CSV files in the repository."""

from pathlib import Path

import polars as pl
import pytest
from alpasim_wizard.scenes.csv_utils import (
    CLIP_CATALOG_COLUMNS,
    SCENE_SUITE_COLUMNS,
    CSVValidationError,
    check_catalog_headers,
    merge_clip_catalog_csv,
    merge_scene_suite_csv,
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
CLIP_CATALOG_CSV = REPO_ROOT / "data" / "scenes" / "clip_catalog.csv"
LEGACY_CLIP_CATALOG_CSV = REPO_ROOT / "data" / "scenes" / "clip_catalog_2505.csv"


@pytest.mark.parametrize(
    "clip_catalog_csvs",
    [CLIP_CATALOG_CSV, LEGACY_CLIP_CATALOG_CSV],
)
def test_scene_csvs_are_valid(clip_catalog_csvs: Path):
    """
    Validate that the repository's clip-catalog CSV files are well-formed.

    This test runs in CI to catch:
    - Duplicate entries
    - Missing required columns
    - Invalid formats (UUIDs, timestamps, clip_ids)
    """
    try:
        validate_csvs(str(clip_catalog_csvs))
    except CSVValidationError as e:
        pytest.fail(f"Clip-catalog CSV validation failed:\n{e}")


def test_curated_split_is_disjoint() -> None:
    """Prevent clip leakage between public training and validation suites."""
    suite_dir = REPO_ROOT / "data" / "scenes" / "scene_suites"
    train = pl.read_csv(suite_dir / "curated_train.csv")
    val = pl.read_csv(suite_dir / "curated_val.csv")
    assert train.join(val, on="clip_id", how="inner").is_empty()


def test_validate_csvs_catches_duplicate_uuids(tmp_path):
    """Verify validation catches duplicate UUIDs."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "dup-uuid,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
        "dup-uuid,clipgt-bbb,0.2.220,path/b,2025-01-01 00:00:00,swiftstack,\n"  # duplicate!
    )
    scene_suite.write_text("clip_id,uuid,event_t0_us\n")

    with pytest.raises(CSVValidationError, match="Duplicate UUIDs"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_catches_orphaned_scene_suite_references(tmp_path):
    """Verify validation catches scene-suite entries referencing non-existent artifacts."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
    )
    scene_suite.write_text(
        "clip_id,uuid,event_t0_us\nclipgt-missing,uuid-missing,AUTO\n"
    )

    with pytest.raises(CSVValidationError, match="pairs not in clip catalog CSV"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_catches_mismatched_clip_and_uuid(tmp_path):
    """Verify a scene-suite UUID must belong to the clip ID beside it."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
        "uuid-2,clipgt-bbb,0.2.220,path/b,2025-01-01 00:00:00,swiftstack,\n"
    )
    scene_suite.write_text("clip_id,uuid,event_t0_us\nclipgt-aaa,uuid-2,AUTO\n")

    with pytest.raises(CSVValidationError, match="pairs not in clip catalog CSV"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_catches_invalid_timestamp_format(tmp_path):
    """Verify validation catches non-ISO timestamp formats."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,01/15/2025 10:30:00,swiftstack,\n"  # wrong format!
    )
    scene_suite.write_text("clip_id,uuid,event_t0_us\n")

    with pytest.raises(CSVValidationError, match="Invalid last_modified format"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_catches_invalid_clip_id_format(tmp_path):
    """Verify validation catches invalid clip_id formats."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,invalid-scene-id,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"  # missing clipgt- prefix
    )
    scene_suite.write_text("clip_id,uuid,event_t0_us\n")

    with pytest.raises(CSVValidationError, match="Invalid clip_id format"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_catches_missing_columns(tmp_path):
    """Verify validation catches missing required columns."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id\n"  # missing nre_version_string, path, last_modified, artifact_repository
        "uuid-1,clipgt-aaa\n"
    )
    scene_suite.write_text("clip_id,uuid,event_t0_us\n")

    with pytest.raises(CSVValidationError, match="missing columns"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_requires_scene_suite_uuid(tmp_path):
    """Verify scene-suite rows must pin an artifact UUID."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
    )
    scene_suite.write_text("clip_id\nclipgt-aaa\n")

    with pytest.raises(CSVValidationError, match="Scene-suite CSV missing columns"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_catches_duplicate_scene_suite_entries(tmp_path):
    """Verify validation catches exact duplicate authored rows."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
    )
    scene_suite.write_text(
        "clip_id,uuid,event_t0_us\nclipgt-aaa,uuid-1,AUTO\nclipgt-aaa,uuid-1,AUTO\n"
    )

    with pytest.raises(CSVValidationError, match="Duplicate"):
        validate_csvs(str(scenes), str(scene_suite))


def test_merge_scene_suite_csv_uses_authored_clip_identity(tmp_path):
    """Rows deduplicate by their complete authored identity."""
    scene_suite = tmp_path / "scene_suite.csv"
    scene_suite.write_text("clip_id,uuid,event_t0_us\nclipgt-aaa,uuid-1,AUTO\n")
    new_rows = pl.DataFrame(
        [
            {
                "clip_id": "clipgt-aaa",
                "uuid": "uuid-1",
                "event_t0_us": "AUTO",
            },
            {
                "clip_id": "clipgt-bbb",
                "uuid": "uuid-2",
                "event_t0_us": "AUTO",
            },
        ]
    )

    added, duplicates = merge_scene_suite_csv(str(scene_suite), new_rows)

    assert (added, duplicates) == (1, 1)
    assert pl.read_csv(scene_suite)["event_t0_us"].to_list() == ["AUTO", "AUTO"]


def test_merge_scene_suite_csv_normalizes_existing_rows_for_deduplication(tmp_path):
    scene_suite = tmp_path / "scene_suite.csv"
    scene_suite.write_text(
        "clip_id,uuid,event_t0_us\n  clipgt-aaa  ,  uuid-1  ,  AUTO  \n"
    )
    new_rows = pl.DataFrame(
        [{"clip_id": "clipgt-aaa", "uuid": "uuid-1", "event_t0_us": "AUTO"}]
    )

    added, duplicates = merge_scene_suite_csv(str(scene_suite), new_rows)

    assert (added, duplicates) == (0, 1)
    assert pl.read_csv(scene_suite).height == 1


def test_merge_scene_suite_csv_accepts_numeric_t0(tmp_path):
    scene_suite = tmp_path / "scene_suite.csv"
    scene_suite.write_text("clip_id,uuid,event_t0_us\nclipgt-aaa,uuid-1,AUTO\n")
    rows = pl.DataFrame(
        [
            {"clip_id": "clipgt-aaa", "uuid": "uuid-1", "event_t0_us": "AUTO"},
            {"clip_id": "clipgt-aaa", "uuid": "uuid-1", "event_t0_us": "1"},
        ]
    )

    added, duplicates = merge_scene_suite_csv(str(scene_suite), rows)

    assert (added, duplicates) == (1, 1)
    assert pl.read_csv(scene_suite)["event_t0_us"].to_list() == ["AUTO", "1"]


def test_validate_csvs_catches_invalid_artifact_repository(tmp_path):
    """Verify validation catches invalid artifact_repository values."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,invalid_repo,\n"  # invalid!
    )
    scene_suite.write_text("clip_id,uuid,event_t0_us\n")

    with pytest.raises(CSVValidationError, match="Invalid artifact_repository"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_catches_missing_hf_revision(tmp_path):
    """Verify validation catches huggingface rows without hf_revision."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa-bbb-ccc,0.2.220-abc123,path/to/file.usdz,2025-01-01 00:00:00,huggingface,\n"
    )
    scene_suite.write_text("clip_id,uuid,event_t0_us\n")

    with pytest.raises(CSVValidationError, match="hf_revision"):
        validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_passes_for_valid_data(tmp_path):
    """Verify validation passes for correctly formatted CSVs."""
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa-bbb-ccc,0.2.220-abc123,alpasim/path/to/file.usdz,2025-01-01 00:00:00,swiftstack,\n"
        "uuid-2,clipgt-ddd-eee-fff,0.2.220-abc123,alpasim/path/to/file2.usdz,2025-01-02 12:30:45,huggingface,v1\n"
    )
    scene_suite.write_text(
        "clip_id,uuid,event_t0_us\n"
        "clipgt-aaa-bbb-ccc,uuid-1,AUTO\n"
        "clipgt-ddd-eee-fff,uuid-2,AUTO\n"
    )

    # Should not raise
    validate_csvs(str(scenes), str(scene_suite))


def test_validate_csvs_normalizes_scene_suite_pair_before_cross_validation(tmp_path):
    scenes = tmp_path / "scenes.csv"
    scene_suite = tmp_path / "scene_suite.csv"

    scenes.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2.220,path/a,2025-01-01 00:00:00,swiftstack,\n"
    )
    scene_suite.write_text(
        "clip_id,uuid,event_t0_us\n  clipgt-aaa  ,  uuid-1  ,  AUTO  \n"
    )

    validate_csvs(str(scenes), str(scene_suite))


def test_merge_clip_catalog_csv_appends_to_a_wider_catalog(tmp_path):
    """A catalog may carry more columns than CLIP_CATALOG_COLUMNS; merging must still work.

    The internal clip catalog has fifteen columns against the seven in CLIP_CATALOG_COLUMNS. Selecting the
    constant rather than the file's own header made the two frames different widths, so nothing
    could be appended to that catalog at all.
    """
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    extra = ["source_artifact_uuid", "session_id", "map_type"]
    header = CLIP_CATALOG_COLUMNS + extra
    existing = pl.DataFrame(
        [
            {
                **{c: "" for c in extra},
                "uuid": "11111111-1111-4111-8111-111111111111",
                "clip_id": "clipgt-11111111-1111-4111-8111-111111111111",
                "nre_version_string": "26.4.96-91b06fb8",
                "path": "alpasim/artifacts/NRE/run/a.usdz",
                "last_modified": "2026-07-31 12:00:00",
                "artifact_repository": "swiftstack",
                "hf_revision": "",
            }
        ]
    ).select(header)
    existing.write_csv(clip_catalog_csvs)

    new_rows = pl.DataFrame(
        [
            {
                "uuid": "22222222-2222-4222-8222-222222222222",
                "clip_id": "clipgt-22222222-2222-4222-8222-222222222222",
                "nre_version_string": "26.4.96-91b06fb8",
                "path": "alpasim/artifacts/NRE/run/b.usdz",
                "last_modified": "2026-07-31 12:00:00",
                "artifact_repository": "swiftstack",
                "hf_revision": "",
            }
        ]
    )

    added, duplicates = merge_clip_catalog_csv(str(clip_catalog_csvs), new_rows)

    assert (added, duplicates) == (1, 0)
    result = pl.read_csv(clip_catalog_csvs)
    assert result.columns == header, "merge must not change the catalog's schema"
    assert result.height == 2


def test_merge_clip_catalog_csv_rejects_rows_missing_required_columns(tmp_path):
    """A caller that omits a required column gets a clear error, not a silent null."""
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in CLIP_CATALOG_COLUMNS}).write_csv(
        clip_catalog_csvs
    )

    with pytest.raises(ValueError, match="missing required columns"):
        merge_clip_catalog_csv(
            str(clip_catalog_csvs),
            pl.DataFrame([{"uuid": "abc", "clip_id": "clipgt-abc"}]),
        )


def test_merge_clip_catalog_csv_rejects_rows_missing_the_dedupe_key(tmp_path):
    """Omitting uuid must hit the same contract, not polars' own column lookup.

    The dedupe filter reads uuid, so validation has to run before it or the caller gets a
    ColumnNotFoundError naming one column instead of a ValueError naming all of them.
    """
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in CLIP_CATALOG_COLUMNS}).write_csv(
        clip_catalog_csvs
    )

    with pytest.raises(ValueError, match="missing required columns"):
        merge_clip_catalog_csv(
            str(clip_catalog_csvs),
            pl.DataFrame([{"clip_id": "clipgt-abc", "path": "alpasim/a.usdz"}]),
        )


def test_merge_scene_suite_csv_rejects_rows_missing_the_dedupe_key(tmp_path):
    """The scene-suite dedupe key gets the same treatment as the clip-catalog one."""
    scene_suite_csv = tmp_path / "scene_suite.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in SCENE_SUITE_COLUMNS}).write_csv(
        scene_suite_csv
    )

    with pytest.raises(ValueError, match="missing columns"):
        merge_scene_suite_csv(
            str(scene_suite_csv),
            pl.DataFrame([{"uuid": "abc", "clip_id": "clipgt-abc"}]),
        )


def test_merge_clip_catalog_csv_rejects_a_catalog_missing_required_columns(tmp_path):
    """A malformed header must fail before the write, not silently drop the values it lacks.

    Taking the file's own header as the target means a catalog narrower than the contract would
    otherwise absorb valid rows and discard the columns it happens to be missing.
    """
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    header = [c for c in CLIP_CATALOG_COLUMNS if c not in ("clip_id", "hf_revision")]
    pl.DataFrame(schema={c: pl.Utf8 for c in header}).write_csv(clip_catalog_csvs)
    before = clip_catalog_csvs.read_text()

    new_rows = pl.DataFrame(
        [
            {
                "uuid": "22222222-2222-4222-8222-222222222222",
                "clip_id": "clipgt-22222222-2222-4222-8222-222222222222",
                "nre_version_string": "26.4.96-91b06fb8",
                "path": "alpasim/artifacts/NRE/run/b.usdz",
                "last_modified": "2026-07-31 12:00:00",
                "artifact_repository": "swiftstack",
                "hf_revision": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="Existing CSV is missing required columns"):
        merge_clip_catalog_csv(str(clip_catalog_csvs), new_rows)

    assert (
        clip_catalog_csvs.read_text() == before
    ), "a rejected merge must not touch the file"


def test_merge_clip_catalog_csv_preserves_numeric_looking_revision_pins(tmp_path):
    """Revision pins are strings; inferring them from the catalog's contents corrupts them.

    A catalog whose hf_revision values all look numeric reads back as Float64 unless the reader
    forces strings, which turns the existing "26.10" into 26.1 and the incoming "v1" into null.
    """
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"

    def row(n: str, revision: str) -> dict[str, str]:
        return {
            "uuid": n * 8,
            "clip_id": f"clipgt-{n * 8}",
            "nre_version_string": "26.4.96-91b06fb8",
            "path": f"alpasim/artifacts/NRE/run/{n}.usdz",
            "last_modified": "2026-07-31 12:00:00",
            "artifact_repository": "huggingface",
            "hf_revision": revision,
        }

    pl.DataFrame([row("1", "26.10"), row("2", "25.05")]).write_csv(clip_catalog_csvs)

    added, duplicates = merge_clip_catalog_csv(
        str(clip_catalog_csvs), pl.DataFrame([row("3", "v1")])
    )

    assert (added, duplicates) == (1, 0)
    result = pl.read_csv(clip_catalog_csvs, infer_schema_length=0)
    assert result["hf_revision"].to_list() == ["26.10", "25.05", "v1"]


def test_check_catalog_headers_rejects_the_pair_before_the_first_write(tmp_path):
    """A bad scene-suite header must stop the clip-catalog merge too, or the pair drifts apart.

    Callers write the clip catalog first, so a scene-suite header caught only when its own merge
    runs would leave the clip-catalog file already updated and no matching scene-suite rows.
    """
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    scene_suite_csv = tmp_path / "scene_suite.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in CLIP_CATALOG_COLUMNS}).write_csv(
        clip_catalog_csvs
    )
    pl.DataFrame(
        schema={c: pl.Utf8 for c in SCENE_SUITE_COLUMNS if c != "uuid"}
    ).write_csv(scene_suite_csv)

    with pytest.raises(
        ValueError,
        match=r"scene_suite\.csv.*missing required columns",
    ):
        check_catalog_headers(
            (str(clip_catalog_csvs), CLIP_CATALOG_COLUMNS),
            (str(scene_suite_csv), SCENE_SUITE_COLUMNS),
        )


def test_check_catalog_headers_ignores_files_that_do_not_exist(tmp_path):
    """Creating a catalog is the merge's job; a missing path is not a malformed one."""
    check_catalog_headers((str(tmp_path / "absent.csv"), CLIP_CATALOG_COLUMNS))


def test_check_catalog_headers_ignores_a_none_path(tmp_path):
    """--scene-suite-csv is optional, so a catalog-only run passes None safely."""
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    pl.DataFrame(schema={c: pl.Utf8 for c in CLIP_CATALOG_COLUMNS}).write_csv(
        clip_catalog_csvs
    )

    check_catalog_headers(
        (str(clip_catalog_csvs), CLIP_CATALOG_COLUMNS), (None, SCENE_SUITE_COLUMNS)
    )
