# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import zipfile
from pathlib import Path

import polars as pl
import pytest
import yaml
from alpasim_wizard.scenes.csv_utils import ArtifactRepository
from alpasim_wizard.scenes.sceneset import (
    LOCAL_SUITE_ID,
    SceneIdAndUuid,
    USDZManager,
    USDZQueryError,
    _deduplicate,
    scan_local_usdz_directory,
)
from alpasim_wizard.schema import ScenesConfig

"""
These tests use stripped out USDZ files which contain just the yaml metadata (fast to download).
If necessary to update, regenerate the stripped USDZ files with `strip_usdz.py` in this directory.
"""

EXAMPLE_USDZ_UUID = "stripped-90db43dd-e5d2-41c4-9a69-400f6c33fb45"
EXAMPLE_SCENE_ID = "clipgt-c370a6ff-e319-4757-8282-09a67fad614e"
EXAMPLE_NRE_VERSION = "stripped-0.2.220-1777390b"
EXAMPLE_USDZ_SS_PATH = "alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz/90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz"  # noqa


@pytest.fixture
def test_csvs(tmp_path: Path) -> tuple[Path, Path]:
    """Create test CSV files with sample data."""
    scenes_csv = tmp_path / "sim_scenes.csv"
    suites_csv = tmp_path / "sim_suites.csv"

    # Create scenes CSV with test data
    scenes_csv.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
        "stripped-2b8f88d4-8348-4faf-9dd3-78564fddde78,clipgt-c045249c-2c01-45b0-87f5-6f631f71f1f1,"
        "stripped-0.2.220-1777390b,alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/"
        "2b8f88d4-8348-4faf-9dd3-78564fddde78.usdz/2b8f88d4-8348-4faf-9dd3-78564fddde78.usdz,2025-02-19 14:18:25,swiftstack\n"  # noqa
        "stripped-90db43dd-e5d2-41c4-9a69-400f6c33fb45,clipgt-c370a6ff-e319-4757-8282-09a67fad614e,"
        "stripped-0.2.220-1777390b,alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/"
        "90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz/90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz,2025-02-19 14:18:28,swiftstack\n"  # noqa
        "stripped-c146251f-16d0-43eb-8905-3f0c037028cb,clipgt-c1ba971e-260f-4a7d-90b1-f60a9caf6acb,"
        "stripped-0.2.220-1777390b,alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/"
        "c146251f-16d0-43eb-8905-3f0c037028cb.usdz/c146251f-16d0-43eb-8905-3f0c037028cb.usdz,2025-02-19 14:18:31,swiftstack\n"  # noqa
    )

    # Create suites CSV with test data
    suites_csv.write_text(
        "test_suite_id,scene_id\n"
        "dev.alpasim.unit_tests.v0,clipgt-c045249c-2c01-45b0-87f5-6f631f71f1f1\n"
        "dev.alpasim.unit_tests.v0,clipgt-c370a6ff-e319-4757-8282-09a67fad614e\n"
        "dev.alpasim.unit_tests.v0,clipgt-c1ba971e-260f-4a7d-90b1-f60a9caf6acb\n"
    )

    return scenes_csv, suites_csv


@pytest.fixture
def usdz_manager(tmp_path: Path, test_csvs: tuple[Path, Path]) -> USDZManager:
    """Create a USDZManager with test CSV files."""
    scenes_csv, suites_csv = test_csvs
    config = ScenesConfig(
        scene_cache=str(tmp_path),
        scenes_csv=str(scenes_csv),
        suites_csv=str(suites_csv),
        artifact_compatibility_matrix={},
    )
    return USDZManager.from_cfg(config)


def test_query_by_scene_ids(usdz_manager: USDZManager):
    """Test querying scenes by scene IDs."""
    results = usdz_manager.query_by_scene_ids(
        scene_ids=[EXAMPLE_SCENE_ID],
        nre_versions=[EXAMPLE_NRE_VERSION],
    )

    assert len(results) == 1
    assert results[0].uuid == EXAMPLE_USDZ_UUID
    assert results[0].scene_id == EXAMPLE_SCENE_ID


def test_query_by_scene_ids_just_an_invalid_uuid(usdz_manager: USDZManager):
    """Test that querying with invalid scene ID raises error."""
    with pytest.raises(USDZQueryError):
        usdz_manager.query_by_scene_ids(
            scene_ids=["invalid-uuid"],
            nre_versions=[EXAMPLE_NRE_VERSION],
        )


def test_query_by_scene_ids_valid_and_invalid_uuid(usdz_manager: USDZManager):
    """Test that querying with mix of valid and invalid scene IDs raises error."""
    with pytest.raises(USDZQueryError):
        usdz_manager.query_by_scene_ids(
            scene_ids=[EXAMPLE_SCENE_ID, "invalid-uuid"],
            nre_versions=[EXAMPLE_NRE_VERSION],
        )


def test_query_by_scene_ids_invalid_nre_version(usdz_manager: USDZManager):
    """Test that querying with invalid NRE version raises error."""
    with pytest.raises(USDZQueryError):
        usdz_manager.query_by_scene_ids(
            scene_ids=[EXAMPLE_SCENE_ID],
            nre_versions=["invalid-version"],
        )


def test_query_by_suite_id(usdz_manager: USDZManager):
    """Test querying scenes by suite ID."""
    results = usdz_manager.query_by_suite_id(
        test_suite_id="dev.alpasim.unit_tests.v0",
        nre_versions=[EXAMPLE_NRE_VERSION],
    )

    assert len(results) == 3
    assert EXAMPLE_SCENE_ID in [r.scene_id for r in results]
    assert EXAMPLE_USDZ_UUID in [r.uuid for r in results]


def test_query_by_suite_id_invalid_id(usdz_manager: USDZManager):
    """Test that querying with invalid suite ID raises error."""
    with pytest.raises(USDZQueryError):
        usdz_manager.query_by_suite_id(
            test_suite_id="invalid-id",
            nre_versions=[EXAMPLE_NRE_VERSION],
        )


def test_query_by_suite_id_invalid_nre_version(usdz_manager: USDZManager):
    """Test that querying suite with invalid NRE version raises error."""
    with pytest.raises(USDZQueryError):
        usdz_manager.query_by_suite_id(
            test_suite_id="dev.alpasim.unit_tests.v0",
            nre_versions=["invalid-version"],
        )


def test_get_paths(usdz_manager: USDZManager):
    """Test getting SwiftStack paths for UUIDs."""
    paths = usdz_manager.get_paths([EXAMPLE_USDZ_UUID])

    assert EXAMPLE_USDZ_UUID in paths
    assert EXAMPLE_USDZ_SS_PATH in paths[EXAMPLE_USDZ_UUID]


def test_get_artifact_info(usdz_manager: USDZManager):
    """Test getting artifact paths and repositories for UUIDs."""
    from alpasim_wizard.scenes.csv_utils import ArtifactRepository

    info = usdz_manager.get_artifact_info([EXAMPLE_USDZ_UUID])

    assert EXAMPLE_USDZ_UUID in info
    path, repo = info[EXAMPLE_USDZ_UUID]
    assert EXAMPLE_USDZ_SS_PATH in path
    assert repo == ArtifactRepository.SWIFTSTACK


def test_deduplicate():
    """Test deduplication by last_modified timestamp."""
    df = pl.DataFrame(
        {
            "scene_id": [
                "clipgt-d8cbf4ca-b7ff-44bd-a5be-260f736a02fe",
                "clipgt-d8cbf4ca-b7ff-44bd-a5be-260f736a02fe",  # duplicate
                "clipgt-d8cbf4ca-b7ff-44bd-a5be-260f736a02ff",  # different
            ],
            "uuid": [
                "dda6ae28-e7f9-493a-b0b6-61e3bbbccbac",
                "fda6ae28-e7f9-493a-b0b6-61e3bbbccbac",
                "ada6ae28-e7f9-493a-b0b6-61e3bbbccbac",
            ],
            "last_modified": [
                "2025-01-18 18:39:30",
                "2025-01-19 18:39:30",  # newer
                "2025-01-19 18:39:31",
            ],
            "nre_version_string": [
                "0.2.335-deadbeef",
                "0.2.221-asdfasdf",
                "0.2.221-asdfasdf",
            ],
        }
    )
    deduplicated = _deduplicate(df)
    assert deduplicated.height == 2
    # Check that the newer record was kept for the duplicate scene_id
    dedup_dict = {
        row["scene_id"]: row["uuid"] for row in deduplicated.iter_rows(named=True)
    }
    assert (
        dedup_dict["clipgt-d8cbf4ca-b7ff-44bd-a5be-260f736a02fe"]
        == "fda6ae28-e7f9-493a-b0b6-61e3bbbccbac"
    )


def test_scene_id_and_uuid_list_from_df():
    """Test SceneIdAndUuid.list_from_df method."""
    df = pl.DataFrame(
        {
            "scene_id": ["clipgt-aaa", "clipgt-bbb"],
            "uuid": ["uuid-1", "uuid-2"],
        }
    )
    results = SceneIdAndUuid.list_from_df(df)
    assert len(results) == 2
    assert results[0].scene_id == "clipgt-aaa"
    assert results[0].uuid == "uuid-1"


def test_scene_id_and_uuid_list_from_df_missing_columns():
    """Test that list_from_df raises error for missing columns."""
    df = pl.DataFrame({"scene_id": ["clipgt-aaa"]})  # missing uuid
    with pytest.raises(ValueError, match="must have columns"):
        SceneIdAndUuid.list_from_df(df)


# Tests for local USDZ directory scanning


def _create_mock_usdz(
    path: Path, uuid: str, scene_id: str, version_string: str = "1.0.0-test"
) -> None:
    """Create a mock USDZ file with metadata."""
    metadata = {
        "uuid": uuid,
        "scene_id": scene_id,
        "version_string": version_string,
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.yaml", yaml.dump(metadata))


@pytest.fixture
def local_usdz_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock USDZ files."""
    usdz_dir = tmp_path / "local_usdzs"
    usdz_dir.mkdir()

    # Create some mock USDZ files
    _create_mock_usdz(
        usdz_dir / "scene1.usdz",
        uuid="uuid-1111-aaaa",
        scene_id="clipgt-scene-1",
        version_string="1.0.0-test",
    )
    _create_mock_usdz(
        usdz_dir / "scene2.usdz",
        uuid="uuid-2222-bbbb",
        scene_id="clipgt-scene-2",
        version_string="1.0.0-test",
    )
    # Create a subdirectory with another USDZ
    subdir = usdz_dir / "subdir"
    subdir.mkdir()
    _create_mock_usdz(
        subdir / "scene3.usdz",
        uuid="uuid-3333-cccc",
        scene_id="clipgt-scene-3",
        version_string="1.0.0-test",
    )

    return usdz_dir


def test_scan_local_usdz_directory(local_usdz_dir: Path):
    """Test scanning a local directory for USDZ files."""
    sim_scenes, sim_suites = scan_local_usdz_directory(str(local_usdz_dir))

    # Check scenes DataFrame
    assert sim_scenes.height == 3
    assert set(sim_scenes.columns) == {
        "uuid",
        "scene_id",
        "nre_version_string",
        "path",
        "last_modified",
        "artifact_repository",
    }

    uuids = set(sim_scenes["uuid"].to_list())
    assert uuids == {"uuid-1111-aaaa", "uuid-2222-bbbb", "uuid-3333-cccc"}

    scene_ids = set(sim_scenes["scene_id"].to_list())
    assert scene_ids == {"clipgt-scene-1", "clipgt-scene-2", "clipgt-scene-3"}

    # All should be marked as "local" repository
    repos = set(sim_scenes["artifact_repository"].to_list())
    assert repos == {"local"}

    # Check suites DataFrame - all scenes should be in the "local" suite
    assert sim_suites.height == 3
    suite_ids = set(sim_suites["test_suite_id"].to_list())
    assert suite_ids == {LOCAL_SUITE_ID}


def test_scan_local_usdz_directory_not_exists():
    """Test that scanning a non-existent directory raises error."""
    with pytest.raises(ValueError, match="does not exist"):
        scan_local_usdz_directory("/nonexistent/path")


def test_scan_local_usdz_directory_empty(tmp_path: Path):
    """Test that scanning an empty directory raises error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No \\*.usdz files found"):
        scan_local_usdz_directory(str(empty_dir))


def test_usdz_manager_from_cfg_with_local_usdz_dir(
    local_usdz_dir: Path, tmp_path: Path
):
    """Test creating USDZManager with local_usdz_dir."""
    # Create dummy CSV files (they shouldn't be used)
    scenes_csv = tmp_path / "sim_scenes.csv"
    suites_csv = tmp_path / "sim_suites.csv"
    scenes_csv.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
    )
    suites_csv.write_text("test_suite_id,scene_id\n")

    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),  # Not used when local_usdz_dir is set
        scenes_csv=str(scenes_csv),  # Not used when local_usdz_dir is set
        suites_csv=str(suites_csv),  # Not used when local_usdz_dir is set
        artifact_compatibility_matrix={},
    )

    manager = USDZManager.from_cfg(config)

    # Should have loaded scenes from local directory
    assert manager.sim_scenes.height == 3

    # Query by the "local" suite should work
    results = manager.query_by_suite_id(
        test_suite_id=LOCAL_SUITE_ID,
        nre_versions=["1.0.0-test"],
    )
    assert len(results) == 3


def test_usdz_manager_query_by_scene_ids_with_local(
    local_usdz_dir: Path, tmp_path: Path
):
    """Test querying by scene IDs when using local USDZ directory."""
    scenes_csv = tmp_path / "sim_scenes.csv"
    suites_csv = tmp_path / "sim_suites.csv"
    scenes_csv.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
    )
    suites_csv.write_text("test_suite_id,scene_id\n")

    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),
        scenes_csv=str(scenes_csv),
        suites_csv=str(suites_csv),
        artifact_compatibility_matrix={},
    )

    manager = USDZManager.from_cfg(config)

    # Query for a subset of scenes
    results = manager.query_by_scene_ids(
        scene_ids=["clipgt-scene-1", "clipgt-scene-2"],
        nre_versions=["1.0.0-test"],
    )
    assert len(results) == 2
    result_scene_ids = {r.scene_id for r in results}
    assert result_scene_ids == {"clipgt-scene-1", "clipgt-scene-2"}


def test_get_artifact_info_local(local_usdz_dir: Path, tmp_path: Path):
    """Test getting artifact info for local artifacts."""
    scenes_csv = tmp_path / "sim_scenes.csv"
    suites_csv = tmp_path / "sim_suites.csv"
    scenes_csv.write_text(
        "uuid,scene_id,nre_version_string,path,last_modified,artifact_repository\n"
    )
    suites_csv.write_text("test_suite_id,scene_id\n")

    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),
        scenes_csv=str(scenes_csv),
        suites_csv=str(suites_csv),
        artifact_compatibility_matrix={},
    )

    manager = USDZManager.from_cfg(config)

    info = manager.get_artifact_info(["uuid-1111-aaaa"])
    assert "uuid-1111-aaaa" in info
    path, repo = info["uuid-1111-aaaa"]
    assert repo == ArtifactRepository.LOCAL
    assert path.endswith("scene1.usdz")
