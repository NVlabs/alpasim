# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest
import yaml
from alpasim_utils.types import ClipDataSource
from alpasim_wizard.scenes.csv_utils import ArtifactRepository
from alpasim_wizard.scenes.sceneset import (
    USDZManager,
    USDZQueryError,
    _deduplicate,
    _load_and_merge_csvs,
    resolve_scene_selection,
    scan_local_usdz_directory,
)
from alpasim_wizard.schema import ScenesConfig, SceneSuiteConfig

"""
These tests use stripped out USDZ files which contain just the yaml metadata (fast to download).
If necessary to update, regenerate the stripped USDZ files with `strip_usdz.py` in this directory.
"""

EXAMPLE_USDZ_UUID = "stripped-90db43dd-e5d2-41c4-9a69-400f6c33fb45"
EXAMPLE_SCENE_ID = "clipgt-c370a6ff-e319-4757-8282-09a67fad614e"
EXAMPLE_NRE_VERSION = "stripped-0.2.220-1777390b"
EXAMPLE_USDZ_SS_PATH = "alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz/90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz"  # noqa


@pytest.fixture
def test_csvs(tmp_path: Path) -> Path:
    """Create a clip catalog with sample data."""
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"

    # Create scenes CSV with test data
    clip_catalog_csvs.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "stripped-2b8f88d4-8348-4faf-9dd3-78564fddde78,clipgt-c045249c-2c01-45b0-87f5-6f631f71f1f1,"
        "stripped-0.2.220-1777390b,alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/"
        "2b8f88d4-8348-4faf-9dd3-78564fddde78.usdz/2b8f88d4-8348-4faf-9dd3-78564fddde78.usdz,2025-02-19 14:18:25,swiftstack,\n"  # noqa
        "stripped-90db43dd-e5d2-41c4-9a69-400f6c33fb45,clipgt-c370a6ff-e319-4757-8282-09a67fad614e,"
        "stripped-0.2.220-1777390b,alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/"
        "90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz/90db43dd-e5d2-41c4-9a69-400f6c33fb45.usdz,2025-02-19 14:18:28,swiftstack,\n"  # noqa
        "stripped-c146251f-16d0-43eb-8905-3f0c037028cb,clipgt-c1ba971e-260f-4a7d-90b1-f60a9caf6acb,"
        "stripped-0.2.220-1777390b,alpasim/artifacts/NRE/unit-tests/25.02.19/three-stripped-scenes/"
        "c146251f-16d0-43eb-8905-3f0c037028cb.usdz/c146251f-16d0-43eb-8905-3f0c037028cb.usdz,2025-02-19 14:18:31,swiftstack,\n"  # noqa
    )

    return clip_catalog_csvs


@pytest.fixture
def usdz_manager(tmp_path: Path, test_csvs: Path) -> USDZManager:
    """Create a USDZManager with test CSV files."""
    config = ScenesConfig(
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[str(test_csvs)],
    )
    return USDZManager.from_cfg(config)


def test_query_by_clip_ids(usdz_manager: USDZManager):
    """Test querying scenes by scene IDs."""
    results = usdz_manager.query_by_clip_ids(
        clip_ids=[EXAMPLE_SCENE_ID],
    )

    assert len(results) == 1
    assert isinstance(results[0].source, ClipDataSource)
    assert results[0].source.usdz_uuid == EXAMPLE_USDZ_UUID
    assert results[0].source.source_id == EXAMPLE_SCENE_ID
    assert results[0].policy_start_us == "AUTO"


def test_query_by_clip_ids_just_an_invalid_uuid(usdz_manager: USDZManager):
    """Test that querying with invalid scene ID raises error."""
    with pytest.raises(USDZQueryError):
        usdz_manager.query_by_clip_ids(
            clip_ids=["invalid-uuid"],
        )


def test_query_by_clip_ids_valid_and_invalid_uuid(usdz_manager: USDZManager):
    """Test that querying with mix of valid and invalid scene IDs raises error."""
    with pytest.raises(USDZQueryError):
        usdz_manager.query_by_clip_ids(
            clip_ids=[EXAMPLE_SCENE_ID, "invalid-uuid"],
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
    path, repo, revision = info[EXAMPLE_USDZ_UUID]
    assert EXAMPLE_USDZ_SS_PATH in path
    assert repo == ArtifactRepository.SWIFTSTACK
    assert revision is None


def test_deduplicate():
    """Test deduplication by last_modified timestamp."""
    df = pl.DataFrame(
        {
            "clip_id": [
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
    # Check that the newer record was kept for the duplicate clip_id
    dedup_dict = {
        row["clip_id"]: row["uuid"] for row in deduplicated.iter_rows(named=True)
    }
    assert (
        dedup_dict["clipgt-d8cbf4ca-b7ff-44bd-a5be-260f736a02fe"]
        == "fda6ae28-e7f9-493a-b0b6-61e3bbbccbac"
    )


def test_load_and_merge_csvs_single(tmp_path: Path):
    """Single CSV loads normally without duplicate checking."""
    csv_a = tmp_path / "a.csv"
    csv_a.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.1,path/a,2025-01-01 00:00:00,huggingface,\n"
        "uuid-2,clipgt-bbb,0.1,path/b,2025-01-01 00:00:00,huggingface,\n"
    )
    result = _load_and_merge_csvs([str(csv_a)], dedup_key="uuid")
    assert result.height == 2


def test_load_and_merge_csvs_multiple_no_overlap(tmp_path: Path):
    """Multiple CSVs with disjoint rows merge successfully."""
    csv_a = tmp_path / "a.csv"
    csv_a.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.1,path/a,2025-01-01 00:00:00,huggingface,\n"
    )
    csv_b = tmp_path / "b.csv"
    csv_b.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-2,clipgt-bbb,0.1,path/b,2025-01-01 00:00:00,swiftstack,\n"
    )
    result = _load_and_merge_csvs([str(csv_a), str(csv_b)], dedup_key="uuid")
    assert result.height == 2
    assert set(result["uuid"].to_list()) == {"uuid-1", "uuid-2"}


def test_load_and_merge_csvs_preserves_optional_catalog_columns(tmp_path: Path):
    """Catalog-specific metadata columns merge with nulls for older catalogs."""
    csv_a = tmp_path / "a.csv"
    csv_a.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.1,path/a,2025-01-01 00:00:00,huggingface,\n"
    )
    csv_b = tmp_path / "b.csv"
    csv_b.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision,map_type\n"
        "uuid-2,clipgt-bbb,0.1,path/b,2025-01-01 00:00:00,swiftstack,,mads_stitched\n"
    )

    result = _load_and_merge_csvs([str(csv_a), str(csv_b)], dedup_key="uuid")

    assert result.height == 2
    assert "map_type" in result.columns
    assert result.filter(pl.col("uuid") == "uuid-1")["map_type"].to_list() == [None]
    assert result.filter(pl.col("uuid") == "uuid-2")["map_type"].to_list() == [
        "mads_stitched"
    ]


def test_load_and_merge_csvs_duplicate_raises(tmp_path: Path):
    """Duplicate UUIDs across files raise ValueError."""
    csv_a = tmp_path / "a.csv"
    csv_a.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.1,path/a,2025-01-01 00:00:00,huggingface,\n"
    )
    csv_b = tmp_path / "b.csv"
    csv_b.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.2,path/a2,2025-02-01 00:00:00,swiftstack,\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        _load_and_merge_csvs([str(csv_a), str(csv_b)], dedup_key="uuid")


def test_from_cfg_multiple_csvs(tmp_path: Path):
    """USDZManager.from_cfg merges multiple clip catalogs."""
    scenes_a = tmp_path / "scenes_a.csv"
    scenes_a.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-1,clipgt-aaa,0.1,path/a,2025-01-01 00:00:00,huggingface,\n"
    )
    scenes_b = tmp_path / "scenes_b.csv"
    scenes_b.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-2,clipgt-bbb,0.1,path/b,2025-01-01 00:00:00,swiftstack,\n"
    )
    config = ScenesConfig(
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[str(scenes_a), str(scenes_b)],
    )
    manager = USDZManager.from_cfg(config)
    assert manager.clip_catalog.height == 2
    # Can query from both catalogs
    results = manager.query_by_clip_ids(["clipgt-aaa", "clipgt-bbb"])
    assert len(results) == 2


# Tests for local USDZ directory scanning


def _create_mock_usdz(
    path: Path,
    uuid: str,
    clip_id: str,
    version_string: str = "1.0.0-test",
    start_us: int = 100_000_000,
    end_us: int = 120_000_000,
) -> None:
    """Create a mock USDZ file with metadata."""
    metadata = {
        "uuid": uuid,
        # NRE USDZ metadata keeps its external scene_id field.
        "scene_id": clip_id,
        "version_string": version_string,
        "time_range": {"start": start_us, "end": end_us},
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
        clip_id="clipgt-scene-1",
        version_string="1.0.0-test",
    )
    _create_mock_usdz(
        usdz_dir / "scene2.usdz",
        uuid="uuid-2222-bbbb",
        clip_id="clipgt-scene-2",
        version_string="1.0.0-test",
    )
    # Create a subdirectory with another USDZ
    subdir = usdz_dir / "subdir"
    subdir.mkdir()
    _create_mock_usdz(
        subdir / "scene3.usdz",
        uuid="uuid-3333-cccc",
        clip_id="clipgt-scene-3",
        version_string="1.0.0-test",
    )

    return usdz_dir


def test_scan_local_usdz_directory(local_usdz_dir: Path):
    """Test scanning a local directory for USDZ files."""
    clip_catalog = scan_local_usdz_directory(str(local_usdz_dir))

    # Check scenes DataFrame
    assert clip_catalog.height == 3
    assert set(clip_catalog.columns) == {
        "uuid",
        "clip_id",
        "nre_version_string",
        "path",
        "last_modified",
        "artifact_repository",
        "hf_revision",
    }

    uuids = set(clip_catalog["uuid"].to_list())
    assert uuids == {"uuid-1111-aaaa", "uuid-2222-bbbb", "uuid-3333-cccc"}

    clip_ids = set(clip_catalog["clip_id"].to_list())
    assert clip_ids == {"clipgt-scene-1", "clipgt-scene-2", "clipgt-scene-3"}

    # All should be marked as "local" repository
    repos = set(clip_catalog["artifact_repository"].to_list())
    assert repos == {"local"}


def test_scan_local_usdz_directory_does_not_write_scene_database(
    local_usdz_dir: Path,
):
    """Local scene catalogs stay in memory so scene mounts can be read-only."""
    scan_local_usdz_directory(str(local_usdz_dir))

    assert not (local_usdz_dir / "clip_catalog.csv").exists()
    assert not (local_usdz_dir / "sim_suites.csv").exists()


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
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    clip_catalog_csvs.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
    )

    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),  # Not used when local_usdz_dir is set
        clip_catalog_csvs=[
            str(clip_catalog_csvs)
        ],  # Not used when local_usdz_dir is set
    )

    manager = USDZManager.from_cfg(config)

    # Should have loaded scenes from local directory
    assert manager.clip_catalog.height == 3
    assert manager.cache_dir == str(tmp_path)

    results = manager.query_by_clip_ids(manager.clip_catalog["clip_id"].to_list())
    assert len(results) == 3


def test_local_usdz_directory_resolves_all_discovered_scenes(
    local_usdz_dir: Path, tmp_path: Path
):
    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
    )
    manager = USDZManager.from_cfg(config)

    scenes = resolve_scene_selection(config, manager)

    assert {scene.source.source_id for scene in scenes} == {
        "clipgt-scene-1",
        "clipgt-scene-2",
        "clipgt-scene-3",
    }


def test_local_usdz_directory_rejects_scene_suites(
    local_usdz_dir: Path, tmp_path: Path
):
    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[],
        suite_configs=[SceneSuiteConfig(csv_path="unused.csv")],
    )
    manager = USDZManager.from_cfg(config)

    with pytest.raises(ValueError, match="local_usdz_dir cannot be combined"):
        resolve_scene_selection(config, manager)


def test_usdz_manager_query_by_clip_ids_with_local(
    local_usdz_dir: Path, tmp_path: Path
):
    """Test querying by scene IDs when using local USDZ directory."""
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    clip_catalog_csvs.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
    )

    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[str(clip_catalog_csvs)],
    )

    manager = USDZManager.from_cfg(config)

    # Query for a subset of scenes
    results = manager.query_by_clip_ids(
        clip_ids=["clipgt-scene-1", "clipgt-scene-2"],
    )
    assert len(results) == 2
    result_clip_ids = {result.source.source_id for result in results}
    assert result_clip_ids == {"clipgt-scene-1", "clipgt-scene-2"}


def test_get_artifact_info_local(local_usdz_dir: Path, tmp_path: Path):
    """Test getting artifact info for local artifacts."""
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    clip_catalog_csvs.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
    )

    config = ScenesConfig(
        local_usdz_dir=str(local_usdz_dir),
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[str(clip_catalog_csvs)],
    )

    manager = USDZManager.from_cfg(config)

    info = manager.get_artifact_info(["uuid-1111-aaaa"])
    assert "uuid-1111-aaaa" in info
    path, repo, revision = info["uuid-1111-aaaa"]
    assert repo == ArtifactRepository.LOCAL
    assert path.endswith("scene1.usdz")


def test_external_artifact_mounts_ignore_regular_cached_artifacts(
    tmp_path: Path,
) -> None:
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path / "cache"))
    all_usdzs = Path(manager.all_usdzs_dir)
    all_usdzs.mkdir(parents=True)
    (all_usdzs / "uuid-a.usdz").write_bytes(b"artifact")

    assert manager.external_artifact_mounts(["uuid-a"], "/mnt/nre-data") == []


def test_external_artifact_mounts_expose_absolute_symlink_target_directory(
    tmp_path: Path,
) -> None:
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path / "cache"))
    all_usdzs = Path(manager.all_usdzs_dir)
    source = tmp_path / "source"
    all_usdzs.mkdir(parents=True)
    source.mkdir()
    target = source / "uuid-a.usdz"
    target.write_bytes(b"artifact")
    (all_usdzs / "uuid-a.usdz").symlink_to(target)

    assert manager.external_artifact_mounts(["uuid-a"], "/mnt/nre-data") == [
        (str(source), str(source))
    ]


def test_external_artifact_mounts_expose_absolute_target_inside_host_cache(
    tmp_path: Path,
) -> None:
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path / "cache"))
    all_usdzs = Path(manager.all_usdzs_dir)
    source = tmp_path / "cache" / "source"
    all_usdzs.mkdir(parents=True)
    source.mkdir()
    target = source / "uuid-a.usdz"
    target.write_bytes(b"artifact")
    (all_usdzs / "uuid-a.usdz").symlink_to(target)

    assert manager.external_artifact_mounts(["uuid-a"], "/mnt/nre-data") == [
        (str(source), str(source))
    ]


def test_external_artifact_mounts_map_relative_symlink_target_directory(
    tmp_path: Path,
) -> None:
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path / "cache"))
    all_usdzs = Path(manager.all_usdzs_dir)
    source = tmp_path / "source"
    all_usdzs.mkdir(parents=True)
    source.mkdir()
    target = source / "uuid-a.usdz"
    target.write_bytes(b"artifact")
    (all_usdzs / "uuid-a.usdz").symlink_to("../../source/uuid-a.usdz")

    assert manager.external_artifact_mounts(["uuid-a"], "/mnt/nre-data") == [
        (str(source), "/mnt/source")
    ]


def test_external_artifact_mounts_reject_conflicting_container_directories(
    tmp_path: Path,
) -> None:
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path / "cache"))
    all_usdzs = Path(manager.all_usdzs_dir)
    all_usdzs.mkdir(parents=True)
    shared = tmp_path / "shared"
    shared.mkdir()
    for uuid, source_name in (("uuid-a", "source-a"), ("uuid-b", "source-b")):
        source = tmp_path / source_name
        source.mkdir()
        target = source / f"{uuid}.usdz"
        target.write_bytes(b"artifact")
        (shared / f"{uuid}.usdz").symlink_to(target)
        (all_usdzs / f"{uuid}.usdz").symlink_to(shared / f"{uuid}.usdz")

    with pytest.raises(ValueError, match="Conflicting USDZ symlink mounts"):
        manager.external_artifact_mounts(
            ["uuid-a", "uuid-b"],
            "/mnt/nre-data",
        )


@pytest.mark.parametrize("nested_uuid", ["uuid-a", "uuid-b"])
def test_external_artifact_mounts_collapse_compatible_nested_directories(
    tmp_path: Path,
    nested_uuid: str,
) -> None:
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path / "cache"))
    all_usdzs = Path(manager.all_usdzs_dir)
    all_usdzs.mkdir(parents=True)
    source = tmp_path / "external"
    (source / "sub").mkdir(parents=True)
    for uuid in ("uuid-a", "uuid-b"):
        target_dir = source / "sub" if uuid == nested_uuid else source
        target = target_dir / f"{uuid}.usdz"
        target.write_bytes(b"artifact")
        (all_usdzs / f"{uuid}.usdz").symlink_to(target)

    assert manager.external_artifact_mounts(
        ["uuid-a", "uuid-b"],
        "/mnt/nre-data",
    ) == [(str(source), str(source))]


def test_external_artifact_mounts_reject_mismatched_nested_directories(
    tmp_path: Path,
) -> None:
    manager = USDZManager(clip_catalog=MagicMock(), cache_dir=str(tmp_path / "cache"))
    all_usdzs = Path(manager.all_usdzs_dir)
    all_usdzs.mkdir(parents=True)
    mount_layout = tmp_path / "mount-layout"
    (mount_layout / "sub").mkdir(parents=True)
    for uuid, link_dir, source_name in (
        ("uuid-a", mount_layout, "source-a"),
        ("uuid-b", mount_layout / "sub", "source-b"),
    ):
        source = tmp_path / source_name
        source.mkdir()
        target = source / f"{uuid}.usdz"
        target.write_bytes(b"artifact")
        (link_dir / f"{uuid}.usdz").symlink_to(target)
        relative_link = Path("../..") / link_dir.relative_to(tmp_path) / target.name
        (all_usdzs / f"{uuid}.usdz").symlink_to(relative_link)

    with pytest.raises(ValueError, match="Conflicting overlapping USDZ symlink mounts"):
        manager.external_artifact_mounts(
            ["uuid-a", "uuid-b"],
            "/mnt/nre-data",
        )


def test_get_artifact_info_includes_hf_revision(tmp_path: Path):
    """Test that get_artifact_info returns hf_revision for HuggingFace artifacts."""
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    clip_catalog_csvs.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-hf,clipgt-hf-scene,1.0,sample_set/26.01/file.usdz,2025-01-01 00:00:00,huggingface,26.01\n"
    )
    config = ScenesConfig(
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[str(clip_catalog_csvs)],
    )
    manager = USDZManager.from_cfg(config)
    info = manager.get_artifact_info(["uuid-hf"])
    path, repo, revision = info["uuid-hf"]
    assert repo == ArtifactRepository.HUGGINGFACE
    assert revision == "26.01"


def test_timing_metadata_range_read_creates_only_metadata_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "clip_catalog.csv"
    catalog.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-a,clipgt-a,1.0,bucket/path/a.usdz,2025-01-01 00:00:00,swiftstack,\n"
    )
    manager = USDZManager.from_cfg(
        ScenesConfig(scene_cache=str(tmp_path), clip_catalog_csvs=[str(catalog)])
    )
    raw_metadata = yaml.safe_dump(
        {
            "uuid": "uuid-a",
            "scene_id": "clipgt-a",
            "time_range": {"start": 100_000_000, "end": 119_733_638},
        }
    ).encode()
    range_read = MagicMock(return_value=raw_metadata)
    monkeypatch.setattr(manager, "_read_metadata_from_swiftstack", range_read)

    metadata = manager.load_timing_metadata(
        [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
    )

    assert metadata["uuid-a"].duration_us == 19_733_638
    range_read.assert_called_once_with("bucket/path/a.usdz")
    assert (tmp_path / "metadata-yaml/uuid-a.yaml").read_bytes() == raw_metadata
    assert not (tmp_path / "all-usdzs/uuid-a.usdz").exists()

    range_read.reset_mock(side_effect=AssertionError("unexpected range read"))
    assert (
        manager.load_timing_metadata(
            [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
        )["uuid-a"].duration_us
        == 19_733_638
    )
    range_read.assert_not_called()


def test_timing_metadata_prefers_existing_full_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "clip_catalog.csv"
    catalog.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-a,clipgt-a,1.0,bucket/path/a.usdz,2025-01-01 00:00:00,swiftstack,\n"
    )
    manager = USDZManager.from_cfg(
        ScenesConfig(scene_cache=str(tmp_path), clip_catalog_csvs=[str(catalog)])
    )
    artifact_path = tmp_path / "all-usdzs/uuid-a.usdz"
    _create_mock_usdz(
        artifact_path,
        uuid="uuid-a",
        clip_id="clipgt-a",
        start_us=100,
        end_us=300,
    )
    range_read = MagicMock(side_effect=AssertionError("unexpected range read"))
    monkeypatch.setattr(manager, "_read_metadata_from_swiftstack", range_read)

    metadata = manager.load_timing_metadata(
        [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
    )

    assert metadata["uuid-a"].duration_us == 200
    range_read.assert_not_called()


def test_timing_metadata_rejects_catalog_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "clip_catalog.csv"
    catalog.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-a,clipgt-a,1.0,bucket/path/a.usdz,2025-01-01 00:00:00,swiftstack,\n"
    )
    manager = USDZManager.from_cfg(
        ScenesConfig(scene_cache=str(tmp_path), clip_catalog_csvs=[str(catalog)])
    )
    monkeypatch.setattr(
        manager,
        "_read_metadata_from_swiftstack",
        MagicMock(
            return_value=yaml.safe_dump(
                {
                    "uuid": "uuid-a",
                    "scene_id": "clipgt-other",
                    "time_range": {"start": 100, "end": 300},
                }
            ).encode()
        ),
    )

    with pytest.raises(ValueError, match="expected 'clipgt-a'"):
        manager.load_timing_metadata(
            [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
        )

    assert not (tmp_path / "metadata-yaml/uuid-a.yaml").exists()


def test_timing_metadata_rejects_zero_length_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "clip_catalog.csv"
    catalog.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-a,clipgt-a,1.0,bucket/path/a.usdz,2025-01-01 00:00:00,swiftstack,\n"
    )
    manager = USDZManager.from_cfg(
        ScenesConfig(scene_cache=str(tmp_path), clip_catalog_csvs=[str(catalog)])
    )
    monkeypatch.setattr(
        manager,
        "_read_metadata_from_swiftstack",
        MagicMock(
            return_value=yaml.safe_dump(
                {
                    "uuid": "uuid-a",
                    "scene_id": "clipgt-a",
                    "time_range": {"start": 100, "end": 100},
                }
            ).encode()
        ),
    )

    with pytest.raises(ValueError, match="invalid time range"):
        manager.load_timing_metadata(
            [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
        )

    assert not (tmp_path / "metadata-yaml/uuid-a.yaml").exists()


@pytest.mark.parametrize(
    ("start_us", "end_us"),
    [
        (True, 300),
        (100, 300.5),
    ],
)
def test_timing_metadata_rejects_non_integer_range_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_us: object,
    end_us: object,
) -> None:
    catalog = tmp_path / "clip_catalog.csv"
    catalog.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-a,clipgt-a,1.0,bucket/path/a.usdz,2025-01-01 00:00:00,swiftstack,\n"
    )
    manager = USDZManager.from_cfg(
        ScenesConfig(scene_cache=str(tmp_path), clip_catalog_csvs=[str(catalog)])
    )
    monkeypatch.setattr(
        manager,
        "_read_metadata_from_swiftstack",
        MagicMock(
            return_value=yaml.safe_dump(
                {
                    "uuid": "uuid-a",
                    "scene_id": "clipgt-a",
                    "time_range": {"start": start_us, "end": end_us},
                }
            ).encode()
        ),
    )

    with pytest.raises(ValueError, match=r"time_range\.(start|end) is not an integer"):
        manager.load_timing_metadata(
            [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
        )

    assert not (tmp_path / "metadata-yaml/uuid-a.yaml").exists()


@pytest.mark.parametrize("cache_metadata", [False, True])
@pytest.mark.parametrize(
    ("repository", "revision", "error"),
    [
        ("unsupported", "", "Invalid artifact_repository 'unsupported'"),
        ("huggingface", "", "is missing hf_revision"),
    ],
)
def test_timing_metadata_rejects_invalid_backend_before_cache_or_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_metadata: bool,
    repository: str,
    revision: str,
    error: str,
) -> None:
    catalog = tmp_path / "clip_catalog.csv"
    catalog.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        f"uuid-a,clipgt-a,1.0,path/a.usdz,2025-01-01 00:00:00,{repository},{revision}\n"
    )
    manager = USDZManager.from_cfg(
        ScenesConfig(scene_cache=str(tmp_path), clip_catalog_csvs=[str(catalog)])
    )
    if cache_metadata:
        metadata_dir = tmp_path / "metadata-yaml"
        metadata_dir.mkdir()
        (metadata_dir / "uuid-a.yaml").write_text(
            yaml.safe_dump(
                {
                    "uuid": "uuid-a",
                    "scene_id": "clipgt-a",
                    "time_range": {"start": 100, "end": 300},
                }
            )
        )
    backend_read = MagicMock(side_effect=AssertionError("unexpected backend read"))
    backend_download = MagicMock(
        side_effect=AssertionError("unexpected backend download")
    )
    monkeypatch.setattr(manager, "_read_metadata_from_swiftstack", backend_read)
    monkeypatch.setattr(manager, "_download_huggingface_artifacts", backend_download)

    with pytest.raises(ValueError, match=error):
        manager.load_timing_metadata(
            [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
        )

    backend_read.assert_not_called()
    backend_download.assert_not_called()


def test_timing_metadata_requires_backend_column_before_cache(tmp_path: Path) -> None:
    catalog = tmp_path / "clip_catalog.csv"
    catalog.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified\n"
        "uuid-a,clipgt-a,1.0,bucket/path/a.usdz,2025-01-01 00:00:00\n"
    )
    manager = USDZManager.from_cfg(
        ScenesConfig(scene_cache=str(tmp_path), clip_catalog_csvs=[str(catalog)])
    )
    metadata_dir = tmp_path / "metadata-yaml"
    metadata_dir.mkdir()
    (metadata_dir / "uuid-a.yaml").write_text(
        yaml.safe_dump(
            {
                "uuid": "uuid-a",
                "scene_id": "clipgt-a",
                "time_range": {"start": 100, "end": 300},
            }
        )
    )

    with pytest.raises(ValueError, match="missing the required artifact_repository"):
        manager.load_timing_metadata(
            [ClipDataSource(clip_id="clipgt-a", usdz_uuid="uuid-a")]
        )


def test_query_by_clip_ids_duplicate_nre_versions_warns(tmp_path: Path, caplog):
    """Warning when a scene has artifacts for multiple NRE versions; newest is picked."""
    clip_catalog_csvs = tmp_path / "clip_catalog.csv"
    clip_catalog_csvs.write_text(
        "uuid,clip_id,nre_version_string,path,last_modified,artifact_repository,hf_revision\n"
        "uuid-aaa,clipgt-scene-1,1.0.0-aaaa,path/a,2025-01-01 00:00:00,swiftstack,\n"
        "uuid-bbb,clipgt-scene-1,2.0.0-bbbb,path/b,2025-01-02 00:00:00,swiftstack,\n"
    )
    config = ScenesConfig(
        scene_cache=str(tmp_path),
        clip_catalog_csvs=[str(clip_catalog_csvs)],
    )
    manager = USDZManager.from_cfg(config)

    with caplog.at_level(logging.WARNING, logger="alpasim_wizard"):
        results = manager.query_by_clip_ids(clip_ids=["clipgt-scene-1"])

    assert len(results) == 1
    assert isinstance(results[0].source, ClipDataSource)
    assert results[0].source.usdz_uuid == "uuid-bbb"  # newest by last_modified
    assert "clipgt-scene-1" in caplog.text
    assert "multiple NRE versions" in caplog.text
