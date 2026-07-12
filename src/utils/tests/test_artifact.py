# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from alpasim_utils import artifact as artifact_module
from alpasim_utils.artifact import Artifact
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import Polyline, RoadLane


def _artifact_with_metadata(path) -> Artifact:
    artifact = Artifact(source=str(path))
    artifact._metadata = SimpleNamespace(scene_id="clipgt-test")
    return artifact


def _write_map_directory(archive: zipfile.ZipFile, directory: str) -> None:
    archive.writestr(f"{directory}/lane.parquet", b"")


def _add_test_lane(vector_map: VectorMap) -> None:
    vector_map.add_map_element(
        RoadLane(
            id="lane_0",
            center=Polyline(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        )
    )


def test_artifact_no_map():
    usdz_file = "tests/data/no_map_artifact/artifact_no_map.usdz"
    artifact = Artifact(source=usdz_file)

    # expect that the map is None (no exceptions)
    assert artifact.map is None
    assert artifact.map_source is None


def test_xodr_artifact():
    usdz_file = "tests/data/xodr_artifact/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz"
    artifact = Artifact(source=usdz_file)

    # expect that the map is not None (no exceptions)
    assert artifact.map is not None
    assert artifact.map_source == "xodr"
    assert (
        artifact.map.map_id
        == "alpasim_usdz:clipgt-026d6a39-bd8f-4175-bc61-fe50ed0403a3"
    )


def test_loads_fastmap_directory(tmp_path, monkeypatch) -> None:
    path = tmp_path / "artifact.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        _write_map_directory(archive, "fastmap")

    loaded_roots = []

    def populate(vector_map, map_root):
        loaded_roots.append(map_root)
        _add_test_lane(vector_map)

    monkeypatch.setattr(artifact_module, "populate_vector_map", populate)
    artifact = _artifact_with_metadata(path)

    with zipfile.ZipFile(path) as archive:
        vector_map = artifact._load_parquet_map(archive, "fastmap")

    assert vector_map is not None
    assert vector_map.map_id == "alpasim_usdz:clipgt-test"
    assert len(loaded_roots) == 1
    assert loaded_roots[0].endswith("/fastmap")


def test_fastmap_precedes_xodr_and_snaps_to_ground(tmp_path, monkeypatch) -> None:
    path = tmp_path / "artifact.usdz"
    with zipfile.ZipFile(path, "w"):
        pass
    artifact = _artifact_with_metadata(path)
    calls = []

    def load_parquet(_archive, directory):
        calls.append(directory)
        if directory == "fastmap":
            return VectorMap(map_id="alpasim_usdz:clipgt-test")
        return None

    monkeypatch.setattr(artifact, "_load_parquet_map", load_parquet)
    monkeypatch.setattr(
        artifact,
        "_load_xodr_map",
        lambda _archive: pytest.fail("XODR should not be loaded"),
    )
    monkeypatch.setattr(artifact, "_snap_map_z", lambda: calls.append("snap"))
    monkeypatch.setattr(artifact, "_finalize_map", lambda: None)

    assert artifact.map is not None
    assert calls == ["map_data", "clipgt", "fastmap", "snap"]
    assert artifact.map_source == "fastmap"


def test_complete_map_data_precedes_fastmap(tmp_path, monkeypatch) -> None:
    path = tmp_path / "artifact.usdz"
    with zipfile.ZipFile(path, "w"):
        pass
    artifact = _artifact_with_metadata(path)
    calls = []

    def load_parquet(_archive, directory):
        calls.append(directory)
        return VectorMap(map_id="alpasim_usdz:clipgt-test")

    monkeypatch.setattr(artifact, "_load_parquet_map", load_parquet)
    monkeypatch.setattr(
        artifact,
        "_snap_map_z",
        lambda: pytest.fail("preferred map must not be ground-snapped"),
    )
    monkeypatch.setattr(artifact, "_finalize_map", lambda: None)

    assert artifact.map is not None
    assert calls == ["map_data"]
    assert artifact.map_source == "map_data"


def test_failed_map_source_does_not_leak_partial_state(tmp_path, monkeypatch) -> None:
    path = tmp_path / "artifact.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        _write_map_directory(archive, "map_data")
        _write_map_directory(archive, "clipgt")
    artifact = _artifact_with_metadata(path)

    def populate(vector_map, map_root):
        if map_root.endswith("/map_data"):
            vector_map.failed_source_marker = True
            raise ValueError("broken map")
        _add_test_lane(vector_map)

    monkeypatch.setattr(artifact_module, "populate_vector_map", populate)
    monkeypatch.setattr(artifact, "_finalize_map", lambda: None)

    vector_map = artifact.map

    assert vector_map is not None
    assert not hasattr(vector_map, "failed_source_marker")
    assert artifact.map_source == "clipgt"


def test_empty_map_source_falls_back_to_fastmap(tmp_path, monkeypatch) -> None:
    path = tmp_path / "artifact.usdz"
    with zipfile.ZipFile(path, "w") as archive:
        _write_map_directory(archive, "map_data")
        _write_map_directory(archive, "fastmap")
    artifact = _artifact_with_metadata(path)
    calls = []

    def populate(vector_map, map_root):
        calls.append(map_root.rsplit("/", 1)[-1])
        if map_root.endswith("/fastmap"):
            _add_test_lane(vector_map)

    monkeypatch.setattr(artifact_module, "populate_vector_map", populate)
    monkeypatch.setattr(artifact, "_snap_map_z", lambda: calls.append("snap"))
    monkeypatch.setattr(artifact, "_finalize_map", lambda: None)

    assert artifact.map is not None
    assert calls == ["map_data", "fastmap", "snap"]
    assert artifact.map_source == "fastmap"


def test_fastmap_requires_ground_mesh(tmp_path) -> None:
    path = tmp_path / "artifact.usdz"
    with zipfile.ZipFile(path, "w"):
        pass
    artifact = _artifact_with_metadata(path)
    artifact._map = VectorMap(map_id="alpasim_usdz:clipgt-test")

    with pytest.raises(KeyError, match=r"mesh_ground\.ply"):
        artifact._snap_map_z()
