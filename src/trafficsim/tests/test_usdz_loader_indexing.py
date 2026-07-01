# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation


import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from alpasim_runtime.scene_loader import ArtifactSceneProvider, SceneLoader
from alpasim_trafficsim.catk.map_adapter import build_env_map_from_vector_map
from alpasim_trafficsim.catk.scene_adapter import CATKSceneAdapter
from alpasim_utils.artifact import Artifact
from trajdata.maps.vec_map_elements import MapElementType


def _metadata_yaml(scene_id: str) -> str:
    return yaml.safe_dump(
        {
            "scene_id": scene_id,
            "version_string": "test",
            "training_date": "2026-01-01",
            "dataset_hash": "test-hash",
            "uuid": "test-uuid",
            "is_resumable": False,
            "sensors": {"camera_ids": [], "lidar_ids": []},
            "logger": {},
            "time_range": {"start": 0.0, "end": 1.0},
        }
    )


def test_runtime_scene_loader_indexes_nested_usdz_files(tmp_path) -> None:
    nested_dir = tmp_path / "all-usdzs"
    nested_dir.mkdir()
    usdz_path = nested_dir / "scene.usdz"
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.writestr("metadata.yaml", _metadata_yaml("clipgt-test-scene"))

    scene_loader = SceneLoader(
        ArtifactSceneProvider.from_path(
            tmp_path,
            smooth_trajectories=False,
        )
    )

    assert "clipgt-test-scene" in scene_loader.scene_ids


def test_vector_map_adapter_resamples_before_catk_segmentation() -> None:
    vector_map = SimpleNamespace(
        elements={
            MapElementType.ROAD_LANE: {
                "lane": SimpleNamespace(
                    center=SimpleNamespace(
                        xyz=np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [10.0, 0.0, 0.0],
                            ],
                            dtype=np.float32,
                        )
                    ),
                    left_edge=None,
                    right_edge=None,
                )
            }
        }
    )

    without_resampling = build_env_map_from_vector_map(
        vector_map,
        ego_xyz=torch.tensor([0.0, 0.0, 0.0]),
        ego_heading=0.0,
        distance_x=0.0,
        distance_y=0.0,
        map_polyline_length_k=1,
        map_resample_interval_m=None,
    )
    with_resampling = build_env_map_from_vector_map(
        vector_map,
        ego_xyz=torch.tensor([0.0, 0.0, 0.0]),
        ego_heading=0.0,
        distance_x=0.0,
        distance_y=0.0,
        map_polyline_length_k=1,
        map_resample_interval_m=1.0,
    )

    assert without_resampling["lane_centers"] is None
    assert with_resampling["lane_centers"]["polylines"].shape == (5, 3, 3)
    torch.testing.assert_close(
        with_resampling["lane_centers"]["polylines"][0, :, 0],
        torch.tensor([0.0, 1.0, 2.0]),
    )


@pytest.mark.integration
def test_catk_scene_adapter_runtime_path_reads_sample_artifact_with_vector_map() -> (
    None
):
    sample = (
        Path(__file__).resolve().parents[3]
        / "data/nre-artifacts/all-usdzs/5001cd19-e936-40d7-a42c-fa1fbf2bb2ba.usdz"
    )
    if not sample.is_file():
        pytest.skip(f"Sample USDZ not found: {sample}")

    env_data = CATKSceneAdapter(motion_stepsize=0.1).load(
        Artifact(str(sample), _smooth_trajectories=False)
    )

    assert type(env_data) is dict
    assert env_data["metadata"]["map_source"] == "trajdata_vector_map"
    assert env_data["agents"]["num_obstacles"] > 0
    for layer_name in ["lanelines", "road_boundaries", "waitlines", "lane_boundaries"]:
        layer = env_data["map"][layer_name]
        assert layer is not None
        assert layer["polylines"].shape[1:] == (3, 3)
        assert torch.isfinite(layer["polylines"]).all()
