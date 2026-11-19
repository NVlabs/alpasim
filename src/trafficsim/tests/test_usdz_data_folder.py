# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Load USDZ archives from the repository ``data/`` folder."""

from pathlib import Path

import pytest
from alpasim_runtime.data_source_loader import (
    ArtifactDataSourceProvider,
    DataSourceLoader,
)
from alpasim_trafficsim.catk.scene_adapter import CATKSceneAdapter


def test_data_dir_exists(usdz_data_dir: Path) -> None:
    assert (
        usdz_data_dir.is_dir()
    ), f"Expected {usdz_data_dir} to exist (create it for local USDZ files)."


def test_runtime_data_source_loader_discovers_scenes(usdz_data_dir: Path) -> None:
    data_source_loader = DataSourceLoader(
        ArtifactDataSourceProvider.from_path(
            usdz_data_dir,
            smooth_trajectories=False,
        )
    )
    assert len(data_source_loader.source_ids) >= 1


@pytest.mark.integration
def test_catk_scene_adapter_loads_training_style_data(
    usdz_data_dir: Path, usdz_from_data_dir: Path
) -> None:
    del usdz_data_dir
    data_source_loader = DataSourceLoader(
        ArtifactDataSourceProvider.from_path(
            usdz_from_data_dir,
            smooth_trajectories=False,
        )
    )
    clip_id = next(iter(data_source_loader.source_ids))
    data_source = data_source_loader.get_data_source(clip_id)
    env_data = CATKSceneAdapter().load(data_source)

    assert env_data["map"], "Expected at least one decoded map layer"
    assert env_data["env"]["curr_t"] == 15
    ego_steps = env_data["ego"]["xyz"].shape[0]
    assert env_data["ego"]["xyz"].ndim == 2
    assert env_data["ego"]["xyz"].shape[1] == 3
    assert ego_steps > env_data["env"]["curr_t"]
    assert env_data["ego"]["heading"].shape == (ego_steps,)
    assert env_data["ego"]["lwh"].shape == (3,)

    agents = env_data["agents"]
    assert agents["valid_mask"].ndim == 2
    assert agents["xyz"].ndim == 3
    assert agents["heading"].ndim == 2
    assert agents["valid_mask"].shape == agents["heading"].shape
    assert agents["xyz"].shape[:2] == agents["valid_mask"].shape
    assert agents["lwh"].shape[0] == agents["valid_mask"].shape[0]
