# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Integration tests for trajdata data source functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_runtime.daemon.engine import (
    DaemonEngine,
    UnknownSceneError,
    build_pending_jobs_from_request,
)


@pytest.fixture
def mock_trajdata_scene():
    """Create a mock trajdata Scene object."""
    scene = MagicMock()
    scene.name = "test_scene_001"
    scene.length_timesteps = 100
    scene.dt = 0.1
    return scene


@pytest.fixture
def mock_trajdata_dataset(mock_trajdata_scene):
    """Create a mock UnifiedDataset."""
    dataset = MagicMock()
    dataset.num_scenes.return_value = 1
    dataset.get_scene.return_value = mock_trajdata_scene
    dataset.cache_class = MagicMock
    dataset.cache_path = "/tmp/cache"
    dataset.augmentations = None
    dataset.obs_format = MagicMock()
    return dataset


def test_build_pending_jobs_with_valid_scene():
    """Test building pending jobs from a simulation request with valid scene IDs."""
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="scene_a", nr_rollouts=2),
            runtime_pb2.RolloutSpec(scenario_id="scene_b", nr_rollouts=1),
        ]
    )

    mock_data_source_a = MagicMock()
    mock_data_source_b = MagicMock()

    def fake_get_data_source(scene_id: str):
        if scene_id == "scene_a":
            return mock_data_source_a
        elif scene_id == "scene_b":
            return mock_data_source_b
        raise UnknownSceneError(scene_id)

    jobs = build_pending_jobs_from_request(request, fake_get_data_source)

    # Should create 3 jobs total: 2 for scene_a, 1 for scene_b
    assert len(jobs) == 3

    # Check first two jobs are for scene_a
    assert jobs[0].scene_id == "scene_a"
    assert jobs[0].rollout_spec_index == 0
    assert jobs[0].data_source is mock_data_source_a
    assert jobs[1].scene_id == "scene_a"
    assert jobs[1].rollout_spec_index == 0
    assert jobs[1].data_source is mock_data_source_a

    # Check third job is for scene_b
    assert jobs[2].scene_id == "scene_b"
    assert jobs[2].rollout_spec_index == 1
    assert jobs[2].data_source is mock_data_source_b


def test_build_pending_jobs_with_unknown_scene():
    """Test that UnknownSceneError is raised for unknown scene IDs."""
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="unknown_scene", nr_rollouts=1),
        ]
    )

    def fake_get_data_source(scene_id: str):
        raise UnknownSceneError(scene_id)

    with pytest.raises(UnknownSceneError) as exc_info:
        build_pending_jobs_from_request(request, fake_get_data_source)

    assert exc_info.value.scene_id == "unknown_scene"


def test_build_pending_jobs_drops_zero_rollouts():
    """Test that specs with nr_rollouts=0 are dropped with a warning."""
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="scene_a", nr_rollouts=1),
            runtime_pb2.RolloutSpec(scenario_id="scene_b", nr_rollouts=0),
        ]
    )

    mock_data_source = MagicMock()

    def fake_get_data_source(scene_id: str):
        return mock_data_source

    jobs = build_pending_jobs_from_request(request, fake_get_data_source)

    # Should only create 1 job (scene_b with 0 rollouts is dropped)
    assert len(jobs) == 1
    assert jobs[0].scene_id == "scene_a"


def test_daemon_engine_get_data_source_caching(
    mock_trajdata_dataset, mock_trajdata_scene
):
    """Test that DaemonEngine uses SceneLoader for caching data sources."""
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )

    # Set up engine with mock SceneLoader
    engine._started = True
    mock_scene_loader = MagicMock()
    mock_data_source = MagicMock()
    mock_scene_loader.get_data_source.return_value = mock_data_source
    engine._scene_loader = mock_scene_loader

    # First call should delegate to SceneLoader
    data_source_1 = engine._get_data_source("test_scene_001")
    assert data_source_1 is mock_data_source
    assert mock_scene_loader.get_data_source.call_count == 1

    # Second call should also delegate (SceneLoader handles caching internally)
    data_source_2 = engine._get_data_source("test_scene_001")
    assert data_source_2 is mock_data_source
    assert mock_scene_loader.get_data_source.call_count == 2


def test_daemon_engine_get_data_source_unknown_scene():
    """Test that _get_data_source raises UnknownSceneError for unknown scenes."""
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )

    engine._started = True
    # Mock SceneLoader that raises UnknownSceneError
    mock_scene_loader = MagicMock()
    mock_scene_loader.get_data_source.side_effect = UnknownSceneError("unknown_scene")
    engine._scene_loader = mock_scene_loader

    with pytest.raises(UnknownSceneError) as exc_info:
        engine._get_data_source("unknown_scene")

    assert exc_info.value.scene_id == "unknown_scene"


def test_daemon_engine_get_data_source_without_dataset():
    """Test that _get_data_source raises RuntimeError if SceneLoader is not initialized."""
    engine = DaemonEngine(
        user_config="u.yaml",
        network_config="n.yaml",
        eval_config="e.yaml",
        log_dir="/tmp/log",
    )

    engine._started = True
    engine._scene_loader = None

    with pytest.raises(RuntimeError, match="SceneLoader not initialized"):
        engine._get_data_source("test_scene")
