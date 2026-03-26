# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_runtime.daemon.engine import build_pending_jobs_from_request
from alpasim_runtime.daemon.exceptions import UnknownSceneError


def test_adapter_expands_nr_rollouts() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=3)]
    )

    mock_data_source = MagicMock()

    def fake_get_data_source(scene_id: str):
        if scene_id == "clipgt-a":
            return mock_data_source
        raise UnknownSceneError(scene_id)

    jobs = build_pending_jobs_from_request(req, fake_get_data_source)
    assert [job.scene_id for job in jobs] == ["clipgt-a", "clipgt-a", "clipgt-a"]
    assert [job.rollout_spec_index for job in jobs] == [0, 0, 0]
    assert all(job.data_source is mock_data_source for job in jobs)


def test_adapter_drops_zero_nr_rollouts_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="alpasim_runtime.daemon.engine")
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="clipgt-a")]
    )

    mock_data_source = MagicMock()

    def fake_get_data_source(scene_id: str):
        return mock_data_source

    jobs = build_pending_jobs_from_request(req, fake_get_data_source)
    assert jobs == []
    assert "Dropping rollout spec with nr_rollouts=0" in caplog.text


def test_adapter_rejects_scene_without_artifact() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-missing", nr_rollouts=1)
        ]
    )

    def fake_get_data_source(scene_id: str):
        if scene_id == "clipgt-missing":
            raise UnknownSceneError(scene_id)
        return MagicMock()

    with pytest.raises(UnknownSceneError):
        build_pending_jobs_from_request(req, fake_get_data_source)


def test_adapter_assigns_rollout_spec_indexes_in_request_order() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1),
            runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=2),
        ]
    )

    mock_data_source_a = MagicMock()
    mock_data_source_b = MagicMock()

    def fake_get_data_source(scene_id: str):
        if scene_id == "clipgt-a":
            return mock_data_source_a
        elif scene_id == "clipgt-b":
            return mock_data_source_b
        raise UnknownSceneError(scene_id)

    jobs = build_pending_jobs_from_request(req, fake_get_data_source)
    assert len(jobs) == 3
    assert [job.scene_id for job in jobs] == ["clipgt-a", "clipgt-b", "clipgt-b"]
    assert [job.rollout_spec_index for job in jobs] == [0, 1, 1]
    assert [job.data_source for job in jobs] == [
        mock_data_source_a,
        mock_data_source_b,
        mock_data_source_b,
    ]


def test_adapter_ignores_zero_rollout_specs_when_indexing() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=0),
            runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=2),
        ]
    )

    mock_data_source_a = MagicMock()
    mock_data_source_b = MagicMock()

    def fake_get_data_source(scene_id: str):
        if scene_id == "clipgt-a":
            return mock_data_source_a
        elif scene_id == "clipgt-b":
            return mock_data_source_b
        raise UnknownSceneError(scene_id)

    jobs = build_pending_jobs_from_request(req, fake_get_data_source)
    assert len(jobs) == 2
    assert [job.scene_id for job in jobs] == ["clipgt-b", "clipgt-b"]
    assert [job.rollout_spec_index for job in jobs] == [1, 1]
