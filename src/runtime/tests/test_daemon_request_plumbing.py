# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_runtime.daemon.engine import (
    UnknownSceneError,
    build_pending_jobs_from_request,
)


def test_adapter_expands_nr_rollouts() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=3)]
    )

    jobs = build_pending_jobs_from_request(
        req,
        scene_id_to_artifact_path={"clipgt-a": "/tmp/clipgt-a.usdz"},
    )
    assert [job.scene_id for job in jobs] == ["clipgt-a", "clipgt-a", "clipgt-a"]
    assert [job.rollout_spec_index for job in jobs] == [0, 0, 0]
    assert all(job.artifact_path == "/tmp/clipgt-a.usdz" for job in jobs)


def test_adapter_drops_zero_nr_rollouts_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="alpasim_runtime.daemon.engine")
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[runtime_pb2.RolloutSpec(scenario_id="clipgt-a")]
    )

    jobs = build_pending_jobs_from_request(
        req,
        scene_id_to_artifact_path={"clipgt-a": "/tmp/clipgt-a.usdz"},
    )
    assert jobs == []
    assert "Dropping rollout spec with nr_rollouts=0" in caplog.text


def test_adapter_rejects_scene_without_artifact() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-missing", nr_rollouts=1)
        ]
    )

    with pytest.raises(UnknownSceneError):
        build_pending_jobs_from_request(
            req,
            scene_id_to_artifact_path={"clipgt-a": "/tmp/clipgt-a.usdz"},
        )


def test_adapter_assigns_rollout_spec_indexes_in_request_order() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1),
            runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=2),
        ]
    )

    jobs = build_pending_jobs_from_request(
        req,
        scene_id_to_artifact_path={
            "clipgt-a": "/tmp/clipgt-a.usdz",
            "clipgt-b": "/tmp/clipgt-b.usdz",
        },
    )
    assert len(jobs) == 3
    assert [job.scene_id for job in jobs] == ["clipgt-a", "clipgt-b", "clipgt-b"]
    assert [job.rollout_spec_index for job in jobs] == [0, 1, 1]
    assert [job.artifact_path for job in jobs] == [
        "/tmp/clipgt-a.usdz",
        "/tmp/clipgt-b.usdz",
        "/tmp/clipgt-b.usdz",
    ]


def test_adapter_ignores_zero_rollout_specs_when_indexing() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=0),
            runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=2),
        ]
    )

    jobs = build_pending_jobs_from_request(
        req,
        scene_id_to_artifact_path={
            "clipgt-a": "/tmp/clipgt-a.usdz",
            "clipgt-b": "/tmp/clipgt-b.usdz",
        },
    )
    assert len(jobs) == 2
    assert [job.scene_id for job in jobs] == ["clipgt-b", "clipgt-b"]
    assert [job.rollout_spec_index for job in jobs] == [1, 1]
