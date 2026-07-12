# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_runtime.daemon.engine import build_pending_jobs_from_request
from alpasim_runtime.errors import UnknownSceneError


def test_adapter_rejects_scene_without_artifact() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-missing", nr_rollouts=1)
        ]
    )

    with pytest.raises(UnknownSceneError):
        build_pending_jobs_from_request(req, "req-1", lambda _scene_id: False)


def test_adapter_assigns_rollout_spec_indexes_in_request_order() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=1),
            runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=2),
        ]
    )

    jobs = build_pending_jobs_from_request(
        req, "req-1", lambda scene_id: scene_id in {"clipgt-a", "clipgt-b"}
    )
    assert len(jobs) == 3
    assert [job.scene_id for job in jobs] == ["clipgt-a", "clipgt-b", "clipgt-b"]
    assert [job.rollout_spec_index for job in jobs] == [0, 1, 1]


def test_adapter_ignores_zero_rollout_specs_when_indexing() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(scenario_id="clipgt-a", nr_rollouts=0),
            runtime_pb2.RolloutSpec(scenario_id="clipgt-b", nr_rollouts=2),
        ]
    )

    jobs = build_pending_jobs_from_request(
        req, "req-1", lambda scene_id: scene_id in {"clipgt-a", "clipgt-b"}
    )
    assert len(jobs) == 2
    assert [job.scene_id for job in jobs] == ["clipgt-b", "clipgt-b"]
    assert [job.rollout_spec_index for job in jobs] == [1, 1]


def test_adapter_propagates_session_uuids_in_order() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(
                scenario_id="clipgt-a",
                nr_rollouts=3,
                session_uuids=["u0", "u1", "u2"],
            )
        ]
    )

    jobs = build_pending_jobs_from_request(req, "req-1", lambda _scene_id: True)
    assert [job.session_uuid for job in jobs] == ["u0", "u1", "u2"]


def test_adapter_rejects_mismatched_session_uuids_length() -> None:
    req = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(
                scenario_id="clipgt-a",
                nr_rollouts=2,
                session_uuids=["u0"],  # length mismatch
            )
        ]
    )

    with pytest.raises(ValueError, match="session_uuids"):
        build_pending_jobs_from_request(req, "req-1", lambda _scene_id: True)
