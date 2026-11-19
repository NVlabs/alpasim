# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from typing import Literal

import pytest
from alpasim_grpc.v0 import runtime_pb2
from alpasim_grpc.v0.common_pb2 import PoseAtTime, Trajectory
from alpasim_runtime.daemon.engine import build_pending_jobs_from_request
from alpasim_runtime.errors import UnknownSceneError
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene, scene_to_proto


def _scene(
    clip_id: str,
    policy_start_us: int | Literal["AUTO"] = T0_AUTO,
    *,
    usdz_uuid: str | None = None,
) -> Scene:
    return Scene(
        ClipDataSource(clip_id, usdz_uuid or f"uuid-for-{clip_id}"),
        policy_start_us,
    )


def _spec(
    clip_id: str,
    nr_rollouts: int,
    policy_start_us: int | Literal["AUTO"] = T0_AUTO,
    *,
    session_uuids: list[str] | None = None,
    usdz_uuid: str | None = None,
    random_seed: int = 0,
) -> runtime_pb2.RolloutSpec:
    return runtime_pb2.RolloutSpec(
        scene=scene_to_proto(_scene(clip_id, policy_start_us, usdz_uuid=usdz_uuid)),
        nr_rollouts=nr_rollouts,
        session_uuids=session_uuids or [],
        random_seed=random_seed,
    )


def _resolve(requested: Scene) -> Scene:
    if requested.source.source_id == "clipgt-unknown-scene":
        raise KeyError(requested.scene_key)
    configured = _scene(requested.source.source_id, requested.policy_start_us)
    if requested.source != configured.source:
        raise ValueError("configured descriptor mismatch")
    return configured


def _resolve_fork(requested: Scene) -> Scene:
    assert requested.policy_start_us == 2_000_000
    return requested


def test_adapter_rejects_unconfigured_scene_address() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[_spec("clipgt-unknown-scene", 1, 3_000_000)]
    )

    with pytest.raises(UnknownSceneError) as exc_info:
        build_pending_jobs_from_request(request, "req-1", _resolve)

    assert exc_info.value.source_id == "clipgt-unknown-scene"
    assert exc_info.value.policy_start_us == 3_000_000
    assert "source_id='clipgt-unknown-scene'" in str(exc_info.value)


def test_adapter_rejects_mismatched_data_source_descriptor() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[_spec("clipgt-a", 1, usdz_uuid="wrong-uuid")]
    )

    with pytest.raises(ValueError, match="descriptor mismatch"):
        build_pending_jobs_from_request(request, "req-1", _resolve)


def test_adapter_assigns_rollout_spec_indexes_in_request_order() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[
            _spec("clipgt-a", 1),
            _spec("clipgt-b", 2, 3_000_000),
        ]
    )

    jobs = build_pending_jobs_from_request(request, "req-1", _resolve)
    assert [job.scene.source.source_id for job in jobs] == [
        "clipgt-a",
        "clipgt-b",
        "clipgt-b",
    ]
    assert [job.scene.policy_start_us for job in jobs] == [
        T0_AUTO,
        3_000_000,
        3_000_000,
    ]
    assert [job.rollout_spec_index for job in jobs] == [0, 1, 1]


def test_adapter_ignores_zero_rollout_specs_when_indexing() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[_spec("clipgt-a", 0), _spec("clipgt-b", 2)]
    )

    jobs = build_pending_jobs_from_request(request, "req-1", _resolve)
    assert [job.scene.source.source_id for job in jobs] == [
        "clipgt-b",
        "clipgt-b",
    ]
    assert [job.rollout_spec_index for job in jobs] == [1, 1]


def test_adapter_propagates_session_uuids_in_order() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[_spec("clipgt-a", 3, session_uuids=["u0", "u1", "u2"])]
    )

    jobs = build_pending_jobs_from_request(request, "req-1", _resolve)
    assert [job.session_uuid for job in jobs] == ["u0", "u1", "u2"]


def test_adapter_rejects_mismatched_session_uuids_length() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[_spec("clipgt-a", 2, session_uuids=["u0"])]
    )

    with pytest.raises(ValueError, match="session_uuids"):
        build_pending_jobs_from_request(request, "req-1", _resolve)


def test_adapter_distinguishes_explicit_zero_from_auto() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[_spec("clipgt-a", 1), _spec("clipgt-a", 1, 0)]
    )

    jobs = build_pending_jobs_from_request(request, "req-1", _resolve)
    assert [job.scene.policy_start_us for job in jobs] == [T0_AUTO, 0]
    assert [job.scene.scene_key for job in jobs] == [
        "clipgt-a@t0=AUTO",
        "clipgt-a@t0=0",
    ]


def test_adapter_resolves_and_propagates_fork_ego_history() -> None:
    ego_history = Trajectory(poses=[PoseAtTime(timestamp_us=2_000_000)])
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[
            runtime_pb2.RolloutSpec(
                scene=scene_to_proto(_scene("clipgt-a", 2_000_000)),
                nr_rollouts=2,
                ego_history=ego_history,
            )
        ]
    )

    jobs = build_pending_jobs_from_request(
        request,
        "req-1",
        _resolve,
        _resolve_fork,
    )

    assert [job.scene.policy_start_us for job in jobs] == [2_000_000, 2_000_000]
    assert all(job.ego_history_bytes == ego_history.SerializeToString() for job in jobs)


def test_adapter_derives_per_rollout_session_seeds_from_spec_seed() -> None:
    request = runtime_pb2.SimulationRequest(
        rollout_specs=[_spec("clipgt-a", 3, random_seed=100)]
    )

    jobs = build_pending_jobs_from_request(request, "req-1", _resolve)
    assert [job.session_seed for job in jobs] == [100, 101, 102]


def test_adapter_leaves_session_seed_unset_when_spec_has_no_seed() -> None:
    request = runtime_pb2.SimulationRequest(rollout_specs=[_spec("clipgt-a", 2)])

    jobs = build_pending_jobs_from_request(request, "req-1", _resolve)
    assert [job.session_seed for job in jobs] == [0, 0]
