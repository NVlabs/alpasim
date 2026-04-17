# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from alpasim_runtime.daemon.interactive_servicer import (
    InteractiveRuntimeServicer,
    _snapshot_to_proto,
    _state_to_proto,
)
from alpasim_runtime.interactive.models import (
    CandidateSummaryModel,
    CheckpointSummaryModel,
    DecisionSummaryModel,
    EgoStateModel,
    SessionSnapshotModel,
    SessionStateModel,
)
from alpasim_grpc.v0.common_pb2 import DynamicState, Pose, Quat, Vec3
from alpasim_grpc.v1 import interactive_runtime_pb2


def _make_pose() -> Pose:
    return Pose(vec=Vec3(x=1.0, y=2.0, z=0.0), quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0))


def _make_decision() -> DecisionSummaryModel:
    return DecisionSummaryModel(
        step_id=3,
        input_snapshot_id="input-3",
        selected_candidate_id="input-3:safe:0",
        candidates=[
            CandidateSummaryModel(
                candidate_id="input-3:fast:0",
                backend_id="fast",
                status="READY",
                selected=False,
            ),
            CandidateSummaryModel(
                candidate_id="input-3:safe:0",
                backend_id="safe",
                status="SELECTED",
                selected=True,
            ),
        ],
        arbitration_reason="priority",
    )


def test_state_to_proto_includes_latest_decision() -> None:
    decision = _make_decision()
    snapshot = SessionSnapshotModel(
        interactive_session_id="sess-1",
        tick_id=4,
        sim_time_us=400_000,
        ego=EgoStateModel(
            pose=_make_pose(),
            dynamics=DynamicState(),
        ),
        actors=[],
        frame_refs=[],
        latest_decision=decision,
    )
    state = SessionStateModel(
        interactive_session_id="sess-1",
        rollout_uuid="rollout-1",
        scene_id="scene-a",
        status="PAUSED",
        current_tick_id=4,
        current_sim_time_us=400_000,
        latest_snapshot=snapshot,
        latest_decision=decision,
        active_backend_ids=["fast", "safe"],
    )

    proto = _state_to_proto(state)

    assert proto.latest_decision.step_id == 3
    assert proto.latest_decision.selected_candidate_id == "input-3:safe:0"
    assert len(proto.latest_decision.candidates) == 2
    assert proto.latest_snapshot.latest_decision.step_id == 3
    assert list(proto.active_backend_ids) == ["fast", "safe"]


def test_snapshot_to_proto_includes_latest_decision() -> None:
    snapshot = SessionSnapshotModel(
        interactive_session_id="sess-1",
        tick_id=4,
        sim_time_us=400_000,
        ego=EgoStateModel(
            pose=_make_pose(),
            dynamics=DynamicState(),
            front_steering_angle_rad=0.15,
        ),
        actors=[],
        frame_refs=[],
        latest_decision=_make_decision(),
    )

    proto = _snapshot_to_proto(snapshot)

    assert proto.latest_decision.input_snapshot_id == "input-3"
    assert proto.latest_decision.candidates[1].selected is True
    assert proto.ego.front_steering_angle_rad == pytest.approx(0.15)


class _AbortContext:
    def __init__(self) -> None:
        self.code = None
        self.details = None

    async def abort(self, code, details):
        self.code = code
        self.details = details
        raise RuntimeError("aborted")


@pytest.mark.asyncio
async def test_list_candidates_returns_candidate_summaries() -> None:
    manager = SimpleNamespace(
        list_candidates=AsyncMock(
            return_value=_make_decision().candidates
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)

    response = await servicer.ListCandidates(
        interactive_runtime_pb2.ListCandidatesRequest(
            interactive_session_id="sess-1"
        ),
        _AbortContext(),
    )

    manager.list_candidates.assert_awaited_once_with("sess-1")
    assert len(response.candidates) == 2
    assert response.candidates[1].backend_id == "safe"
    assert response.candidates[1].selected is True


@pytest.mark.asyncio
async def test_list_sessions_returns_session_states() -> None:
    manager = SimpleNamespace(
        list_sessions=AsyncMock(
            return_value=[
                SessionStateModel(
                    interactive_session_id="sess-1",
                    rollout_uuid="rollout-1",
                    scene_id="scene-a",
                    status="PAUSED",
                    current_tick_id=4,
                    current_sim_time_us=400_000,
                    latest_snapshot=None,
                    latest_decision=None,
                    active_backend_ids=["safe"],
                )
            ]
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)

    response = await servicer.ListSessions(
        interactive_runtime_pb2.ListSessionsRequest(),
        _AbortContext(),
    )

    manager.list_sessions.assert_awaited_once_with()
    assert len(response.sessions) == 1
    assert response.sessions[0].interactive_session_id == "sess-1"
    assert response.sessions[0].scene_id == "scene-a"


@pytest.mark.asyncio
async def test_set_active_backends_returns_updated_state() -> None:
    manager = SimpleNamespace(
        set_active_backends=AsyncMock(
            return_value=SessionStateModel(
                interactive_session_id="sess-1",
                rollout_uuid="rollout-1",
                scene_id="scene-a",
                status="PAUSED",
                current_tick_id=4,
                current_sim_time_us=400_000,
                latest_snapshot=None,
                latest_decision=None,
                active_backend_ids=["safe"],
            )
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)

    response = await servicer.SetActiveBackends(
        interactive_runtime_pb2.SetActiveBackendsRequest(
            interactive_session_id="sess-1",
            backend_ids=["safe"],
        ),
        _AbortContext(),
    )

    manager.set_active_backends.assert_awaited_once_with("sess-1", ["safe"])
    assert list(response.active_backend_ids) == ["safe"]


@pytest.mark.asyncio
async def test_list_checkpoints_returns_checkpoint_summaries() -> None:
    manager = SimpleNamespace(
        list_checkpoints=AsyncMock(
            return_value=[
                CheckpointSummaryModel(
                    checkpoint_id="tick-4",
                    tick_id=4,
                    sim_time_us=400_000,
                    status="PAUSED",
                    restore_supported=False,
                    unsupported_backend_ids=["default_driver"],
                )
            ]
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)

    response = await servicer.ListCheckpoints(
        interactive_runtime_pb2.ListCheckpointsRequest(
            interactive_session_id="sess-1"
        ),
        _AbortContext(),
    )

    manager.list_checkpoints.assert_awaited_once_with("sess-1")
    assert len(response.checkpoints) == 1
    assert response.checkpoints[0].checkpoint_id == "tick-4"
    assert response.checkpoints[0].restore_supported is False
    assert list(response.checkpoints[0].unsupported_backend_ids) == ["default_driver"]


@pytest.mark.asyncio
async def test_recompute_candidate_returns_updated_state() -> None:
    manager = SimpleNamespace(
        recompute_candidate=AsyncMock(
            return_value=SessionStateModel(
                interactive_session_id="sess-1",
                rollout_uuid="rollout-1",
                scene_id="scene-a",
                status="PAUSED",
                current_tick_id=4,
                current_sim_time_us=400_000,
                latest_snapshot=None,
                latest_decision=None,
                active_backend_ids=["safe"],
            )
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)

    response = await servicer.RecomputeCandidate(
        interactive_runtime_pb2.RecomputeCandidateRequest(
            interactive_session_id="sess-1",
            backend_id="safe",
        ),
        _AbortContext(),
    )

    manager.recompute_candidate.assert_awaited_once_with("sess-1", "safe")
    assert response.current_tick_id == 4


@pytest.mark.asyncio
async def test_select_candidate_returns_updated_state() -> None:
    manager = SimpleNamespace(
        select_candidate=AsyncMock(
            return_value=SessionStateModel(
                interactive_session_id="sess-1",
                rollout_uuid="rollout-1",
                scene_id="scene-a",
                status="PAUSED",
                current_tick_id=4,
                current_sim_time_us=400_000,
                latest_snapshot=None,
                latest_decision=None,
                active_backend_ids=["safe"],
            )
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)

    response = await servicer.SelectCandidate(
        interactive_runtime_pb2.SelectCandidateRequest(
            interactive_session_id="sess-1",
            candidate_id="input-4:fast:0",
        ),
        _AbortContext(),
    )

    manager.select_candidate.assert_awaited_once_with("sess-1", "input-4:fast:0")
    assert response.current_tick_id == 4


@pytest.mark.asyncio
async def test_restore_checkpoint_returns_updated_state() -> None:
    manager = SimpleNamespace(
        restore_checkpoint=AsyncMock(
            return_value=SessionStateModel(
                interactive_session_id="sess-1",
                rollout_uuid="rollout-1",
                scene_id="scene-a",
                status="PAUSED",
                current_tick_id=4,
                current_sim_time_us=400_000,
                latest_snapshot=None,
                latest_decision=None,
                active_backend_ids=["safe"],
            )
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)

    response = await servicer.RestoreCheckpoint(
        interactive_runtime_pb2.RestoreCheckpointRequest(
            interactive_session_id="sess-1",
            checkpoint_id="tick-4",
        ),
        _AbortContext(),
    )

    manager.restore_checkpoint.assert_awaited_once_with("sess-1", "tick-4")
    assert response.current_tick_id == 4


@pytest.mark.asyncio
async def test_restore_checkpoint_returns_failed_precondition_for_unsupported_restore() -> None:
    manager = SimpleNamespace(
        restore_checkpoint=AsyncMock(
            side_effect=RuntimeError("Checkpoint restore is not supported for backends: default_driver")
        )
    )
    servicer = InteractiveRuntimeServicer(manager=manager)
    context = _AbortContext()

    with pytest.raises(RuntimeError, match="aborted"):
        await servicer.RestoreCheckpoint(
            interactive_runtime_pb2.RestoreCheckpointRequest(
                interactive_session_id="sess-1",
                checkpoint_id="tick-4",
            ),
            context,
        )

    assert context.code.name == "FAILED_PRECONDITION"
