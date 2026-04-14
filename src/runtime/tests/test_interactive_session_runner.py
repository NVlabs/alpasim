# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_runtime.decision import (
    CandidateDecision,
    CandidateStatus,
    DecisionBundle,
    DecisionSnapshot,
)
from alpasim_runtime.interactive.frame_store import FrameStore
from alpasim_runtime.interactive.models import (
    CandidateSummaryModel,
    CheckpointSummaryModel,
    DecisionSummaryModel,
    EgoStateModel,
    FrameDataModel,
    SessionSnapshotModel,
)
from alpasim_runtime.interactive.session_runner import InteractiveSessionRunner
from alpasim_grpc.v0.common_pb2 import DynamicState, Pose, Quat, Vec3


class _AsyncLock:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_pose() -> Pose:
    return Pose(vec=Vec3(x=1.0, y=2.0, z=0.0), quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0))


def _make_snapshot() -> SessionSnapshotModel:
    return SessionSnapshotModel(
        interactive_session_id="sess-1",
        tick_id=4,
        sim_time_us=400_000,
        ego=EgoStateModel(pose=_make_pose(), dynamics=DynamicState()),
        actors=[],
        frame_refs=[],
        latest_decision=DecisionSummaryModel(
            step_id=4,
            input_snapshot_id="input-4",
            selected_candidate_id="input-4:safe:0",
            candidates=[
                CandidateSummaryModel(
                    candidate_id="input-4:safe:0",
                    backend_id="safe",
                    status="SELECTED",
                    selected=True,
                )
            ],
        ),
    )


@pytest.mark.asyncio
async def test_list_and_restore_checkpoint_updates_state() -> None:
    runner = object.__new__(InteractiveSessionRunner)
    runner._lock = _AsyncLock()
    runner.pause = AsyncMock()
    runner._publish = AsyncMock()
    runner._build_state = MagicMock(
        return_value=SimpleNamespace(
            interactive_session_id="sess-1",
            rollout_uuid="rollout-1",
            scene_id="scene-a",
            status="PAUSED",
            current_tick_id=4,
            current_sim_time_us=400_000,
            latest_snapshot=_make_snapshot(),
            latest_decision=_make_snapshot().latest_decision,
            active_backend_ids=["safe"],
            error="",
        )
    )
    runner._capture_images = AsyncMock()
    runner._frame_store = FrameStore(max_retained_ticks=4)
    frame = FrameDataModel(
        sensor_id="cam_front",
        frame_start_us=300_000,
        frame_end_us=333_000,
        frame_encoding="JPEG",
        content_type="image/jpeg",
        image_bytes=b"abc",
    )
    runner._frame_store.restore(OrderedDict([(4, {"cam_front": frame})]))
    runner._pending_frames = [frame]
    runner._tick_id = 4
    runner._latest_snapshot = _make_snapshot()
    runner._status = "COMPLETED"
    runner._error = "stale"
    checkpoint = SimpleNamespace(
        checkpoint_id="tick-4",
        tick_id=4,
        sim_time_us=400_000,
        runtime_checkpoint=SimpleNamespace(unsupported_backend_ids=[]),
        frame_store_snapshot=OrderedDict([(4, {"cam_front": frame})]),
        latest_snapshot=_make_snapshot(),
        pending_frames=[frame],
        status="PAUSED",
    )
    runner._checkpoints = OrderedDict([("tick-4", checkpoint)])
    rollout = SimpleNamespace(
        restore_runtime_checkpoint=AsyncMock(),
        current_state=SimpleNamespace(rendered_images_handler=None),
    )
    runner._rollout = rollout

    checkpoints = await runner.list_checkpoints()
    restored_state = await runner.restore_checkpoint("tick-4")

    assert checkpoints == [
        CheckpointSummaryModel(
            checkpoint_id="tick-4",
            tick_id=4,
            sim_time_us=400_000,
            status="PAUSED",
            restore_supported=True,
            unsupported_backend_ids=[],
        )
    ]
    runner.pause.assert_awaited_once()
    rollout.restore_runtime_checkpoint.assert_awaited_once_with(checkpoint.runtime_checkpoint)
    assert runner._tick_id == 4
    assert runner._status == "PAUSED"
    assert runner._error == ""
    assert runner._frame_store.get_frame("cam_front", 4).image_bytes == b"abc"
    assert restored_state.status == "PAUSED"


@pytest.mark.asyncio
async def test_restore_checkpoint_rejects_unsupported_backend_restore() -> None:
    runner = object.__new__(InteractiveSessionRunner)
    runner._lock = _AsyncLock()
    runner.pause = AsyncMock()
    runner._publish = AsyncMock()
    runner._rollout = SimpleNamespace()
    runner._checkpoints = OrderedDict(
        [
            (
                "tick-4",
                SimpleNamespace(
                    checkpoint_id="tick-4",
                    tick_id=4,
                    sim_time_us=400_000,
                    runtime_checkpoint=SimpleNamespace(
                        unsupported_backend_ids=["default_driver"]
                    ),
                    frame_store_snapshot=OrderedDict(),
                    latest_snapshot=None,
                    pending_frames=[],
                    status="PAUSED",
                ),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="default_driver"):
        await runner.restore_checkpoint("tick-4")


@pytest.mark.asyncio
async def test_recompute_candidate_updates_latest_decision_bundle() -> None:
    runner = object.__new__(InteractiveSessionRunner)
    runner._lock = _AsyncLock()
    runner.pause = AsyncMock()
    runner._publish = AsyncMock()
    runner._build_state = MagicMock(return_value=SimpleNamespace(status="PAUSED"))
    runner._record_checkpoint_locked = MagicMock()
    runner._latest_snapshot = _make_snapshot()
    snapshot = DecisionSnapshot(
        step_id=4,
        input_snapshot_id="input-4",
        time_now_us=400_000,
        time_query_us=500_000,
        ego_pose_history_timestamps_us=[300_000, 400_000],
        traffic_actor_ids=[],
        route_waypoints_in_rig=[],
        planner_context=None,
        renderer_data=None,
        camera_frame_timestamps_us={},
    )
    old_candidate = CandidateDecision(
        candidate_id="input-4:safe:0",
        step_id=4,
        input_snapshot_id="input-4",
        backend_id="safe",
        status=CandidateStatus.SELECTED,
        trajectory=None,
    )
    updated_bundle = DecisionBundle(
        snapshot=snapshot,
        candidates=[
            old_candidate,
            CandidateDecision(
                candidate_id="input-4:safe:1",
                step_id=4,
                input_snapshot_id="input-4",
                backend_id="safe",
                status=CandidateStatus.READY,
                trajectory=None,
                recompute_count=1,
            ),
        ],
        selected_candidate_id="input-4:safe:0",
        arbitration_reason="recomputed:safe",
    )
    orchestrator = SimpleNamespace(recompute_candidate=AsyncMock(return_value=updated_bundle))
    rollout = SimpleNamespace(
        _build_default_driver_orchestrator=MagicMock(return_value=orchestrator),
        driver=SimpleNamespace(),
        current_state=SimpleNamespace(
            last_committed_decision_bundle=DecisionBundle(
                snapshot=snapshot,
                candidates=[old_candidate],
                selected_candidate_id="input-4:safe:0",
                arbitration_reason="priority",
            ),
            available_driver_backend_ids=["safe"],
        ),
    )
    runner._rollout = rollout

    state = await runner.recompute_candidate("safe")

    runner.pause.assert_awaited_once()
    orchestrator.recompute_candidate.assert_awaited_once()
    assert runner._rollout.current_state.last_committed_decision_bundle == updated_bundle
    assert runner._latest_snapshot.latest_decision is not None
    assert len(runner._latest_snapshot.latest_decision.candidates) == 2
    runner._record_checkpoint_locked.assert_called_once_with(runner._latest_snapshot)
    assert state.status == "PAUSED"


@pytest.mark.asyncio
async def test_select_candidate_updates_latest_decision_bundle() -> None:
    runner = object.__new__(InteractiveSessionRunner)
    runner._lock = _AsyncLock()
    runner.pause = AsyncMock()
    runner._publish = AsyncMock()
    runner._build_state = MagicMock(return_value=SimpleNamespace(status="PAUSED"))
    runner._record_checkpoint_locked = MagicMock()
    runner._latest_snapshot = _make_snapshot()
    snapshot = DecisionSnapshot(
        step_id=4,
        input_snapshot_id="input-4",
        time_now_us=400_000,
        time_query_us=500_000,
        ego_pose_history_timestamps_us=[300_000, 400_000],
        traffic_actor_ids=[],
        route_waypoints_in_rig=[],
        planner_context=None,
        renderer_data=None,
        camera_frame_timestamps_us={},
    )
    rollout = SimpleNamespace(
        current_state=SimpleNamespace(
            last_committed_decision_bundle=DecisionBundle(
                snapshot=snapshot,
                candidates=[
                    CandidateDecision(
                        candidate_id="input-4:safe:0",
                        step_id=4,
                        input_snapshot_id="input-4",
                        backend_id="safe",
                        status=CandidateStatus.SELECTED,
                        trajectory=None,
                    ),
                    CandidateDecision(
                        candidate_id="input-4:fast:0",
                        step_id=4,
                        input_snapshot_id="input-4",
                        backend_id="fast",
                        status=CandidateStatus.READY,
                        trajectory=None,
                    ),
                ],
                selected_candidate_id="input-4:safe:0",
                arbitration_reason="priority",
            )
        )
    )
    runner._rollout = rollout

    state = await runner.select_candidate("input-4:fast:0")

    runner.pause.assert_awaited_once()
    assert runner._rollout.current_state.last_committed_decision_bundle.selected_candidate_id == "input-4:fast:0"
    assert runner._latest_snapshot.latest_decision is not None
    assert runner._latest_snapshot.latest_decision.selected_candidate_id == "input-4:fast:0"
    runner._record_checkpoint_locked.assert_called_once_with(runner._latest_snapshot)
    assert state.status == "PAUSED"
