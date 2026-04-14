# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from alpasim_runtime.decision import (
    BackendMetadata,
    CandidateDecision,
    CandidateStatus,
    DecisionBundle,
    DecisionSnapshot,
    DriverServiceBackendAdapter,
    DriverBackendRegistry,
    MultiBackendDriverOrchestrator,
    build_input_snapshot_id,
    select_candidate_in_bundle,
)
from alpasim_utils.geometry import Pose, Trajectory
import numpy as np
from unittest.mock import AsyncMock


def _make_candidate(step_id: int, snapshot_id: str, backend_id: str) -> CandidateDecision:
    trajectory = Trajectory.from_poses(
        timestamps=np.array([100_000], dtype=np.uint64),
        poses=[
            Pose(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
        ],
    )
    return CandidateDecision(
        candidate_id=f"{snapshot_id}:{backend_id}:0",
        step_id=step_id,
        input_snapshot_id=snapshot_id,
        backend_id=backend_id,
        status=CandidateStatus.READY,
        trajectory=trajectory,
        generated_at_us=100_000,
    )


class _FakeBackend:
    def __init__(self, backend_id: str, priority: int, *, supports_restore: bool = False) -> None:
        self.metadata = BackendMetadata(
            backend_id=backend_id,
            backend_type="fake",
            supports_parallel=True,
            supports_restore=supports_restore,
            priority=priority,
        )
        self.infer = AsyncMock()
        self.capture_backend_state = MagicMock(return_value={"backend_id": backend_id})
        self.restore_backend_state = AsyncMock()


def _make_snapshot(step_id: int = 1) -> DecisionSnapshot:
    snapshot_id = build_input_snapshot_id(
        step_id=step_id,
        time_now_us=100_000,
        time_query_us=200_000,
        planner_context={"ego": {"yaw": 0.0}},
        route_waypoints_in_rig=[[0.0, 0.0, 0.0]],
        traffic_actor_ids=["actor_a"],
        ego_pose_history_timestamps_us=[0, 100_000],
        camera_frame_timestamps_us={"cam_front": 100_000},
        renderer_data=b"",
    )
    return DecisionSnapshot(
        step_id=step_id,
        input_snapshot_id=snapshot_id,
        time_now_us=100_000,
        time_query_us=200_000,
        ego_pose_history_timestamps_us=[0, 100_000],
        traffic_actor_ids=["actor_a"],
        route_waypoints_in_rig=[[0.0, 0.0, 0.0]],
        planner_context={"ego": {"yaw": 0.0}},
        renderer_data=b"",
        camera_frame_timestamps_us={"cam_front": 100_000},
    )


def test_build_input_snapshot_id_is_deterministic() -> None:
    first = build_input_snapshot_id(
        step_id=1,
        time_now_us=100_000,
        time_query_us=200_000,
        planner_context={"a": 1},
        route_waypoints_in_rig=[[0.0, 1.0, 2.0]],
        traffic_actor_ids=["b", "a"],
        ego_pose_history_timestamps_us=[0, 100_000],
        camera_frame_timestamps_us={"cam_front": 100_000},
        renderer_data=b"payload",
    )
    second = build_input_snapshot_id(
        step_id=1,
        time_now_us=100_000,
        time_query_us=200_000,
        planner_context={"a": 1},
        route_waypoints_in_rig=[[0.0, 1.0, 2.0]],
        traffic_actor_ids=["b", "a"],
        ego_pose_history_timestamps_us=[0, 100_000],
        camera_frame_timestamps_us={"cam_front": 100_000},
        renderer_data=b"payload",
    )
    assert first == second


@pytest.mark.asyncio
async def test_multi_backend_orchestrator_selects_highest_priority_ready_candidate() -> None:
    snapshot = _make_snapshot()
    backend_fast = _FakeBackend("fast", priority=10)
    backend_safe = _FakeBackend("safe", priority=1)
    backend_fast.infer.return_value = _make_candidate(snapshot.step_id, snapshot.input_snapshot_id, "fast")
    backend_safe.infer.return_value = _make_candidate(snapshot.step_id, snapshot.input_snapshot_id, "safe")

    orchestrator = MultiBackendDriverOrchestrator(
        DriverBackendRegistry([backend_fast, backend_safe])
    )
    bundle = await orchestrator.generate_candidates(snapshot)
    selected = await orchestrator.select_candidate(bundle)

    assert len(bundle.candidates) == 2
    assert selected.backend_id == "safe"
    assert selected.status == CandidateStatus.SELECTED


@pytest.mark.asyncio
async def test_multi_backend_orchestrator_filters_requested_backend_ids() -> None:
    snapshot = _make_snapshot()
    backend_fast = _FakeBackend("fast", priority=10)
    backend_safe = _FakeBackend("safe", priority=1)
    backend_fast.infer.return_value = _make_candidate(snapshot.step_id, snapshot.input_snapshot_id, "fast")
    backend_safe.infer.return_value = _make_candidate(snapshot.step_id, snapshot.input_snapshot_id, "safe")

    orchestrator = MultiBackendDriverOrchestrator(
        DriverBackendRegistry([backend_fast, backend_safe])
    )
    bundle = await orchestrator.generate_candidates(snapshot, backend_ids=["fast"])

    assert [candidate.backend_id for candidate in bundle.candidates] == ["fast"]
    backend_fast.infer.assert_awaited_once()
    backend_safe.infer.assert_not_called()


@pytest.mark.asyncio
async def test_driver_service_backend_adapter_sets_next_model_override() -> None:
    driver = SimpleNamespace(
        drive=AsyncMock(return_value=_make_candidate(1, "snapshot", "backend").trajectory),
        set_next_model_for_next_drive=MagicMock(),
        set_planner_context_for_next_drive=MagicMock(),
    )
    adapter = DriverServiceBackendAdapter(
        driver,
        backend_id="pdm_backend",
        model_type_override="pdm",
    )
    snapshot = _make_snapshot()

    candidate = await adapter.infer(snapshot)

    driver.set_next_model_for_next_drive.assert_called_once_with("pdm")
    driver.set_planner_context_for_next_drive.assert_called_once()
    assert candidate.backend_id == "pdm_backend"
    assert candidate.diagnostics["model_type_override"] == "pdm"


@pytest.mark.asyncio
async def test_driver_service_backend_adapter_injects_observation_window_summary() -> None:
    driver = SimpleNamespace(
        drive=AsyncMock(return_value=_make_candidate(1, "snapshot", "backend").trajectory),
        set_next_model_for_next_drive=MagicMock(),
        set_planner_context_for_next_drive=MagicMock(),
    )
    adapter = DriverServiceBackendAdapter(
        driver,
        backend_id="vla_backend",
        observation_window_summary_getter=lambda snapshot_id, window_size: {
            "anchor_snapshot_id": snapshot_id,
            "window_size": window_size,
            "available_frames": 2,
            "frames": [{"input_snapshot_id": "prev"}, {"input_snapshot_id": snapshot_id}],
        },
        observation_window_summary_size=4,
    )
    snapshot = _make_snapshot()

    await adapter.infer(snapshot)

    planner_context = driver.set_planner_context_for_next_drive.call_args.args[0]
    assert (
        planner_context["decision_metadata"]["observation_window"]["anchor_snapshot_id"]
        == snapshot.input_snapshot_id
    )
    assert planner_context["decision_metadata"]["observation_window"]["window_size"] == 4


def test_multi_backend_orchestrator_captures_restoreable_backend_state() -> None:
    backend_fast = _FakeBackend("fast", priority=10, supports_restore=False)
    backend_safe = _FakeBackend("safe", priority=1, supports_restore=True)
    orchestrator = MultiBackendDriverOrchestrator(
        DriverBackendRegistry([backend_fast, backend_safe])
    )

    checkpoint, unsupported = orchestrator.capture_backend_checkpoint()

    assert checkpoint == {"safe": {"backend_id": "safe"}}
    assert unsupported == ["fast"]
    backend_safe.capture_backend_state.assert_called_once()
    backend_fast.capture_backend_state.assert_not_called()


@pytest.mark.asyncio
async def test_multi_backend_orchestrator_restores_restoreable_backend_state() -> None:
    backend_safe = _FakeBackend("safe", priority=1, supports_restore=True)
    orchestrator = MultiBackendDriverOrchestrator(
        DriverBackendRegistry([backend_safe])
    )

    await orchestrator.restore_backend_checkpoint({"safe": {"checkpoint": 1}})

    backend_safe.restore_backend_state.assert_awaited_once_with({"checkpoint": 1})


@pytest.mark.asyncio
async def test_multi_backend_orchestrator_recompute_appends_new_candidate_and_stales_old() -> None:
    snapshot = _make_snapshot()
    backend_safe = _FakeBackend("safe", priority=1)
    original = _make_candidate(snapshot.step_id, snapshot.input_snapshot_id, "safe")
    backend_safe.infer.return_value = original
    orchestrator = MultiBackendDriverOrchestrator(
        DriverBackendRegistry([backend_safe])
    )
    bundle = await orchestrator.generate_candidates(snapshot)
    backend_safe.infer.return_value = _make_candidate(
        snapshot.step_id, snapshot.input_snapshot_id, "safe"
    )

    updated = await orchestrator.recompute_candidate(bundle, "safe")

    assert [candidate.status for candidate in updated.candidates] == [
        CandidateStatus.STALE,
        CandidateStatus.READY,
    ]
    assert updated.candidates[-1].candidate_id.endswith(":1")
    assert updated.candidates[-1].recompute_count == 1


def test_select_candidate_in_bundle_marks_previous_selection_rejected() -> None:
    snapshot = _make_snapshot()
    bundle = DecisionBundle(
        snapshot=snapshot,
        candidates=[
            CandidateDecision(
                candidate_id=f"{snapshot.input_snapshot_id}:safe:0",
                step_id=snapshot.step_id,
                input_snapshot_id=snapshot.input_snapshot_id,
                backend_id="safe",
                status=CandidateStatus.SELECTED,
                trajectory=None,
            ),
            CandidateDecision(
                candidate_id=f"{snapshot.input_snapshot_id}:fast:0",
                step_id=snapshot.step_id,
                input_snapshot_id=snapshot.input_snapshot_id,
                backend_id="fast",
                status=CandidateStatus.READY,
                trajectory=None,
            ),
        ],
        selected_candidate_id=f"{snapshot.input_snapshot_id}:safe:0",
        arbitration_reason="priority",
    )

    updated = select_candidate_in_bundle(bundle, f"{snapshot.input_snapshot_id}:fast:0")

    assert updated.selected_candidate_id == f"{snapshot.input_snapshot_id}:fast:0"
    assert updated.arbitration_reason == "manual_selection"
    assert [candidate.status for candidate in updated.candidates] == [
        CandidateStatus.REJECTED,
        CandidateStatus.SELECTED,
    ]
