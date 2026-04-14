# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import numpy as np

from alpasim_runtime.decision import (
    CandidateDecision,
    CandidateStatus,
    DecisionBundle,
    DecisionSnapshot,
)
from alpasim_runtime.interactive.session_runner import _decision_summary_from_bundle
from alpasim_utils.geometry import Pose, Trajectory


def test_decision_summary_from_bundle_marks_selected_candidate() -> None:
    trajectory = Trajectory.from_poses(
        timestamps=np.array([100_000], dtype=np.uint64),
        poses=[
            Pose(
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
        ],
    )
    snapshot = DecisionSnapshot(
        step_id=1,
        input_snapshot_id="input-1",
        time_now_us=100_000,
        time_query_us=200_000,
        ego_pose_history_timestamps_us=[0, 100_000],
        traffic_actor_ids=[],
        route_waypoints_in_rig=[],
        planner_context=None,
        renderer_data=None,
        camera_frame_timestamps_us={"cam_front": 100_000},
    )
    fast = CandidateDecision(
        candidate_id="input-1:fast:0",
        step_id=1,
        input_snapshot_id="input-1",
        backend_id="fast",
        status=CandidateStatus.READY,
        trajectory=trajectory,
    )
    safe = CandidateDecision(
        candidate_id="input-1:safe:0",
        step_id=1,
        input_snapshot_id="input-1",
        backend_id="safe",
        status=CandidateStatus.SELECTED,
        trajectory=trajectory,
    )
    summary = _decision_summary_from_bundle(
        DecisionBundle(
            snapshot=snapshot,
            candidates=[fast, safe],
            selected_candidate_id=safe.candidate_id,
            arbitration_reason="priority",
        )
    )

    assert summary.step_id == 1
    assert summary.selected_candidate_id == "input-1:safe:0"
    assert [candidate.backend_id for candidate in summary.candidates] == ["fast", "safe"]
    assert [candidate.selected for candidate in summary.candidates] == [False, True]
