# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Trajectory selection inside the Alpamayo prediction path.

Uses a stub inferencer so the candidate handling, waypoint timestamps and plan
handover can be tested without a checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from alpasim_driver.models.alpamayo_base import AlpamayoBaseModel
from alpasim_driver.models.base import DriveCommand, PredictionInput
from alpasim_driver.models.trajectory_selection import (
    TrajectorySelectionStrategy,
    plan_in_local_frame,
)
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3
from alpasim_utils.geometry import Pose as GeometryPose
from alpasim_utils.geometry import Trajectory

CAMERA_ID = "camera_front_wide_120fov"
NUM_WAYPOINTS = 20
SPEED_M_S = 10.0


class _StubAlpamayo(AlpamayoBaseModel):
    """Alpamayo variant whose inference returns fixed candidates."""

    def __init__(
        self,
        candidates: np.ndarray,
        selection_strategy: TrajectorySelectionStrategy,
        max_num_distance_points: int = 64,
        skip_first_n_distance_points: int = 0,
        waypoint_dt: float = 0.1,
    ) -> None:
        self._candidates = candidates
        self._init_common(
            model=SimpleNamespace(
                action_space=SimpleNamespace(
                    get_action_space_dims=lambda: (NUM_WAYPOINTS, 3), dt=waypoint_dt
                )
            ),
            processor=SimpleNamespace(apply_chat_template=lambda *args, **kwargs: {}),
            helper_module=SimpleNamespace(to_device=lambda inputs, device: inputs),
            device=torch.device("cpu"),
            camera_ids=[CAMERA_ID],
            context_length=1,
            num_traj_samples=len(candidates),
            selection_strategy=selection_strategy,
            max_num_distance_points=max_num_distance_points,
            skip_first_n_distance_points=skip_first_n_distance_points,
        )

    def _create_chat_message(
        self, image_frames: torch.Tensor, nav_text: str | None
    ) -> list:
        return []

    def _run_inference(self, model_inputs: dict, nav_text: str | None) -> tuple:
        positions = torch.from_numpy(self._candidates).float()[None, None]
        rotations = torch.eye(3).expand(*positions.shape[:-1], 3, 3)
        # Text outputs are generated per candidate, shaped [B, sets, K].
        cot = np.array([[[f"reasoning {i}" for i in range(len(self._candidates))]]])
        return positions, rotations, {"cot": cot}


def _straight_candidate(lateral_offset: float) -> np.ndarray:
    """Waypoints of shape (T, 3) driving forward at a constant lateral offset."""
    forward = SPEED_M_S * 0.1 * np.arange(1, NUM_WAYPOINTS + 1)
    return np.stack(
        [forward, np.full(NUM_WAYPOINTS, lateral_offset), np.zeros(NUM_WAYPOINTS)],
        axis=-1,
    )


def _driving_straight_poses(latest_us: int) -> list[PoseAtTime]:
    """Ego poses along the local x axis, covering the required history window."""
    return [
        PoseAtTime(
            timestamp_us=timestamp_us,
            pose=Pose(
                vec=Vec3(x=SPEED_M_S * timestamp_us / 1e6, y=0.0, z=0.0),
                quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        )
        for timestamp_us in range(0, latest_us + 1, 100_000)
    ]


def _plan_following(candidate: np.ndarray, t0_us: int) -> Trajectory:
    """The plan a previous cycle would have returned for *candidate*."""
    return plan_in_local_frame(
        candidate,
        np.tile(np.eye(3), (len(candidate), 1, 1)),
        t0_us + np.arange(1, len(candidate) + 1, dtype=np.uint64) * 100_000,
        pose_local_to_rig_t0=GeometryPose(
            np.array([SPEED_M_S * t0_us / 1e6, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
    )


def _prediction_input(
    latest_us: int, previous_plan: Trajectory | None
) -> PredictionInput:
    return PredictionInput(
        camera_images={CAMERA_ID: [(latest_us, np.zeros((8, 8, 3), dtype=np.uint8))]},
        command=DriveCommand.STRAIGHT,
        speed=SPEED_M_S,
        acceleration=0.0,
        ego_pose_history=_driving_straight_poses(latest_us),
        inference_seed=0,
        previous_plan=previous_plan,
        route=None,
    )


def test_always_first_returns_the_first_candidate() -> None:
    candidates = np.stack([_straight_candidate(2.0), _straight_candidate(0.0)])
    model = _StubAlpamayo(candidates, TrajectorySelectionStrategy.ALWAYS_FIRST)

    prediction = model.predict(_prediction_input(2_000_000, previous_plan=None))

    np.testing.assert_allclose(prediction.selected_positions, candidates[0])


def test_previous_plan_selects_the_matching_candidate() -> None:
    """A plan from an earlier cycle keeps the driver on the same candidate."""
    candidates = np.stack(
        [
            _straight_candidate(2.0),
            _straight_candidate(-2.0),
            _straight_candidate(0.0),
        ]
    )
    model = _StubAlpamayo(candidates, TrajectorySelectionStrategy.CLOSEST_LATERAL)
    previous_plan = _plan_following(candidates[2], t0_us=1_500_000)

    first = model.predict(_prediction_input(2_000_000, previous_plan=previous_plan))
    np.testing.assert_allclose(first.selected_positions, candidates[2])
    # The reasoning trace follows the selected candidate.
    assert first.reasoning_text == "reasoning 2"

    # The plan it returns keeps that choice stable in the following cycle.
    second = model.predict(
        _prediction_input(2_500_000, previous_plan=first.selected_plan)
    )
    np.testing.assert_allclose(second.selected_positions, candidates[2])


def test_plan_waypoints_follow_the_latest_camera_frame() -> None:
    candidates = np.stack([_straight_candidate(0.0)])
    model = _StubAlpamayo(candidates, TrajectorySelectionStrategy.ALWAYS_FIRST)
    latest_us = 2_000_000

    prediction = model.predict(_prediction_input(latest_us, previous_plan=None))

    expected_us = latest_us + np.arange(1, NUM_WAYPOINTS + 1) * 100_000
    np.testing.assert_array_equal(prediction.selected_plan.timestamps_us, expected_us)
    # Driving straight along local x, the plan continues from the ego position.
    np.testing.assert_allclose(
        prediction.selected_plan.positions[:, 0],
        SPEED_M_S * latest_us / 1e6 + candidates[0][:, 0],
        atol=1e-3,
    )


def test_selection_strategy_requires_multiple_samples() -> None:
    with pytest.raises(ValueError, match="needs at least 2 samples"):
        _StubAlpamayo(
            np.stack([_straight_candidate(0.0)]),
            TrajectorySelectionStrategy.CLOSEST_3D,
        )


@pytest.mark.parametrize(
    ("window", "match"),
    [
        ({"max_num_distance_points": 0}, "max_num_distance_points must be at least 1"),
        (
            {"skip_first_n_distance_points": -1},
            "skip_first_n_distance_points must not be negative",
        ),
    ],
)
def test_distance_window_must_cover_waypoints(
    window: dict[str, int], match: str
) -> None:
    """An empty or reversed window scores nothing, or the wrong waypoints."""
    with pytest.raises(ValueError, match=match):
        _StubAlpamayo(
            np.stack([_straight_candidate(0.0), _straight_candidate(2.0)]),
            TrajectorySelectionStrategy.CLOSEST_3D,
            **window,
        )


def test_waypoint_step_must_match_the_reported_frequency() -> None:
    """A checkpoint with a different step would be read back as scaled motion."""
    with pytest.raises(ValueError, match="predicts a waypoint every"):
        _StubAlpamayo(
            np.stack([_straight_candidate(0.0)]),
            TrajectorySelectionStrategy.ALWAYS_FIRST,
            waypoint_dt=0.25,
        )


def test_waypoint_step_accepts_a_float32_round_trip() -> None:
    """A step that went through float32 is the same 10 Hz, not a mismatch."""
    _StubAlpamayo(
        np.stack([_straight_candidate(0.0)]),
        TrajectorySelectionStrategy.ALWAYS_FIRST,
        waypoint_dt=float(np.float32(0.1)),
    )
