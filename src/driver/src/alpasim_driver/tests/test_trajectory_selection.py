# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for selecting one of several sampled trajectory candidates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from alpasim_driver.main import DriveJob, EgoDriverService
from alpasim_driver.models.base import DriveCommand, ModelPrediction, PredictionInput
from alpasim_driver.models.trajectory_selection import (
    MIN_OVERLAP_STEPS,
    TrajectorySelectionStrategy,
    plan_in_local_frame,
    select_trajectory,
)
from alpasim_utils.geometry import Pose, Trajectory

STEP_US = 100_000
NUM_WAYPOINTS = 20


def _yaw_pose(x: float, y: float, yaw: float = 0.0) -> Pose:
    """Pose at (x, y, 0) rotated by *yaw* about Z."""
    return Pose(
        np.array([x, y, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32),
    )


def _waypoint_timestamps(t0_us: int, num_waypoints: int = NUM_WAYPOINTS) -> np.ndarray:
    return t0_us + np.arange(1, num_waypoints + 1, dtype=np.uint64) * STEP_US


def _straight_candidate(
    lateral_offset: float,
    speed_m_s: float = 10.0,
    num_waypoints: int = NUM_WAYPOINTS,
) -> np.ndarray:
    """Waypoints of shape (T, 3) driving forward at a constant lateral offset."""
    forward = speed_m_s * STEP_US / 1e6 * np.arange(1, num_waypoints + 1)
    return np.stack(
        [forward, np.full(num_waypoints, lateral_offset), np.zeros(num_waypoints)],
        axis=-1,
    )


def _identity_rotations(num_waypoints: int = NUM_WAYPOINTS) -> np.ndarray:
    return np.tile(np.eye(3), (num_waypoints, 1, 1))


def _plan(candidate: np.ndarray, t0_us: int, pose_local_to_rig_t0: Pose) -> Trajectory:
    return plan_in_local_frame(
        candidate,
        _identity_rotations(len(candidate)),
        _waypoint_timestamps(t0_us, len(candidate)),
        pose_local_to_rig_t0,
    )


def _select(
    candidates: np.ndarray,
    strategy: TrajectorySelectionStrategy,
    *,
    previous_plan: Trajectory | None,
    t0_us: int = 5_000_000,
    pose_local_to_rig_t0: Pose | None = None,
    max_num_distance_points: int = 64,
    skip_first_n_distance_points: int = 0,
) -> int:
    return select_trajectory(
        candidate_positions=candidates,
        candidate_timestamps_us=_waypoint_timestamps(t0_us, candidates.shape[1]),
        previous_plan_in_local=previous_plan,
        pose_local_to_rig_t0=pose_local_to_rig_t0 or Pose.identity(),
        strategy=strategy,
        max_num_distance_points=max_num_distance_points,
        skip_first_n_distance_points=skip_first_n_distance_points,
    )


class TestAlwaysFirst:
    def test_ignores_previous_plan(self) -> None:
        candidates = np.stack([_straight_candidate(3.0), _straight_candidate(0.0)])
        previous_plan = _plan(_straight_candidate(0.0), 4_000_000, Pose.identity())

        assert (
            _select(
                candidates,
                TrajectorySelectionStrategy.ALWAYS_FIRST,
                previous_plan=previous_plan,
            )
            == 0
        )


class TestClosestToPreviousPlan:
    """Both distance strategies score candidates against the previous plan."""

    @pytest.mark.parametrize(
        "strategy",
        [
            TrajectorySelectionStrategy.CLOSEST_3D,
            TrajectorySelectionStrategy.CLOSEST_LATERAL,
        ],
    )
    def test_first_cycle_selects_first_candidate(
        self, strategy: TrajectorySelectionStrategy
    ) -> None:
        candidates = np.stack([_straight_candidate(3.0), _straight_candidate(0.0)])

        assert _select(candidates, strategy, previous_plan=None) == 0

    @pytest.mark.parametrize(
        "strategy",
        [
            TrajectorySelectionStrategy.CLOSEST_3D,
            TrajectorySelectionStrategy.CLOSEST_LATERAL,
        ],
    )
    def test_selects_candidate_matching_previous_plan(
        self, strategy: TrajectorySelectionStrategy
    ) -> None:
        candidates = np.stack(
            [
                _straight_candidate(3.0),
                _straight_candidate(-2.0),
                _straight_candidate(0.2),
            ]
        )
        previous_plan = _plan(_straight_candidate(0.0), 4_500_000, Pose.identity())

        assert _select(candidates, strategy, previous_plan=previous_plan) == 2

    def test_previous_plan_is_compared_in_the_current_rig_frame(self) -> None:
        """The previous plan is stored in the local frame and must be moved back.

        The ego has driven 5 m and turned left since the previous cycle, so the
        previous plan runs to the right of the current heading and the candidate
        with a matching right-hand offset wins.
        """
        previous_plan = _plan(
            _straight_candidate(0.0), 4_000_000, pose_local_to_rig_t0=Pose.identity()
        )
        candidates = np.stack([_straight_candidate(0.0), _straight_candidate(-3.0)])

        assert (
            _select(
                candidates,
                TrajectorySelectionStrategy.CLOSEST_3D,
                previous_plan=previous_plan,
                t0_us=5_000_000,
                pose_local_to_rig_t0=_yaw_pose(5.0, 0.0, yaw=0.3),
            )
            == 1
        )

    def test_lateral_strategy_ignores_longitudinal_distance(self) -> None:
        """A slower candidate on the previous path beats a faster one beside it."""
        previous_plan = _plan(
            _straight_candidate(0.0, speed_m_s=10.0), 4_500_000, Pose.identity()
        )
        candidates = np.stack(
            [
                _straight_candidate(0.0, speed_m_s=4.0),
                _straight_candidate(1.5, speed_m_s=10.0),
            ]
        )

        assert (
            _select(
                candidates,
                TrajectorySelectionStrategy.CLOSEST_LATERAL,
                previous_plan=previous_plan,
            )
            == 0
        )
        assert (
            _select(
                candidates,
                TrajectorySelectionStrategy.CLOSEST_3D,
                previous_plan=previous_plan,
            )
            == 1
        )

    def test_distance_window_restricts_the_compared_waypoints(self) -> None:
        """Only waypoints inside the window influence the choice."""
        previous_plan = _plan(_straight_candidate(0.0), 4_500_000, Pose.identity())
        # Candidate 0 matches the previous plan early then diverges; candidate 1
        # does the opposite.
        early_match = _straight_candidate(0.0)
        early_match[5:, 1] = 4.0
        late_match = _straight_candidate(4.0)
        late_match[5:, 1] = 0.0
        candidates = np.stack([early_match, late_match])

        assert (
            _select(
                candidates,
                TrajectorySelectionStrategy.CLOSEST_LATERAL,
                previous_plan=previous_plan,
                max_num_distance_points=5,
            )
            == 0
        )
        assert (
            _select(
                candidates,
                TrajectorySelectionStrategy.CLOSEST_LATERAL,
                previous_plan=previous_plan,
                skip_first_n_distance_points=5,
            )
            == 1
        )

    def test_rejects_previous_plan_that_barely_overlaps(self) -> None:
        t0_us = 5_000_000
        candidates = np.stack([_straight_candidate(0.0), _straight_candidate(3.0)])
        # The previous plan ends just after the candidates start.
        previous_t0_us = t0_us - (NUM_WAYPOINTS - MIN_OVERLAP_STEPS + 1) * STEP_US
        previous_plan = _plan(_straight_candidate(0.0), previous_t0_us, Pose.identity())

        with pytest.raises(ValueError, match="overlaps the current candidates"):
            _select(
                candidates,
                TrajectorySelectionStrategy.CLOSEST_3D,
                previous_plan=previous_plan,
                t0_us=t0_us,
            )

    def test_rejects_previous_plan_from_the_future(self) -> None:
        candidates = np.stack([_straight_candidate(0.0), _straight_candidate(3.0)])
        previous_plan = _plan(_straight_candidate(0.0), 6_000_000, Pose.identity())

        with pytest.raises(ValueError, match="planning time must advance"):
            _select(
                candidates,
                TrajectorySelectionStrategy.CLOSEST_3D,
                previous_plan=previous_plan,
                t0_us=5_000_000,
            )


class TestSessionPlanHandoff:
    """The servicer feeds the returned plan back into the next inference."""

    class _PlanningModel:
        """Model that returns a fresh plan and records what it was given."""

        def __init__(self) -> None:
            self.seen_previous_plans: list[Trajectory | None] = []
            self.plans = [
                _plan(_straight_candidate(0.0), t0_us, Pose.identity())
                for t0_us in (1_000_000, 2_000_000)
            ]

        def predict_batch(self, inputs: list[PredictionInput]) -> list[ModelPrediction]:
            self.seen_previous_plans.extend(inp.previous_plan for inp in inputs)
            return [
                ModelPrediction(
                    candidate_positions=np.zeros((1, 0, 3)),
                    candidate_rotations=np.zeros((1, 0, 3, 3)),
                    selected_plan=self.plans[len(self.seen_previous_plans) - 1],
                )
                for _ in inputs
            ]

    def test_plan_is_carried_into_the_next_inference(self) -> None:
        service = EgoDriverService.__new__(EgoDriverService)
        model = self._PlanningModel()
        service._model = model
        service._get_speed_and_acceleration = lambda session: (0.0, 0.0)
        service._prepare_camera_images = lambda session: {}
        session = SimpleNamespace(
            seed=0,
            inference_count=0,
            poses=[],
            last_selected_plan=None,
            route=None,
            frames_trail_request_warned=False,
        )

        def _job() -> DriveJob:
            return DriveJob(
                session_id="session",
                session=session,
                command=DriveCommand.STRAIGHT,
                pose=None,
                timestamp_us=0,
                result=None,  # type: ignore[arg-type]
            )

        service._run_batch([_job()])
        service._run_batch([_job()])

        assert model.seen_previous_plans == [None, model.plans[0]]
        assert session.last_selected_plan is model.plans[1]


class TestPlanInLocalFrame:
    def test_places_waypoints_relative_to_the_rig_pose(self) -> None:
        candidate = _straight_candidate(0.0, num_waypoints=4)
        # Rig heading 90 degrees left of the local x axis, 10 m along y.
        pose_local_to_rig_t0 = _yaw_pose(0.0, 10.0, yaw=np.pi / 2)

        plan = _plan(candidate, 1_000_000, pose_local_to_rig_t0)

        # Forward motion in the rig frame becomes +y motion in the local frame.
        np.testing.assert_allclose(plan.positions[:, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(
            plan.positions[:, 1], 10.0 + candidate[:, 0], atol=1e-4
        )
        np.testing.assert_array_equal(
            plan.timestamps_us, _waypoint_timestamps(1_000_000, len(candidate))
        )
