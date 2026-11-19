# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math

import pytest

from eval.aggregation.scene_score import score_rollout
from eval.schema import SceneScoreConfig


def _score_row(**overrides: object) -> dict[str, object]:
    row = {
        "progress_clipped_rel": 0.4,
        "progress_rel": 0.4,
        "gt_dist_traveled_m": 10.0,
        "collision_at_fault": 0.0,
        "offroad": 0.0,
        "dist_to_gt_trajectory": 0.0,
        "left_corridor_laterally": 0.0,
    }
    row.update(overrides)
    return row


def test_score_rollout_continuous_progress_score() -> None:
    result = score_rollout(_score_row(progress_clipped_rel=0.4), SceneScoreConfig())

    assert result.score == pytest.approx(0.5)
    assert result.progress_score == pytest.approx(0.5)
    assert result.failure_reason is None


def test_score_rollout_short_gt_distance_gets_full_progress_score() -> None:
    result = score_rollout(
        _score_row(progress_clipped_rel=0.0, gt_dist_traveled_m=4.9),
        SceneScoreConfig(),
    )

    assert result.score == 1.0
    assert result.progress_score == 1.0
    assert result.failure_reason is None


def test_score_rollout_deviation_does_not_hard_fail() -> None:
    result = score_rollout(
        _score_row(progress_clipped_rel=0.8, dist_to_gt_trajectory=11.0),
        SceneScoreConfig(),
    )

    assert result.score == 1.0
    assert result.progress_score == 1.0
    assert result.failure_reason is None


@pytest.mark.parametrize(
    "failure_metric",
    ["collision_at_fault", "offroad", "left_corridor_laterally"],
)
def test_score_rollout_collision_or_offroad_hard_fails(failure_metric: str) -> None:
    result = score_rollout(
        _score_row(progress_clipped_rel=0.8, **{failure_metric: 1.0}),
        SceneScoreConfig(),
    )

    assert result.score == 0.0
    assert result.progress_score == 1.0
    assert result.failure_reason == failure_metric


def test_score_rollout_requires_finite_metrics() -> None:
    row = _score_row(progress_clipped_rel=math.nan)

    with pytest.raises(
        ValueError,
        match="metric 'progress_clipped_rel' has non-finite value",
    ):
        score_rollout(row, SceneScoreConfig())


def test_score_rollout_lateral_corridor_exit_hard_fails() -> None:
    result = score_rollout(_score_row(left_corridor_laterally=1.0), SceneScoreConfig())

    assert result.score == 0.0
    assert result.failure_reason == "left_corridor_laterally"


def test_score_rollout_forward_corridor_exit_does_not_fail() -> None:
    # Far from the recording, but only because the ego drove past its end.
    result = score_rollout(
        _score_row(
            progress_clipped_rel=1.0,
            dist_to_gt_trajectory=20.0,
            left_corridor_laterally=0.0,
        ),
        SceneScoreConfig(),
    )

    assert result.failure_reason is None
    assert result.score == 1.0
