# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math
from dataclasses import dataclass

from eval.schema import SceneScoreConfig

REQUIRED_SCENE_SCORE_METRICS = (
    "progress_clipped_rel",
    "gt_dist_traveled_m",
    "collision_at_fault",
    "offroad",
)


@dataclass(frozen=True)
class SceneScoreResult:
    score: float
    progress_score: float
    failure_reason: str | None


def _required_finite_float(row: dict[str, object], key: str) -> float:
    if key not in row:
        raise ValueError(f"Cannot compute scene score: missing metric '{key}'.")
    value = row[key]
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"Cannot compute scene score: metric '{key}' has non-numeric value "
            f"{value!r}."
        )
    value_float = float(value)
    if math.isnan(value_float) or math.isinf(value_float):
        raise ValueError(
            f"Cannot compute scene score: metric '{key}' has non-finite value "
            f"{value!r}."
        )
    return value_float


def _required_scene_score_values(row: dict[str, object]) -> dict[str, float]:
    return {
        key: _required_finite_float(row, key) for key in REQUIRED_SCENE_SCORE_METRICS
    }


def _progress_score(
    values: dict[str, float],
    scene_score_config: SceneScoreConfig,
) -> float:
    if (
        values["gt_dist_traveled_m"]
        < scene_score_config.min_gt_distance_for_full_score_m
    ):
        return 1.0

    progress_clipped_rel = min(max(values["progress_clipped_rel"], 0.0), 1.0)
    return min(
        progress_clipped_rel / scene_score_config.progress_saturation_threshold,
        1.0,
    )


def _hard_failure_reason(
    values: dict[str, float],
) -> str | None:
    if values["collision_at_fault"] != 0.0:
        return "collision_at_fault"
    if values["offroad"] != 0.0:
        return "offroad"
    return None


def score_rollout(
    row: dict[str, object],
    scene_score_config: SceneScoreConfig,
) -> SceneScoreResult:
    values = _required_scene_score_values(row)
    progress_score = _progress_score(values, scene_score_config)
    failure_reason = _hard_failure_reason(values)
    if failure_reason is not None:
        return SceneScoreResult(
            score=0.0,
            progress_score=progress_score,
            failure_reason=failure_reason,
        )
    return SceneScoreResult(
        score=progress_score,
        progress_score=progress_score,
        failure_reason=None,
    )
