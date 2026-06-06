# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Helpers for representing rollout failures in aggregation outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FailedRollout:
    run_name: str | None
    run_uuid: str | None
    clipgt_id: str
    rollout_id: str
    error: str | None


FailedRolloutInput = FailedRollout | Mapping[str, object]


def _get_failure_value(failure: FailedRolloutInput, key: str) -> object:
    if isinstance(failure, FailedRollout):
        return getattr(failure, key, None)
    return failure.get(key)


def failed_rollout_summary_rows(
    failed_rollouts: Iterable[FailedRolloutInput] | None,
    *,
    scene_score_enabled: bool = True,
) -> list[dict[str, object]]:
    if not failed_rollouts:
        return []

    rows = []
    for idx, failure in enumerate(failed_rollouts):
        error = _get_failure_value(failure, "error") or _get_failure_value(
            failure,
            "failure_reason",
        )
        run_uuid = _get_failure_value(failure, "run_uuid")
        run_name = _get_failure_value(failure, "run_name")
        clipgt_id = _get_failure_value(failure, "clipgt_id")
        rollout_id = _get_failure_value(failure, "rollout_id") or f"failed-{idx}"
        metrics = {
            "run_name": run_name,
            "run_uuid": run_uuid,
            "clipgt_id": clipgt_id,
            "rollout_id": rollout_id,
            "error": error,
        }
        row = {
            "run_uuid": run_uuid,
            "run_name": run_name,
            "clipgt_id": clipgt_id,
            "rollout_id": rollout_id,
            "status": "fail",
            "passed": False,
            "failure_reason": error,
            "metrics": metrics,
        }
        if scene_score_enabled:
            row.update(
                {
                    "score": 0.0,
                    "score_metrics": {
                        "progress_clipped_rel": None,
                        "progress_rel": None,
                        "progress_score": 0.0,
                        "collision_at_fault": None,
                        "offroad": None,
                        "dist_to_gt_trajectory": None,
                        "gt_dist_traveled_m": None,
                    },
                }
            )
        rows.append(row)
    return rows
