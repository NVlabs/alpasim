# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json

import pytest

from eval.aggregation.failed_rollouts import FailedRollout
from eval.aggregation.main import _run_aggregation_core


@pytest.mark.parametrize("scoring", [False, True])
def test_all_failed_cohort_writes_summary(tmp_path, scoring, default_eval_config):
    failure = FailedRollout(
        run_name="run",
        run_uuid="uuid",
        scene_key="clip-a@t0=100",
        rollout_id="attempt",
        error="evaluation failed",
    )
    output = tmp_path / "aggregate"
    default_eval_config.scene_score.enabled = scoring
    result = _run_aggregation_core(
        [tmp_path],
        output,
        default_eval_config,
        failed_rollouts=[failure],
    )
    assert result is None
    summary = json.loads((output / "results-summary.json").read_text())
    assert summary["scene_score_enabled"] is scoring
    assert summary["metrics_results"] == []
    assert len(summary["rollouts"]) == 1
    row = summary["rollouts"][0]
    assert row["scene_key"] == failure.scene_key
    assert row["failure_reason"] == failure.error
    assert row["status"] == "fail"
    if scoring:
        assert row["score"] == 0.0
    else:
        assert "score" not in row


def test_empty_cohort_without_known_failure_raises(tmp_path, default_eval_config):
    with pytest.raises(ValueError, match="No metrics files"):
        _run_aggregation_core([tmp_path], tmp_path / "aggregate", default_eval_config)
