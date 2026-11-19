# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import inspect
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from eval.runtime_evaluator import RuntimeEvaluator
from eval.schema import FrameQualityScorerConfig


def test_runtime_evaluator_has_no_quality_service_dependency() -> None:
    parameters = inspect.signature(RuntimeEvaluator).parameters

    assert "frame_quality_service" not in parameters
    assert not hasattr(RuntimeEvaluator, "_score_frame_quality")
    assert not hasattr(RuntimeEvaluator, "_add_frame_quality_result")


def test_runtime_quality_config_has_no_client_batch_size() -> None:
    config = FrameQualityScorerConfig()

    assert not hasattr(config, "batch_size")
    assert config.rpc_timeout_s == 60.0
    with pytest.raises(ValueError, match="rpc_timeout_s must be positive"):
        FrameQualityScorerConfig(rpc_timeout_s=0)


@pytest.mark.asyncio
async def test_evaluation_failure_propagates_with_original_cause(
    monkeypatch, tmp_path, default_eval_config
):
    evaluator = RuntimeEvaluator(default_eval_config, "attempt", str(tmp_path), None)
    evaluator.build_eval_input = Mock(return_value=object())
    original = ValueError("scorer failed")

    def fail(**kwargs):
        raise original

    monkeypatch.setattr("eval.runtime_evaluator._evaluate_in_subprocess", fail)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(
            RuntimeError, match="Evaluation failed for session attempt"
        ) as error:
            await evaluator.run_evaluation(executor)
    assert error.value.__cause__ is original


def test_score_only_import_does_not_initialize_video_module():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import eval.runtime_evaluator; import eval.aggregation.main; "
            "assert 'eval.video' not in sys.modules",
        ],
        check=True,
    )
