# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
ASL file loader that produces ScenarioEvalInput.

This module provides functionality to load ASL (Alpasim Simulation Log) files
and convert them to ScenarioEvalInput, enabling a unified evaluation code path
for both post-eval (from ASL files) and runtime-eval (from memory).
"""

import logging
from typing import Any

from alpasim_utils.artifact import Artifact
from alpasim_utils.logs import async_read_pb_log, clip_id_from_session_metadata

from eval.accumulator import EvalDataAccumulator
from eval.data import ScenarioEvalInput
from eval.schema import EvalConfig

logger = logging.getLogger("alpasim_eval.asl_loader")


async def load_scenario_eval_input_from_asl(
    asl_file_path: str,
    cfg: EvalConfig,
    artifacts: dict[str, Artifact],
    run_metadata: dict[str, Any],
) -> ScenarioEvalInput:
    """
    Load an ASL file and return ScenarioEvalInput for evaluation.

    This function provides a unified loading path for ASL files, returning
    a ScenarioEvalInput that can be evaluated using ScenarioEvaluator.evaluate().

    Args:
        asl_file_path: Path to the ASL file to load.
        cfg: Evaluation configuration.
        artifacts: Artifacts (maps, etc.) keyed by clip ID.
        run_metadata: Run metadata containing run_uuid and run_name.

    Returns:
        ScenarioEvalInput ready for evaluation via ScenarioEvaluator.
    """
    # Create accumulator to process messages
    accumulator = EvalDataAccumulator(cfg=cfg)

    # Feed all messages to accumulator
    async for message in async_read_pb_log(asl_file_path):
        accumulator.handle_message(message)

    vec_map = None
    if accumulator.session_metadata is not None:
        clip_id = clip_id_from_session_metadata(accumulator.session_metadata)
        if clip_id in artifacts:
            vec_map = artifacts[clip_id].map

    # Build and return ScenarioEvalInput
    return accumulator.build_scenario_eval_input(
        run_uuid=run_metadata["run_uuid"],
        run_name=run_metadata["run_name"],
        vec_map=vec_map,
    )
