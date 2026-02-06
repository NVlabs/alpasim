# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Provides a ScenarioEvaluator class that can be called externally by the runtime
to compute metrics for a completed scenario without needing to read from disk.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import polars as pl

from eval.data import (
    MetricReturn,
    ScenarioEvalInput,
    SimulationResult,
    create_metrics_dataframe,
)
from eval.schema import EvalConfig
from eval.scorers import create_scorer_group

logger = logging.getLogger("alpasim_eval")


@dataclasses.dataclass
class ScenarioEvalResult:
    """
    Result of evaluating a scenario.

    Contains both per-timestep metrics and aggregated metrics.
    """

    # Per-timestep metrics
    timestep_metrics: list[MetricReturn]

    # Aggregated metrics (dict of metric_name -> aggregated_value)
    aggregated_metrics: dict[str, float]

    # Raw polars dataframe with all metric data (for further processing)
    metrics_df: Optional[pl.DataFrame] = None


class ScenarioEvaluator:
    """
    Evaluates a completed scenario and returns metrics.

    This class provides a simple interface for the runtime to compute metrics
    for a completed scenario without needing to read data from disk.

    Example usage:

        # Create evaluator with config
        evaluator = ScenarioEvaluator(eval_config)

        # From runtime, after scenario completion:
        result = evaluator.evaluate(scenario_input)

        # Access results
        print(result.timestep_metrics)  # List[MetricReturn]
        print(result.aggregated_metrics)  # {"collision_any": 0.0, ...}
    """

    def __init__(self, cfg: EvalConfig) -> None:
        """
        Initialize the evaluator with configuration.

        Args:
            cfg: Evaluation configuration (controls which metrics to compute,
                 vehicle parameters, etc.)
        """
        self.cfg = cfg
        self._scorer_group = create_scorer_group(cfg)

    def evaluate(self, scenario_input: ScenarioEvalInput) -> ScenarioEvalResult:
        """
        Evaluate a completed scenario from ScenarioEvalInput data.

        Args:
            scenario_input: Input data containing trajectories, metadata, etc.

        Returns:
            ScenarioEvalResult with per-timestep metrics and aggregated metrics.

        Note:
            If you need the SimulationResult (e.g., for video rendering), use
            SimulationResult.from_scenario_input(scenario_input, self.cfg) directly.
        """
        # Convert input to SimulationResult using factory method
        simulation_result = SimulationResult.from_scenario_input(
            scenario_input, self.cfg
        )

        # Compute timestep metrics directly using the scorer group
        timestep_metrics = self._scorer_group.calculate(simulation_result)

        # Aggregate metrics over time
        aggregated_metrics = {m.name: m.aggregate() for m in timestep_metrics}

        # Create metrics dataframe with run metadata for aggregation
        clipgt_id = simulation_result.session_metadata.scene_id
        rollout_id = simulation_result.session_metadata.session_uuid
        metrics_df = create_metrics_dataframe(
            metric_results=timestep_metrics,
            clipgt_id=clipgt_id,
            batch_id=scenario_input.batch_id,
            rollout_id=rollout_id,
            run_uuid=scenario_input.run_uuid,
            run_name=scenario_input.run_name,
        )

        return ScenarioEvalResult(
            timestep_metrics=timestep_metrics,
            aggregated_metrics=aggregated_metrics,
            metrics_df=metrics_df,
        )
