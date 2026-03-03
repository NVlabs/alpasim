# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging

import numpy as np

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.schema import EvalConfig, PlanDeviationScorerConfig
from eval.scorers.base import Scorer

logger = logging.getLogger(__name__)


class PlanDeviationScorer(Scorer):
    """Scorer for plan deviation.

    Measures the consistency of ego's plans by computing the L2 distance between
    planned points for two consecutive timestamps (only on points for which we
    have a prediction at both timestamps).

    For aggregation of the L2 distances of each planned waypoint, we use an
    exponentially decaying average, i.e. weighing planned waypoints closer in
    time higher.

    Configuration:
    * incl_z: Whether to include the z-axis in the plan consistency computation.
    * avg_decay_rate: The decay rate for the plan consistency metric.
    """

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg)
        scorer_config: PlanDeviationScorerConfig = cfg.scorers.plan_deviation

        self.incl_z = scorer_config.incl_z
        self.avg_decay_rate = scorer_config.avg_decay_rate
        self.min_timesteps = scorer_config.min_timesteps

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:

        plan_deviation_results = []
        result_timestamps = []

        for ts_idx in range(1, len(simulation_result.timestamps_us)):
            ts = simulation_result.timestamps_us[ts_idx]
            prev_ts = simulation_result.timestamps_us[ts_idx - 1]

            # First element is current time, so we have to slice it off
            driver_response_pred_at_time = (
                simulation_result.driver_responses.get_driver_response_for_time(
                    ts, "now"
                )
            )
            driver_response_pred_at_prev_time = (
                simulation_result.driver_responses.get_driver_response_for_time(
                    prev_ts, "now"
                )
            )
            if (
                driver_response_pred_at_time is None
                or driver_response_pred_at_prev_time is None
            ):
                continue

            driver_plan_pred_at_time = driver_response_pred_at_time.selected_trajectory
            driver_plan_pred_at_prev_time = (
                driver_response_pred_at_prev_time.selected_trajectory
            )

            # Exclude the first one, which is the actual position at current time, not planned.
            current_plan_timestamps = driver_plan_pred_at_time.timestamps_us[1:]
            # The timesteps might not _exactly_ align, so we find the ones that
            # are the same time-range and interpolate both trajectories to them.
            common_timestamps = current_plan_timestamps[
                current_plan_timestamps
                <= driver_plan_pred_at_prev_time.timestamps_us[-1]
            ]

            driver_waypoints_at_prev_time = np.asarray(
                driver_plan_pred_at_prev_time.interpolate_to_timestamps(
                    common_timestamps
                ).positions
            )
            driver_waypoints_at_time = np.asarray(
                driver_plan_pred_at_time.interpolate_to_timestamps(
                    common_timestamps
                ).positions
            )

            delta_xyz = driver_waypoints_at_time - driver_waypoints_at_prev_time

            distances = np.linalg.norm(
                delta_xyz if self.incl_z else delta_xyz[..., :2], axis=-1
            )

            weights = np.exp(-self.avg_decay_rate * np.arange(len(distances)))
            weighted_avg = np.average(distances, weights=weights)
            plan_deviation_results.append(weighted_avg.item())
            result_timestamps.append(ts)
        return [
            MetricReturn(
                name="plan_deviation",
                values=plan_deviation_results,
                valid=[True] * len(plan_deviation_results),
                timestamps_us=result_timestamps,
                time_aggregation=AggregationType.MEAN,
            ),
        ]
