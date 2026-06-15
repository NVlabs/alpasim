# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.scorers.base import Scorer


class SafetyScorer(Scorer):
    """Scorer for safety metrics (currently from the optional safety monitor output).

    Adds the following metrics:
    * safety_monitor_triggered: Indication of the safety monitor triggering
    """

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        """Calculate metrics for entire trajectory."""

        triggers = []

        response_timestamps = simulation_result.driver_responses.timestamps_us
        for ts in response_timestamps:
            driver_response_at_time = (
                simulation_result.driver_responses.get_driver_response_for_time(
                    ts, "now"
                )
            )
            if (driver_response_at_time is None) or (
                driver_response_at_time.safety_monitor_safe is None
            ):
                triggers.append(False)
            else:
                triggers.append(not (driver_response_at_time.safety_monitor_safe))

        return [
            MetricReturn(
                name="safety_monitor_triggered",
                values=triggers,
                valid=[True] * len(triggers),
                timestamps_us=list(response_timestamps),
                time_aggregation=AggregationType.MAX,
            )
        ]
