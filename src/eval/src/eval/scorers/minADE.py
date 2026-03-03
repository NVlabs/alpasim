# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging

import numpy as np

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.schema import EvalConfig, MinADEScorerConfig, MinADEScorerTarget
from eval.scorers.base import Scorer

logger = logging.getLogger(__name__)


class MinADEScorer(Scorer):
    """Scorer for minimum average displacement error (minADE).

    Configuration:
    * time_deltas: The time deltas to compute minADE for.
    * incl_z: Whether to include the z-axis in the minADE computation.
    * target: The target to compute minADE for. Can be `AUTO`, `GT` or `SELF`.
        * `GT`: This is the 'normal' minADE metric comparing to the ground truth
            ego trajectory. However, this is pretty meaningless unless the
            driver exactly follows the logs - and hence makes predictions
            comparable to the ground truth.
        * `SELF`: This is the minADE metric comparing to the _simulated_ ego
            trajectory. If we don't use log-replay, this gives a sense of how
            well the model actually followed through on it's plans.
        * `AUTO`: Automatically select the target based on the Hausdorff
            distance between the ego trajectory and the ground truth trajectory.
            I.e. for small distance, `GT` is selected, otherwise `SELF`.
    * auto_target_threshold: The threshold for the auto target selection.
    """

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg)
        scorer_config: MinADEScorerConfig = cfg.scorers.min_ade

        self.time_deltas = scorer_config.time_deltas
        self.incl_z = scorer_config.incl_z
        self.target = scorer_config.target

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:

        min_ade_results = {
            f"min_ade@{time_delta}s": {
                "values": [],
                "timestamps_us": [],
                "valid": [],
            }
            for time_delta in self.time_deltas
        }

        for ts_idx, ts in enumerate(simulation_result.timestamps_us):
            # First element is current time
            driver_response_at_time = (
                simulation_result.driver_responses.get_driver_response_for_time(
                    ts, "now"
                )
            )
            if driver_response_at_time is None:
                continue

            # Index into the resulting ade array created at current time.
            # Need subtract additionally -1 because we exclude the current timestamp
            ade_timestamps_idx_delta = np.array(
                [
                    np.searchsorted(simulation_result.timestamps_us, ts + delta * 1e6)
                    - ts_idx
                    - 1
                    for delta in self.time_deltas
                ]
            )
            # Max timestamp for which we need to query the ground truth trajectory
            max_ts = min(
                ts + max(self.time_deltas) * 1e6, simulation_result.timestamps_us[-1]
            )
            # Filter timestamps to be in relevant range. Note that we exclude
            # the current timestamp and start at the query time.
            filtered_timestamps = [
                t
                for t in simulation_result.timestamps_us
                if driver_response_at_time.time_query_us <= t <= max_ts
            ]

            if self.target == MinADEScorerTarget.GT:
                gt_trajectory = simulation_result.ego_recorded_ground_truth_trajectory
                relevant_comparison_trajectory_part = (
                    gt_trajectory.interpolate_to_timestamps(
                        np.array(filtered_timestamps)
                    )
                )
            elif self.target == MinADEScorerTarget.SELF:
                # Alternative: Use the simulated EGO trajectory
                relevant_comparison_trajectory_part = (
                    simulation_result.actor_trajectories[
                        "EGO"
                    ].interpolate_to_timestamps(np.array(filtered_timestamps))
                )
            else:
                raise NotImplementedError(f"Invalid target: {self.target}")

            # Skip if no sampled trajectories are available
            if len(driver_response_at_time.sampled_trajectories) == 0:
                continue

            relevant_sampled_trajectory_waypoints = np.array(
                [
                    np.asarray(
                        sampled_trajectory.interpolate_to_timestamps(
                            np.array(filtered_timestamps)
                        ).positions
                    )
                    for sampled_trajectory in driver_response_at_time.sampled_trajectories
                ]
            )

            # [nr_samples, T, 3]
            delta_xyz = (
                relevant_sampled_trajectory_waypoints
                - relevant_comparison_trajectory_part.positions[None]
            )
            # [nr_samples, T]
            distances = np.linalg.norm(
                delta_xyz if self.incl_z else delta_xyz[:, :, :2],
                axis=-1,
            )

            # [nr_samples, T]
            cumulative_avg = (
                np.cumsum(distances, axis=-1)
                / np.arange(1, distances.shape[-1] + 1)[None]
            )
            # [T]
            cumulative_avg_min_over_samples = np.min(cumulative_avg, axis=0)

            # At the end of the trajectory, we don't have the full gt anymore,
            # need to shorted the minADE computation.
            ade_timestamps_idx_delta = ade_timestamps_idx_delta[
                ade_timestamps_idx_delta < cumulative_avg.shape[1]
            ]
            for time_delta, idx_delta in zip(
                self.time_deltas, ade_timestamps_idx_delta
            ):
                min_ade_results[f"min_ade@{time_delta}s"]["values"].append(
                    cumulative_avg_min_over_samples[idx_delta].item()
                )
                min_ade_results[f"min_ade@{time_delta}s"]["timestamps_us"].append(ts)
                min_ade_results[f"min_ade@{time_delta}s"]["valid"].append(True)

        metric_returns = []
        for name, result in min_ade_results.items():
            metric_returns.append(
                MetricReturn(
                    name=name + f"({self.target})",
                    values=result["values"],
                    valid=result["valid"],
                    timestamps_us=result["timestamps_us"],
                    time_aggregation=AggregationType.MEAN,
                )
            )

        return metric_returns
