# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.scorers.base import Scorer

_NO_ACTORS_SENTINEL = float("inf")


class MinDistanceToObstacleScorer(Scorer):
    """Computes minimum Euclidean distance from ego to any other actor at each timestep.

    Metric name: ``min_distance_to_obstacle_m``

    The per-timestep value is 0.0 when ego intersects or touches any actor
    (i.e. a collision is occurring) and positive otherwise.  Timesteps with no
    other actors present are marked invalid and excluded from aggregation.

    Aggregated with MIN over the episode, giving the closest the ego came to
    any obstacle.
    """

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        distances: list[float] = []
        valids: list[bool] = []

        for ts in simulation_result.timestamps_us:
            timestep_polygons = simulation_result.actor_polygons.get_polygons_at_time(
                ts
            )
            ego_polygon = timestep_polygons.get_polygon_for_agent("EGO")
            ego_idx = timestep_polygons.get_idx_for_agent("EGO")

            non_ego_indices = [
                i for i in range(len(timestep_polygons.bbox_polygons)) if i != ego_idx
            ]

            if not non_ego_indices:
                distances.append(_NO_ACTORS_SENTINEL)
                valids.append(False)
                continue

            min_dist = min(
                ego_polygon.distance(timestep_polygons.bbox_polygons[i])
                for i in non_ego_indices
            )
            distances.append(min_dist)
            valids.append(True)

        return [
            MetricReturn(
                name="min_distance_to_obstacle_m",
                values=distances,
                valid=valids,
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MIN,
            )
        ]
