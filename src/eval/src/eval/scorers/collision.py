# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.scorers.base import Scorer


class CollisionScorer(Scorer):
    """Scorer for collision metrics.

    Adds the following metrics:
    * collision_front: The front of the ego bbox intersects with the other bbox
    * collision_rear: The rear of the ego bbox intersects with the other bbox
    * collision_lateral: The ego bbox intersects with the other, but neither
        front nor rear
    * collision_any: Any of the above
    """

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        """Calculate metrics for entire trajectory."""

        front_collisions = []
        rear_collisions = []
        lateral_collisions = []

        for ts in simulation_result.timestamps_us:
            timestep_polygons = simulation_result.actor_polygons.get_polygons_at_time(
                ts
            )
            srt_tree = timestep_polygons.str_tree
            ego_polygon = timestep_polygons.get_polygon_for_agent("EGO")
            ego_front_bumper = timestep_polygons.get_front_bumper_line_for_agent("EGO")
            ego_rear_bumper = timestep_polygons.get_rear_bumper_line_for_agent("EGO")
            intersecting_indices = [
                i
                for i in srt_tree.query(ego_polygon, predicate="intersects")
                if i != timestep_polygons.get_idx_for_agent("EGO")
            ]
            if len(intersecting_indices) == 0:
                front_collisions.append(False)
                rear_collisions.append(False)
                lateral_collisions.append(False)
                continue

            for idx in intersecting_indices:
                other_polygon = timestep_polygons.bbox_polygons[idx]
                front_collision, rear_collision, lateral_collision = False, False, False
                if other_polygon.intersects(ego_front_bumper):
                    front_collision = True
                elif other_polygon.intersects(ego_rear_bumper):
                    rear_collision = True
                else:
                    lateral_collision = True
            front_collisions.append(front_collision)
            rear_collisions.append(rear_collision)
            lateral_collisions.append(lateral_collision)
        valids = [True] * len(front_collisions)
        any_collision = [
            any([front, rear, lateral])
            for front, rear, lateral in zip(
                front_collisions, rear_collisions, lateral_collisions
            )
        ]

        return [
            MetricReturn(
                name="collision_front",
                values=front_collisions,
                valid=valids,
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="collision_rear",
                values=rear_collisions,
                valid=valids,
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="collision_lateral",
                values=lateral_collisions,
                valid=valids,
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="collision_any",
                values=any_collision,
                valid=valids,
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
        ]
