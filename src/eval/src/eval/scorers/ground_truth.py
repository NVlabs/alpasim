# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np
from shapely.geometry import LineString, Point

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.scorers.base import Scorer


def _end_direction(gt_linestring: LineString) -> np.ndarray | None:
    """Final travel direction supported by at least 1 m of displacement.

    A stopped recording can end with duplicate positions or millimetre jitter.
    Look backwards for meaningful motion rather than normalizing that noise.
    Paths confined within 1 m of their endpoint have no supported direction.
    """
    coords = np.asarray(gt_linestring.coords, dtype=np.float64)[:, :2]
    for start_point in coords[-2::-1]:
        direction = coords[-1] - start_point
        norm = float(np.linalg.norm(direction))
        if norm >= 1.0:
            return direction / norm
    return None


def _lateral_distance_to_gt(
    gt_linestring: LineString,
    point: Point,
    end_direction: np.ndarray | None,
) -> float:
    """Distance to the recorded path union its forward endpoint ray.

    The ray separates longitudinal overshoot from lateral departure even when
    endpoint jitter puts the closest recorded point before the final station.
    Ordinary distance still determines the supported aggregation window.
    """
    distance = float(gt_linestring.distance(point))
    if end_direction is None:
        return distance

    end_point = np.asarray(gt_linestring.coords, dtype=np.float64)[-1, :2]
    offset = np.array([point.x, point.y], dtype=np.float64) - end_point
    if float(offset @ end_direction) <= 0:
        return distance
    lateral = float(abs(offset[0] * end_direction[1] - offset[1] * end_direction[0]))
    return min(distance, lateral)


class GroundTruthScorer(Scorer):
    """Scorer for metrics comparing to the ground truth trajectory.

    Adds the following metrics:
    * progress: The progress along the _full_ ground truth trajectory.
    * progress_rel_to_total: Progress for scene scoring, relative to the ground
        truth reachable within the simulated window. May exceed 1.0.
    * progress_rel: The progress along the current ground truth trajectory up to
        the current timestamp. Gives a better sense of progression during the
        simulation.
    * dist_to_gt_trajectory: Projected distance to the ground truth trajectory.
    * dist_to_gt_location: The distance to the ground truth ego location at the
        current timestamp.
    """

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        full_gt_trajectory = simulation_result.ego_recorded_ground_truth_trajectory
        full_gt_linestring = full_gt_trajectory.to_linestring()
        full_gt_distance_traveled_m = full_gt_linestring.length

        # Heuristically set the first two timestamps. For
        # `progress_along_full_gt` we set them at the end by interpolation.
        progress_along_current_gt = [1.0, 1.0]
        progress_along_full_gt = []
        distance_to_gt_trajectory = [0.0, 0.0]
        lateral_distance_to_gt_trajectory = [0.0, 0.0]
        distance_to_current_gt_point = [0.0, 0.0]
        distance_traveled = [0.0, 0.0]
        gt_end_direction = _end_direction(full_gt_linestring)
        corridor_m = self.cfg.aggregation_modifiers.max_dist_to_gt_trajectory
        gt_distance_traveled = [full_gt_distance_traveled_m] * len(
            simulation_result.timestamps_us
        )

        # Skip first two timestamps to avoid errors in shapely's project function
        for idx in range(2, len(simulation_result.timestamps_us)):
            ts = simulation_result.timestamps_us[idx]

            ego_polygon = (
                simulation_result.actor_polygons.get_polygon_for_agent_at_time(
                    "EGO", ts
                )
            )
            ego_trajectory = simulation_result.actor_trajectories["EGO"].to_linestring()
            current_gt_linestring = full_gt_trajectory.interpolate_to_timestamps(
                simulation_result.timestamps_us[: idx + 1]
            ).to_linestring()

            progress_along_full_gt.append(
                full_gt_linestring.project(ego_polygon.centroid, normalized=True)
            )
            progress_along_current_gt.append(
                current_gt_linestring.project(ego_polygon.centroid, normalized=True)
            )
            distance_traveled.append(ego_trajectory.project(ego_polygon.centroid))

            current_gt_point = full_gt_trajectory.interpolate_to_timestamps(
                np.array([ts])
            ).to_point()

            distance_to_current_gt_point.append(
                current_gt_point.distance(ego_polygon.centroid)
            )
            distance_to_gt_trajectory.append(
                full_gt_linestring.distance(ego_polygon.centroid)
            )
            lateral_distance_to_gt_trajectory.append(
                _lateral_distance_to_gt(
                    full_gt_linestring,
                    ego_polygon.centroid,
                    gt_end_direction,
                )
            )

        # Heuristically interpolate the first two timestamps
        if len(progress_along_full_gt) > 0:
            progress_along_full_gt = (
                list(np.linspace(0, progress_along_full_gt[0], 3)[:2])
                + progress_along_full_gt
            )

        # Normalize by the GT the rollout can actually reach: the rollout stops at
        # the last whole control step that fits the recording, so a perfect replay
        # still falls short of the full path -- badly so when the ego is
        # accelerating and most of the path lies in that trailing step.
        reachable_gt_distance_m = full_gt_distance_traveled_m
        if len(simulation_result.timestamps_us) > 0 and full_gt_distance_traveled_m > 0:
            last_gt_point = full_gt_trajectory.interpolate_to_timestamps(
                np.array([simulation_result.timestamps_us[-1]])
            ).to_point()
            reachable_gt_distance_m = float(full_gt_linestring.project(last_gt_point))

        if reachable_gt_distance_m > 0:
            reachable_scale = full_gt_distance_traveled_m / reachable_gt_distance_m
        else:
            reachable_scale = 1.0
        # Deliberately unclipped so out-running the recording stays visible; the
        # scene score clamps to [0, 1] itself.
        progress_rel_to_total = [
            value * reachable_scale for value in progress_along_full_gt
        ]

        return [
            MetricReturn(
                name="progress",
                values=progress_along_full_gt,
                valid=[True] * len(progress_along_full_gt),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.LAST,
            ),
            MetricReturn(
                name="progress_rel_to_total",
                values=progress_rel_to_total,
                valid=[True] * len(progress_rel_to_total),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.LAST,
            ),
            MetricReturn(
                name="progress_rel",
                values=progress_along_current_gt,
                valid=[True] * len(progress_along_current_gt),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MIN,
            ),
            MetricReturn(
                name="dist_to_gt_trajectory",
                values=distance_to_gt_trajectory,
                valid=[True] * len(distance_to_gt_trajectory),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="lateral_dist_to_gt_trajectory",
                values=lateral_distance_to_gt_trajectory,
                valid=[True] * len(lateral_distance_to_gt_trajectory),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="left_corridor_laterally",
                values=[
                    float(value >= corridor_m)
                    for value in lateral_distance_to_gt_trajectory
                ],
                valid=[True] * len(lateral_distance_to_gt_trajectory),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="dist_to_gt_location",
                values=distance_to_current_gt_point,
                valid=[True] * len(distance_to_current_gt_point),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.MAX,
            ),
            MetricReturn(
                name="dist_traveled_m",
                values=distance_traveled,
                valid=[True] * len(distance_traveled),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.LAST,
            ),
            MetricReturn(
                name="gt_dist_traveled_m",
                values=gt_distance_traveled,
                valid=[True] * len(gt_distance_traveled),
                timestamps_us=list(simulation_result.timestamps_us),
                time_aggregation=AggregationType.LAST,
            ),
        ]
