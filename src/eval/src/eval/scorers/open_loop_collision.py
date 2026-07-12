# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np

from eval.data import (
    AggregationType,
    MetricReturn,
    RenderableTrajectory,
    SimulationResult,
)
from eval.schema import EvalConfig, OpenLoopCollisionScorerConfig
from eval.scorers.base import Scorer


class OpenLoopCollisionScorer(Scorer):
    """Scores whether each planned trajectory collides with logged agents.

    At every inference step the driver returns a planned trajectory. This
    scorer sweeps the ego bounding box along the first ``horizon_s`` seconds of
    that plan and, at each planned future pose, checks whether it intersects the
    recorded bounding box of any surrounding agent at the same absolute time.
    Unlike ``CollisionScorer`` (which tests the executed ego pose), this is an
    open-loop check of the plan against the agents' recorded future motion.

    Metric name: ``open_loop_collision`` — 1.0 at an inference step whose plan
    collides with any agent within the horizon, else 0.0.  Aggregated with MEAN,
    giving the fraction of inference steps that plan into a collision.  Steps
    whose plan has no future pose within the horizon are marked invalid and
    excluded from aggregation.
    """

    def __init__(self, cfg: EvalConfig) -> None:
        super().__init__(cfg)
        scorer_config: OpenLoopCollisionScorerConfig = cfg.scorers.open_loop_collision
        self.horizon_us = round(scorer_config.horizon_s * 1e6)
        if self.horizon_us <= 0:
            raise ValueError(
                "scorers.open_loop_collision.horizon_s must be > 0 (>= 1e-6 s), got "
                f"{scorer_config.horizon_s}"
            )

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        other_trajectories = {
            agent_id: trajectory
            for agent_id, trajectory in simulation_result.actor_trajectories.items()
            if agent_id != "EGO"
        }

        timestamps_us: list[int] = []
        collisions: list[bool] = []
        valids: list[bool] = []

        responses = simulation_result.driver_responses.per_timestep_driver_responses
        for response in responses:
            now_us = response.now_time_us
            plan = response.selected_trajectory
            timestamps_us.append(now_us)

            # The first plan pose is the current ego pose, so only strictly
            # future poses within the horizon form the open-loop check.
            plan_timestamps = np.asarray(plan.timestamps_us)
            in_horizon = (plan_timestamps > now_us) & (
                plan_timestamps <= now_us + self.horizon_us
            )
            future_timestamps = plan_timestamps[in_horizon]
            if len(future_timestamps) == 0:
                collisions.append(False)
                valids.append(False)
                continue

            valids.append(True)
            collisions.append(
                self._plan_collides(plan, future_timestamps, other_trajectories)
            )

        return [
            MetricReturn(
                name="open_loop_collision",
                values=collisions,
                valid=valids,
                timestamps_us=timestamps_us,
                time_aggregation=AggregationType.MEAN,
            )
        ]

    @staticmethod
    def _plan_collides(
        plan: RenderableTrajectory,
        future_timestamps: np.ndarray,
        other_trajectories: dict[str, RenderableTrajectory],
    ) -> bool:
        """Return whether the plan intersects any agent within the horizon.

        For each future plan timestamp the ego box is placed at the planned
        pose and tested against every agent present at that time. Returns on the
        first intersection.
        """
        for ts in future_timestamps:
            ts = int(ts)
            ego_polygon = plan.get_polygon_at_time(ts)
            for trajectory in other_trajectories.values():
                if ts not in trajectory.time_range_us:
                    continue
                if ego_polygon.intersects(trajectory.get_polygon_at_time(ts)):
                    return True
        return False
