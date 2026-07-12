# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Behavioral tests for the open-loop collision scorer.

Builds real ``RenderableTrajectory`` / ``DriverResponseAtTime`` objects (the
types the scorer computes on) and a light stand-in for the surrounding
``SimulationResult`` fields it reads.
"""

from types import SimpleNamespace

import numpy as np

from eval.data import RAABB, DriverResponseAtTime, RenderableTrajectory
from eval.scorers.open_loop_collision import OpenLoopCollisionScorer

# Identity orientation in the (x, y, z, w) convention used by the geometry
# layer: yaw 0, so a box's length runs along +x and width along +y.
_IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def _raabb() -> RAABB:
    # No corner rounding so boxes are exact 4.5 x 2.0 rectangles.
    return RAABB(size_x=4.5, size_y=2.0, size_z=1.5, corner_radius_m=0.0)


def _trajectory(
    timestamps_us: list[int], xy: list[tuple[float, float]]
) -> RenderableTrajectory:
    positions = np.array([[x, y, 0.0] for x, y in xy], dtype=np.float32)
    quaternions = np.array([_IDENTITY_QUAT] * len(xy), dtype=np.float32)
    return RenderableTrajectory(
        timestamps_us=np.array(timestamps_us, dtype=np.uint64),
        positions=positions,
        quaternions=quaternions,
        raabb=_raabb(),
    )


def _driver_response(now_us: int, plan: RenderableTrajectory) -> DriverResponseAtTime:
    return DriverResponseAtTime(
        now_time_us=now_us,
        time_query_us=now_us,
        selected_trajectory=plan,
        sampled_trajectories=[],
    )


def _scorer(horizon_s: float) -> OpenLoopCollisionScorer:
    cfg = SimpleNamespace(
        scorers=SimpleNamespace(
            open_loop_collision=SimpleNamespace(horizon_s=horizon_s)
        )
    )
    return OpenLoopCollisionScorer(cfg)


def _sim_result(
    responses: list[DriverResponseAtTime], actor_trajectories: dict
) -> SimpleNamespace:
    return SimpleNamespace(
        actor_trajectories=actor_trajectories,
        driver_responses=SimpleNamespace(per_timestep_driver_responses=responses),
    )


def test_flags_step_whose_plan_drives_into_a_logged_agent() -> None:
    """A plan passing through a static agent within the horizon scores 1.0."""
    # Static agent parked at x=20; ego plan drives straight through x=20 at t=2s.
    agent = _trajectory([0, 1_000_000, 2_000_000, 3_000_000], [(20.0, 0.0)] * 4)
    plan = _trajectory(
        [0, 1_000_000, 2_000_000, 3_000_000],
        [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
    )
    sim_result = _sim_result([_driver_response(0, plan)], {"agent": agent})

    [metric] = _scorer(horizon_s=3.0).calculate(sim_result)

    assert metric.name == "open_loop_collision"
    assert metric.values == [True]
    assert metric.valid == [True]
    assert metric.aggregate() == 1.0


def test_clean_plan_and_horizon_clipping() -> None:
    """A plan that only collides after the horizon scores 0.0."""
    # Agent parked at x=20. The plan is clear within a 1s horizon (only reaches
    # x=10) but would collide at t=2s, so a 1s horizon must still score 0.
    agent = _trajectory([0, 1_000_000, 2_000_000, 3_000_000], [(20.0, 0.0)] * 4)
    plan = _trajectory(
        [0, 1_000_000, 2_000_000, 3_000_000],
        [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
    )
    sim_result = _sim_result([_driver_response(0, plan)], {"agent": agent})

    [metric] = _scorer(horizon_s=1.0).calculate(sim_result)

    assert metric.values == [False]
    assert metric.valid == [True]


def test_mean_aggregates_collision_rate() -> None:
    """MEAN over steps gives the fraction of plans that collide."""
    agent = _trajectory(
        [0, 1_000_000, 2_000_000, 3_000_000, 4_000_000], [(20.0, 0.0)] * 5
    )
    colliding_plan = _trajectory(
        [0, 1_000_000, 2_000_000], [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    )
    # Second plan swerves to y=10, clearing the agent (width 2.0 -> |y|<=1).
    clean_plan = _trajectory(
        [1_000_000, 2_000_000, 3_000_000, 4_000_000],
        [(10.0, 0.0), (20.0, 10.0), (30.0, 10.0), (40.0, 10.0)],
    )
    sim_result = _sim_result(
        [_driver_response(0, colliding_plan), _driver_response(1_000_000, clean_plan)],
        {"agent": agent},
    )

    [metric] = _scorer(horizon_s=3.0).calculate(sim_result)

    assert metric.values == [True, False]
    assert metric.timestamps_us == [0, 1_000_000]
    assert metric.aggregate() == 0.5


def test_ego_actor_is_skipped() -> None:
    """The ego's own recorded trajectory is never treated as an obstacle.

    The plan drives straight along the ego's OWN recorded path, so the only
    box overlap is ego-vs-ego. With EGO the sole actor, the metric must be 0.0;
    it would be 1.0 if the ``agent_id != "EGO"`` filter were dropped.
    """
    ego_recorded = _trajectory(
        [0, 1_000_000, 2_000_000], [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    )
    plan = _trajectory(
        [0, 1_000_000, 2_000_000], [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    )
    sim_result = _sim_result([_driver_response(0, plan)], {"EGO": ego_recorded})

    [metric] = _scorer(horizon_s=3.0).calculate(sim_result)

    assert metric.values == [False]
    assert metric.valid == [True]


def test_step_without_future_plan_is_invalid() -> None:
    """A plan holding only the current pose cannot be scored and is invalid."""
    agent = _trajectory([0, 1_000_000], [(20.0, 0.0)] * 2)
    plan = _trajectory([0], [(0.0, 0.0)])
    sim_result = _sim_result([_driver_response(0, plan)], {"agent": agent})

    [metric] = _scorer(horizon_s=3.0).calculate(sim_result)

    assert metric.valid == [False]
    assert np.isnan(metric.aggregate())
