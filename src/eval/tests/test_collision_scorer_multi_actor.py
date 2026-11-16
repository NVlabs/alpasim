# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Regression coverage for simultaneous multi-actor collision attribution.

Re-initializing flags inside the actor loop retained only the category of the
last actor yielded by the spatial tree. These checks verify category retention
(both front and rear contacts persist) and permutation invariance (query order
cannot change collision flags or downstream at-fault attribution).
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl

from eval.aggregation.processing import DEFAULT_MODIFIERS
from eval.data import RAABB, ActorPolygons, RenderableTrajectory
from eval.scorers.collision import CollisionScorer


def test_front_and_rear_contacts_are_order_invariant(default_eval_config) -> None:
    """Retain both contacts and identical at-fault penalties in either order.

    At t=0 s, the front contact makes native at-fault attribution 1.0 even when
    a rear contact is processed afterward. At t=1 s, both actors have moved
    clear of the ego vehicle, so all flags reset and at-fault returns to 0.0.
    """
    # Timestamps are in microseconds: simultaneous contacts at 0 s, clear at 1 s.
    timestamps = np.array([0, 1_000_000], dtype=np.uint64)

    def trajectory(x: float) -> RenderableTrajectory:
        return RenderableTrajectory(
            timestamps_us=timestamps,
            # Other actors move away at 1 s; ego (x=0) stays stationary.
            positions=np.array([[x, 0.0, 0.0], [10 * x, 0.0, 0.0]]),
            # Identity rotation: the vehicle's front points along positive x.
            quaternions=np.array([[0.0, 0.0, 0.0, 1.0]] * 2),
            # Length, width, height and corner radius, in metres.
            raabb=RAABB(2.0, 1.0, 1.0, 0.0),
        )

    # Ego spans x=[-1, 1]. The actors at +/-1.5 overlap its respective
    # bumpers by 0.5 m; at 1 s their centres are at +/-15 m, clear of ego.
    actors = {
        "EGO": trajectory(0.0),
        "front": trajectory(1.5),
        "rear": trajectory(-1.5),
    }
    # Use the production rule: front OR lateral contact is at fault; rear-only
    # contact is not. Losing the front flag would therefore change attribution.
    at_fault = next(
        m for m in DEFAULT_MODIFIERS if getattr(m, "name", None) == "collision_at_fault"
    )
    results = []
    for order in [("front", "rear"), ("rear", "front")]:
        polygons = ActorPolygons.from_actor_trajectories(
            {name: actors[name] for name in ("EGO", *order)}
        )
        contacts = polygons.get_polygons_at_time(0)
        tree = contacts.str_tree
        hits = tree.query(contacts.get_polygon_for_agent("EGO"), predicate="intersects")
        # Real geometry must detect both actors and ego itself. The scorer
        # excludes ego (index 0) before classifying the other contacts.
        assert set(hits) == {0, 1, 2}

        # STRtree traversal need not follow insertion order. Reorder real hits
        # so both actor visitation orders are exercised on every GEOS version.
        # Sorted indices visit front/rear in the first run and rear/front in
        # the second. Only the t=0 query order is controlled; geometry and the
        # separate t=1 spatial tree remain unchanged.
        with patch.object(tree, "query", return_value=np.sort(hits)):
            metrics = CollisionScorer(default_eval_config).calculate(
                SimpleNamespace(timestamps_us=timestamps, actor_polygons=polygons)
            )
        flags = {metric.name: metric.values for metric in metrics}
        fault = at_fault.apply(pl.DataFrame(flags))["collision_at_fault"].to_list()
        results.append((flags, fault))

    # Check both order invariance and correctness: two identical but incorrect
    # results must not pass. Each list below contains [t=0, t=1] values.
    assert results[0] == results[1]
    flags, fault = results[0]
    assert flags == {
        "collision_front": [True, False],
        "collision_rear": [True, False],
        "collision_lateral": [False, False],
        "collision_any": [True, False],
    }
    assert fault == [1.0, 0.0]
