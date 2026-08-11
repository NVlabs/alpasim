# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Navigation strings derived from route geometry."""

from __future__ import annotations

import numpy as np
from alpasim_driver.models.nav_text import route_to_nav_text

WAYPOINT_SPACING_M = 4.0


def _route(headings_deg: list[float]) -> np.ndarray:
    """Route walking one waypoint per heading, starting at the ego origin."""
    positions = [np.zeros(2)]
    for heading_deg in headings_deg:
        step = WAYPOINT_SPACING_M * np.array(
            [np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg))]
        )
        positions.append(positions[-1] + step)
    return np.pad(np.array(positions), ((0, 0), (0, 1)))


def test_straight_route() -> None:
    assert route_to_nav_text(_route([0.0] * 10)) == "Continue straight"


def test_left_turn_reports_distance_to_the_turn() -> None:
    """The distance is measured to where the heading change is realised."""
    # Straight out to 16 m, then left: the first turning segment ends at
    # (16, 4), 16.5 m away.
    nav_text = route_to_nav_text(_route([0.0] * 4 + [90.0] * 5))

    assert nav_text == "Turn left in 16m"


def test_right_turn() -> None:
    assert route_to_nav_text(_route([0.0] * 4 + [-90.0] * 5)).startswith("Turn right")


def test_gentle_curvature_is_not_a_turn() -> None:
    """Lane curvature stays under the heading threshold at every distance."""
    assert (
        route_to_nav_text(_route(list(np.arange(1, 11) * 2.0))) == "Continue straight"
    )


def test_route_at_an_angle_to_the_ego_is_straight() -> None:
    """A straight road the ego is not yet aligned with is not a turn."""
    assert route_to_nav_text(_route([40.0] * 10)) == "Continue straight"


def test_turn_beyond_the_lookahead_is_reported_later() -> None:
    # The turn starts at 40 m, past the default 40 m detection window once the
    # heading change is realised at the segment's far endpoint.
    assert route_to_nav_text(_route([0.0] * 10 + [90.0] * 3)) == "Continue straight"


def test_nan_padding_is_ignored() -> None:
    """Routes shorter than the requested lookahead arrive NaN-padded."""
    route = _route([0.0] * 4 + [90.0] * 5)
    padded = np.vstack([route, np.full((6, 3), np.nan)])

    assert route_to_nav_text(padded) == route_to_nav_text(route)


def test_route_without_usable_waypoints() -> None:
    assert route_to_nav_text(np.zeros((0, 3))) is None
    assert route_to_nav_text(np.full((20, 3), np.nan)) is None
