# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Turn a route polyline into the navigation string Alpamayo 1.5 conditions on.

Alpamayo 1.5 takes navigation intent as text ("Turn left in 20m"), not as a
route tensor, so a route only reaches the model through this classifier.
Without it the model has to guess at intersections, and a driver that keeps its
trajectory choice consistent across cycles will then commit to whichever branch
the first cycle happened to sample.

Turns are classified by heading change rather than lateral offset: a fixed
lateral threshold means ~24 degrees of heading at 5 m but only ~3 degrees at
40 m, so it reads gentle drift and parallel lane offsets as turns up close and
misses real turns further out.

The wording ("Turn left in {N}m", "Continue straight") is the wording the model
was trained on, so it is fixed rather than a formatting choice.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Shortest segment that still defines a bearing.  Resampled routes can repeat
# their last waypoint as padding.
_MIN_SEGMENT_LENGTH_M = 1e-3


def route_to_nav_text(
    route_in_rig: np.ndarray,
    *,
    min_lookahead_m: float = 5.0,
    turn_angle_deg: float = 30.0,
    distance_lookahead_m: float = 40.0,
) -> str | None:
    """Classify the route ahead as a left turn, a right turn, or straight.

    Args:
        route_in_rig: Route waypoints of shape ``(N, 3)`` in the rig frame at
            t0 (+x forward, +y left).  Rows may be NaN, which is how a
            partially known route is padded.
        min_lookahead_m: Turns detected nearer than this are ignored; the ego is
            already in them and the training labels start a few metres out.
        turn_angle_deg: Heading change that separates a turn from lane
            curvature.
        distance_lookahead_m: Turns detected beyond this are ignored; they are
            outside the trained label range and get reported on a later cycle as
            the ego approaches.

    Returns:
        ``"Turn left in {N}m"``, ``"Turn right in {N}m"`` or
        ``"Continue straight"``, or None if the route holds no usable waypoint,
        in which case the model runs unconditioned.
    """
    if route_in_rig.size == 0:
        return None

    xy = route_in_rig[:, :2]
    points = xy[~np.isnan(xy).any(axis=1)]
    if len(points) == 0:
        return None

    if len(points) >= 2:
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = (
            np.linalg.norm(np.diff(points, axis=0), axis=1) > _MIN_SEGMENT_LENGTH_M
        )
        points = points[keep]

    # Measuring a heading change needs two segments.  With one segment the route
    # is indistinguishable from a straight road at an angle to the ego, or from
    # a parallel lane offset.
    if len(points) < 3:
        return "Continue straight"

    distances = np.linalg.norm(points, axis=1)

    # Take the initial road heading as the chord across the near zone rather
    # than the first segment alone, so that a constant yaw offset between ego
    # and road does not read as a turn.
    near = np.flatnonzero(distances <= min_lookahead_m)
    start, end = (near[0], near[-1]) if len(near) >= 2 else (0, 1)
    base = points[end] - points[start]
    if np.linalg.norm(base) < _MIN_SEGMENT_LENGTH_M:
        base = points[1] - points[0]
    initial_heading = np.arctan2(base[1], base[0])

    # Segment k runs points[k] -> points[k + 1].  The turn is attributed to the
    # far endpoint, where the heading change has been realised, which is also
    # the distance to report.
    segments = np.diff(points, axis=0)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    deviations = np.remainder(headings - initial_heading + np.pi, 2 * np.pi) - np.pi
    endpoint_distances = distances[1:]

    turning = (
        (endpoint_distances >= min_lookahead_m)
        & (endpoint_distances <= distance_lookahead_m)
        & (np.abs(deviations) > np.radians(turn_angle_deg))
    )
    if not turning.any():
        return "Continue straight"

    index = int(np.argmax(turning))
    direction = "left" if deviations[index] > 0 else "right"
    return f"Turn {direction} in {max(1, round(endpoint_distances[index]))}m"
