# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Navigation utilities for determining driving commands from route geometry."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from alpasim_grpc.v0.common_pb2 import Vec3
from alpasim_grpc.v0.egodriver_pb2 import Route

from .models.base import DriveCommand

logger = logging.getLogger(__name__)


def determine_command_from_route(
    route: Route,
    command_distance_threshold: float = 2.0,
    min_lookahead_distance: float = 5.0,
) -> DriveCommand:
    """Determine semantic driving command from route geometry.

    Analyzes route waypoints (in rig frame) to determine whether the
    vehicle should turn left, go straight, or turn right.

    Args:
        route: Route containing waypoints in the rig frame.
        command_distance_threshold: Lateral distance threshold (meters) for
            determining turn commands. Waypoints beyond this threshold
            trigger LEFT/RIGHT commands.
        min_lookahead_distance: Minimum forward distance (meters) to consider
            a waypoint as the target for command derivation.

    Returns:
        Semantic DriveCommand (LEFT, STRAIGHT, RIGHT, or UNKNOWN).
    """
    if len(route.waypoints) < 1:
        return DriveCommand.UNKNOWN

    # Find the first waypoint that is at least min_lookahead_distance ahead
    target_waypoint: Optional["Vec3"] = None
    for wp in route.waypoints:
        distance = np.hypot(wp.x, wp.y)
        if distance >= min_lookahead_distance:
            target_waypoint = wp
            break

    if target_waypoint is None:
        return DriveCommand.STRAIGHT

    # In rig frame, positive Y is left
    dy_rig = target_waypoint.y

    if dy_rig > command_distance_threshold:
        command = DriveCommand.LEFT
    elif dy_rig < -command_distance_threshold:
        command = DriveCommand.RIGHT
    else:
        command = DriveCommand.STRAIGHT

    logger.debug(
        "Command: %s (lateral displacement: %.2fm at distance: %.2fm)",
        command.name,
        dy_rig,
        np.hypot(target_waypoint.x, target_waypoint.y),
    )

    return command
