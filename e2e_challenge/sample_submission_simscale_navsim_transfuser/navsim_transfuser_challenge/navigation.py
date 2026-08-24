# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from enum import IntEnum

import numpy as np
from alpasim_grpc.v0 import egodriver_pb2


class DriveCommand(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    UNKNOWN = 3


def command_one_hot(command: DriveCommand) -> np.ndarray:
    """Encode a drive command as a four-element float32 vector."""
    one_hot = np.zeros(len(DriveCommand), dtype=np.float32)
    one_hot[int(command)] = 1.0
    return one_hot


def command_from_route(
    route: egodriver_pb2.Route,
    lateral_threshold_m: float = 2.0,
    min_lookahead_m: float = 5.0,
) -> np.ndarray:
    """Derive a drive command from the first sufficiently distant waypoint."""
    if not route.waypoints:
        return command_one_hot(DriveCommand.UNKNOWN)

    target = next(
        (
            waypoint
            for waypoint in route.waypoints
            if np.hypot(waypoint.x, waypoint.y) >= min_lookahead_m
        ),
        None,
    )
    if target is None:
        return command_one_hot(DriveCommand.STRAIGHT)
    if target.y > lateral_threshold_m:
        return command_one_hot(DriveCommand.LEFT)
    if target.y < -lateral_threshold_m:
        return command_one_hot(DriveCommand.RIGHT)
    return command_one_hot(DriveCommand.STRAIGHT)
