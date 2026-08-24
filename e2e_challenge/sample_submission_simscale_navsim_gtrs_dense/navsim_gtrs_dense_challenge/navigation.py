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
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[int(command)] = 1.0
    return one_hot


def command_from_route(
    route: egodriver_pb2.Route,
    *,
    lateral_threshold_m: float = 2.0,
    min_lookahead_m: float = 5.0,
) -> np.ndarray:
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

    if abs(target.y) <= lateral_threshold_m:
        command = DriveCommand.STRAIGHT
    elif target.y > 0:
        command = DriveCommand.LEFT
    else:
        command = DriveCommand.RIGHT
    return command_one_hot(command)
