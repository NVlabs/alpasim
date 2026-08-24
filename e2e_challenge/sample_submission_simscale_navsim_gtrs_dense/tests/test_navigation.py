# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2, egodriver_pb2
from navsim_gtrs_dense_challenge.navigation import (
    DriveCommand,
    command_from_route,
    command_one_hot,
)


def _route(waypoints: list[tuple[float, float]]) -> egodriver_pb2.Route:
    return egodriver_pb2.Route(
        waypoints=[common_pb2.Vec3(x=x, y=y) for x, y in waypoints]
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (DriveCommand.LEFT, [1, 0, 0, 0]),
        (DriveCommand.STRAIGHT, [0, 1, 0, 0]),
        (DriveCommand.RIGHT, [0, 0, 1, 0]),
        (DriveCommand.UNKNOWN, [0, 0, 0, 1]),
    ],
)
def test_command_one_hot(command: DriveCommand, expected: list[int]) -> None:
    actual = command_one_hot(command)

    assert actual.shape == (4,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.float32))


@pytest.mark.parametrize(
    ("waypoints", "expected"),
    [
        ([], [0, 0, 0, 1]),
        ([(10, 3)], [1, 0, 0, 0]),
        ([(10, 0.5)], [0, 1, 0, 0]),
        ([(10, -3)], [0, 0, 1, 0]),
        ([(1, 1), (10, -3)], [0, 0, 1, 0]),
        ([(3, 4)], [1, 0, 0, 0]),
        ([(10, 2)], [0, 1, 0, 0]),
        ([(10, -2)], [0, 1, 0, 0]),
        ([(1, 1), (2, 2)], [0, 1, 0, 0]),
    ],
)
def test_command_from_route(
    waypoints: list[tuple[float, float]], expected: list[int]
) -> None:
    actual = command_from_route(_route(waypoints))

    np.testing.assert_array_equal(actual, np.array(expected, dtype=np.float32))


def test_command_from_route_with_custom_thresholds() -> None:
    route = _route([(3, 5), (8, 3)])

    actual = command_from_route(
        route,
        lateral_threshold_m=4.0,
        min_lookahead_m=6.0,
    )

    np.testing.assert_array_equal(actual, np.array([0, 1, 0, 0], dtype=np.float32))


def test_nan_padded_short_route_defaults_to_straight() -> None:
    route = _route([(1.0, 0.0), (4.0, 1.0), (float("nan"), float("nan"))])

    actual = command_from_route(route)

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, np.array([0, 1, 0, 0], dtype=np.float32))
