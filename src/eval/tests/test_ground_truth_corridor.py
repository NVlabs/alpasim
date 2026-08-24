# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from shapely.geometry import LineString, Point

from eval.scorers.ground_truth import _end_direction, _lateral_distance_to_gt


def _lat(x: float, y: float) -> float:
    gt = LineString([(0, 0), (50, 0)])
    return _lateral_distance_to_gt(gt, Point(x, y), gt.length, _end_direction(gt))


def test_forward_overshoot_is_not_lateral() -> None:
    # `LineString.distance` clamps to the endpoint, so these read as 6 m and
    # 20 m away; laterally the ego never left the corridor.
    assert _lat(56.0, 0.0) == 0.0
    assert _lat(70.0, 0.0) == 0.0


def test_sideways_excursion_is_lateral() -> None:
    assert _lat(25.0, 4.5) == 4.5


def test_overshoot_and_sideways_reports_only_the_sideways_part() -> None:
    assert _lat(56.0, 5.0) == 5.0


def test_degenerate_gt_falls_back_to_plain_distance() -> None:
    degenerate = LineString([(0, 0), (0, 0)])
    assert (
        _lateral_distance_to_gt(
            degenerate, Point(3.0, 4.0), 0.0, _end_direction(degenerate)
        )
        == 5.0
    )
