# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon, box

from eval.scorers.offroad import (
    ROAD_AREA_QUERY_DIST_M,
    _road_area_union,
    _road_areas_near_ego,
)
from trajdata.maps.vec_map_elements import MapElementType


class _FakeVectorMap:
    def __init__(self, polygons: dict[str, object]) -> None:
        self._polygons = polygons

    def get_road_area_polygon_2d(self, road_area_id: str):  # noqa: ANN201
        return self._polygons[road_area_id]


def _road_area_union_for(*polygons):  # noqa: ANN002, ANN202
    road_areas = [SimpleNamespace(id=str(index)) for index in range(len(polygons))]
    simulation_result = SimpleNamespace(
        vec_map=_FakeVectorMap(
            {road_area.id: polygon for road_area, polygon in zip(road_areas, polygons)}
        )
    )
    return _road_area_union(simulation_result, road_areas)


def test_road_area_union_closes_submillimetre_seam() -> None:
    seam_m = 0.0005
    road_area_union = _road_area_union_for(
        box(0.0, 0.0, 10.0, 10.0),
        box(10.0 + seam_m, 0.0, 20.0, 10.0),
    )

    assert road_area_union.covers(box(9.0, 4.0, 11.0 + seam_m, 6.0))


def test_road_area_union_does_not_allow_larger_outer_incursion() -> None:
    road_area_union = _road_area_union_for(box(0.0, 0.0, 20.0, 10.0))

    assert not road_area_union.covers(box(18.0, 4.0, 20.0015, 6.0))


def test_road_area_union_repairs_invalid_input_before_union() -> None:
    self_intersecting_area = Polygon(
        [(0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0), (0.0, 0.0)]
    )

    road_area_union = _road_area_union_for(
        self_intersecting_area,
        box(2.0, 0.0, 4.0, 2.0),
    )

    assert road_area_union.is_valid


def test_road_area_query_uses_competition_map_radius() -> None:
    class FakeVectorMap:
        search_kdtrees = {MapElementType.ROAD_AREA: object()}

        def __init__(self) -> None:
            self.query = None

        def get_road_areas_within(self, xyz, dist):  # noqa: ANN001, ANN201
            self.query = (xyz, dist)
            return []

    vector_map = FakeVectorMap()
    simulation_result = SimpleNamespace(vec_map=vector_map)

    assert (
        _road_areas_near_ego(
            simulation_result,
            np.asarray([1.0, 2.0, 3.0, 0.5]),
        )
        == []
    )
    assert vector_map.query is not None
    xyz, distance_m = vector_map.query
    np.testing.assert_array_equal(xyz, np.asarray([1.0, 2.0, 3.0]))
    assert distance_m == ROAD_AREA_QUERY_DIST_M == 25.0


def test_road_area_query_remains_disabled_without_road_area_index() -> None:
    class FakeVectorMap:
        search_kdtrees = {}

        def get_road_areas_within(self, xyz, dist):  # noqa: ANN001, ANN201
            raise AssertionError("road-area lookup should not run without an index")

    simulation_result = SimpleNamespace(vec_map=FakeVectorMap())

    assert (
        _road_areas_near_ego(
            simulation_result,
            np.asarray([1.0, 2.0, 3.0, 0.5]),
        )
        is None
    )
