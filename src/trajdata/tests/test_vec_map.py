# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
from shapely import contains_xy, dwithin, linearrings, points, polygons

from trajdata import MapAPI, VectorMap
from trajdata.maps.vec_map_elements import MapElementType


class TestVectorMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cache_path = Path("~/.unified_data_cache").expanduser()
        cls.map_api = MapAPI(cache_path)
        cls.proto_loading_kwargs = {
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": True,
            "incl_ped_walkways": True,
        }

        cls.location_dict: Dict[str, List[str]] = {
            "nusc_mini": ["boston-seaport", "singapore-onenorth"],
        }

    # TODO(pkarkus) this assumes we already have the maps cached. It would be better
    # to attempt to cache them if the cache does not yet exists.
    def test_map_existence(self):
        for env_name, map_names in self.location_dict.items():
            for map_name in map_names:
                vec_map: VectorMap = self.map_api.get_map(
                    f"{env_name}:{map_name}", **self.proto_loading_kwargs
                )
                assert vec_map is not None

    def test_proto_equivalence(self):
        for env_name, map_names in self.location_dict.items():
            for map_name in map_names:
                vec_map: VectorMap = self.map_api.get_map(
                    f"{env_name}:{map_name}", **self.proto_loading_kwargs
                )

                assert maps_equal(
                    VectorMap.from_proto(
                        vec_map.to_proto(), **self.proto_loading_kwargs
                    ),
                    vec_map,
                )

    def test_road_area_queries(self):
        env_name = next(self.location_dict.keys().__iter__())
        map_name = self.location_dict[env_name][0]

        vec_map: VectorMap = self.map_api.get_map(
            f"{env_name}:{map_name}", **self.proto_loading_kwargs
        )

        if vec_map.search_rtrees is None:
            return

        point = vec_map.lanes[0].center.xy[0, :]
        closest_area = vec_map.get_closest_area(
            point, elem_type=MapElementType.ROAD_AREA
        )
        holes = closest_area.interior_holes
        if len(holes) == 0:
            holes = None
        closest_area_polygon = polygons(closest_area.exterior_polygon.xy, holes=holes)
        self.assertTrue(contains_xy(closest_area_polygon, point[None, :2]))

        rnd_points = np.random.uniform(
            low=vec_map.extent[:2], high=vec_map.extent[3:5], size=(10, 2)
        )

        NEARBY_DIST = 150.0
        for point in rnd_points:
            nearby_areas = vec_map.get_areas_within(
                point, elem_type=MapElementType.ROAD_AREA, dist=NEARBY_DIST
            )
            for area in nearby_areas:
                holes = [linearrings(hole.xy) for hole in area.interior_holes]
                if len(holes) == 0:
                    holes = None
                area_polygon = polygons(area.exterior_polygon.xy, holes=holes)
                point_pt = points(point)
                self.assertTrue(dwithin(area_polygon, point_pt, distance=NEARBY_DIST))

        for elem_type in [
            MapElementType.PED_CROSSWALK,
            MapElementType.PED_WALKWAY,
        ]:
            for point in rnd_points:
                nearby_areas = vec_map.get_areas_within(
                    point, elem_type=elem_type, dist=NEARBY_DIST
                )
                for area in nearby_areas:
                    area_polygon = polygons(area.polygon.xy)
                    point_pt = points(point)
                    if not dwithin(area_polygon, point_pt, distance=NEARBY_DIST):
                        print(
                            f"{elem_type.name} at {area_polygon} is not within {NEARBY_DIST} of {point_pt}",
                        )

    # TODO(bivanovic): Add more!


def maps_equal(map1: VectorMap, map2: VectorMap) -> bool:
    elements1_set = set([elem.id for elem in map1.iter_elems()])
    elements2_set = set([elem.id for elem in map2.iter_elems()])
    return elements1_set == elements2_set


if __name__ == "__main__":
    unittest.main()
