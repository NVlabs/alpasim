# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from trajdata.maps.vec_map import VectorMap

import numpy as np
from shapely import LinearRing, Polygon, STRtree, linearrings, points, polygons
from tqdm import tqdm

from trajdata.maps.vec_map_elements import (
    MapElement,
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    RoadArea,
)


def polygon_with_holes_geometry(map_element: MapElement) -> Polygon:
    assert isinstance(map_element, RoadArea)
    points = linearrings(map_element.exterior_polygon.xy)
    holes: Optional[List[LinearRing]] = None
    if len(map_element.interior_holes) > 0:
        holes = [linearrings(hole.xy) for hole in map_element.interior_holes]

    return polygons(points, holes=holes)


def polygon_geometry(map_element: MapElement) -> Polygon:
    assert isinstance(map_element, (PedWalkway, PedCrosswalk))
    return polygons(map_element.polygon.xy)


# Dictionary mapping map_elem_type to function returning
# shapely polygon for that map element
MAP_ELEM_TO_GEOMETRY: Dict[MapElementType, Callable[[MapElement], Polygon]] = {
    MapElementType.ROAD_AREA: polygon_with_holes_geometry,
    MapElementType.PED_CROSSWALK: polygon_geometry,
    MapElementType.PED_WALKWAY: polygon_geometry,
}


class MapElementSTRTree:
    """
    Constructs an Rtree of Polygonal MapElements and exposes fast lookup functions.

    Inheriting classes need to implement the _extract_geometry function which for a MapElement
    returns the geometry we want to store
    """

    def __init__(
        self,
        vector_map: VectorMap,
        elem_type: MapElementType,
        verbose: bool = False,
    ) -> None:
        # Build R-tree
        self.strtree, self.elem_ids = self._build_strtree(
            vector_map, elem_type, verbose
        )

    def _build_strtree(
        self,
        vector_map: VectorMap,
        elem_type: MapElementType,
        verbose: bool = False,
    ) -> Tuple[STRtree, np.ndarray]:
        geometries: List[Polygon] = []
        ids: List[str] = []
        geometry_fn = MAP_ELEM_TO_GEOMETRY[elem_type]

        map_elem: MapElement
        for id, map_elem in tqdm(
            vector_map.elements[elem_type].items(),
            desc=f"Building STR Tree for {elem_type.name} elements",
            leave=False,
            disable=not verbose,
        ):
            ids.append(id)
            geometries.append(geometry_fn(map_elem))

        return STRtree(geometries), np.array(ids)

    def query_point(
        self,
        point: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns ID of all elements of type elem_type
        that intersect with query point

        Args:
            point (np.ndarray): point to query
            elem_type (MapElementType): type of elem to query
            kwargs: passed on to STRtree.query(), see
                https://pygeos.readthedocs.io/en/latest/strtree.html
                Can be used for predicate based queries, e.g.
                 predicate="dwithin", distance=100.
                returns all elements which are within 100m of query point

        Returns:
            np.ndarray[str]: 1d array of ids of all elements matching query
        """
        indices = self.strtree.query(points(point), **kwargs)
        return self.elem_ids[indices]

    def nearest_area(
        self,
        point: np.ndarray,
        **kwargs,
    ) -> str:
        """
        Returns ID of the elements of type elem_type
        that are closest to point.

        Args:
            point (np.ndarray): point to query
            elem_type (MapElementType): type of elem to query
            kwargs: passed on to STRtree.nearest(), see
                https://pygeos.readthedocs.io/en/latest/strtree.html

        Returns:
            str: element_id of all elements matching query
        """
        idx = self.strtree.nearest(points(point), **kwargs)
        return self.elem_ids[idx]
