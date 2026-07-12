# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_utils.ground_snap import snap_vector_map_to_ground
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import MapElementType, Polyline, RoadEdge, RoadLane


def _flat_mesh(z: float = 0.0, extent: float = 10.0) -> np.ndarray:
    return np.array(
        [
            [-extent, -extent, z],
            [extent, -extent, z],
            [extent, extent, z],
            [-extent, extent, z],
            [0.0, 0.0, z],
        ],
        dtype=np.float64,
    )


def test_snap_vector_map_to_ground_rejects_invalid_vertices() -> None:
    vector_map = VectorMap(map_id="test:test")

    with pytest.raises(ValueError, match="shape"):
        snap_vector_map_to_ground(vector_map, np.zeros((10, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        snap_vector_map_to_ground(vector_map, np.zeros((0, 3)))


def test_snap_vector_map_to_ground_updates_lane_geometry() -> None:
    vector_map = VectorMap(map_id="test:test")
    vector_map.extent = np.array([-1.0, -1.0, -99.0, 1.0, 1.0, -99.0])
    lane = RoadLane(
        id="lane_0",
        center=Polyline(
            np.array([[0.0, 0.0, -99.0], [1.0, 0.0, -99.0]], dtype=np.float64)
        ),
        left_edge=Polyline(
            np.array([[0.0, 0.5, -99.0], [1.0, 0.5, -99.0]], dtype=np.float64)
        ),
        right_edge=Polyline(
            np.array([[0.0, -0.5, -99.0], [1.0, -0.5, -99.0]], dtype=np.float64)
        ),
    )
    vector_map.elements[MapElementType.ROAD_LANE][lane.id] = lane

    snap_vector_map_to_ground(vector_map, _flat_mesh(z=1.0))

    np.testing.assert_allclose(lane.center.points[:, 2], 1.0)
    np.testing.assert_allclose(lane.left_edge.points[:, 2], 1.0)
    np.testing.assert_allclose(lane.right_edge.points[:, 2], 1.0)
    np.testing.assert_allclose(vector_map.extent[[2, 5]], 1.0)


def test_snap_vector_map_to_ground_uses_nearest_mesh_vertex() -> None:
    vector_map = VectorMap(map_id="test:test")
    edge = RoadEdge(
        id="edge_0",
        polyline=Polyline(
            np.array(
                [[0.1, 0.1, -50.0], [4.9, 0.1, -50.0], [0.1, 4.9, -50.0]],
                dtype=np.float64,
            )
        ),
    )
    vector_map.elements[MapElementType.ROAD_EDGE][edge.id] = edge
    vertices = np.array(
        [[0.0, 0.0, 1.0], [5.0, 0.0, 2.0], [0.0, 5.0, 3.0]],
        dtype=np.float64,
    )

    snap_vector_map_to_ground(vector_map, vertices)

    np.testing.assert_allclose(edge.polyline.points[:, 2], [1.0, 2.0, 3.0])
