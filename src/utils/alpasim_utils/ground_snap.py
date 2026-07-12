# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Align vector-map geometry with an AlpaSim ground mesh."""

import logging

import numpy as np
from scipy.spatial import cKDTree
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import RoadEdge, RoadLane, WaitLine

logger = logging.getLogger(__name__)


def snap_vector_map_to_ground(
    vector_map: VectorMap, ground_mesh_vertices: np.ndarray
) -> None:
    """Snap supported vector-map polylines to the ground mesh in place."""
    if (
        ground_mesh_vertices.ndim != 2
        or ground_mesh_vertices.shape[1] != 3
        or ground_mesh_vertices.shape[0] == 0
    ):
        raise ValueError(
            "ground_mesh_vertices must be a non-empty array with shape (N, 3), "
            f"got {ground_mesh_vertices.shape}"
        )

    ground_tree = cKDTree(ground_mesh_vertices[:, :2])
    ground_z = ground_mesh_vertices[:, 2]
    snapped_points = 0
    minimum_height = np.inf
    maximum_height = -np.inf

    for element in vector_map.iter_elems():
        polylines: list[np.ndarray] = []
        if isinstance(element, RoadLane):
            polylines.extend(
                polyline.points
                for polyline in (element.center, element.left_edge, element.right_edge)
                if polyline is not None
            )
        elif isinstance(element, (RoadEdge, WaitLine)):
            if element.polyline is not None:
                polylines.append(element.polyline.points)
        for points in polylines:
            if points.shape[0] == 0 or points.shape[1] < 3:
                continue
            _, indices = ground_tree.query(np.asarray(points[:, :2], dtype=np.float64))
            heights = ground_z[indices]
            points[:, 2] = heights
            minimum_height = min(minimum_height, float(heights.min()))
            maximum_height = max(maximum_height, float(heights.max()))
            snapped_points += points.shape[0]

    if snapped_points and vector_map.extent is not None:
        vector_map.extent[2] = minimum_height
        vector_map.extent[5] = maximum_height

    logger.info("Snapped %d map points to the ground mesh", snapped_points)
