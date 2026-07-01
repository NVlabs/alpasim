# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math

import numpy as np
import torch
from alpasim_utils.geometry import Polyline
from trajdata.maps.vec_map_elements import MapElementType

MAP_ELEMENT_NAME2_TYPEID = {
    "lane_lines": 0,
    "road_boundaries": 1,
    "wait_lines": 2,
    "crosswalks": 3,
    "lane_boundaries": 4,
    "lane_centers": 5,
    "intersection_areas": 6,
    "road_islands": 7,
    "road_markings": 8,
    "poles": 9,
    "traffic_signs": 10,
}

ENVDATA_MAP_KEY_TO_CATK_MAP_ELEMENT_NAME = {
    "lanelines": "lane_lines",
    "road_boundaries": "road_boundaries",
    "waitlines": "wait_lines",
    "lane_boundaries": "lane_boundaries",
    "lane_centers": "lane_centers",
    "crosswalks": "crosswalks",
    "intersection_areas": "intersection_areas",
    "road_islands": "road_islands",
}


def build_env_map_from_vector_map(
    vector_map,
    *,
    ego_xyz: torch.Tensor,
    ego_heading: torch.Tensor | float,
    distance_x: float,
    distance_y: float,
    map_polyline_length_k: int,
    map_resample_interval_m: float | None = None,
) -> dict:
    """Adapt a trajdata VectorMap to the current runtime map contract."""
    raw_layers = _vector_map_polylines_by_env_key(vector_map)
    env_map: dict = {}
    for (
        env_key,
        catk_map_element_name,
    ) in ENVDATA_MAP_KEY_TO_CATK_MAP_ELEMENT_NAME.items():
        env_map[env_key] = _build_env_map_element(
            _map_element_from_arrays(
                raw_layers.get(env_key, []),
                map_resample_interval_m=map_resample_interval_m,
            ),
            catk_map_element_name=catk_map_element_name,
            ego_xyz=ego_xyz,
            ego_heading=ego_heading,
            distance_x=distance_x,
            distance_y=distance_y,
            map_polyline_length_k=map_polyline_length_k,
        )
    return env_map


def _vector_map_polylines_by_env_key(vector_map) -> dict[str, list[np.ndarray]]:
    layers: dict[str, list[np.ndarray]] = {
        "lanelines": [],
        "road_boundaries": [],
        "waitlines": [],
        "lane_boundaries": [],
        "lane_centers": [],
        "crosswalks": [],
        "intersection_areas": [],
        "road_islands": [],
    }

    for lane in vector_map.elements.get(MapElementType.ROAD_LANE, {}).values():
        layers["lane_centers"].append(lane.center.xyz)
        for edge in (lane.left_edge, lane.right_edge):
            if edge is None:
                continue
            layers["lane_boundaries"].append(edge.xyz)
            # VectorMap does not distinguish lane markings from lane edges for
            # all sources. Preserve the CATK laneline layer from the available
            # lane-edge geometry.
            layers["lanelines"].append(edge.xyz)

    for road_edge in vector_map.elements.get(MapElementType.ROAD_EDGE, {}).values():
        layers["road_boundaries"].append(road_edge.polyline.xyz)

    for wait_line in vector_map.elements.get(MapElementType.WAIT_LINE, {}).values():
        layers["waitlines"].append(wait_line.polyline.xyz)

    for crosswalk in vector_map.elements.get(MapElementType.PED_CROSSWALK, {}).values():
        layers["crosswalks"].append(crosswalk.polygon.xyz)

    for road_area in vector_map.elements.get(MapElementType.ROAD_AREA, {}).values():
        layers["intersection_areas"].append(road_area.exterior_polygon.xyz)
        for hole in road_area.interior_holes:
            layers["road_islands"].append(hole.xyz)

    return layers


def _map_element_from_arrays(
    polylines: list[np.ndarray],
    *,
    map_resample_interval_m: float | None,
) -> dict | None:
    if not polylines:
        return None
    cleaned = []
    for polyline in polylines:
        arr = np.asarray(polyline, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        if arr.shape[1] == 2:
            arr = np.concatenate(
                [arr, np.zeros((arr.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
        if map_resample_interval_m is not None:
            arr = _resample_polyline_by_interval(arr, map_resample_interval_m)
        cleaned.append(arr[..., :3])
    if not cleaned:
        return None
    max_vertices = max(polyline.shape[0] for polyline in cleaned)
    stacked = np.full((len(cleaned), max_vertices, 3), np.nan, dtype=np.float32)
    for idx, polyline in enumerate(cleaned):
        stacked[idx, : polyline.shape[0], :] = polyline
    return {"polylines": torch.from_numpy(stacked)}


def _resample_polyline_by_interval(
    polyline: np.ndarray,
    interval_m: float,
) -> np.ndarray:
    if interval_m <= 0:
        raise ValueError(f"interval_m must be positive, got {interval_m}")
    return (
        Polyline(polyline)
        .resample_by_spacing(interval_m)
        .waypoints.astype(
            np.float32,
            copy=False,
        )
    )


def _build_env_map_element(
    map_element: dict | None,
    *,
    catk_map_element_name: str,
    ego_xyz: torch.Tensor,
    ego_heading: torch.Tensor | float,
    distance_x: float,
    distance_y: float,
    map_polyline_length_k: int,
) -> dict | None:
    if not isinstance(map_element, dict):
        return None

    polylines = map_element.get("polylines")
    if not torch.is_tensor(polylines) or polylines.numel() == 0:
        return None

    fixed_polylines = []
    for polyline in polylines:
        finite_polyline = _finite_polyline_vertices(polyline)
        if finite_polyline is None:
            continue
        if not _polyline_hits_local_box(
            finite_polyline,
            ego_xyz=ego_xyz,
            ego_heading=ego_heading,
            distance_x=distance_x,
            distance_y=distance_y,
        ):
            continue
        fixed_polylines.append(
            _build_map_polyline_segments(finite_polyline, map_polyline_length_k)
        )

    if not fixed_polylines:
        return None

    stacked = torch.cat(fixed_polylines, dim=0).to(dtype=torch.float32)
    if stacked.shape[0] == 0:
        return None
    label = torch.full(
        (stacked.shape[0],),
        MAP_ELEMENT_NAME2_TYPEID[catk_map_element_name],
        dtype=torch.long,
        device=stacked.device,
    )
    return {
        "polylines": stacked,
        "label": label,
        "polylines_styles": None,
        "polylines_colors": None,
    }


def _finite_polyline_vertices(polyline: torch.Tensor) -> torch.Tensor | None:
    if polyline.ndim != 2 or polyline.shape[-1] < 3:
        return None
    finite_mask = torch.isfinite(polyline).all(dim=-1)
    finite_polyline = polyline[finite_mask, :3].clone()
    if finite_polyline.shape[0] < 2:
        return None

    deltas = finite_polyline[1:] - finite_polyline[:-1]
    keep_mask = torch.cat(
        [
            torch.ones((1,), dtype=torch.bool, device=finite_polyline.device),
            torch.linalg.norm(deltas, dim=-1) > 1e-6,
        ]
    )
    finite_polyline = finite_polyline[keep_mask]
    if finite_polyline.shape[0] < 2:
        return None
    return finite_polyline


def _build_map_polyline_segments(
    polyline: torch.Tensor, map_polyline_length_k: int
) -> torch.Tensor:
    """Segment map polylines while keeping EnvData layers."""
    if map_polyline_length_k < 1:
        raise ValueError(
            f"map_polyline_length_k must be >= 1, got {map_polyline_length_k}"
        )
    if map_polyline_length_k > 1:
        polyline = _downsample_polyline_vertices(polyline, factor=map_polyline_length_k)
    return _overlapping_three_point_segments(polyline)


def _downsample_polyline_vertices(
    polyline: torch.Tensor,
    *,
    factor: int,
) -> torch.Tensor:
    if factor <= 1 or polyline.shape[0] <= 1:
        return polyline
    indices = torch.arange(0, polyline.shape[0], factor, device=polyline.device)
    last_idx = polyline.shape[0] - 1
    if int(indices[-1].item()) != last_idx:
        indices = torch.cat([indices, indices.new_tensor([last_idx])])
    return polyline[indices]


def _overlapping_three_point_segments(polyline: torch.Tensor) -> torch.Tensor:
    if polyline.shape[0] < 3:
        return polyline.new_empty((0, 3, 3))

    starts = torch.arange(0, polyline.shape[0] - 2, 2, device=polyline.device)
    tail_start = polyline.shape[0] - 3
    if int(starts[-1].item()) != tail_start:
        starts = torch.cat([starts, starts.new_tensor([tail_start])])

    offsets = torch.arange(3, device=polyline.device)
    return polyline[starts[:, None] + offsets[None, :]]


def _polyline_hits_local_box(
    polyline: torch.Tensor,
    *,
    ego_xyz: torch.Tensor,
    ego_heading: torch.Tensor | float,
    distance_x: float,
    distance_y: float,
) -> bool:
    if distance_x <= 0 or distance_y <= 0:
        return True

    local_xy = _world_xy_to_local(
        polyline[:, :2],
        ego_xyz=ego_xyz.to(device=polyline.device, dtype=polyline.dtype),
        ego_heading=ego_heading,
    )
    inside_x = (local_xy[:, 0] >= -distance_x) & (local_xy[:, 0] <= distance_x)
    inside_y = (local_xy[:, 1] >= -distance_y) & (local_xy[:, 1] <= distance_y)
    return bool((inside_x & inside_y).any().item())


def _world_xy_to_local(
    xy: torch.Tensor,
    *,
    ego_xyz: torch.Tensor,
    ego_heading: torch.Tensor | float,
) -> torch.Tensor:
    heading = float(ego_heading)
    rot = xy.new_tensor(
        [
            [math.cos(heading), -math.sin(heading)],
            [math.sin(heading), math.cos(heading)],
        ]
    )
    return (xy - ego_xyz[:2]) @ rot
