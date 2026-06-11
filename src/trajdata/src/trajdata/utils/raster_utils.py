# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Final

if TYPE_CHECKING:
    from trajdata.maps import VectorMap


from math import ceil
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from trajdata.maps.raster_map import RasterizedMap, RasterizedMapMetadata
from trajdata.maps.vec_map import MapElement, MapElementType
from trajdata.utils import map_utils

# Sub-pixel drawing precision constants.
# See https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/rasterization/semantic_rasterizer.py#L16
CV2_SUB_VALUES: Final[Dict[str, Any]] = {"shift": 9, "lineType": cv2.LINE_AA}
CV2_SHIFT_VALUE: Final[int] = 2 ** CV2_SUB_VALUES["shift"]
DEFAULT_PX_PER_M: Final[float] = 2.0


def cv2_subpixel(coords: np.ndarray) -> np.ndarray:
    """
    Cast coordinates to numpy.int but keep fractional part by previously multiplying by 2**CV2_SHIFT
    cv2 calls will use shift to restore original values with higher precision

    Args:
        coords (np.ndarray): XY coords as float

    Returns:
        np.ndarray: XY coords as int for cv2 shift draw
    """
    return (coords * CV2_SHIFT_VALUE).astype(int)


def world_to_subpixel(pts: np.ndarray, raster_from_world: np.ndarray):
    return cv2_subpixel(map_utils.transform_points(pts, raster_from_world))


def cv2_draw_polygons(
    polygon_pts: List[np.ndarray],
    onto_img: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    cv2.fillPoly(
        img=onto_img,
        pts=polygon_pts,
        color=color,
        **CV2_SUB_VALUES,
    )


def cv2_draw_polylines(
    polyline_pts: List[np.ndarray],
    onto_img: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    cv2.polylines(
        img=onto_img,
        pts=polyline_pts,
        isClosed=False,
        color=color,
        **CV2_SUB_VALUES,
    )


def rasterize_world_polygon(
    polygon_pts: np.ndarray,
    onto_img: np.ndarray,
    raster_from_world: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    subpixel_area: np.ndarray = world_to_subpixel(
        polygon_pts[..., :2], raster_from_world
    )

    # Drawing general road areas.
    cv2_draw_polygons(polygon_pts=[subpixel_area], onto_img=onto_img, color=color)


def rasterize_world_polylines(
    polyline_pts: List[np.ndarray],
    onto_img: np.ndarray,
    raster_from_world: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    subpixel_pts: List[np.ndarray] = [
        world_to_subpixel(pts[..., :2], raster_from_world) for pts in polyline_pts
    ]

    # Drawing line.
    cv2_draw_polylines(
        polyline_pts=subpixel_pts,
        onto_img=onto_img,
        color=color,
    )


def rasterize_lane(
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    onto_img_area: np.ndarray,
    onto_img_line: np.ndarray,
    raster_from_world: np.ndarray,
    area_color: Tuple[int, int, int],
    line_color: Tuple[int, int, int],
) -> None:
    lane_edges: List[np.ndarray] = [left_edge[:, :2], right_edge[::-1, :2]]

    # Drawing lane area.
    rasterize_world_polygon(
        np.concatenate(lane_edges, axis=0),
        onto_img_area,
        raster_from_world,
        color=area_color,
    )

    # Drawing lane lines.
    rasterize_world_polylines(lane_edges, onto_img_line, raster_from_world, line_color)


def rasterize_map(
    vec_map: VectorMap, resolution: float, **pbar_kwargs
) -> RasterizedMap:
    """Renders the semantic map at the given resolution.

    Args:
        vec_map (VectorMap): _description_
        resolution (float): The rasterized image's resolution in pixels per meter.

    Returns:
        np.ndarray: The rasterized RGB image.
    """
    # extents is [min_x, min_y, min_z, max_x, max_y, max_z]
    min_x, min_y, _, max_x, max_y, _ = vec_map.extent
    world_center_m: Tuple[float, float] = (
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
    )

    raster_size_x: int = ceil((max_x - min_x) * resolution)
    raster_size_y: int = ceil((max_y - min_y) * resolution)

    raster_from_local: np.ndarray = np.array(
        [
            [resolution, 0, raster_size_x / 2],
            [0, resolution, raster_size_y / 2],
            [0, 0, 1],
        ]
    )

    # Compute pose from its position and rotation.
    pose_from_world: np.ndarray = np.array(
        [
            [1, 0, -world_center_m[0]],
            [0, 1, -world_center_m[1]],
            [0, 0, 1],
        ]
    )

    raster_from_world: np.ndarray = raster_from_local @ pose_from_world

    lane_area_img: np.ndarray = np.zeros(
        shape=(raster_size_y, raster_size_x, 3), dtype=np.uint8
    )
    lane_line_img: np.ndarray = np.zeros(
        shape=(raster_size_y, raster_size_x, 3), dtype=np.uint8
    )
    ped_area_img: np.ndarray = np.zeros(
        shape=(raster_size_y, raster_size_x, 3), dtype=np.uint8
    )

    map_elem: MapElement
    for map_elem in tqdm(
        vec_map.iter_elems(),
        desc=f"Rasterizing Map at {resolution:.2f} px/m",
        total=len(vec_map),
        **pbar_kwargs,
    ):
        if map_elem.elem_type == MapElementType.ROAD_LANE:
            if map_elem.left_edge is not None and map_elem.right_edge is not None:
                # Heading doesn't matter for rasterization.
                left_pts: np.ndarray = map_elem.left_edge.xyz
                right_pts: np.ndarray = map_elem.right_edge.xyz

                # Need to for-loop because doing it all at once can make holes.
                # Drawing lane.
                rasterize_lane(
                    left_pts,
                    right_pts,
                    lane_area_img,
                    lane_line_img,
                    raster_from_world,
                    area_color=(255, 0, 0),
                    line_color=(0, 255, 0),
                )

            # # This code helps visualize centerlines to check if the inferred headings are correct.
            # center_pts = cv2_subpixel(
            #     transform_points(
            #         proto_to_np(map_elem.road_lane.center, incl_heading=False),
            #         raster_from_world,
            #     )
            # )[..., :2]

            # # Drawing lane centerlines.
            # cv2.polylines(
            #     img=lane_line_img,
            #     pts=center_pts[None, :, :],
            #     isClosed=False,
            #     color=(255, 0, 0),
            #     **CV2_SUB_VALUES,
            # )

            # headings = np.asarray(map_elem.road_lane.center.h_rad)
            # delta = cv2_subpixel(30*np.array([np.cos(headings[0]), np.sin(headings[0])]))
            # cv2.arrowedLine(img=lane_line_img, pt1=tuple(center_pts[0]), pt2=tuple(center_pts[0] + 10*(center_pts[1] - center_pts[0])), color=(255, 0, 0), shift=9, line_type=cv2.LINE_AA)
            # cv2.arrowedLine(img=lane_line_img, pt1=tuple(center_pts[0]), pt2=tuple(center_pts[0] + delta), color=(0, 255, 0), shift=9, line_type=cv2.LINE_AA)

        elif map_elem.elem_type == MapElementType.ROAD_AREA:
            # Drawing general road areas.
            rasterize_world_polygon(
                map_elem.exterior_polygon.xy,
                lane_area_img,
                raster_from_world,
                color=(255, 0, 0),
            )

            for interior_hole in map_elem.interior_holes:
                # Removing holes.
                rasterize_world_polygon(
                    interior_hole.xy, lane_area_img, raster_from_world, color=(0, 0, 0)
                )

        elif map_elem.elem_type in {
            MapElementType.PED_CROSSWALK,
            MapElementType.PED_WALKWAY,
        }:
            # Drawing crosswalks and walkways.
            rasterize_world_polygon(
                map_elem.polygon.xy, ped_area_img, raster_from_world, color=(0, 0, 255)
            )

    map_data: np.ndarray = (lane_area_img + lane_line_img + ped_area_img).astype(
        np.float32
    ).transpose(2, 0, 1) / 255

    rasterized_map_info = RasterizedMapMetadata(
        name=vec_map.map_name,
        shape=map_data.shape,
        layers=["drivable_area", "lane_divider", "ped_area"],
        layer_rgb_groups=([0], [1], [2]),
        resolution=resolution,
        map_from_world=raster_from_world,
    )

    return RasterizedMap(rasterized_map_info, map_data)
