# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import bokeh
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, GlyphRenderer
from bokeh.plotting import curdoc, figure
from PIL import Image
from shapely.geometry import LineString, Polygon

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.state import StateArray
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    RoadArea,
    RoadEdge,
    RoadLane,
)
from trajdata.utils.arr_utils import (
    batch_nd_transform_points_np,
    batch_nd_transform_points_pt,
    transform_coords_2d_np,
)


def apply_default_settings(fig: figure) -> None:
    # Pixel dimensions match data dimensions,
    # a 1x1 area in data space is a square in pixels.
    fig.match_aspect = True

    # No gridlines.
    fig.grid.visible = False

    # Setting the scroll wheel to active by default.
    fig.toolbar.active_scroll = fig.tools[1]

    # Set autohide to true to only show the toolbar when mouse is over plot.
    fig.toolbar.autohide = True

    # Setting the match_aspect property of bokeh's default BoxZoomTool.
    fig.tools[2].match_aspect = True

    fig.xaxis.axis_label_text_font_size = "10pt"
    fig.xaxis.major_label_text_font_size = "10pt"

    fig.yaxis.axis_label_text_font_size = "10pt"
    fig.yaxis.major_label_text_font_size = "10pt"

    fig.title.text_font_size = "13pt"


def calculate_figure_sizes(
    data_bbox: Tuple[float, float, float, float],
    data_margin: float = 10,
    aspect_ratio: float = 16 / 9,
) -> Tuple[float, float, float, float]:
    """_summary_

    Args:
        data_bbox (Tuple[float, float, float, float]): x_min, x_max, y_min, y_max (in data units)
        data_margin (float, optional): _description_. Defaults to 10.
        aspect_ratio (float, optional): _description_. Defaults to 16/9.

    Returns:
        Tuple[float, float, float, float]: Visualization x_min, x_max, y_min, y_max (in data units) matching the desired aspect ratio and clear margin around data points.
    """
    x_min, x_max, y_min, y_max = data_bbox

    x_range = x_max - x_min
    x_center = (x_min + x_max) / 2

    y_range = y_max - y_min
    y_center = (y_min + y_max) / 2

    radius = (x_range / 2 if x_range > y_range else y_range / 2) + data_margin
    return (
        x_center - radius,
        x_center + radius,
        y_center - radius / aspect_ratio,
        y_center + radius / aspect_ratio,
    )


def pretty_print_agent_type(agent_type: AgentType):
    return str(agent_type).capitalize()


def agent_type_to_str(agent_type_int: int) -> str:
    return pretty_print_agent_type(AgentType(agent_type_int).name)


def get_agent_type_color(agent_type: AgentType) -> str:
    palette = sns.color_palette("husl", 8).as_hex()
    if agent_type == AgentType.VEHICLE:
        return palette[3]
    elif agent_type == AgentType.PEDESTRIAN:
        return "darkorange"
    elif agent_type == AgentType.BICYCLE:
        return palette[5]
    elif agent_type == AgentType.MOTORCYCLE:
        return palette[7]
    elif agent_type == "EGO":
        return palette[0]
    else:
        return "#A9A9A9"


def get_map_patch_color(map_elem_type: MapElementType) -> str:
    if map_elem_type == MapElementType.ROAD_AREA:
        return "gray"
    elif map_elem_type == MapElementType.ROAD_LANE:
        return "red"
    elif map_elem_type == MapElementType.PED_CROSSWALK:
        return "gold"  # "blue"
    elif map_elem_type == MapElementType.PED_WALKWAY:
        return "green"
    else:
        raise ValueError()


def get_multi_line_bbox(
    lines_data: ColumnDataSource,
) -> Tuple[float, float, float, float]:
    """_summary_

    Args:
        lines_data (ColumnDataSource): _description_

    Returns:
        Tuple[float, float, float, float]: x_min, x_max, y_min, y_max
    """
    all_xs = np.concatenate(lines_data.data["xs"], axis=0)
    all_ys = np.concatenate(lines_data.data["ys"], axis=0)
    all_xy = np.stack((all_xs, all_ys), axis=1)
    x_min, y_min = np.nanmin(all_xy, axis=0)
    x_max, y_max = np.nanmax(all_xy, axis=0)
    return (
        x_min.item(),
        x_max.item(),
        y_min.item(),
        y_max.item(),
    )


def compute_agent_rect_coords(
    agent_type: int, heading: float, length: float, width: float
) -> Tuple[np.ndarray, np.ndarray]:
    agent_rect_coords = transform_coords_2d_np(
        np.array(
            [
                [-length / 2, -width / 2],
                [-length / 2, width / 2],
                [length / 2, width / 2],
                [length / 2, -width / 2],
            ]
        ),
        angle=heading,
    )

    size = 1.0
    if agent_type == AgentType.PEDESTRIAN or agent_type == AgentType.BICYCLE:
        size = 0.25

    dir_patch_coords = transform_coords_2d_np(
        np.array(
            [
                [0, np.sqrt(3) / 3],
                [-1 / 2, -np.sqrt(3) / 6],
                [1 / 2, -np.sqrt(3) / 6],
            ]
        )
        * size,
        angle=heading - np.pi / 2,
    )

    return agent_rect_coords, dir_patch_coords


def compute_agent_rects_coords(
    agent_type: int, hs: np.ndarray, lengths: np.ndarray, widths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    raw_rect_coords = np.stack(
        (
            np.stack((-lengths / 2, -widths / 2), axis=-1),
            np.stack((-lengths / 2, widths / 2), axis=-1),
            np.stack((lengths / 2, widths / 2), axis=-1),
            np.stack((lengths / 2, -widths / 2), axis=-1),
        ),
        axis=-2,
    )

    agent_rect_coords = transform_coords_2d_np(
        raw_rect_coords,
        angle=hs[:, None].repeat(raw_rect_coords.shape[-2], axis=-1),
    )

    size = 1.0
    if agent_type == AgentType.PEDESTRIAN or agent_type == AgentType.BICYCLE:
        size = 0.25

    raw_tri_coords = size * np.array(
        [
            [
                [0, np.sqrt(3) / 3],
                [-1 / 2, -np.sqrt(3) / 6],
                [1 / 2, -np.sqrt(3) / 6],
            ]
        ]
    ).repeat(hs.shape[0], axis=0)

    dir_patch_coords = transform_coords_2d_np(
        raw_tri_coords,
        angle=hs[:, None].repeat(raw_tri_coords.shape[-2], axis=-1) - np.pi / 2,
    )

    return agent_rect_coords, dir_patch_coords


def extract_full_agent_data_df(
    batch: AgentBatch, batch_idx: int, violations
) -> pd.DataFrame:
    main_data_dict = defaultdict(list)

    # Historical information
    ## Agent
    H = batch.agent_hist_len[batch_idx].item()
    agent_type = batch.agent_type[batch_idx].item()
    agent_extent: np.ndarray = batch.agent_hist_extent[batch_idx, -H:].cpu().numpy()
    agent_hist_np: StateArray = batch.agent_hist[batch_idx, -H:].cpu().numpy()

    speed_mps = np.linalg.norm(agent_hist_np.velocity, axis=1)

    xs = agent_hist_np.get_attr("x")
    ys = agent_hist_np.get_attr("y")
    hs = agent_hist_np.get_attr("h")

    lengths = agent_extent[:, 0]
    widths = agent_extent[:, 1]

    agent_rect_coords, dir_patch_coords = compute_agent_rects_coords(
        agent_type, hs, lengths, widths
    )

    main_data_dict["id"].extend([0] * H)
    main_data_dict["t"].extend(range(-H + 1, 1))
    main_data_dict["x"].extend(xs)
    main_data_dict["y"].extend(ys)
    main_data_dict["h"].extend(hs)
    main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
    main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
    main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
    main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
    main_data_dict["speed_mps"].extend(speed_mps)
    main_data_dict["speed_kph"].extend(speed_mps * 3.6)
    main_data_dict["type"].extend([agent_type_to_str(agent_type)] * H)
    main_data_dict["length"].extend(lengths)
    main_data_dict["width"].extend(widths)
    main_data_dict["pred_agent"].extend([True] * H)
    # main_data_dict["color"].extend([get_agent_type_color(agent_type)] * H)
    main_data_dict["color"] = ["lightblue"] * H
    main_data_dict["violation"] = [""] * H

    ## Neighbors
    num_neighbors: int = batch.num_neigh[batch_idx].item()

    for n_neigh in range(num_neighbors):
        H = batch.neigh_hist_len[batch_idx, n_neigh].item()
        agent_type = batch.neigh_types[batch_idx, n_neigh].item()
        agent_extent: np.ndarray = (
            batch.neigh_hist_extents[batch_idx, n_neigh, -H:].cpu().numpy()
        )
        agent_hist_np: StateArray = (
            batch.neigh_hist[batch_idx, n_neigh, -H:].cpu().numpy()
        )

        speed_mps = np.linalg.norm(agent_hist_np.velocity, axis=1)

        xs = agent_hist_np.get_attr("x")
        ys = agent_hist_np.get_attr("y")
        hs = agent_hist_np.get_attr("h")

        lengths = agent_extent[:, 0]
        widths = agent_extent[:, 1]

        agent_rect_coords, dir_patch_coords = compute_agent_rects_coords(
            agent_type, hs, lengths, widths
        )

        main_data_dict["id"].extend([n_neigh + 1] * H)
        main_data_dict["t"].extend(range(-H + 1, 1))
        main_data_dict["x"].extend(xs)
        main_data_dict["y"].extend(ys)
        main_data_dict["h"].extend(hs)
        main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
        main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
        main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
        main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
        main_data_dict["speed_mps"].extend(speed_mps)
        main_data_dict["speed_kph"].extend(speed_mps * 3.6)
        main_data_dict["type"].extend([agent_type_to_str(agent_type)] * H)
        main_data_dict["length"].extend(lengths)
        main_data_dict["width"].extend(widths)
        main_data_dict["pred_agent"].extend([False] * H)
        main_data_dict["color"].extend([get_agent_type_color(agent_type)] * H)
        main_data_dict["violation"].extend([""] * H)

    # Future information
    ## Agent
    T = batch.agent_fut_len[batch_idx].item()
    agent_type = batch.agent_type[batch_idx].item()
    agent_extent: np.ndarray = batch.agent_fut_extent[batch_idx, :T].cpu().numpy()
    agent_fut_np: StateArray = batch.agent_fut[batch_idx, :T].cpu().numpy()

    speed_mps = np.linalg.norm(agent_fut_np.velocity, axis=1)

    xs = agent_fut_np.get_attr("x")
    ys = agent_fut_np.get_attr("y")
    hs = agent_fut_np.get_attr("h")

    lengths = agent_extent[:, 0]
    widths = agent_extent[:, 1]

    agent_rect_coords, dir_patch_coords = compute_agent_rects_coords(
        agent_type, hs, lengths, widths
    )

    main_data_dict["id"].extend([0] * T)
    main_data_dict["t"].extend(range(1, T + 1))
    main_data_dict["x"].extend(xs)
    main_data_dict["y"].extend(ys)
    main_data_dict["h"].extend(hs)
    main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
    main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
    main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
    main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
    main_data_dict["speed_mps"].extend(speed_mps)
    main_data_dict["speed_kph"].extend(speed_mps * 3.6)
    main_data_dict["type"].extend([agent_type_to_str(agent_type)] * T)
    main_data_dict["length"].extend(lengths)
    main_data_dict["width"].extend(widths)
    main_data_dict["pred_agent"].extend([True] * T)
    main_data_dict["color"].extend(["lightblue"] * T)
    main_data_dict["violation"].extend([""] * T)
    # main_data_dict["color"].extend([get_agent_type_color(agent_type)] * T)
    if violations is not None:
        main_data_dict["color"][
            -T : len(main_data_dict["violation"]) - T + len(violations)
        ] = [
            "lightblue" if len(violation) == 0 else get_agent_type_color("EGO")
            for violation in violations
        ]
        main_data_dict["violation"][
            -T : len(main_data_dict["violation"]) - T + len(violations)
        ] = [",".join(violation) for violation in violations]
    ## Neighbors
    num_neighbors: int = batch.num_neigh[batch_idx].item()

    for n_neigh in range(num_neighbors):
        T = batch.neigh_fut_len[batch_idx, n_neigh].item()
        agent_type = batch.neigh_types[batch_idx, n_neigh].item()
        agent_extent: np.ndarray = (
            batch.neigh_fut_extents[batch_idx, n_neigh, :T].cpu().numpy()
        )
        agent_fut_np: StateArray = batch.neigh_fut[batch_idx, n_neigh, :T].cpu().numpy()

        speed_mps = np.linalg.norm(agent_fut_np.velocity, axis=1)

        xs = agent_fut_np.get_attr("x")
        ys = agent_fut_np.get_attr("y")
        hs = agent_fut_np.get_attr("h")

        lengths = agent_extent[:, 0]
        widths = agent_extent[:, 1]

        agent_rect_coords, dir_patch_coords = compute_agent_rects_coords(
            agent_type, hs, lengths, widths
        )

        main_data_dict["id"].extend([n_neigh + 1] * T)
        main_data_dict["t"].extend(range(1, T + 1))
        main_data_dict["x"].extend(xs)
        main_data_dict["y"].extend(ys)
        main_data_dict["h"].extend(hs)
        main_data_dict["rect_xs"].extend(agent_rect_coords[..., 0] + xs[:, None])
        main_data_dict["rect_ys"].extend(agent_rect_coords[..., 1] + ys[:, None])
        main_data_dict["dir_patch_xs"].extend(dir_patch_coords[..., 0] + xs[:, None])
        main_data_dict["dir_patch_ys"].extend(dir_patch_coords[..., 1] + ys[:, None])
        main_data_dict["speed_mps"].extend(speed_mps)
        main_data_dict["speed_kph"].extend(speed_mps * 3.6)
        main_data_dict["type"].extend([agent_type_to_str(agent_type)] * T)
        main_data_dict["length"].extend(lengths)
        main_data_dict["width"].extend(widths)
        main_data_dict["pred_agent"].extend([False] * T)
        main_data_dict["color"].extend([get_agent_type_color(agent_type)] * T)
        main_data_dict["violation"].extend([""] * T)

        # Fix: Ensure all lists are of the same length
        min_lenght = min(len(v) for v in main_data_dict.values())
        for k in main_data_dict.keys():
            main_data_dict[k] = main_data_dict[k][:min_lenght]

    return pd.DataFrame(main_data_dict)


def convert_to_gpd(vec_map: VectorMap) -> gpd.GeoDataFrame:
    geo_data = defaultdict(list)
    for elem in vec_map.iter_elems():
        geo_data["id"].append(elem.id)
        geo_data["type"].append(elem.elem_type)
        if isinstance(elem, RoadLane):
            geo_data["geometry"].append(LineString(elem.center.xyz))
        elif isinstance(elem, PedCrosswalk) or isinstance(elem, PedWalkway):
            geo_data["geometry"].append(Polygon(shell=elem.polygon.xyz))
        elif isinstance(elem, RoadArea):
            geo_data["geometry"].append(
                Polygon(
                    shell=elem.exterior_polygon.xyz,
                    holes=[hole.xyz for hole in elem.interior_holes],
                )
            )
        elif isinstance(elem, RoadEdge):
            geo_data["geometry"].append(LineString(elem.polyline.xyz))

    return gpd.GeoDataFrame(geo_data)


def get_map_cds(
    map_from_world_tf: np.ndarray,
    vec_map: VectorMap,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[
    ColumnDataSource,
    ColumnDataSource,
    ColumnDataSource,
    ColumnDataSource,
    ColumnDataSource,
    ColumnDataSource,
]:
    road_lane_data = defaultdict(list)
    lane_center_data = defaultdict(list)
    ped_crosswalk_data = defaultdict(list)
    ped_walkway_data = defaultdict(list)
    road_area_data = defaultdict(list)
    road_edge_data = defaultdict(list)

    map_gpd = convert_to_gpd(vec_map)
    affine_tf_params = (
        map_from_world_tf[:2, :2].flatten().tolist()
        + map_from_world_tf[:2, -1].flatten().tolist()
    )
    map_gpd["geometry"] = map_gpd["geometry"].affine_transform(affine_tf_params)

    elems_gdf: gpd.GeoDataFrame
    if bbox is not None:
        elems_gdf = map_gpd.cx[bbox[0] : bbox[1], bbox[2] : bbox[3]]
    else:
        elems_gdf = map_gpd

    for row_idx, row in elems_gdf.iterrows():
        if row["type"] == MapElementType.PED_CROSSWALK:
            xy = np.stack(row["geometry"].exterior.xy, axis=1)
            ped_crosswalk_data["xs"].append(xy[..., 0])
            ped_crosswalk_data["ys"].append(xy[..., 1])
        if row["type"] == MapElementType.PED_WALKWAY:
            xy = np.stack(row["geometry"].exterior.xy, axis=1)
            ped_walkway_data["xs"].append(xy[..., 0])
            ped_walkway_data["ys"].append(xy[..., 1])
        elif row["type"] == MapElementType.ROAD_LANE:
            xy = np.stack(row["geometry"].xy, axis=1)
            lane_center_data["xs"].append(xy[..., 0])
            lane_center_data["ys"].append(xy[..., 1])
            lane_obj: RoadLane = vec_map.elements[MapElementType.ROAD_LANE][row["id"]]
            if lane_obj.left_edge is not None and lane_obj.right_edge is not None:
                left_xy = lane_obj.left_edge.xy
                right_xy = lane_obj.right_edge.xy[::-1]
                patch_xy = np.concatenate((left_xy, right_xy), axis=0)

                transformed_xy: np.ndarray = transform_coords_2d_np(
                    patch_xy,
                    offset=map_from_world_tf[:2, -1],
                    rot_mat=map_from_world_tf[:2, :2],
                )

                road_lane_data["xs"].append(transformed_xy[..., 0])
                road_lane_data["ys"].append(transformed_xy[..., 1])
        elif row["type"] == MapElementType.ROAD_AREA:
            xy = np.stack(row["geometry"].exterior.xy, axis=1)
            holes_xy: List[np.ndarray] = [
                np.stack(interior.xy, axis=1) for interior in row["geometry"].interiors
            ]

            road_area_data["xs"].append(
                [[xy[..., 0]] + [hole[..., 0] for hole in holes_xy]]
            )
            road_area_data["ys"].append(
                [[xy[..., 1]] + [hole[..., 1] for hole in holes_xy]]
            )
        elif row["type"] == MapElementType.ROAD_EDGE:
            xy = np.stack(row["geometry"].xy, axis=1)
            road_edge_data["xs"].append(xy[..., 0])
            road_edge_data["ys"].append(xy[..., 1])
    return (
        ColumnDataSource(data=lane_center_data),
        ColumnDataSource(data=road_lane_data),
        ColumnDataSource(data=ped_crosswalk_data),
        ColumnDataSource(data=ped_walkway_data),
        ColumnDataSource(data=road_area_data),
        ColumnDataSource(data=road_edge_data),
    )


def draw_map_elems(
    fig: figure,
    vec_map: VectorMap,
    map_from_world_tf: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    **kwargs,
) -> Tuple[GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer]:
    """_summary_

    Args:
        fig (Figure): _description_
        vec_map (VectorMap): _description_
        map_from_world_tf (np.ndarray): _description_
        bbox (Tuple[float, float, float, float]): x_min, x_max, y_min, y_max

    Returns:
        Tuple[GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer]: _description_
    """
    (
        lane_center_cds,
        road_lane_cds,
        ped_crosswalk_cds,
        ped_walkway_cds,
        road_area_cds,
        road_edge_cds,
    ) = get_map_cds(map_from_world_tf, vec_map, bbox)

    road_areas = fig.multi_polygons(
        source=road_area_cds,
        line_color="black",
        line_width=0.3,
        fill_alpha=0.1,
        fill_color=get_map_patch_color(MapElementType.ROAD_AREA),
    )

    road_lanes = fig.patches(
        source=road_lane_cds,
        line_color="black",
        line_width=0.3,
        fill_alpha=0.1,
        fill_color=get_map_patch_color(MapElementType.ROAD_LANE),
    )

    ped_crosswalks = fig.patches(
        source=ped_crosswalk_cds,
        line_color="black",
        line_width=0.3,
        fill_alpha=0.5,
        fill_color=get_map_patch_color(MapElementType.PED_CROSSWALK),
    )

    ped_walkways = fig.patches(
        source=ped_walkway_cds,
        line_color="black",
        line_width=0.3,
        fill_alpha=0.3,
        fill_color=get_map_patch_color(MapElementType.PED_WALKWAY),
    )

    lane_centers = fig.multi_line(
        source=lane_center_cds,
        line_color="gray",
        line_alpha=0.5,
    )

    if kwargs.get("incl_road_edges", True):
        road_edges = fig.multi_line(
            source=road_edge_cds,
            line_color="red",
            line_alpha=0.5,
        )

    return road_areas, road_lanes, ped_crosswalks, ped_walkways, lane_centers
