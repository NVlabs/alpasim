# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path

import numpy as np
from bokeh.models import ColumnDataSource

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.data_structures.state import StateArray, StateTensor
from trajdata.maps.map_api import MapAPI
from trajdata.utils import vis_utils
from trajdata.utils.arr_utils import transform_coords_2d_np
from trajdata.visualization.interactive_figure import InteractiveFigure


def plot_agent_batch_interactive(
    batch: AgentBatch, batch_idx: int, cache_path: Path, **kwargs
) -> None:
    fig = InteractiveFigure(
        tooltips=[
            ("Class", "@type"),
            ("Position", "(@x, @y) m"),
            ("Speed", "@speed_mps m/s (@speed_kph km/h)"),
        ]
    )

    agent_type: int = batch.agent_type[batch_idx].item()
    num_neighbors: int = batch.num_neigh[batch_idx].item()
    agent_hist_np: StateArray = batch.agent_hist[batch_idx].cpu().numpy()
    neigh_hist_np: StateArray = batch.neigh_hist[batch_idx].cpu().numpy()
    neigh_types = batch.neigh_types[batch_idx].cpu().numpy()
    agent_histories = ColumnDataSource(
        data={
            "xs": [agent_hist_np.get_attr("x")]
            + [
                neigh_hist_np[n_neigh].get_attr("x") for n_neigh in range(num_neighbors)
            ],
            "ys": [agent_hist_np.get_attr("y")]
            + [
                neigh_hist_np[n_neigh].get_attr("y") for n_neigh in range(num_neighbors)
            ],
            "line_dash": ["dashed"] * (num_neighbors + 1),
            "line_color": [vis_utils.get_agent_type_color(agent_type)]
            + [
                vis_utils.get_agent_type_color(neigh_types[n_neigh])
                for n_neigh in range(num_neighbors)
            ],
        }
    )

    agent_fut_np: StateArray = batch.agent_fut[batch_idx].cpu().numpy()
    neigh_fut_np: StateArray = batch.neigh_fut[batch_idx].cpu().numpy()
    agent_futures = ColumnDataSource(
        data={
            "xs": [agent_fut_np.get_attr("x")]
            + [neigh_fut_np[n_neigh].get_attr("x") for n_neigh in range(num_neighbors)],
            "ys": [agent_fut_np.get_attr("y")]
            + [neigh_fut_np[n_neigh].get_attr("y") for n_neigh in range(num_neighbors)],
            "line_dash": ["solid"] * (num_neighbors + 1),
            "line_color": [vis_utils.get_agent_type_color(agent_type)]
            + [
                vis_utils.get_agent_type_color(neigh_types[n_neigh])
                for n_neigh in range(num_neighbors)
            ],
        }
    )

    agent_state: StateArray = batch.agent_hist[batch_idx, -1].cpu().numpy()
    x, y = agent_state.position

    if batch.map_names is not None:
        map_vis_radius: float = 50.0
        mapAPI = MapAPI(cache_path)
        fig.add_map(
            batch.agents_from_world_tf[batch_idx].cpu().numpy(),
            mapAPI.get_map(
                batch.map_names[batch_idx],
                incl_road_lanes=True,
                incl_road_areas=True,
                incl_ped_crosswalks=True,
                incl_ped_walkways=True,
                incl_road_edges=True if kwargs.get("incl_road_edges", False) else False,
            ),
            # x_min, x_max, y_min, y_max
            bbox=(
                x - map_vis_radius,
                x + map_vis_radius,
                y - map_vis_radius,
                y + map_vis_radius,
            ),
            kwargs=kwargs,
        )

    fig.add_lines(agent_histories)
    if agent_fut_np.shape[0] > 0:
        fig.add_lines(agent_futures)

    agent_extent: np.ndarray = batch.agent_hist_extent[batch_idx, -1]
    if agent_extent.isnan().any():
        raise ValueError("Agent extents cannot be NaN!")

    length = agent_extent[0].item()
    width = agent_extent[1].item()

    heading: float = agent_state.heading.item()
    speed_mps: float = np.linalg.norm(agent_state.velocity).item()

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

    agent_rects_data = {
        "x": [x],
        "y": [y],
        "xs": [agent_rect_coords[:, 0] + x],
        "ys": [agent_rect_coords[:, 1] + y],
        # "fill_color": [vis_utils.get_agent_type_color(agent_type)],
        "fill_color": [vis_utils.get_agent_type_color("EGO")],  # ego color
        "line_color": ["black"],
        "fill_alpha": [0.7],
        "type": [str(AgentType(agent_type))[len("AgentType.") :]],
        "speed_mps": [speed_mps],
        "speed_kph": [speed_mps * 3.6],
    }

    size = 1.0
    if agent_type == AgentType.PEDESTRIAN:
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
    dir_patches_data = {
        "xs": [dir_patch_coords[:, 0] + x],
        "ys": [dir_patch_coords[:, 1] + y],
        "fill_color": [vis_utils.get_agent_type_color("EGO")],  # ego color
        # "fill_color": [vis_utils.get_agent_type_color(agent_type)],
        "line_color": ["black"],
        "alpha": [0.7],
    }

    for n_neigh in range(num_neighbors):
        agent_type: int = batch.neigh_types[batch_idx, n_neigh].item()
        agent_state: StateArray = batch.neigh_hist[batch_idx, n_neigh, -1].cpu().numpy()
        agent_extent: np.ndarray = batch.neigh_hist_extents[batch_idx, n_neigh, -1]

        if agent_extent.isnan().any():
            continue
            # raise ValueError("Agent extents cannot be NaN!")

        length = agent_extent[0].item()
        width = agent_extent[1].item()

        x, y = agent_state.position
        heading: float = agent_state.heading.item()
        speed_mps: float = np.linalg.norm(agent_state.velocity).item()

        agent_rect_coords, dir_patch_coords = vis_utils.compute_agent_rect_coords(
            agent_type, heading, length, width
        )

        agent_rects_data["x"].append(x)
        agent_rects_data["y"].append(y)
        agent_rects_data["xs"].append(agent_rect_coords[:, 0] + x)
        agent_rects_data["ys"].append(agent_rect_coords[:, 1] + y)
        agent_rects_data["fill_color"].append(
            vis_utils.get_agent_type_color(agent_type)
        )
        agent_rects_data["line_color"].append("black")
        agent_rects_data["fill_alpha"].append(0.7)
        agent_rects_data["type"].append(str(AgentType(agent_type))[len("AgentType.") :])
        agent_rects_data["speed_mps"].append(speed_mps)
        agent_rects_data["speed_kph"].append(speed_mps * 3.6)

        dir_patches_data["xs"].append(dir_patch_coords[:, 0] + x)
        dir_patches_data["ys"].append(dir_patch_coords[:, 1] + y)
        dir_patches_data["fill_color"].append(
            vis_utils.get_agent_type_color(agent_type)
        )
        dir_patches_data["line_color"].append("black")
        dir_patches_data["alpha"].append(0.7)

    rects, _ = fig.add_agents(
        ColumnDataSource(data=agent_rects_data), ColumnDataSource(data=dir_patches_data)
    )

    fig.raw_figure.hover.renderers = [rects]
    fig.show()
