# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import List, Optional, Tuple
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Polygon
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch, SceneBatch
from trajdata.data_structures.state import StateTensor
from trajdata.maps import RasterizedMap
from trajdata.maps.vec_map_elements import RoadLane


def draw_agent(
    ax: Axes,
    agent_type: AgentType,
    agent_state: StateTensor,
    agent_extent: Tensor,
    agent_to_world_tf: Tensor,
    **kwargs,
) -> None:
    """Draws a path with the correct location, heading, and dimensions onto the given axes

    Args:
        ax (Axes): _description_
        agent_type (AgentType): _description_
        agent_state (Tensor): _description_
        agent_extent (Tensor): _description_
        agent_to_world_tf (Tensor): _description_
    """

    if torch.any(torch.isnan(agent_extent)):
        if agent_type == AgentType.VEHICLE:
            length = 4.3
            width = 1.8
        elif agent_type == AgentType.PEDESTRIAN:
            length = 0.5
            width = 0.5
        elif agent_type == AgentType.BICYCLE:
            length = 1.9
            width = 0.5
        else:
            length = 1.0
            width = 1.0
    else:
        length = agent_extent[0].item()
        width = agent_extent[1].item()

    xy = agent_state.position
    heading = agent_state.heading

    patch = FancyBboxPatch([-length / 2, -width / 2], length, width, **kwargs)
    transform = (
        mtransforms.Affine2D().rotate(heading[0].item()).translate(xy[0], xy[1])
        + mtransforms.Affine2D(matrix=agent_to_world_tf.cpu().numpy())
        + ax.transData
    )
    patch.set_transform(transform)

    kwargs["label"] = None
    size = 1.0
    angles = [0, 2 * np.pi / 3, np.pi, 4 * np.pi / 3]
    pts = np.stack([size * np.cos(angles), size * np.sin(angles)], axis=-1)
    center_patch = Polygon(pts, zorder=10.0, **kwargs)
    center_patch.set_transform(transform)

    ax.add_patch(patch)
    ax.add_patch(center_patch)


def draw_history(
    ax: Axes,
    agent_type: AgentType,
    agent_history: StateTensor,
    agent_extent: Tensor,
    agent_to_world_tf: Tensor,
    start_alpha: float = 0.2,
    end_alpha: float = 0.5,
    **kwargs,
):
    T = agent_history.shape[0]
    alphas = np.linspace(start_alpha, end_alpha, T)
    for t in range(T):
        draw_agent(
            ax,
            agent_type,
            agent_history[t],
            agent_extent,
            agent_to_world_tf,
            alpha=alphas[t],
            **kwargs,
        )


def draw_map(
    ax: Axes, map: Tensor, base_frame_from_map_tf: Tensor, alpha=1.0, **kwargs
):
    patch_size: int = map.shape[-1]
    map_array = RasterizedMap.to_img(map.cpu())
    brightened_map_array = map_array * 0.2 + 0.8

    im = ax.imshow(
        brightened_map_array,
        extent=[0, patch_size, patch_size, 0],
        clip_on=True,
        **kwargs,
    )
    transform = (
        mtransforms.Affine2D(matrix=base_frame_from_map_tf.cpu().numpy()) + ax.transData
    )
    im.set_transform(transform)

    coords = np.array(
        [[0, 0, 1], [patch_size, 0, 1], [patch_size, patch_size, 1], [0, patch_size, 1]]
    )
    world_frame_corners = base_frame_from_map_tf.cpu().numpy() @ coords[:, :, None]
    xmin = np.min(world_frame_corners[:, 0, 0])
    xmax = np.max(world_frame_corners[:, 0, 0])
    ymin = np.min(world_frame_corners[:, 1, 0])
    ymax = np.max(world_frame_corners[:, 1, 0])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def draw_lanes(
    ax: Axes,
    lanes: List[RoadLane],
    centered_agent_from_world_tf: Tensor,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
):
    transform = (
        mtransforms.Affine2D(matrix=centered_agent_from_world_tf.cpu().numpy())
        + ax.transData
    )
    for lane in lanes:
        ax.plot(
            lane.center.xy[:, 0],
            lane.center.xy[:, 1],
            linestyle="--",
            color=color,
            transform=transform,
        )


def plot_agent_batch_all(
    batch: AgentBatch,
    ax: Optional[Axes] = None,
    show: bool = True,
    close: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    # Use first agent as common reference frame
    base_frame_from_world_tf = batch.agents_from_world_tf[0].cpu()

    # plot maps over each other with proper transformations:
    for i in range(len(batch.agent_name)):
        base_frame_from_map_tf = base_frame_from_world_tf @ torch.linalg.inv(
            batch.rasters_from_world_tf[i].cpu()
        )
        draw_map(ax, batch.maps[i], base_frame_from_map_tf, alpha=1.0)

    for i in range(len(batch.agent_name)):
        agent_type = batch.agent_type[i]
        agent_name = batch.agent_name[i]
        agent_hist = batch.agent_hist[i, :, :].cpu()
        agent_fut = batch.agent_fut[i, :, :].cpu()
        agent_extent = batch.agent_hist_extent[i, -1, :].cpu()
        base_frame_from_agent_tf = base_frame_from_world_tf @ torch.linalg.inv(
            batch.agents_from_world_tf[i].cpu()
        )

        palette = sns.color_palette("husl", 4)
        if agent_type == AgentType.VEHICLE:
            color = palette[0]
        elif agent_type == AgentType.PEDESTRIAN:
            color = palette[1]
        elif agent_type == AgentType.BICYCLE:
            color = palette[2]
        else:
            color = palette[3]

        transform = (
            mtransforms.Affine2D(matrix=base_frame_from_agent_tf.numpy()) + ax.transData
        )
        draw_history(
            ax,
            agent_type,
            agent_hist[:-1, :],
            agent_extent,
            base_frame_from_agent_tf,
            facecolor="None",
            edgecolor=color,
            linewidth=0,
        )
        ax.plot(
            agent_hist[:, 0],
            agent_hist[:, 1],
            linestyle="--",
            color=color,
            transform=transform,
        )
        draw_agent(
            ax,
            agent_type,
            agent_hist[-1, :],
            agent_extent,
            base_frame_from_agent_tf,
            facecolor=color,
            edgecolor="k",
        )
        ax.plot(
            agent_fut[:, 0],
            agent_fut[:, 1],
            linestyle="-",
            color=color,
            transform=transform,
        )

    ax.set_ylim(-30, 40)
    ax.set_xlim(-30, 40)
    ax.grid(False)

    if show:
        plt.show()

    if close:
        plt.close()


def plot_agent_batch(
    batch: AgentBatch,
    batch_idx: int,
    ax: Optional[Axes] = None,
    legend: bool = True,
    show: bool = True,
    close: bool = True,
) -> None:
    if ax is None:
        _, ax = plt.subplots()

    agent_name: str = batch.agent_name[batch_idx]
    agent_type: AgentType = AgentType(batch.agent_type[batch_idx].item())
    current_state = batch.curr_agent_state[batch_idx].cpu().numpy()
    ax.set_title(
        f"{str(agent_type)}/{agent_name}\nat x={current_state[0]:.2f},y={current_state[1]:.2f},h={current_state[-1]:.2f}"
    )

    agent_from_world_tf: Tensor = batch.agents_from_world_tf[batch_idx].cpu()

    if batch.maps is not None:
        world_from_raster_tf: Tensor = torch.linalg.inv(
            batch.rasters_from_world_tf[batch_idx].cpu()
        )

        agent_from_raster_tf: Tensor = agent_from_world_tf @ world_from_raster_tf

        draw_map(ax, batch.maps[batch_idx], agent_from_raster_tf, alpha=1.0)

    agent_hist = batch.agent_hist[batch_idx].cpu()
    agent_fut = batch.agent_fut[batch_idx].cpu()
    agent_extent = batch.agent_hist_extent[batch_idx, -1, :].cpu()
    base_frame_from_agent_tf = torch.eye(3)

    palette = sns.color_palette("husl", 4)
    if agent_type == AgentType.VEHICLE:
        color = palette[0]
    elif agent_type == AgentType.PEDESTRIAN:
        color = palette[1]
    elif agent_type == AgentType.BICYCLE:
        color = palette[2]
    else:
        color = palette[3]

    draw_history(
        ax,
        agent_type,
        agent_hist[:-1],
        agent_extent,
        base_frame_from_agent_tf,
        facecolor=color,
        edgecolor=None,
        linewidth=0,
    )
    ax.plot(
        agent_hist.get_attr("x"),
        agent_hist.get_attr("y"),
        linestyle="--",
        color=color,
        label="Agent History",
    )
    draw_agent(
        ax,
        agent_type,
        agent_hist[-1],
        agent_extent,
        base_frame_from_agent_tf,
        facecolor=color,
        edgecolor="k",
        label="Agent Current",
    )
    ax.plot(
        agent_fut.get_attr("x"),
        agent_fut.get_attr("y"),
        linestyle="-",
        color=color,
        label="Agent Future",
    )

    num_neigh = batch.num_neigh[batch_idx]
    if num_neigh > 0:
        neighbor_hist = batch.neigh_hist[batch_idx].cpu()
        neighbor_fut = batch.neigh_fut[batch_idx].cpu()
        neighbor_extent = batch.neigh_hist_extents[batch_idx, :, -1, :].cpu()
        neighbor_type = batch.neigh_types[batch_idx].cpu()

        ax.plot([], [], c="olive", ls="--", label="Neighbor History")
        ax.plot([], [], c="darkgreen", label="Neighbor Future")

        for n in range(num_neigh):
            if torch.isnan(neighbor_hist[n, -1, :]).any():
                # this neighbor does not exist at the current timestep
                continue
            ax.plot(
                neighbor_hist.get_attr("x")[n, :],
                neighbor_hist.get_attr("y")[n, :],
                c="olive",
                ls="--",
            )
            draw_agent(
                ax,
                neighbor_type[n],
                neighbor_hist[n, -1],
                neighbor_extent[n, :],
                base_frame_from_agent_tf,
                facecolor="olive",
                edgecolor="k",
                alpha=0.7,
            )
            ax.plot(
                neighbor_fut.get_attr("x")[n, :],
                neighbor_fut.get_attr("y")[n, :],
                c="darkgreen",
            )

    if batch.robot_fut is not None and batch.robot_fut.shape[1] > 0:
        ax.plot(
            batch.robot_fut.get_attr("x")[batch_idx, 1:],
            batch.robot_fut.get_attr("y")[batch_idx, 1:],
            label="Ego Future",
            c="blue",
        )
        ax.scatter(
            batch.robot_fut.get_attr("x")[batch_idx, 0],
            batch.robot_fut.get_attr("y")[batch_idx, 0],
            s=20,
            c="blue",
            label="Ego Current",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")

    # Doing this because the imshow above makes the map origin at the top.
    # TODO(pkarkus) we should just modify imshow not to change the origin instead.
    ax.invert_yaxis()

    if legend:
        ax.legend(loc="best", frameon=True)

    if show:
        plt.show()

    if close:
        plt.close()

    return ax


def plot_scene_batch(
    batch: SceneBatch,
    batch_idx: int,
    ax: Optional[Axes] = None,
    plot_vec_map: bool = False,
    vec_map_search_radius: float = 100,
    show: bool = True,
    close: bool = True,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    num_agents: int = batch.num_agents[batch_idx].item()

    agent_from_world_tf: Tensor = batch.centered_agent_from_world_tf[batch_idx].cpu()

    if plot_vec_map and batch.vector_maps is not None:
        try:
            search_point = (
                batch.centered_agent_state.position3d[batch_idx].cpu().numpy()
            )
        except ValueError:
            warn(
                "could not compute 3d position. try adding 'z' component to state format, "
                "e.g. state_format='x,y,z,xd,yd,xdd,ydd,h'"
            )
            raise
        vec_map = batch.vector_maps[batch_idx]
        lanes = vec_map.get_lanes_within(search_point, vec_map_search_radius)
        draw_lanes(ax, lanes, agent_from_world_tf)
    elif batch.maps is not None:
        centered_agent_id = 0
        world_from_raster_tf: Tensor = torch.linalg.inv(
            batch.rasters_from_world_tf[batch_idx, centered_agent_id].cpu()
        )

        agent_from_raster_tf: Tensor = agent_from_world_tf @ world_from_raster_tf

        draw_map(
            ax,
            batch.maps[batch_idx, centered_agent_id],
            agent_from_raster_tf,
            alpha=1.0,
        )

    base_frame_from_agent_tf = torch.eye(3)
    agent_hist = batch.agent_hist[batch_idx]
    agent_type = batch.agent_type[batch_idx]
    agent_extent = batch.agent_hist_extent[batch_idx, :, -1]
    agent_fut = batch.agent_fut[batch_idx]

    for agent_id in range(num_agents):
        ax.plot(
            agent_hist.get_attr("x")[agent_id],
            agent_hist.get_attr("y")[agent_id],
            c="orange",
            ls="--",
            label="Agent History" if agent_id == 0 else None,
        )
        draw_agent(
            ax,
            agent_type[agent_id],
            agent_hist[agent_id, -1],
            agent_extent[agent_id],
            base_frame_from_agent_tf,
            facecolor="olive",
            edgecolor="k",
            alpha=0.7,
            label="Agent Current" if agent_id == 0 else None,
        )
        ax.plot(
            agent_fut.get_attr("x")[agent_id],
            agent_fut.get_attr("y")[agent_id],
            c="violet",
            label="Agent Future" if agent_id == 0 else None,
        )

    if batch.robot_fut is not None and batch.robot_fut.shape[1] > 0:
        ax.plot(
            batch.robot_fut.get_attr("x")[batch_idx, 1:],
            batch.robot_fut.get_attr("y")[batch_idx, 1:],
            label="Ego Future",
            c="blue",
        )
        ax.scatter(
            batch.robot_fut.get_attr("x")[batch_idx, 0],
            batch.robot_fut.get_attr("y")[batch_idx, 0],
            s=20,
            c="blue",
            label="Ego Current",
        )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)

    # Doing this because the imshow above makes the map origin at the top.
    ax.invert_yaxis()

    if show:
        plt.show()

    if close:
        plt.close()

    return ax
