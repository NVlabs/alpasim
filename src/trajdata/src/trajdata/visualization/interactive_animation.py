# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import logging
import os
import socket
import threading
import time
import warnings
from collections import defaultdict
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import panel as pn
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.document import Document, without_document_lock
from bokeh.io import export_png
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import column, row
from bokeh.models import (
    BooleanFilter,
    Button,
    CDSView,
    ColumnDataSource,
    HoverTool,
    Legend,
    LegendItem,
    RangeSlider,
    Select,
    Slider,
)
from bokeh.plotting import curdoc, figure
from bokeh.server.server import Server
from selenium import webdriver
from tornado import gen
from tornado.ioloop import IOLoop
from tqdm import trange

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.state import StateArray
from trajdata.maps.map_api import MapAPI
from trajdata.utils import vis_utils


def animate_agent_batch_interactive(
    batch: AgentBatch,
    batch_idx: int,
    cache_path: Path,
    render_immediately: bool = False,
    incl_road_edges: bool = False,
    embeded: bool = False,
    image_files: List[str] = None,
    violations=None,
) -> None:
    agent_data_df = vis_utils.extract_full_agent_data_df(batch, batch_idx, violations)

    # figure creation and a few initial settings.
    width: int = 1280
    aspect_ratio: float = 16 / 9
    data_vis_margin: float = 10.0

    x_min = agent_data_df["x"].min()
    x_max = agent_data_df["x"].max()

    y_min = agent_data_df["y"].min()
    y_max = agent_data_df["y"].max()

    (
        x_range_min,
        x_range_max,
        y_range_min,
        y_range_max,
    ) = vis_utils.calculate_figure_sizes(
        data_bbox=(x_min, x_max, y_min, y_max),
        data_margin=data_vis_margin,
        aspect_ratio=aspect_ratio,
    )

    kwargs = {
        "x_range": (x_range_min, x_range_max),
        "y_range": (y_range_min, y_range_max),
    }

    fig = figure(
        width=width,
        height=int(width / aspect_ratio),
        output_backend="canvas",
        **kwargs,
    )
    vis_utils.apply_default_settings(fig)

    agent_name: str = batch.agent_name[batch_idx]
    agent_type: AgentType = AgentType(batch.agent_type[batch_idx].item())
    current_state: StateArray = batch.curr_agent_state[batch_idx].cpu().numpy()
    map_id: str = batch.map_names[batch_idx]
    env_name, map_name = map_id.split(":")
    scene_id: str = batch.scene_ids[batch_idx]
    fig.title = (
        f"Dataset: {env_name}, Location: {map_name}, Scene: {scene_id}"
        + "\n"
        + f"Agent ID: {agent_name} ({vis_utils.pretty_print_agent_type(agent_type)}) at x = {current_state[0]:.2f} m, y = {current_state[1]:.2f} m, heading = {current_state[-1]:.2f} rad ({np.rad2deg(current_state[-1]):.2f} deg)"
    )

    # Map plotting.
    if batch.map_names is not None:
        mapAPI = MapAPI(cache_path)

        vec_map = mapAPI.get_map(
            batch.map_names[batch_idx],
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True,
            incl_road_edges=True if incl_road_edges else False,
        )

        (
            road_areas,
            road_lanes,
            ped_crosswalks,
            ped_walkways,
            lane_centers,
        ) = vis_utils.draw_map_elems(
            fig,
            vec_map,
            batch.agents_from_world_tf[batch_idx].cpu().numpy(),
            bbox=(
                x_min - data_vis_margin,
                x_max + data_vis_margin,
                y_min - data_vis_margin,
                y_max + data_vis_margin,
            ),
        )

    # Preparing agent information for fast slicing with the time_slider.
    agent_cds = ColumnDataSource(agent_data_df)
    curr_time_view = CDSView(filter=BooleanFilter((agent_cds.data["t"] == 0).tolist()))

    # Some neighbors can have more history than the agent to be predicted
    # (the data-collecting agent has observed the neighbors for longer).
    if batch.neigh_hist_len[batch_idx].shape[0] == 0:
        full_H = batch.agent_hist_len[batch_idx].item()
        full_T = batch.agent_fut_len[batch_idx].item()
    else:
        full_H = max(
            batch.agent_hist_len[batch_idx].item(),
            *batch.neigh_hist_len[batch_idx].tolist(),
        )
        full_T = max(
            batch.agent_fut_len[batch_idx].item(),
            *batch.neigh_fut_len[batch_idx].tolist(),
        )

    def create_multi_line_data(agents_df: pd.DataFrame) -> Dict[str, List]:
        lines_data = defaultdict(list)
        for agent_id, agent_df in agents_df.groupby(by="id"):
            xs, ys, color = (
                agent_df.x.to_numpy(),
                agent_df.y.to_numpy(),
                agent_df.color.iat[0],
            )

            if agent_id == 0:
                pad_before = full_H - batch.agent_hist_len[batch_idx].item()
                pad_after = full_T - batch.agent_fut_len[batch_idx].item()

            else:
                pad_before = (
                    full_H - batch.neigh_hist_len[batch_idx, agent_id - 1].item()
                )
                pad_after = full_T - batch.neigh_fut_len[batch_idx, agent_id - 1].item()

            xs = np.pad(xs, (pad_before, pad_after), constant_values=np.nan)
            ys = np.pad(ys, (pad_before, pad_after), constant_values=np.nan)

            lines_data["xs"].append(xs)
            lines_data["ys"].append(ys)
            lines_data["color"].append(color)

        return lines_data

    def slice_multi_line_data(
        multi_line_df: Dict[str, Any], slice_obj, check_idx: int
    ) -> Dict[str, List]:
        lines_data = defaultdict(list)
        for i in range(len(multi_line_df["xs"])):
            sliced_xs = multi_line_df["xs"][i][slice_obj]
            sliced_ys = multi_line_df["ys"][i][slice_obj]
            if (
                sliced_xs.shape[0] > 0
                and sliced_ys.shape[0] > 0
                and np.isfinite(sliced_xs[check_idx])
                and np.isfinite(sliced_ys[check_idx])
            ):
                lines_data["xs"].append(sliced_xs)
                lines_data["ys"].append(sliced_ys)
                lines_data["color"].append(multi_line_df["color"][i])

        return lines_data

    # Getting initial historical and future trajectory information ready for plotting.
    history_line_data_df = create_multi_line_data(agent_data_df)
    history_lines_cds = ColumnDataSource(
        slice_multi_line_data(history_line_data_df, slice(None, full_H), check_idx=-1)
    )
    future_line_data_df = history_line_data_df.copy()
    future_lines_cds = ColumnDataSource(
        slice_multi_line_data(future_line_data_df, slice(full_H, None), check_idx=0)
    )

    history_lines = fig.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_dash="dashed",
        line_width=2,
        source=history_lines_cds,
    )

    future_lines = fig.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_dash="solid",
        line_width=2,
        source=future_lines_cds,
    )

    # Agent rectangles/directional arrows at the current timestep.
    agent_rects = fig.patches(
        xs="rect_xs",
        ys="rect_ys",
        fill_color="color",
        line_color="black",
        # fill_alpha=0.7,
        source=agent_cds,
        view=curr_time_view,
    )

    agent_dir_patches = fig.patches(
        xs="dir_patch_xs",
        ys="dir_patch_ys",
        fill_color="color",
        line_color="black",
        # fill_alpha=0.7,
        source=agent_cds,
        view=curr_time_view,
    )
    if batch.extras is not None and "action_sample" in batch.extras:
        action_sample = batch.extras["action_sample"][batch_idx, 0]
        action_sample = action_sample.cpu().numpy()
        action_sample = np.concatenate(
            [
                np.full([full_H, *action_sample.shape[1:]], fill_value=np.nan),
                action_sample,
            ],
            axis=0,
        )
        action_sample_cds = ColumnDataSource(
            dict(
                xs=[action_sample[0, i, :, 0] for i in range(action_sample.shape[1])],
                ys=[action_sample[0, i, :, 1] for i in range(action_sample.shape[1])],
            )
        )
        action_sample_lines = fig.multi_line(
            xs="xs",
            ys="ys",
            line_color="green",
            line_dash="dashed",
            line_width=1.5,
            source=action_sample_cds,
        )
    else:
        action_sample_lines = None
    scene_ts: int = batch.scene_ts[batch_idx].item()

    # Controlling the timestep shown to users.
    end_time: int = (
        min(agent_cds.data["t"].max(), len(violations))
        if violations is not None
        else agent_cds.data["t"].max()
    )
    total_timesteps: int = end_time - agent_cds.data["t"].min() + 1
    abs_total_timesteps: int = agent_cds.data["t"].max() - agent_cds.data["t"].min() + 1
    time_slider = Slider(
        start=agent_cds.data["t"].min(),
        end=end_time,
        step=1,
        value=0,
        title=f"Current Timestep (scene timestep {scene_ts})",
    )

    dt: float = batch.dt[batch_idx].item()

    # adding image if available
    if image_files is not None:
        from PIL import Image

        def pil_image_to_rgba(image_file):
            image = Image.open(image_file)
            image = image.convert("RGBA")  # Ensure the image is in RGBA mode
            img_array = np.array(image)  # Convert the image to a numpy array
            # Flatten the RGBA array into a 1D array in the format Bokeh expects (row-major order)
            img_flat = np.flipud(img_array).flatten()
            return img_flat.view(np.uint32).reshape((image.height, image.width))

        img_source = ColumnDataSource(
            data={"image": [pil_image_to_rgba(image_files[0])]}
        )

        # Set up the figure for displaying the image
        p = figure(
            x_range=(0, 1),
            y_range=(0, 1),
            width=Image.open(image_files[0]).width,
            height=Image.open(image_files[0]).height,
        )
        p.image_rgba(image="image", source=img_source, x=0, y=0, dw=1, dh=1)
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        p.outline_line_color = None

    # Ensuring that information gets updated upon a cahnge in the slider value.
    def time_callback(attr, old, new) -> None:
        curr_time_view.filter = BooleanFilter((agent_cds.data["t"] == new).tolist())
        history_lines_cds.data = slice_multi_line_data(
            history_line_data_df, slice(None, new + full_H), check_idx=-1
        )
        future_lines_cds.data = slice_multi_line_data(
            future_line_data_df, slice(new + full_H, None), check_idx=0
        )
        if action_sample_lines is not None:
            action_sample_cds.data = dict(
                xs=[
                    action_sample[new + full_H, i, :, 0]
                    for i in range(action_sample.shape[1])
                ],
                ys=[
                    action_sample[new + full_H, i, :, 1]
                    for i in range(action_sample.shape[1])
                ],
            )

        if new == 0:
            time_slider.title = f"Current Timestep (scene timestep {scene_ts})"
        else:
            n_steps = abs(new)
            time_slider.title = f"{n_steps} timesteps ({n_steps * dt:.2f} s) into the {'future' if new > 0 else 'past'}"

        if image_files is not None:
            new_image_idx = new * len(image_files) // abs_total_timesteps
            img_source.data = {"image": [pil_image_to_rgba(image_files[new_image_idx])]}

    time_slider.on_change("value", time_callback)

    # Adding tooltips on mouse hover.
    fig.add_tools(
        HoverTool(
            tooltips=[
                ("Class", "@type"),
                ("Position", "(@x, @y) m"),
                ("Speed", "@speed_mps m/s (@speed_kph km/h)"),
                ("Violation", "@violation"),
            ],
            renderers=[agent_rects],
        )
    )

    exit_button = Button(label="Exit", button_type="danger", width=60)

    def button_callback():
        # Stop the server.
        import sys

        from tornado.ioloop import IOLoop

        sys.exit()

    exit_button.on_click(button_callback)

    # Writing animation callback functions so that the play/pause button animate the
    # data according to its native dt.
    def animate_update():
        t = time_slider.value + 1

        if t > time_slider.end:
            # If slider value + 1 is above max, reset to 0.
            t = 0

        time_slider.value = t

    play_cb_manager = [None]

    def animate():
        if play_button.label.startswith("►"):
            play_button.label = "❚❚ Pause"

            play_cb_manager[0] = play_button.document.add_periodic_callback(
                animate_update, period_milliseconds=int(dt * 1000)
            )
        else:
            play_button.label = "► Play"
            play_button.document.remove_periodic_callback(play_cb_manager[0])

    play_button = Button(label="► Play", width=100)
    play_button.on_click(animate)

    # Creating the legend elements and connecting them to their original elements
    # (allows us to hide them on click later!)
    agent_legend_elems = [
        fig.rect(fill_color="lightblue", line_color="black", name="EGO"),
        fig.rect(
            fill_color=vis_utils.get_agent_type_color("EGO"),
            line_color="black",
            name="EGO_Violation",
        ),
    ]
    agent_legend_elems.extend(
        [
            fig.rect(
                fill_color=vis_utils.get_agent_type_color(x),
                line_color="black",
                name=vis_utils.agent_type_to_str(x),
            )
            for x in AgentType
        ]
    )

    map_legend_elems = [LegendItem(label="Lane Center", renderers=[lane_centers])]

    map_area_legend_elems = [
        LegendItem(label="Road Area", renderers=[road_areas]),
        LegendItem(label="Road Lanes", renderers=[road_lanes]),
        LegendItem(label="Crosswalks", renderers=[ped_crosswalks]),
        LegendItem(label="Sidewalks", renderers=[ped_walkways]),
    ]

    hist_future_legend_elems = [
        LegendItem(
            label="Past Motion",
            renderers=[
                history_lines,
                fig.multi_line(
                    line_color="black", line_dash="dashed", line_alpha=1.0, line_width=2
                ),
            ],
        ),
        LegendItem(
            label="Future Motion",
            renderers=[
                future_lines,
                fig.multi_line(
                    line_color="black", line_dash="solid", line_alpha=1.0, line_width=2
                ),
            ],
        ),
    ]

    # Adding the legend to the figure.
    legend = Legend(
        items=[
            LegendItem(label=legend_item.name, renderers=[legend_item])
            for legend_item in agent_legend_elems
        ]
        + hist_future_legend_elems
        + map_legend_elems
        + map_area_legend_elems,
        click_policy="hide",
        label_text_font_size="15pt",
        spacing=10,
    )
    fig.add_layout(legend, "right")

    # Video rendering functions.
    video_button = Button(
        label="Render Video",
        width=120,
    )

    render_range_slider = RangeSlider(
        value=(0, time_slider.end),
        start=time_slider.start,
        end=time_slider.end,
        title=f"Timesteps to Render",
    )

    filetype_select = Select(
        title="Filetype:", value=".mp4", options=[".mp4", ".avi"], width=80
    )

    def reset_buttons() -> None:
        video_button.label = "Render Video"
        video_button.disabled = False
        time_slider.disabled = False
        render_range_slider.disabled = False
        filetype_select.disabled = False
        play_button.disabled = False
        fig.toolbar_location = "right"

        logging.basicConfig(level=logging.WARNING, force=True)

    def after_frame_save(label: str) -> None:
        video_button.label = label
        animate_update()

    def execute_save_animation(file_path: Path) -> None:
        raise NotImplementedError()

    @gen.coroutine
    @without_document_lock
    def save_animation(filename: str) -> None:
        video_button.label = "Rendering..."
        video_button.disabled = True
        time_slider.disabled = True
        render_range_slider.disabled = True
        filetype_select.disabled = True
        play_button.disabled = True
        fig.toolbar_location = None

        # Bokeh logs a lot of warnings here related to some figure elements not having
        # 'x', 'y', etc attributes set (most of these are legend items for which this
        # is intentional). Accordingly, ignore WARNINGs and ERRORs now and re-enable
        # them after.
        logging.basicConfig(level=logging.CRITICAL, force=True)

        # Stop any ongoing animation.
        if play_button.label.startswith("❚❚"):
            animate()

        # Reset the current timestep to the left end of the range.
        time_slider.value = render_range_slider.value[0]

        threading.Thread(
            target=execute_save_animation,
            args=(Path(filename + filetype_select.value),),
        ).start()

    video_button_fn = partial(
        save_animation,
        filename=("_".join([env_name, map_name, scene_id, f"t{scene_ts}", agent_name])),
    )
    video_button.on_click(video_button_fn)

    layout = column(
        fig if image_files is None else row(fig, p),
        # row(play_button, time_slider, exit_button),
        row(play_button, time_slider),
        row(video_button, render_range_slider, filetype_select),
    )
    bokeh_pane = pn.pane.Bokeh(layout)

    if embeded:
        return bokeh_pane

    server = pn.serve(bokeh_pane)
    if render_immediately:
        video_button_fn()

    return server
