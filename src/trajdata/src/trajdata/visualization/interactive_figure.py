# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Optional, Tuple

import bokeh.plotting as plt
import numpy as np
import torch
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting import figure
from torch import Tensor

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.state import StateTensor
from trajdata.maps import VectorMap
from trajdata.utils import vis_utils


class InteractiveFigure:
    def __init__(self, **kwargs) -> None:
        self.aspect_ratio: float = kwargs.get("aspect_ratio", 16 / 9)
        self.width: int = kwargs.pop("width", 1280)
        self.height: int = kwargs.pop("height", int(self.width / self.aspect_ratio))

        # We'll be tracking the maxes and mins of data with these.
        self.x_min = np.inf
        self.x_max = -np.inf
        self.y_min = np.inf
        self.y_max = -np.inf

        self.raw_figure = figure(width=self.width, height=self.height, **kwargs)
        vis_utils.apply_default_settings(self.raw_figure)

    def update_mins_maxs(self, x_min, x_max, y_min, y_max) -> None:
        self.x_min = min(self.x_min, x_min)
        self.x_max = max(self.x_max, x_max)
        self.y_min = min(self.y_min, y_min)
        self.y_max = max(self.y_max, y_max)

    def show(self) -> None:
        if np.isfinite((self.x_min, self.x_max, self.y_min, self.y_max)).all():
            (
                x_range_min,
                x_range_max,
                y_range_min,
                y_range_max,
            ) = vis_utils.calculate_figure_sizes(
                data_bbox=(self.x_min, self.x_max, self.y_min, self.y_max),
                aspect_ratio=self.aspect_ratio,
            )

            self.raw_figure.x_range = Range1d(x_range_min, x_range_max)
            self.raw_figure.y_range = Range1d(y_range_min, y_range_max)

        plt.show(self.raw_figure)

    def add_line(self, states: StateTensor, **kwargs) -> GlyphRenderer:
        xy_pos = states.position.cpu().numpy()

        x_min, y_min = np.nanmin(xy_pos, axis=0)
        x_max, y_max = np.nanmax(xy_pos, axis=0)
        self.update_mins_maxs(x_min.item(), x_max.item(), y_min.item(), y_max.item())

        return self.raw_figure.line(xy_pos[:, 0], xy_pos[:, 1], **kwargs)

    def add_lines(self, lines_data: ColumnDataSource, **kwargs) -> GlyphRenderer:
        self.update_mins_maxs(*vis_utils.get_multi_line_bbox(lines_data))
        return self.raw_figure.multi_line(
            source=lines_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            **{x: x for x in lines_data.column_names},
            **kwargs,
        )

    def add_map(
        self,
        map_from_world_tf: np.ndarray,
        vec_map: VectorMap,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Tuple[
        GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer
    ]:
        """_summary_

        Args:
            map_from_world_tf (np.ndarray): _description_
            vec_map (VectorMap): _description_
            bbox (Tuple[float, float, float, float]): x_min, x_max, y_min, y_max

        Returns:
            Tuple[ GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer, GlyphRenderer ]: _description_
        """
        return vis_utils.draw_map_elems(
            self.raw_figure, vec_map, map_from_world_tf, bbox, **kwargs
        )

    def add_agent(
        self,
        agent_type: AgentType,
        agent_state: StateTensor,
        agent_extent: Tensor,
        **kwargs,
    ) -> Tuple[GlyphRenderer, GlyphRenderer]:
        """Draws an agent at the given location, heading, and dimensions.

        Args:
            agent_type (AgentType): _description_
            agent_state (Tensor): _description_
            agent_extent (Tensor): _description_
        """
        if torch.any(torch.isnan(agent_extent)):
            raise ValueError("Agent extents cannot be NaN!")

        length = agent_extent[0].item()
        width = agent_extent[1].item()

        x, y = agent_state.position.cpu().numpy()
        heading = agent_state.heading.cpu().numpy()

        agent_rect_coords, dir_patch_coords = vis_utils.compute_agent_rect_coords(
            agent_type, heading, length, width
        )

        source = {
            "x": agent_rect_coords[:, 0] + x,
            "y": agent_rect_coords[:, 1] + y,
            "type": [vis_utils.pretty_print_agent_type(agent_type)],
            "speed": [torch.linalg.norm(agent_state.velocity).item()],
        }

        r = self.raw_figure.patch(
            x="x",
            y="y",
            source=source,
            **kwargs,
        )
        p = self.raw_figure.patch(
            x=dir_patch_coords[:, 0] + x, y=dir_patch_coords[:, 1] + y, **kwargs
        )

        return r, p

    def add_agents(
        self,
        agent_rects_data: ColumnDataSource,
        dir_patches_data: ColumnDataSource,
        **kwargs,
    ) -> Tuple[GlyphRenderer, GlyphRenderer]:
        r = self.raw_figure.patches(
            source=agent_rects_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            xs="xs",
            ys="ys",
            fill_alpha="fill_alpha",
            fill_color="fill_color",
            line_color="line_color",
            **kwargs,
        )

        p = self.raw_figure.patches(
            source=dir_patches_data,
            # This is to ensure that the columns given in the
            # ColumnDataSource are respected (e.g., "line_color").
            **{x: x for x in dir_patches_data.column_names},
            **kwargs,
        )

        return r, p
