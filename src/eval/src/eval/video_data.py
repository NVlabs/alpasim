# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from shapely import LineString, Point, STRtree
from trajdata import maps
from trajdata.maps import VectorMap

from eval.schema import EvalConfig


@dataclasses.dataclass
class PointPlot:
    """Represents a point to be plotted. Used to plot the vec_map."""

    name: str
    points: list[Point]
    color: str
    marker: str
    size: float = 6


@dataclasses.dataclass
class PolylinePlot:
    """Represents a polyline to be plotted. Used to plot the vec_map."""

    name: str
    linestrings: list[LineString]
    linewidth: float
    style: str
    alpha: float = 1.0


@dataclasses.dataclass
class RenderableLineString:
    """Represents a description of a line string to be plotted."""

    linestring: LineString
    name: str | None = None
    linewidth: float | None = None
    style: str | None = None
    alpha: float | None = None
    artists: list[plt.Artist] | None = None
    zorder: float | None = None
    color: str | None = None

    def set_plot_style(
        self,
        name: str,
        linewidth: float,
        style: str,
        alpha: float,
        color: str | None = None,
        zorder: float | None = None,
    ) -> None:
        self.name: str = name
        self.linewidth: float = linewidth
        self.style: str = style
        self.alpha: float = alpha
        self.color: str | None = color
        self.zorder: float | None = zorder
        self.artists: list[plt.Artist] | None = None

    def remove_artist(self) -> None:
        if self.artists is not None:
            self.artists[0].remove()
            self.artists = None

    def render(self, ax: plt.Axes) -> list[plt.Artist]:
        assert (
            self.name is not None
            and self.linewidth is not None
            and self.style is not None
            and self.alpha is not None
        ), "Before rendering, you must call set_plot_style"
        if self.artists is not None:
            raise RuntimeError("You should not re-render an existing linestring.")
        plot_kwargs = {
            "linewidth": self.linewidth,
            "alpha": self.alpha,
        }
        if self.color is not None:
            plot_kwargs["color"] = self.color
        if self.zorder is not None:
            plot_kwargs["zorder"] = self.zorder

        self.artists = ax.plot(
            self.linestring.xy[0],
            self.linestring.xy[1],
            self.style,
            **plot_kwargs,
        )
        return self.artists


@dataclasses.dataclass
class ShapelyMap:
    """Represents a map with shapely objects."""

    renderable_linestrings: list[RenderableLineString]
    # TODOs: Make work for points
    # * Split traffic signs by type
    # * Add points to str_tree
    # * Rendering error where using rotation transform does something weird.
    # point_plots: list[PointPlot]
    str_tree: STRtree
    currently_rendered_linestring_ids: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([])
    )

    @staticmethod
    def from_vec_map(vec_map: VectorMap | None) -> "ShapelyMap":
        if vec_map is None:
            return ShapelyMap(
                renderable_linestrings=[],
                str_tree=STRtree([]),
            )
        renderable_linestrings = []

        road_lane_elements = vec_map.elements[
            maps.vec_map_elements.MapElementType.ROAD_LANE
        ]
        road_edge_elements = vec_map.elements[
            maps.vec_map_elements.MapElementType.ROAD_EDGE
        ]
        wait_line_elements = vec_map.elements[
            maps.vec_map_elements.MapElementType.WAIT_LINE
        ]
        stop_line_elements = [
            e for e in wait_line_elements.values() if e.wait_line_type == "STOP"
        ]
        other_line_elements = [
            e for e in wait_line_elements.values() if e.wait_line_type != "STOP"
        ]
        # traffic_sign_elements = vec_map.elements[
        #     maps.vec_map_elements.MapElementType.TRAFFIC_SIGN
        # ]

        for element in road_lane_elements.values():
            renderable_linestrings.append(
                RenderableLineString(
                    linestring=LineString(element.center.xy),
                    name="road_lane_center",
                    linewidth=1,
                    style="b--",
                    alpha=0.5,
                )
            )

        for element in road_lane_elements.values():
            if element.left_edge is not None:
                renderable_linestrings.append(
                    RenderableLineString(
                        linestring=LineString(element.left_edge.xy),
                        name="road_lane_left_edge",
                        linewidth=1,
                        style="b-",
                        alpha=1.0,
                    )
                )

        for element in road_lane_elements.values():
            if element.right_edge is not None:
                renderable_linestrings.append(
                    RenderableLineString(
                        linestring=LineString(element.right_edge.xy),
                        name="road_lane_right_edge",
                        linewidth=1,
                        style="b-",
                        alpha=1.0,
                    )
                )

        for element in road_edge_elements.values():
            renderable_linestrings.append(
                RenderableLineString(
                    linestring=LineString(element.polyline.xy),
                    name="road_edge",
                    linewidth=1,
                    style="k-",
                    alpha=1.0,
                )
            )

        for element in stop_line_elements:
            renderable_linestrings.append(
                RenderableLineString(
                    linestring=LineString(element.polyline.xy),
                    name="stop_line",
                    linewidth=1,
                    style="r--",
                    alpha=1.0,
                )
            )

        for element in other_line_elements:
            renderable_linestrings.append(
                RenderableLineString(
                    linestring=LineString(element.polyline.xy),
                    name="other_line",
                    linewidth=1,
                    style="y--",
                    alpha=1.0,
                )
            )

        # TODO: Currently doesn't work with transform...
        # point_plots = []
        # point_plots.append(
        #     PointPlot(
        #         name="traffic_sign",
        #         points=[
        #             Point(traffic_sign.position[:2])
        #             for traffic_sign in traffic_sign_elements.values()
        #         ],
        #         color="orange",
        #         marker="8",
        #     )
        # )

        str_tree = STRtree([r.linestring for r in renderable_linestrings])
        return ShapelyMap(
            renderable_linestrings,
            str_tree,
        )

    def get_linestring_idxs_in_radius(self, center: Point, radius: float) -> np.ndarray:
        return self.str_tree.query(center.buffer(radius), "intersects")

    def render(
        self,
        ax: plt.Axes,
        cfg: EvalConfig,
        center: Point | None = None,
        max_dist: float | None = None,
    ) -> dict[str, list[plt.Artist]]:
        """Render the map elements.

        Subsequent calls to this function will update the existing artist for
        elements inside the drawing radius and remove unnecessary artists for
        elements outside the drawing radius.

        Args:
            ax: The axis to render the map elements on.
            cfg: The configuration for the map elements.
            center: The center of the map elements. Only required if max_dist is
                provided.
            max_dist: The maximum distance to the center of the map elements.
        """
        assert (
            max_dist is None or center is not None
        ), "center must be provided if max_dist is provided"

        current_linestrings_ids: Iterable[int] = (
            np.arange(len(self.renderable_linestrings))
            if max_dist is None
            else self.get_linestring_idxs_in_radius(center, max_dist)
        )

        # Filter for elements to plot
        if cfg.video.map_video.map_elements_to_plot is not None:
            current_linestring_ids = [
                id
                for id in current_linestrings_ids
                if self.renderable_linestrings[id].name
                in cfg.video.map_video.map_elements_to_plot
            ]

        # Remove artists for linestrings that are no longer in the radius
        for linestring_id in set(self.currently_rendered_linestring_ids) - set(
            current_linestring_ids
        ):
            self.renderable_linestrings[linestring_id].remove_artist()

        # Render new linestrings
        for linestring_id in set(current_linestring_ids) - set(
            self.currently_rendered_linestring_ids
        ):
            self.renderable_linestrings[linestring_id].render(ax)
        self.currently_rendered_linestring_ids = current_linestring_ids

        return {
            "map": [
                artist
                for linestring_id in current_linestring_ids
                for artist in self.renderable_linestrings[linestring_id].artists
            ],
        }
