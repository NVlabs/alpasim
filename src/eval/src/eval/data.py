# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import dataclasses
import io
import logging
import pickle
from enum import StrEnum
from typing import Callable, Iterable, Literal, Optional

import matplotlib.image as mpimg
import matplotlib.transforms as transforms
import numpy as np
import polars as pl
import shapely
from alpasim_grpc.v0 import common_pb2
from alpasim_grpc.v0.egodriver_pb2 import DriveResponse, RolloutCameraImage, Route
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasReturn, CameraSpec
from alpasim_utils import geometry
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from scipy.spatial.transform import Rotation as R
from trajdata.maps import VectorMap

from eval.schema import EvalConfig, MetricVehicleConfig
from eval.video_data import RenderableLineString

logger = logging.getLogger("eval.data")

BOTTOM_CENTER_EGO_LOC_FACTOR = 0.25


@dataclasses.dataclass
class RAABB:
    """Represents a AABB with optional shrinkage and corner rounding."""

    size_x: float
    size_y: float
    size_z: float
    corner_radius_m: float

    @staticmethod
    def from_grpc(
        aabb: common_pb2.AABB, metric_vehicle_config: MetricVehicleConfig
    ) -> "RAABB":
        """Create a RAABB from a grpc AABB and vehicle config.

        Args:
            aabb: grpc AABB
            metric_vehicle_config: Config for vehicles in metrics. In
            particular, we use the fields `vehicle_shrink_factor` and
            `vehicle_corner_roundness` to shrink the AABB and round the corners.
            Both values must be between 0.0 and 1.0, with 0.0 meaning no shrinkage
            or rounding and 1.0 meaning the maximum shrinkage or rounding.
        Returns:
            RAABB
        """
        assert 0.0 <= metric_vehicle_config.vehicle_shrink_factor <= 1.0
        assert 0.0 <= metric_vehicle_config.vehicle_corner_roundness <= 1.0
        # Make sure we're not applying the shrinkage factor twice.
        assert isinstance(aabb, common_pb2.AABB) and not isinstance(aabb, RAABB)

        size_factor = 1.0 - metric_vehicle_config.vehicle_shrink_factor
        min_side_length = min(aabb.size_x, aabb.size_y) * size_factor

        corner_radius_m = (
            metric_vehicle_config.vehicle_corner_roundness * min_side_length / 2
        )
        return RAABB(
            size_x=aabb.size_x * size_factor,
            size_y=aabb.size_y * size_factor,
            size_z=aabb.size_z,
            corner_radius_m=corner_radius_m,
        )


class RenderableTrajectory:
    """Represents a grpc trajectory with bbox.

    Also handles its own rendering by managing it's visual appearance as well as
    it's matplotlib artists.

    Uses composition to wrap a Trajectory instance since Rust types cannot be subclassed.
    """

    def __init__(
        self,
        timestamps_us: np.ndarray,
        positions: np.ndarray,
        quaternions: np.ndarray,
        raabb: RAABB | None = None,
        polygon_artists: dict[str, list[plt.Artist]] | None = None,
        renderable_linestring: RenderableLineString | None = None,
        fill_color: str = "black",
        fill_alpha: float = 0.1,
    ):
        positions = np.asarray(positions, dtype=np.float32)
        quaternions = np.asarray(quaternions, dtype=np.float32)

        # Ensure 2D arrays
        if positions.ndim == 1 and len(positions) == 3:
            positions = positions.reshape(1, 3)
        if quaternions.ndim == 1 and len(quaternions) == 4:
            quaternions = quaternions.reshape(1, 4)

        # Store wrapped Trajectory instance (composition instead of inheritance)
        self._trajectory = geometry.Trajectory(
            np.asarray(timestamps_us, dtype=np.uint64),
            positions,
            quaternions,
        )

        self.raabb = raabb
        self.polygon_artists = polygon_artists
        self.renderable_linestring = renderable_linestring
        self.fill_color = fill_color
        self.fill_alpha = fill_alpha

    # Delegate properties to wrapped Trajectory
    @property
    def timestamps_us(self) -> np.ndarray:
        return self._trajectory.timestamps_us

    @property
    def positions(self) -> np.ndarray:
        return self._trajectory.positions

    @property
    def quaternions(self) -> np.ndarray:
        return self._trajectory.quaternions

    @property
    def yaws(self) -> np.ndarray:
        return self._trajectory.yaws

    @property
    def time_range_us(self) -> range:
        return self._trajectory.time_range_us

    def __len__(self) -> int:
        return len(self._trajectory)

    def is_empty(self) -> bool:
        return self._trajectory.is_empty()

    def get_pose(self, idx: int) -> geometry.Pose:
        return self._trajectory.get_pose(idx)

    def interpolate_pose(self, at_us: int) -> geometry.Pose:
        return self._trajectory.interpolate_pose(at_us)

    def interpolate(self, target_timestamps: np.ndarray) -> "RenderableTrajectory":
        """Interpolates trajectory to target timestamps."""
        interp = self._trajectory.interpolate(
            np.asarray(target_timestamps, dtype=np.uint64)
        )
        return RenderableTrajectory.from_trajectory(interp, self.raabb)

    @staticmethod
    def from_grpc_with_aabb(
        traj: common_pb2.Trajectory, raabb: RAABB
    ) -> "RenderableTrajectory":
        """Creates trajectory with bbox from grpc trajectory and aabb."""
        return RenderableTrajectory.from_trajectory(
            geometry.trajectory_from_grpc(traj), raabb
        )

    @staticmethod
    def from_trajectory(
        traj: geometry.Trajectory, raabb: RAABB
    ) -> "RenderableTrajectory":
        """Creates trajectory with bbox from trajectory and aabb."""
        return RenderableTrajectory(
            timestamps_us=traj.timestamps_us,
            positions=traj.positions,
            quaternions=traj.quaternions,
            raabb=raabb,
        )

    @staticmethod
    def create_empty_with_bbox(raabb: RAABB) -> "RenderableTrajectory":
        """Creates empty trajectory with specified bbox from aabb."""
        return RenderableTrajectory.from_trajectory(
            geometry.Trajectory.create_empty(), raabb
        )

    def update_absolute(self, timestamp_us: int, pose: geometry.Pose) -> None:
        """Append a new pose with absolute coordinates."""
        self._trajectory.update_absolute(timestamp_us, pose)

    def transform(
        self, transform: geometry.Pose, is_relative: bool = False
    ) -> "RenderableTrajectory":
        """Transforms trajectory with bbox by pose."""
        transformed = self._trajectory.transform(transform, is_relative)
        return RenderableTrajectory.from_trajectory(transformed, self.raabb)

    def interpolate_to_timestamps(
        self, ts_target: np.ndarray
    ) -> "RenderableTrajectory":
        """Interpolates trajectory to target timestamps."""
        return self.interpolate(ts_target)

    @property
    def corners(self) -> np.ndarray:
        """Returns bbox corners from positions, aabb and yaw. Shape (T, 4, 2)"""
        positions = self.positions
        yaws = self.yaws
        cx = positions[..., 0]
        cy = positions[..., 1]
        cos = np.cos(yaws)
        sin = np.sin(yaws)
        length = self.raabb.size_x
        width = self.raabb.size_y
        dx = length / 2
        dy = width / 2

        assert cx.shape == cy.shape == cos.shape == sin.shape
        assert cx.ndim == 1

        corners = np.array(
            [
                # Top-right corner, then clockwise?
                (cx + dx * cos - dy * sin, cy + dx * sin + dy * cos),
                (cx + dx * cos + dy * sin, cy + dx * sin - dy * cos),
                (cx - dx * cos + dy * sin, cy - dx * sin - dy * cos),
                (cx - dx * cos - dy * sin, cy - dx * sin + dy * cos),
            ]
        )

        return np.moveaxis(corners, -1, 0)  # (T, 4, 2)

    def to_linestring(self) -> shapely.LineString:
        """Returns shapely linestring from trajectory."""
        if self.is_empty():
            return shapely.LineString()
        return shapely.LineString(self.positions[:, 0:2])

    def to_polygons(self) -> list[shapely.Polygon]:
        """Returns list of shapely polygons from bbox corners."""
        if self.is_empty():
            return [shapely.Polygon()]
        polygons = shapely.creation.polygons(self.corners)
        # Shrinkage must happen first, because it will remove corner rounding.
        if self.raabb.corner_radius_m > 0.0:
            polygons = [
                polygon.buffer(-self.raabb.corner_radius_m).buffer(
                    self.raabb.corner_radius_m
                )
                for polygon in polygons
            ]
        return polygons  # type: ignore

    def to_point(self) -> shapely.Point:
        """Returns shapely point from trajectory.

        Trajectory must be a single timestamp.
        """
        if self.is_empty():
            return shapely.Point()
        positions = self.positions
        assert len(positions) == 1
        return shapely.Point(positions[0, 0], positions[0, 1])

    def _maybe_rounded_bumper_lines(
        self, lines: list[shapely.LineString]
    ) -> list[shapely.geometry.base.BaseGeometry] | list[shapely.LineString]:
        if self.raabb.corner_radius_m == 0.0:
            return lines
        polygons = self.to_polygons()
        bumper_geom = [
            line.buffer(self.raabb.corner_radius_m).intersection(polygon)
            for line, polygon in zip(lines, polygons, strict=True)
        ]
        return bumper_geom

    def to_front_bumper_lines(
        self,
    ) -> list[shapely.geometry.base.BaseGeometry] | list[shapely.LineString]:
        """Returns list of shapely linestrings from front bumper corners."""
        if self.is_empty():
            return [shapely.LineString()]
        lines = shapely.creation.linestrings(self.corners[:, 0:2])
        return self._maybe_rounded_bumper_lines(lines)  # type: ignore

    def to_rear_bumper_lines(self) -> list[shapely.LineString | shapely.Polygon]:
        """Returns list of shapely linestrings from rear bumper corners."""
        if self.is_empty():
            return [shapely.LineString()]
        lines = shapely.creation.linestrings(self.corners[:, 2:4])
        return self._maybe_rounded_bumper_lines(lines)  # type: ignore

    def get_polygon_at_time(self, time: int) -> shapely.Polygon:
        return self.interpolate_to_timestamps(np.array([time])).to_polygons()[0]

    def set_polygon_plot_style(
        self, fill_color: str | None = None, fill_alpha: float | None = None
    ) -> "RenderableTrajectory":
        """Only needs to be called if we want to plot the polygon."""
        if fill_color is not None:
            self.fill_color = fill_color
        if fill_alpha is not None:
            self.fill_alpha = fill_alpha
        return self

    def set_linestring_plot_style(
        self,
        name: str,
        linewidth: float,
        style: str,
        alpha: float,
        color: str | None = None,
        zorder: float | None = None,
    ) -> "RenderableTrajectory":
        """Only needs to be called if we want to plot the linestring."""
        if self.renderable_linestring is None:
            self.renderable_linestring = RenderableLineString(
                linestring=self.to_linestring(),
                name=name,
                linewidth=linewidth,
                style=style,
                alpha=alpha,
                color=color,
                zorder=zorder,
            )
        else:
            self.renderable_linestring.set_plot_style(
                name, linewidth, style, alpha, color, zorder
            )
        return self

    def remove_artists(self) -> None:
        """Removes artists from the axis. Needed for fast video rendering."""
        if self.polygon_artists is not None:
            for artist_list in self.polygon_artists.values():
                for artist in artist_list:
                    artist.remove()
            self.polygon_artists = None

    def render_linestring(self, ax: plt.Axes) -> dict[str, list[plt.Artist]]:
        """Render Trajectory as line. Should only be called once."""
        assert (
            self.renderable_linestring is not None
        ), "Before rendering, you must call set_linestring_plot_style"
        return {self.renderable_linestring.name: self.renderable_linestring.render(ax)}

    def render_polygon_at_time(
        self,
        ax: plt.Axes,
        time: int,
    ) -> dict[str, list[plt.Artist]]:
        """Render Trajectory as polygon.

        Can be called repeatedly to update the polygon to current time.
        """
        polygon = self.get_polygon_at_time(time)
        if self.polygon_artists is not None:
            self.polygon_artists["border"][0].set_data(
                polygon.exterior.xy[0],
                polygon.exterior.xy[1],
            )
            self.polygon_artists["fill"][0].set_xy(polygon.exterior.coords)
            return self.polygon_artists

        current_artists = {}
        current_artists["border"] = ax.plot(
            polygon.exterior.xy[0],
            polygon.exterior.xy[1],
            "k-",
            linewidth=1,
            alpha=0.5,
        )
        current_artists["fill"] = ax.fill(
            polygon.exterior.xy[0],
            polygon.exterior.xy[1],
            color=self.fill_color,
            alpha=self.fill_alpha,
        )
        self.polygon_artists = current_artists
        return current_artists


@dataclasses.dataclass
class DriverResponseAtTime:
    """Represents a driver response at a given time.

    `now_time_us` is the time when the response was predicted.
    `time_query_us` is the time _for which_ the response was predicted.
    """

    now_time_us: int
    time_query_us: int
    # driver_responses and sampled_driver_trajectories are already converted to
    # AABB frame. List over timesteps.
    selected_trajectory: RenderableTrajectory
    # List over timesteps. Each element is a list of sampled trajectories.
    sampled_trajectories: list[RenderableTrajectory]
    # Safety monitor safe (not triggered) status.
    safety_monitor_safe: Optional[bool] = None
    # Command name from driver debug info (e.g. "LEFT", "RIGHT", "STRAIGHT").
    command_name: Optional[str] = None
    # Optional reasoning text from driver debug info.
    reasoning_text: Optional[str] = None

    @staticmethod
    def _extract_debug_extra(driver_response: DriveResponse) -> dict | None:
        """Extract and unpickle the debug *extra* dict from a driver response.

        Args:
            driver_response: The :pyclass:`DriveResponse` containing the
                ``unstructured_debug_info`` bytes.

        Returns:
            The unpickled dictionary if available and valid, otherwise ``None``.
        """
        try:
            dbg_bytes = driver_response.debug_info.unstructured_debug_info
            if not dbg_bytes:
                # Only info, as this spams a lot
                logger.info("No unstructured debug info found")
                return None
            extra = pickle.loads(dbg_bytes)
            if isinstance(extra, dict):
                return extra
            logger.warning(
                "Expected dict in unstructured_debug_info, got %s", type(extra)
            )
            return None
        except Exception as exc:  # pragma: no cover – defensive, should rarely trigger
            logger.warning(
                "Failed to parse unstructured debug info for driver: %s", exc
            )
            return None

    @staticmethod
    def _parse_recovery_info(
        extra: dict | None, num_sampled: int
    ) -> tuple[bool, int | None]:
        """Return whether recovery is active and selected index.

        Args:
            extra: The debug dict (output of `_extract_debug_extra`).
            num_sampled: Number of sampled trajectories – used to validate
                `select_ix`.

        Returns:
            Tuple `(recovery_active, select_ix_when_recovery_active)` where
            `recovery_active` is `True` if a recovery trajectory is present
            and `select_ix_when_recovery_active` is the validated index or
            `None`.
        """

        if extra is None or extra.get("recovery_trajectory") is None:
            return False, None

        sel_ix = extra.get("select_ix")
        if isinstance(sel_ix, int) and 0 <= sel_ix < num_sampled:
            return True, sel_ix
        logger.warning("Recovery active but invalid index")
        return True, None

    @staticmethod
    def _apply_plot_styles(
        selected_traj: "RenderableTrajectory",
        sampled_trajs: list["RenderableTrajectory"],
        recovery_active: bool,
        sel_ix_when_recovery_active: int | None,
    ) -> None:
        """Assign colour/linewidth/alpha/z-order based on recovery status."""

        selected_traj.set_linestring_plot_style(
            name="selected_trajectory",
            linewidth=3.5 if recovery_active else 3.0,
            style="-",
            alpha=1.0,
            color="red" if recovery_active else "orange",
            zorder=12 if recovery_active else 11,
        )

        for idx, st in enumerate(sampled_trajs):
            if recovery_active and idx == sel_ix_when_recovery_active:
                color, linewidth, alpha, z = "orange", 3.0, 1.0, 11
            else:
                color, linewidth, alpha, z = "blue", 2.0, 0.5, 5

            st.set_linestring_plot_style(
                name=f"sampled_trajectory_{idx}",
                linewidth=linewidth,
                style="-",
                alpha=alpha,
                color=color,
                zorder=z,
            )

    @staticmethod
    def from_drive_response(
        driver_response: DriveResponse,
        now_time_us: int,
        query_time_us: int,
        ego_raabb: RAABB,
        ego_coords_rig_to_aabb_center: geometry.Pose,
    ) -> "DriverResponseAtTime":
        """Helper function. Create DriverResponseAtTime from DriveResponse."""
        safety_monitor_safe = None
        command_name = None
        reasoning_text = None
        extra = DriverResponseAtTime._extract_debug_extra(driver_response)
        if extra is not None:
            if "safe_trajectory" in extra:
                safety_monitor_safe = extra["safe_trajectory"]
            if "command_name" in extra:
                command_name = extra["command_name"]
            if "reasoning_text" in extra:
                reasoning_text = extra["reasoning_text"]

        # Selected trajectory
        selected_traj = RenderableTrajectory.from_grpc_with_aabb(
            driver_response.trajectory, ego_raabb
        ).transform(ego_coords_rig_to_aabb_center, is_relative=True)

        # Sampled trajectories
        sampled_trajs = [
            RenderableTrajectory.from_grpc_with_aabb(t, ego_raabb).transform(
                ego_coords_rig_to_aabb_center, is_relative=True
            )
            for t in driver_response.debug_info.sampled_trajectories
        ]

        recovery_active, sel_ix_when_recovery_active = (
            DriverResponseAtTime._parse_recovery_info(extra, len(sampled_trajs))
        )

        # Apply colours / z-orders
        DriverResponseAtTime._apply_plot_styles(
            selected_traj,
            sampled_trajs,
            recovery_active,
            sel_ix_when_recovery_active,
        )

        return DriverResponseAtTime(
            now_time_us=now_time_us,
            time_query_us=query_time_us,
            selected_trajectory=selected_traj,
            sampled_trajectories=sampled_trajs,
            safety_monitor_safe=safety_monitor_safe,
            command_name=command_name,
            reasoning_text=reasoning_text,
        )


@dataclasses.dataclass
class DriverResponses:
    """Represents driver responses for all timesteps.

    Elements:
        * `ego_coords_rig_to_aabb_center`: Rig frame coordinates of AABB center.
            Used when creating `DriverResponseAtTime`s.
        * `ego_trajectory_local`: Ego trajectory in local frame, including RAABB.
        * `timestamps_us`: List of timestamps when the response was predicted.
        * `query_times_us`: List of timestamps for which the response was
            predicted.
        * `per_timestep_driver_responses`: List of `DriverResponseAtTime`s.
        * `artists`: Artists for the driver responses. For videos, those are
            repeatedly updated to capture the new driver response.
    """

    ego_coords_rig_to_aabb_center: geometry.Pose
    ego_trajectory_local: RenderableTrajectory

    @property
    def ego_raabb(self) -> RAABB:
        """RAABB of EGO, derived from ego_trajectory_local."""
        return self.ego_trajectory_local.raabb

    timestamps_us: list[int] = dataclasses.field(default_factory=list)
    query_times_us: list[int] = dataclasses.field(default_factory=list)
    per_timestep_driver_responses: list[DriverResponseAtTime] = dataclasses.field(
        default_factory=list
    )
    artists: dict[str, list[plt.Artist]] | None = None
    camera_artists_by_ax: dict[int, dict[str, list[plt.Artist] | plt.Artist | None]] = (
        dataclasses.field(default_factory=dict)
    )

    def add_drive_response(
        self, driver_response: DriveResponse, now_time_us: int, query_time_us: int
    ) -> None:
        """Helper class to fill in the driver responses when parsing ASL."""
        assert (
            len(self.timestamps_us) == 0 or query_time_us > self.timestamps_us[-1]
        ), "Driver responses must be added in chronological order"
        if len(driver_response.trajectory.poses) == 0:
            # Empty trajectory happens in first few timesteps
            return
        self.timestamps_us.append(now_time_us)
        self.query_times_us.append(query_time_us)
        self.per_timestep_driver_responses.append(
            DriverResponseAtTime.from_drive_response(
                driver_response,
                now_time_us,
                query_time_us,
                self.ego_raabb,
                self.ego_coords_rig_to_aabb_center,
            )
        )

    def render_at_time(
        self,
        ax: plt.Axes,
        time: int,
        which_time: Literal["now", "query"] = "now",
    ) -> dict[str, list[plt.Artist]]:
        """Render driver responses at a given time.

        Can be called repeatedly to update the driver responses to current time.
        `which_time` declares whether the `time` is the query or prediction time.
        """
        driver_response_at_time = self.get_driver_response_for_time(time, which_time)
        if driver_response_at_time is None:
            return {}
        # Styling information is already encoded inside each RenderableTrajectory
        if self.artists is not None:
            # Update geometry (and style in case step-to-step recovery toggles)
            sel_artist = self.artists["selected_trajectory_artist"][0]
            sel_ls = driver_response_at_time.selected_trajectory.renderable_linestring
            sel_positions = np.asarray(
                driver_response_at_time.selected_trajectory.positions
            )
            sel_artist.set_data(
                sel_positions[:, 0],
                sel_positions[:, 1],
            )
            sel_artist.set_color(sel_ls.color)
            sel_artist.set_linewidth(sel_ls.linewidth)
            sel_artist.set_alpha(sel_ls.alpha)
            if sel_ls.zorder is not None:
                sel_artist.set_zorder(sel_ls.zorder)

            for sampled_trajectory, artist in zip(
                driver_response_at_time.sampled_trajectories,
                self.artists["sampled_trajectory_artists"],
                strict=True,
            ):
                samp_positions = sampled_trajectory.positions
                artist.set_data(
                    samp_positions[:, 0],
                    samp_positions[:, 1],
                )
                samp_ls = sampled_trajectory.renderable_linestring
                artist.set_color(samp_ls.color)
                artist.set_linewidth(samp_ls.linewidth)
                artist.set_alpha(samp_ls.alpha)
                if samp_ls.zorder is not None:
                    artist.set_zorder(samp_ls.zorder)

            return self.artists

        # First-time rendering – use each trajectory's own render method
        current_artists: dict[str, list[plt.Artist]] = {}

        # Selected
        current_artists["selected_trajectory_artist"] = (
            driver_response_at_time.selected_trajectory.render_linestring(ax)[
                "selected_trajectory"
            ]
        )

        # Sampled
        current_artists["sampled_trajectory_artists"] = []
        for st in driver_response_at_time.sampled_trajectories:
            art = st.render_linestring(ax)[st.renderable_linestring.name]
            current_artists["sampled_trajectory_artists"].extend(art)

        self.artists = current_artists
        return self.artists

    def render_on_camera(
        self,
        ax: plt.Axes,
        projector: "CameraProjector",
        time: int,
        which_time: Literal["now", "query"] = "now",
    ) -> list[plt.Artist]:
        """Render planner trajectories onto a camera axis."""
        driver_response_at_time = self.get_driver_response_for_time(
            time, which_time=which_time
        )
        artists = self.camera_artists_by_ax.setdefault(
            id(ax), {"selected": None, "sampled": []}
        )

        overlay_artists: list[plt.Artist] = []

        def _to_rig(traj: RenderableTrajectory) -> RenderableTrajectory:
            """Convert stored AABB-frame trajectory back to the current rig frame."""
            # Stored responses are local->aabb; undo the aabb offset to get local->rig.
            traj_rig = traj.transform(
                self.ego_coords_rig_to_aabb_center.inverse(), is_relative=True
            )
            # Ego pose is logged as local->aabb; convert to local->rig, then express
            # the planned points in the ego rig frame by left-multiplying its inverse.
            ego_pose_local_aabb = self.ego_trajectory_local.interpolate_pose(time)
            ego_pose_local_rig = (
                ego_pose_local_aabb @ self.ego_coords_rig_to_aabb_center.inverse()
            )
            rig_traj = traj_rig.transform(
                ego_pose_local_rig.inverse(), is_relative=False
            )
            rig_traj.renderable_linestring = traj.renderable_linestring
            return rig_traj

        def _upsert_line(
            artist: plt.Artist | None,
            pixels: np.ndarray,
            color: str,
            linewidth: float,
            alpha: float,
        ) -> plt.Artist | None:
            if pixels.shape[0] < 2:
                if artist is not None:
                    artist.set_data([], [])
                    artist.set_alpha(0.0)
                return artist
            if artist is None:
                artist = ax.plot(
                    pixels[:, 0],
                    pixels[:, 1],
                    "-",
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )[0]
            else:
                artist.set_data(pixels[:, 0], pixels[:, 1])
                artist.set_color(color)
                artist.set_linewidth(linewidth)
                artist.set_alpha(alpha)
            return artist

        if driver_response_at_time is None:
            if artists["selected"] is not None:
                artists["selected"].set_data([], [])
                artists["selected"].set_alpha(0.0)
                overlay_artists.append(artists["selected"])
            for art in artists["sampled"]:
                art.set_data([], [])
                art.set_alpha(0.0)
                overlay_artists.append(art)
            return overlay_artists

        sel_ls = driver_response_at_time.selected_trajectory.renderable_linestring
        sel_rig = _to_rig(driver_response_at_time.selected_trajectory)
        sel_pixels, _ = projector.project_points(sel_rig.positions[:, :3])
        artists["selected"] = _upsert_line(
            artists["selected"],
            sel_pixels,
            sel_ls.color,
            sel_ls.linewidth,
            sel_ls.alpha,
        )
        if artists["selected"] is not None:
            overlay_artists.append(artists["selected"])

        needed = len(driver_response_at_time.sampled_trajectories)
        while len(artists["sampled"]) < needed:
            artists["sampled"].append(ax.plot([], [], "-")[0])
        while len(artists["sampled"]) > needed:
            extra = artists["sampled"].pop()
            extra.remove()

        for traj, artist in zip(
            driver_response_at_time.sampled_trajectories,
            artists["sampled"],
            strict=True,
        ):
            ls = traj.renderable_linestring
            rig_traj = _to_rig(traj)
            pixels, _ = projector.project_points(rig_traj.positions[:, :3])
            updated = _upsert_line(
                artist,
                pixels,
                ls.color,
                ls.linewidth,
                ls.alpha,
            )
            if updated is not None:
                overlay_artists.append(updated)

        return overlay_artists

    def get_driver_response_for_time(
        self, time: int, which_time: Literal["now", "query"] = "now"
    ) -> DriverResponseAtTime | None:
        """Note that this returns the driver response for the query time.

        I.e. not the time when the response was predicted.
        """
        timestamps_to_search = (
            self.timestamps_us if which_time == "now" else self.query_times_us
        )
        idx = np.searchsorted(timestamps_to_search, time)
        # Corner case: We don't have a response for the last timestamp:
        if (
            which_time == "now"
            and idx == len(timestamps_to_search)
            and time == self.query_times_us[-1]
        ):
            return None
        # Too early, haven't received response yet
        if (
            timestamps_to_search[idx] != time
            and not timestamps_to_search[0] < time < timestamps_to_search[-1]
        ):
            return None
        assert (
            timestamps_to_search[idx] == time
        ), f"{time=} not {timestamps_to_search=}, interpolation is not supported."
        return self.per_timestep_driver_responses[idx]


@dataclasses.dataclass
class ActorPolygonsAtTime:
    """Captures actor polygons at a given time. Crucially also has an STRtree.

    Elements:
        * `bbox_polygons`: List of bounding box polygons for each agent.
        * `yaws`: List of yaws for each agent.
        * `front_bumper_lines`: List of front bumper lines for each agent.
        * `rear_bumper_lines`: List of rear bumper lines for each agent.
        * `agent_ids`: List of agent ids.
        * `str_tree`: STRtree for the bounding box polygons.
        * `timestamp_us`: Timestamp in microseconds.
    """

    # List of polygons for each agent at one point in time
    bbox_polygons: list[shapely.Polygon]
    yaws: list[float]
    front_bumper_lines: list[shapely.LineString] | list[shapely.Polygon]
    rear_bumper_lines: list[shapely.LineString] | list[shapely.Polygon]
    agent_ids: list[str]
    str_tree: shapely.STRtree
    timestamp_us: int

    @staticmethod
    def from_actor_trajectories(
        actor_trajectories: dict[str, RenderableTrajectory],
        time: int,
    ) -> "ActorPolygonsAtTime":
        """Helper function. Create ActorPolygonsAtTime from actor trajectories."""
        bbox_polygons = []
        front_bumper_lines = []
        rear_bumper_lines = []
        agent_ids = []
        yaws = []
        for agent_id, agent_traj in actor_trajectories.items():
            if int(time) in agent_traj.time_range_us:
                agent_ids.append(agent_id)
                interpolated_trajectory = agent_traj.interpolate_to_timestamps(
                    np.array([time])
                )
                bbox_polygons.append(interpolated_trajectory.to_polygons()[0])
                front_bumper_lines.append(
                    interpolated_trajectory.to_front_bumper_lines()[0]
                )
                rear_bumper_lines.append(
                    interpolated_trajectory.to_rear_bumper_lines()[0]
                )
                yaws.append(float(interpolated_trajectory.yaws[0]))
        return ActorPolygonsAtTime(
            bbox_polygons=bbox_polygons,
            yaws=yaws,
            front_bumper_lines=front_bumper_lines,
            rear_bumper_lines=rear_bumper_lines,
            agent_ids=agent_ids,
            str_tree=shapely.STRtree(bbox_polygons),
            timestamp_us=time,
        )

    def get_agent_for_idx(self, idx: int) -> str:
        """Get the agent id for a given index."""
        return self.agent_ids[idx]

    def get_idx_for_agent(self, agent_id: str) -> int:
        """Get the index for a given agent id."""
        return self.agent_ids.index(agent_id)

    def get_polygon_for_agent(self, agent_id: str) -> shapely.Polygon:
        """Get the polygon for a given agent id."""
        return self.bbox_polygons[self.get_idx_for_agent(agent_id)]

    def get_yaw_for_agent(self, agent_id: str) -> float:
        """Get the yaw for a given agent id."""
        return self.yaws[self.get_idx_for_agent(agent_id)]

    def get_front_bumper_line_for_agent(self, agent_id: str) -> shapely.LineString:
        """Get the front bumper line for a given agent id."""
        return self.front_bumper_lines[self.get_idx_for_agent(agent_id)]

    def get_rear_bumper_line_for_agent(self, agent_id: str) -> shapely.LineString:
        """Get the rear bumper line for a given agent id."""
        return self.rear_bumper_lines[self.get_idx_for_agent(agent_id)]

    def get_polygons_in_radius(
        self, center: shapely.Point, radius: float
    ) -> tuple[list[shapely.Polygon], list[str]]:
        """Get the polygons in a given radius."""
        indices = self.str_tree.query(center.buffer(radius), "intersects")
        return [self.bbox_polygons[i] for i in indices], [
            self.agent_ids[i] for i in indices
        ]

    def render(
        self,
        ax: plt.Axes,
        old_agent_artists: dict[str, list[plt.Artist]],
        center: shapely.Point | None = None,
        max_dist: float | None = None,
        only_agents: list[str] | None = None,
    ) -> dict[str, list[plt.Artist]]:
        """Render the actor polygons.

        Can be called repeatedly to update the polygons to current time.

        Args:
            * `ax`: The axis to render on.
            * `old_agent_artists`: Dict of artists of the previously rendered
                timestamp.
            * `center`: Center of the plot. Used to query STRtree for agents to
                render
            * `max_dist`: Maximum distance to render. Only needed if `center` is
                provided.
            * `only_agents`: List of agent ids to render. If provided, only
                these agents will be rendered. Can only be provided if `center`
                and `max_dist` are NOT provided.

        Returns:
            Dict of new artists for each agent.
        """
        new_agent_artists = {}
        assert (
            only_agents is None or max_dist is None
        ), "only_agents and max_dist cannot both be provided."
        assert (
            max_dist is None or center is not None
        ), "center must be provided if max_dist is provided"
        if only_agents is not None:
            polygons, agent_ids = zip(
                *(
                    (polygon, agent_id)
                    for polygon, agent_id in zip(self.bbox_polygons, self.agent_ids)
                    if agent_id in only_agents
                )
            )
        else:
            polygons, agent_ids = (
                (self.bbox_polygons, self.agent_ids)
                if max_dist is None
                else self.get_polygons_in_radius(center, max_dist)
            )

        for polygon, agent_id in zip(polygons, agent_ids, strict=True):
            if agent_id in old_agent_artists:
                old_agent_artists[agent_id][0].set_data(
                    polygon.exterior.xy[0], polygon.exterior.xy[1]
                )
                old_agent_artists[agent_id][1].set_xy(polygon.exterior.coords)
                new_agent_artists[agent_id] = old_agent_artists[agent_id]
            else:
                new_artists = []
                new_artists.extend(
                    ax.plot(
                        polygon.exterior.xy[0],
                        polygon.exterior.xy[1],
                        "k-",
                        linewidth=1,
                    )
                )
                new_artists.extend(
                    ax.fill(
                        polygon.exterior.xy[0],
                        polygon.exterior.xy[1],
                        color="k" if agent_id != "EGO" else "limegreen",
                        alpha=0.1 if agent_id != "EGO" else 0.3,
                    )
                )
                new_agent_artists[agent_id] = new_artists

        # Remove unused artists
        for agent_artist in list(
            set(old_agent_artists.keys()) - set(new_agent_artists.keys())
        ):
            for artist in old_agent_artists[agent_artist]:
                artist.remove()
        return new_agent_artists


@dataclasses.dataclass
class ActorPolygons:
    """Captures actor polygons for all timesteps.

    For rendering, manages the artists over time.
    """

    # List of polygons for each agent at all times
    timestamps_us: np.ndarray
    per_timestep_polygons: list[ActorPolygonsAtTime]
    currently_rendered_agent_ids: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([])
    )
    artists: dict[str, list[plt.Artist]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_actor_trajectories(
        actor_trajectories: dict[str, RenderableTrajectory],
    ) -> "ActorPolygons":
        """Helper function. Create ActorPolygons from actor trajectories."""
        timestamps_us = actor_trajectories["EGO"].timestamps_us
        per_timestep_polygons = []
        for time in timestamps_us:
            per_timestep_polygons.append(
                ActorPolygonsAtTime.from_actor_trajectories(
                    actor_trajectories,
                    time,
                )
            )
        return ActorPolygons(timestamps_us, per_timestep_polygons)

    def get_polygons_at_time(self, time: int) -> ActorPolygonsAtTime:
        """Get the polygons at a given time."""
        idx = np.searchsorted(self.timestamps_us, time)
        assert (
            self.timestamps_us[idx] == time
        ), f"{time=} not {self.timestamps_us=}, interpolation is not supported."
        return self.per_timestep_polygons[idx]

    def get_polygon_for_agent_at_time(
        self, agent_id: str, time: int
    ) -> shapely.Polygon:
        """Get the polygon for a given agent at a given time."""
        return self.get_polygons_at_time(time).get_polygon_for_agent(agent_id)

    def get_yaw_for_agent_at_time(self, agent_id: str, time: int) -> float:
        """Get the yaw for a given agent at a given time."""
        return self.get_polygons_at_time(time).get_yaw_for_agent(agent_id)

    def render_at_time(
        self,
        ax: plt.Axes,
        time: int,
        center: shapely.Point | None = None,
        max_dist: float | None = None,
        only_agents: list[str] | None = None,
    ) -> dict[str, list[plt.Artist]]:
        """Render the actor polygons at a given time.

        Can be called repeatedly to update the polygons to current time.

        Args:
            * `ax`: The axis to render on.
            * `time`: The time to render at.
            * `center`: Center of the plot. Used to query STRtree for agents to
                render
            * `max_dist`: Maximum distance to render. Only needed if `center` is
                provided.
            * `only_agents`: List of agent ids to render. If provided, only
                these agents will be rendered. Can only be provided if `center`
                and `max_dist` are NOT provided.

        Returns:
            Dict of new artists for each agent.
        """
        polygons_at_time = self.get_polygons_at_time(time)
        self.artists = polygons_at_time.render(
            ax, self.artists, center, max_dist, only_agents
        )
        return self.artists

    def set_axis_limits_around_agent(
        self,
        ax: plt.Axes,
        agent_id: str,
        time: int,
        cfg: EvalConfig,
        axis_transform: transforms.Affine2D | None = None,
    ) -> shapely.Point:
        """Set the axis limits around the agent.

        Args:
            ax: The axis to set the limits on.
            agent_id: The id of the agent to set the limits around.
            time: The time at which to query agent position.
            cfg: The evaluation config. Needs:
                - `video.map_video.map_radius_m`
                - `video.map_video.ego_loc`
            axis_transform: The transform to apply to the axis. Used e.g. to
                rotate the map s.t. the ego always faces up.

        Returns:
            The center point of the axis limits in the data coordinate frame.
        """
        padding = cfg.video.map_video.map_radius_m
        loc = cfg.video.map_video.ego_loc
        if axis_transform is None:
            axis_transform = transforms.Affine2D()
        agent_polygon = self.get_polygon_for_agent_at_time(agent_id, time)
        # (2, 1) -> (2,)
        image_center_xy = np.array(agent_polygon.centroid.xy).squeeze()

        if loc == "bottom_center":
            delta_xy_in_ego_frame = (padding * (1 - BOTTOM_CENTER_EGO_LOC_FACTOR), 0)
        else:
            delta_xy_in_ego_frame = (0, 0)
        yaw = self.get_yaw_for_agent_at_time(agent_id, time)
        delta_xy_in_axis_frame = (
            transforms.Affine2D().rotate(yaw).transform(delta_xy_in_ego_frame)
        )
        image_center_xy += delta_xy_in_axis_frame
        x, y = axis_transform.transform(image_center_xy)

        ax.set_xlim(x - padding, x + padding)
        ax.set_ylim(y - padding, y + padding)

        return shapely.Point(image_center_xy)


@dataclasses.dataclass
class CameraCalibration:
    """Calibration info for a camera."""

    logical_id: str
    intrinsics: CameraSpec
    rig_to_camera: geometry.Pose

    @staticmethod
    def from_available_camera(
        available_camera: AvailableCamerasReturn.AvailableCamera,
    ) -> "CameraCalibration":
        """Create calibration from an AvailableCamera proto."""
        intrinsics = CameraSpec()
        intrinsics.CopyFrom(available_camera.intrinsics)
        return CameraCalibration(
            logical_id=available_camera.logical_id,
            intrinsics=intrinsics,
            rig_to_camera=geometry.pose_from_grpc(available_camera.rig_to_camera),
        )


MIN_DEPTH_M = 1e-3


@dataclasses.dataclass
class CameraProjector:
    """Projects rig-frame 3D points to pixel coordinates using camera calibration."""

    calibration: CameraCalibration
    actual_resolution: tuple[int, int] | None = None
    _param_kind: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._param_kind = self.calibration.intrinsics.WhichOneof("camera_param")
        # In logs, rig_to_camera is effectively camera->rig; use inverse only.
        self._rig_to_cam = self.calibration.rig_to_camera.as_se3()  # (4,4)
        self._cam_to_rig = np.linalg.inv(self._rig_to_cam)
        if self._param_kind in ["opencv_pinhole_param", "opencv_fisheye_param"]:
            self._init_pinhole_like()
        elif self._param_kind == "ftheta_param":
            self._init_ftheta()
        else:
            raise ValueError(
                f"Camera model {self._param_kind} not supported for projection "
                f"(camera {self.calibration.logical_id})"
            )
        self._maybe_scale_intrinsics()

    def _init_pinhole_like(self) -> None:
        """Initialize pinhole / fisheye parameters (distortion ignored)."""
        if self._param_kind == "opencv_fisheye_param":
            logger.warning(
                "Using pinhole approximation for camera %s; distortion ignored",
                self.calibration.logical_id,
            )
        param = self.calibration.intrinsics.opencv_pinhole_param
        self.fx = param.focal_length_x
        self.fy = param.focal_length_y
        self.cx = param.principal_point_x
        self.cy = param.principal_point_y
        self.img_w = int(self.calibration.intrinsics.resolution_w)
        self.img_h = int(self.calibration.intrinsics.resolution_h)

    def _init_ftheta(self) -> None:
        """Initialize f-theta projection parameters."""
        intr = self.calibration.intrinsics.ftheta_param
        self._angle_to_pix = np.asarray(intr.angle_to_pixeldist_poly, dtype=float)
        if self._angle_to_pix.size == 0:
            raise ValueError(
                f"F-theta calibration missing angle_to_pixeldist_poly "
                f"(camera {self.calibration.logical_id})"
            )
        self._pix_to_angle = np.asarray(intr.pixeldist_to_angle_poly, dtype=float)
        self._ftheta_cx = intr.principal_point_x
        self._ftheta_cy = intr.principal_point_y
        self._ftheta_max_angle = intr.max_angle if intr.max_angle > 0 else None
        if intr.HasField("linear_cde"):
            linear_c = intr.linear_cde.linear_c
            linear_d = intr.linear_cde.linear_d
            linear_e = intr.linear_cde.linear_e
        else:
            linear_c, linear_d, linear_e = 1.0, 0.0, 0.0
        self._ftheta_linear_matrix = np.array(
            [[linear_c, linear_d], [linear_e, 1.0]], dtype=float
        )
        self.img_w = int(self.calibration.intrinsics.resolution_w)
        self.img_h = int(self.calibration.intrinsics.resolution_h)

    def _maybe_scale_intrinsics(self) -> None:
        """Adjust intrinsics if actual image resolution differs from calibration."""
        if self.actual_resolution is None:
            return
        actual_w, actual_h = self.actual_resolution
        if actual_w == self.img_w and actual_h == self.img_h:
            return
        sx = actual_w / self.img_w
        sy = actual_h / self.img_h
        if self._param_kind in ["opencv_pinhole_param", "opencv_fisheye_param"]:
            self.fx *= sx
            self.fy *= sy
            self.cx *= sx
            self.cy *= sy
            self.img_w = int(actual_w)
            self.img_h = int(actual_h)
        elif self._param_kind == "ftheta_param":
            self._ftheta_cx *= sx
            self._ftheta_cy *= sy
            scale = (sx + sy) * 0.5
            self._angle_to_pix *= scale
            self.img_w = int(actual_w)
            self.img_h = int(actual_h)

    def _rig_to_camera(
        self, points_rig: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        """Transform rig-frame points (N,3) to camera frame."""
        rot = transform[:3, :3]
        trans = transform[:3, 3]
        return (rot @ points_rig.T).T + trans

    def project_points(self, points_rig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project rig-frame points to pixel coords.

        Returns:
            pixels: (M, 2) array of valid pixel coordinates.
            mask: boolean mask of length N indicating which input points are valid.
        """
        if points_rig.size == 0:
            return np.empty((0, 2)), np.zeros((0,), dtype=bool)

        pixels_pref, mask_pref = self._project_points_with_transform(
            points_rig, self._cam_to_rig
        )
        return pixels_pref, mask_pref

    def _project_points_with_transform(
        self, points_rig: np.ndarray, transform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._param_kind == "ftheta_param":
            return self._project_points_ftheta(points_rig, transform)
        return self._project_points_pinhole(points_rig, transform)

    def _project_points_pinhole(
        self,
        points_rig: np.ndarray,
        transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pts_cam = self._rig_to_camera(points_rig, transform)
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        valid = z > MIN_DEPTH_M
        if not np.any(valid):
            return np.empty((0, 2)), valid

        xu = x[valid] / z[valid]
        yu = y[valid] / z[valid]
        u = self.fx * xu + self.cx
        v = self.fy * yu + self.cy

        in_frame = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        # Keep all points; axis limits will clip. Only depth/FOV filter applied.
        in_frame[:] = True

        mask = np.zeros_like(valid)
        mask_indices = np.flatnonzero(valid)
        mask[mask_indices[in_frame]] = True

        pixels = np.stack([u[in_frame], v[in_frame]], axis=-1)
        return pixels, mask

    def _project_points_ftheta(
        self,
        points_rig: np.ndarray,
        transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project using f-theta theta→radius polynomial."""
        pts_cam = self._rig_to_camera(points_rig, transform)
        x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        xy_norm = np.hypot(x, y)

        theta = np.zeros_like(xy_norm)
        positive_z = z > MIN_DEPTH_M
        theta[positive_z] = np.arctan2(xy_norm[positive_z], z[positive_z])

        if self._ftheta_max_angle is not None:
            within_fov = theta <= self._ftheta_max_angle + 1e-6
        else:
            within_fov = positive_z

        radii = np.polynomial.polynomial.polyval(theta, self._angle_to_pix)
        with np.errstate(divide="ignore", invalid="ignore"):
            scales = np.divide(
                radii, xy_norm, out=np.zeros_like(radii), where=xy_norm > 1e-9
            )

        offsets = np.stack([x, y], axis=-1) * scales[:, None]
        pixel_offsets = offsets @ self._ftheta_linear_matrix.T
        u = pixel_offsets[:, 0] + self._ftheta_cx
        v = pixel_offsets[:, 1] + self._ftheta_cy

        in_frame = np.ones_like(u, dtype=bool)

        valid = positive_z & within_fov & in_frame
        mask = valid
        pixels = np.stack([u[mask], v[mask]], axis=-1)
        return pixels, mask

    def project_trajectory(self, trajectory: RenderableTrajectory) -> np.ndarray:
        """Project an entire trajectory to pixel space; returns (M,2) pixels."""
        if trajectory.is_empty():
            return np.empty((0, 2))
        pixels, _ = self.project_points(trajectory.positions[:, :3])
        return pixels

    def project_trajectories(
        self, trajectories: Iterable[RenderableTrajectory]
    ) -> list[np.ndarray]:
        """Project multiple trajectories."""
        return [self.project_trajectory(traj) for traj in trajectories]


@dataclasses.dataclass
class Camera:
    """Captures camera images for all timesteps.

    For rendering, manages the artist over time.
    """

    logical_id: str
    timestamps_us: list[int]
    images_bytes_list: list[bytes]
    artist: mpimg.AxesImage | None = None

    @staticmethod
    def create_empty(id: str) -> "Camera":
        """Helper function. Create empty camera object."""
        return Camera(logical_id=id, timestamps_us=[], images_bytes_list=[])

    def add_image(self, camera_image: RolloutCameraImage.CameraImage) -> None:
        """Helper function. Add an image to the camera."""
        if self.timestamps_us and camera_image.frame_end_us < self.timestamps_us[-1]:
            idx = np.searchsorted(self.timestamps_us, camera_image.frame_end_us)
            self.timestamps_us.insert(idx, camera_image.frame_end_us)
            self.images_bytes_list.insert(idx, camera_image.image_bytes)
        else:
            self.timestamps_us.append(camera_image.frame_end_us)
            self.images_bytes_list.append(camera_image.image_bytes)

    def image_at_time(self, time: int) -> Image.Image | None:
        """Get the image as a PIL Image.

        Returns None if the time is not in the camera's time range.
        """
        idx = np.searchsorted(self.timestamps_us, time)
        if idx == len(self.timestamps_us):
            return None
        image_bytes = self.images_bytes_list[idx]
        try:
            return Image.open(io.BytesIO(image_bytes))
        except UnidentifiedImageError:
            logger.warning("Failed to open image at time %d", time)
            return None

    def render_image_at_time(self, time: int, ax: plt.Axes) -> plt.Artist:
        """Render the image at a given time.

        Can be called repeatedly to update the image to current time.
        """
        image = self.image_at_time(time)

        if self.artist is not None:
            if image is not None:
                self.artist.set_data(image)
                return self.artist
            else:
                # If no image available, draw a black image of the same size
                if self.artist is not None:
                    # Get the shape from the existing artist
                    height, width = self.artist.get_array().shape[:2]
                    black_image = np.zeros((height, width, 3), dtype=np.uint8)
                    self.artist.set_data(black_image)
                    return self.artist

        self.artist = ax.imshow(image)
        return self.artist


@dataclasses.dataclass
class Cameras:
    """Captures cameras for all timesteps."""

    camera_by_logical_id: dict[str, Camera] = dataclasses.field(default_factory=dict)
    calibrations_by_logical_id: dict[str, CameraCalibration] = dataclasses.field(
        default_factory=dict
    )

    def add_camera_image(self, camera_image: RolloutCameraImage.CameraImage) -> None:
        camera = self.camera_by_logical_id.setdefault(
            camera_image.logical_id, Camera.create_empty(camera_image.logical_id)
        )
        camera.add_image(camera_image)

    def add_calibration(
        self, available_camera: AvailableCamerasReturn.AvailableCamera
    ) -> None:
        """Store calibration for a camera logical id."""
        calibration = CameraCalibration.from_available_camera(available_camera)
        self.calibrations_by_logical_id[calibration.logical_id] = calibration


@dataclasses.dataclass
class Routes:
    """Captures routes for all timesteps.

    Routes are per-timestamp, because they are based on where the EGO _thinks_
    it is, which might be a noisy estimate.

    For rendering, manages the artists over time.
    """

    timestamps_us: list[int] = dataclasses.field(default_factory=lambda: [])
    routes_in_rig_frame: list[np.ndarray] = dataclasses.field(
        default_factory=lambda: []
    )
    routes_in_global_frame: list[np.ndarray] = dataclasses.field(
        default_factory=lambda: []
    )
    artists: dict[str, list[plt.Artist]] | None = None
    camera_artists_by_ax: dict[int, plt.Artist | None] = dataclasses.field(
        default_factory=dict
    )
    # Set after convert_routes_to_global_frame is called; used for camera projection.
    _ego_trajectory: Optional["RenderableTrajectory"] = None
    _ego_coords_rig_to_aabb_center: Optional[geometry.Pose] = None

    def add_route(self, route: Route) -> None:
        """Add a route to the routes.

        Used during ASL parsing.
        Routes must be added in chronological order.
        """
        if len(route.waypoints) == 0:
            logger.warning("Route %d has no waypoints", route.timestamp_us)
            return
        assert (
            len(self.timestamps_us) == 0 or route.timestamp_us > self.timestamps_us[-1]
        ), "Routes must be added in chronological order"

        def _vec3_to_np_array(vec3: common_pb2.Vec3) -> np.ndarray:
            return np.array([vec3.x, vec3.y, vec3.z])

        self.timestamps_us.append(route.timestamp_us)
        self.routes_in_rig_frame.append(
            np.array([_vec3_to_np_array(waypoint) for waypoint in route.waypoints])
        )

    def convert_routes_to_global_frame(
        self,
        ego_trajectory: "RenderableTrajectory",
        ego_coords_rig_to_aabb_center: geometry.Pose,
    ) -> None:
        """Convert the routes to the global frame and store them.

        Used during ASL parsing. Also stores references needed for camera projection.
        """
        # Store references for camera projection
        self._ego_trajectory = ego_trajectory
        self._ego_coords_rig_to_aabb_center = ego_coords_rig_to_aabb_center

        for timestamp_us, route_in_rig_frame in zip(
            self.timestamps_us, self.routes_in_rig_frame, strict=True
        ):
            # Transform waypoints from rig frame to AABB frame (just translation)
            route_in_aabb_frame = (
                route_in_rig_frame - ego_coords_rig_to_aabb_center.vec3
            )

            # Get ego pose at this timestamp
            ego_pose = ego_trajectory.interpolate_pose(timestamp_us)

            # Transform waypoints to global frame: rotate by ego orientation, then translate
            rotation = R.from_quat(ego_pose.quat).as_matrix()
            route_in_global_frame = (rotation @ route_in_aabb_frame.T).T + ego_pose.vec3

            self.routes_in_global_frame.append(route_in_global_frame)

    def get_route_at_time(self, time: int, strict: bool = True) -> np.ndarray | None:
        """Get the route at a given time."""
        idx = np.searchsorted(self.timestamps_us, time)
        if strict:
            assert (
                self.timestamps_us[idx] == time
            ), f"{time=} not {self.timestamps_us=}, interpolation is not supported."
        if idx >= len(self.routes_in_global_frame):
            return None
        return self.routes_in_global_frame[idx]

    def get_route_at_time_in_rig_frame(self, time: int) -> np.ndarray | None:
        """Get the route in rig frame at a given time.

        Used for camera projection where rig frame coordinates are needed.
        """
        idx = np.searchsorted(self.timestamps_us, time)
        if idx >= len(self.routes_in_rig_frame):
            return None
        return self.routes_in_rig_frame[idx]

    def remove_artists(self) -> None:
        """Remove the artists for the routes."""
        if self.artists is not None:
            for artist_list in self.artists.values():
                for artist in artist_list:
                    artist.remove()
            self.artists = None

    def render_on_camera(
        self,
        ax: plt.Axes,
        projector: "CameraProjector",
        time: int,
    ) -> list[plt.Artist]:
        """Render route waypoints onto a camera axis.

        Transforms the route from global frame to current rig frame, projects
        to pixel coordinates, and draws it as a line on the given axis.
        Caches the artist for efficient updates.
        """
        artist = self.camera_artists_by_ax.get(id(ax))
        overlay_artists: list[plt.Artist] = []

        def _clear_artist() -> list[plt.Artist]:
            if artist is not None:
                artist.set_data([], [])
                artist.set_alpha(0.0)
                overlay_artists.append(artist)
            return overlay_artists

        # Get route in global frame
        route_global = self.get_route_at_time(time, strict=False)
        if route_global is None or len(route_global) < 2:
            return _clear_artist()

        # Need ego trajectory and AABB offset to transform to rig frame
        if self._ego_trajectory is None or self._ego_coords_rig_to_aabb_center is None:
            return _clear_artist()

        # Transform from global frame to current rig frame:
        # 1. Get current ego pose (local->AABB)
        # 2. Convert to local->rig
        # 3. Apply inverse to get global points in rig frame
        ego_pose_local_aabb = self._ego_trajectory.interpolate_pose(time)
        ego_pose_local_rig = (
            ego_pose_local_aabb @ self._ego_coords_rig_to_aabb_center.inverse()
        )

        # Transform route points from global to rig frame using SE3 matrix
        T_world_to_rig = ego_pose_local_rig.inverse().as_se3()  # 4x4 matrix
        route_homogeneous = np.hstack(
            [route_global[:, :3], np.ones((len(route_global), 1))]
        )
        route_rig = (route_homogeneous @ T_world_to_rig.T)[:, :3]

        pixels, _ = projector.project_points(route_rig)

        if pixels.shape[0] < 2:
            return _clear_artist()

        if artist is None:
            artist = ax.plot(
                pixels[:, 0],
                pixels[:, 1],
                "-",
                color="lime",
                linewidth=2.0,
                alpha=0.8,
            )[0]
            self.camera_artists_by_ax[id(ax)] = artist
        else:
            artist.set_data(pixels[:, 0], pixels[:, 1])
            artist.set_alpha(0.8)

        overlay_artists.append(artist)
        return overlay_artists

    def render_at_time(
        self,
        ax: plt.Axes,
        time: int,
    ) -> dict[str, list[plt.Artist]]:
        """Render the route at a given time.

        Can be called repeatedly to update the route to current time.
        Includes a connecting line from the ego position to the first waypoint.
        """
        route = self.get_route_at_time(time, strict=False)
        if route is None:
            return {}

        # Get ego position for connecting line to first waypoint
        ego_pos: np.ndarray | None = None
        if self._ego_trajectory is not None and len(route) > 0:
            try:
                ego_pose = self._ego_trajectory.interpolate_pose(time)
                ego_pos = ego_pose.vec3
            except ValueError:
                pass  # Time outside trajectory range

        if self.artists is not None:
            self.artists["route"][0].set_data(route[:, 0], route[:, 1])
            # Update connecting line
            if ego_pos is not None:
                self.artists["route_to_first_wp"][0].set_data(
                    [ego_pos[0], route[0, 0]], [ego_pos[1], route[0, 1]]
                )
            else:
                self.artists["route_to_first_wp"][0].set_data([], [])
            return self.artists

        current_artists: dict[str, list[plt.Artist]] = {
            "route": ax.plot(route[:, 0], route[:, 1], "g-")
        }
        # Add connecting line from ego to first waypoint
        if ego_pos is not None:
            current_artists["route_to_first_wp"] = ax.plot(
                [ego_pos[0], route[0, 0]],
                [ego_pos[1], route[0, 1]],
                "g--",
                alpha=0.6,
            )
        else:
            current_artists["route_to_first_wp"] = ax.plot([], [], "g--", alpha=0.6)
        self.artists = current_artists
        return current_artists


# =============================================================================
# Evaluation Data Flow: ScenarioEvalInput -> SimulationResult
# =============================================================================
#
# ScenarioEvalInput and SimulationResult serve different roles in the pipeline:
#
# ScenarioEvalInput (raw input):
#   - A "transfer object" designed to be easy to construct from multiple sources
#     (runtime memory via BoundRollout, or ASL files via asl_loader)
#   - Uses simple types: raw Trajectory objects, explicit AABB dimensions as tuples
#   - Many fields are optional (vec_map, cameras, routes, driver_responses)
#   - Contains run metadata for aggregation (run_uuid, run_name, batch_id)
#
# SimulationResult (processed state):
#   - A "computed object" with enriched data ready for scoring and video rendering
#   - Uses RenderableTrajectory (includes RAABB with corner radius, rendering info)
#   - Contains pre-computed ActorPolygons (spatial index for fast collision detection)
#   - All fields are required (defaults created during conversion)
#
# The conversion (SimulationResult.from_scenario_input) is config-dependent:
#   - Applies vehicle_shrink_factor from EvalConfig
#   - Computes corner_radius from vehicle_corner_roundness
#   - Pre-computes spatial indices for collision detection
#
# =============================================================================


@dataclasses.dataclass
class ScenarioEvalInput:
    """
    Raw input data for evaluating a completed scenario.

    This is a "transfer object" containing data from a completed simulation.
    It can be constructed from runtime memory (BoundRollout) or loaded from
    ASL files (asl_loader). Uses simple types that are readily available from
    data sources.

    To run evaluation or render videos, convert to SimulationResult first:
        sim_result = SimulationResult.from_scenario_input(scenario_input, cfg)
    """

    # Session metadata
    session_metadata: RolloutMetadata.SessionMetadata

    # Run metadata for aggregation (required)
    run_uuid: str = dataclasses.field()
    run_name: str = dataclasses.field()
    batch_id: str = dataclasses.field()  # Usually "0" for single-batch runs

    # Transformation from Rig frame to AABB center frame
    ego_coords_rig_to_aabb_center: geometry.Pose

    # Ego's bounding box dimensions
    ego_aabb_x_m: float
    ego_aabb_y_m: float
    ego_aabb_z_m: float

    # Actor trajectories (dict of actor_id -> trajectory)
    # Trajectories should be in AABB frame (center of bounding box)
    actor_trajectories: dict[
        str, tuple[geometry.Trajectory, tuple[float, float, float]]
    ]
    # Dict mapping actor_id to (trajectory, (aabb_x, aabb_y, aabb_z))

    # Ground truth ego trajectory (recorded original trajectory)
    ego_recorded_ground_truth_trajectory: geometry.Trajectory

    # Driver responses (optional, needed for some metrics)
    driver_responses: Optional[DriverResponses] = None

    # Vector map (needed for offroad detection)
    vec_map: Optional[VectorMap] = None

    # Cameras data (optional, needed for image-based metrics)
    cameras: Optional[Cameras] = None

    # Routes data (optional)
    routes: Optional[Routes] = None


@dataclasses.dataclass
class SimulationResult:
    """
    Processed simulation state ready for scoring and video rendering.

    This is a "computed object" with enriched data derived from ScenarioEvalInput.
    It contains RenderableTrajectory objects (with RAABB and rendering info) and
    pre-computed ActorPolygons (spatial index for fast collision detection).

    Create from raw input using the factory method:
        sim_result = SimulationResult.from_scenario_input(scenario_input, cfg)

    See the section comment above ScenarioEvalInput for the full data flow.
    """

    session_metadata: RolloutMetadata.SessionMetadata
    # Transformation from Rig frame to AABB center frame
    ego_coords_rig_to_aabb_center: geometry.Pose
    # Trajectories for all agents, including EGO. Mapping id -> trajectory
    actor_trajectories: dict[str, RenderableTrajectory]
    # This might deviate from the ground truth trajectory due to noise.
    driver_estimated_trajectory: RenderableTrajectory
    # Driver responses (with selected and sampled trajectories) for each timestep
    driver_responses: DriverResponses
    ego_recorded_ground_truth_trajectory: RenderableTrajectory
    vec_map: VectorMap
    # Shapely polygons and pre-cached STRtrees at each ts for fast spatial queries
    actor_polygons: ActorPolygons
    cameras: Cameras
    routes: Routes

    @property
    def timestamps_us(self) -> np.ndarray:
        """Utility property to get all timestamps from the ego trajectory.
        This assumes that all other agents also fall on the same steps.
        """
        return self.actor_polygons.timestamps_us

    @classmethod
    def from_scenario_input(
        cls, scenario_input: ScenarioEvalInput, cfg: EvalConfig
    ) -> "SimulationResult":
        """
        Create a SimulationResult from ScenarioEvalInput.

        This factory method converts raw input data into a fully processed
        SimulationResult with RenderableTrajectory objects, pre-computed
        actor polygons, and config-dependent vehicle parameters applied.

        Args:
            scenario_input: Raw input data containing trajectories, metadata, etc.
            cfg: Evaluation configuration (used for vehicle shrink factor,
                 corner roundness, etc.)

        Returns:
            SimulationResult ready for use in scoring and video rendering.
        """
        # Build actor trajectories as RenderableTrajectory
        actor_trajectories: dict[str, RenderableTrajectory] = {}

        for actor_id, (
            trajectory,
            aabb_dims,
        ) in scenario_input.actor_trajectories.items():
            # Centralized RAABB construction (apply shrink/roundness once)
            raabb = RAABB.from_grpc(
                common_pb2.AABB(
                    size_x=aabb_dims[0], size_y=aabb_dims[1], size_z=aabb_dims[2]
                ),
                cfg.vehicle,
            )
            actor_trajectories[actor_id] = RenderableTrajectory.from_trajectory(
                trajectory, raabb
            )

        # Create ego RAABB (centralized)
        ego_raabb = RAABB.from_grpc(
            common_pb2.AABB(
                size_x=scenario_input.ego_aabb_x_m,
                size_y=scenario_input.ego_aabb_y_m,
                size_z=scenario_input.ego_aabb_z_m,
            ),
            cfg.vehicle,
        )

        # Create ego recorded ground truth trajectory
        ego_recorded_ground_truth_trajectory = RenderableTrajectory.from_trajectory(
            scenario_input.ego_recorded_ground_truth_trajectory, ego_raabb
        )

        # Create driver estimated trajectory (use EGO trajectory if not specified)
        ego_trajectory = actor_trajectories.get("EGO")
        if ego_trajectory is not None:
            driver_estimated_trajectory = ego_trajectory
        else:
            driver_estimated_trajectory = RenderableTrajectory.from_trajectory(
                geometry.Trajectory.create_empty(), ego_raabb
            )

        # Create driver responses if not provided
        driver_responses = scenario_input.driver_responses
        if driver_responses is None:
            driver_responses = DriverResponses(
                ego_coords_rig_to_aabb_center=scenario_input.ego_coords_rig_to_aabb_center,
                ego_trajectory_local=(
                    ego_trajectory if ego_trajectory else driver_estimated_trajectory
                ),
            )

        # Create actor polygons from trajectories
        actor_polygons = ActorPolygons.from_actor_trajectories(actor_trajectories)

        # Create empty cameras and routes if not provided
        cameras = (
            scenario_input.cameras if scenario_input.cameras is not None else Cameras()
        )
        routes = (
            scenario_input.routes if scenario_input.routes is not None else Routes()
        )

        return cls(
            session_metadata=scenario_input.session_metadata,
            ego_coords_rig_to_aabb_center=scenario_input.ego_coords_rig_to_aabb_center,
            actor_trajectories=actor_trajectories,
            driver_estimated_trajectory=driver_estimated_trajectory,
            driver_responses=driver_responses,
            ego_recorded_ground_truth_trajectory=ego_recorded_ground_truth_trajectory,
            vec_map=scenario_input.vec_map,
            actor_polygons=actor_polygons,
            cameras=cameras,
            routes=routes,
        )


class AggregationType(StrEnum):
    """How should the values of a metric be aggregated over time?"""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    LAST = "last"

    def get_numpy_func(self) -> Callable[[np.ndarray], float]:
        """Return the numpy function for this aggregation type."""
        if self == AggregationType.MEAN:
            return np.mean
        elif self == AggregationType.MEDIAN:
            return np.median
        elif self == AggregationType.MAX:
            return np.max
        elif self == AggregationType.MIN:
            return np.min
        elif self == AggregationType.LAST:
            return lambda x: x[-1]
        else:
            raise ValueError(f"Unknown aggregation type: {self}")

    def get_polars_agg_expr(self, col_name: str) -> pl.Expr:
        """Return the Polars aggregation expression for this aggregation type.

        Args:
            col_name: The column name to aggregate.

        Returns:
            A Polars expression that aggregates the column.
        """
        if self == AggregationType.MEAN:
            return pl.col(col_name).mean()
        elif self == AggregationType.MEDIAN:
            return pl.col(col_name).median()
        elif self == AggregationType.MAX:
            return pl.col(col_name).max()
        elif self == AggregationType.MIN:
            return pl.col(col_name).min()
        elif self == AggregationType.LAST:
            return pl.col(col_name).last()
        else:
            raise ValueError(f"Unknown aggregation type: {self}")


@dataclasses.dataclass
class MetricReturn:
    """A metric return value.

    This is used to store the values of one metric over time.
    """

    # Should be unique.
    name: str
    # Lists are over timesteps, values and valids.
    timestamps_us: list[int]
    values: list[float | bool]
    # The valid field can be used when the computation of a metric is impossible
    # for some timesteps. This can then be used in aggregation (not implemented
    # yet). For example, computing a stop-sign metric only makes sense if we're
    # close to a stop-sign. Any aggregation of a stop-sign metric should take
    # into account how many stop-signs we actually encountered.
    valid: list[bool]
    # How should the values be aggregated over the simulation time?
    time_aggregation: AggregationType
    # Arbitrary info about the metric. Currently not used.
    info: str | None = None

    def aggregate(self) -> float:
        """Aggregate values over time according to the time_aggregation type.

        Only valid values are included in the aggregation.

        Returns:
            Aggregated metric value. Returns NaN if no valid values exist.

        Note:
            This method provides consistent aggregation logic that matches
            the Polars-based aggregation in `processing.aggregate_and_write_metrics_results_txt`.
        """
        values = np.array(self.values, dtype=np.float64)
        valid = np.array(self.valid, dtype=bool)
        valid_values = values[valid]

        if len(valid_values) == 0:
            return float("nan")

        return float(self.time_aggregation.get_numpy_func()(valid_values))


def create_metrics_dataframe(
    metric_results: list["MetricReturn"],
    clipgt_id: str,
    batch_id: str,
    rollout_id: str,
    run_uuid: str,
    run_name: str,
) -> pl.DataFrame:
    """
    Create a polars DataFrame from metric results with run metadata.

    This is a shared helper used by both post-eval (main.py) and in-runtime
    evaluation (scenario_evaluator.py) to ensure consistent DataFrame format.

    Args:
        metric_results: List of MetricReturn objects with per-timestep metrics.
        clipgt_id: Clip/ground truth identifier (typically scene_id).
        batch_id: Batch identifier.
        rollout_id: Rollout/session identifier.
        run_uuid: Unique identifier for the evaluation run.
        run_name: Human-readable name for the evaluation run.

    Returns:
        DataFrame with columns: name, timestamps_us, values, valid,
        time_aggregation, clipgt_id, batch_id, rollout_id, run_uuid, run_name
    """
    dictionaries = []
    for mr in metric_results:
        # Use dataclasses.asdict for forward compatibility when MetricReturn fields change
        mr_dict = dataclasses.asdict(mr)
        # Convert values to float64 for consistency
        mr_dict["values"] = np.array(mr_dict["values"], dtype=np.float64).tolist()
        # Convert time_aggregation enum to string
        mr_dict["time_aggregation"] = str(mr_dict["time_aggregation"])
        # Remove fields not needed in the dataframe
        mr_dict.pop("info", None)
        # Add run metadata
        mr_dict.update(
            {
                "clipgt_id": clipgt_id,
                "batch_id": batch_id,
                "rollout_id": rollout_id,
                "run_uuid": run_uuid,
                "run_name": run_name,
            }
        )
        dictionaries.append(mr_dict)

    if not dictionaries:
        return pl.DataFrame()

    # pl.from_dicts() creates one row per dict with List columns.
    # Explode to get one row per timestamp (long format) as expected by aggregation.
    return pl.from_dicts(dictionaries).explode(["values", "timestamps_us", "valid"])
