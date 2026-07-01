# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Typed per-service session configuration models.

Each service that requires session-specific parameters gets a frozen
dataclass here.  These replace the untyped ``additional_args`` dict that
was previously threaded through ``SessionInfo``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasReturn
from alpasim_utils.geometry import Trajectory
from alpasim_utils.scenario import AABB, TrafficObjects
from alpasim_utils.scene_data_source import SceneDataSource

if TYPE_CHECKING:
    from alpasim_runtime.types import RuntimeCamera


@dataclass(frozen=True)
class DriverSessionConfig:
    """Typed session configuration for the driver service."""

    sensorsim_cameras: list[AvailableCamerasReturn.AvailableCamera]
    scene_id: str | None = None


@dataclass(frozen=True)
class TrafficSessionConfig:
    """Typed session configuration for the traffic service."""

    traffic_objs: TrafficObjects
    scene_id: str
    ego_aabb: AABB
    gt_ego_aabb_trajectory: Trajectory
    start_timestamp_us: int
    force_gt_duration_us: int
    control_timestep_us: int


@dataclass(frozen=True)
class RendererSessionConfig:
    """Common per-rollout session configuration for renderer services.

    Every renderer service (built-in sensorsim/video_model or plugin-provided)
    receives this when its ``rollout_session`` is entered.  Each
    service reads only the fields it needs and is responsible for populating
    the shared ``camera_catalog`` with definitions for this scene before
    render events fire.
    """

    data_source: SceneDataSource
    runtime_cameras: list["RuntimeCamera"]
    gt_ego_trajectory: Trajectory
    image_format: str
    ego_mask_rig_config_id: str | None
