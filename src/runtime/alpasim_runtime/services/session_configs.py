# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Typed per-service session configuration models.

Each service that requires session-specific parameters gets a frozen
dataclass here.  These replace the untyped ``additional_args`` dict that
was previously threaded through ``SessionInfo``.
"""

from dataclasses import dataclass

from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasReturn
from alpasim_utils.geometry import Trajectory
from alpasim_utils.scenario import AABB, TrafficObjects


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
