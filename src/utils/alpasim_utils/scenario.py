# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
A stub for defining simulation scenarios. For now a scenario is just a ground truth rig trajectory
to be followed for a given duration, after which the ego agent is given full control.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Optional, Self

import csaps
import numpy as np
from alpasim_grpc.v0.common_pb2 import AABB as ProtoAABB
from alpasim_utils.geometry import Pose, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class VehicleConfig:
    """
    Dimensions of the vehicle. It matters mostly for collision detection (physics simulation and KPIs)
    """

    # defaults follow hyperion_8_daimler_s223 spec from the NDAS repo:
    # https://git-av.nvidia.com/r/plugins/gitiles/ndas/+/refs/heads/main/data/rig/hyperion_8_daimler_s223/vehicle.json
    # For more information on the conventions, see:
    # https://git-av.nvidia.com/r/plugins/gitiles/ndas/+/refs/heads/main/src/dw/rig/Vehicle.h
    aabb_x_m: float = 5.393  # length
    aabb_y_m: float = 2.109  # width
    aabb_z_m: float = 1.503  # height

    # the _rig_ coordinate system used by drivesim places the origin "on the ground under the center of the rear axle"
    # Below is an offset to transform from DS _rig_ coordinates to DS _bbox_ coordinates which specify the center of
    # its rear bottom edge (NOT center of the bbox).
    # To get center of bbox futher processing is necessary (see `ds_rig_to_aabb_center_transform`)
    aabb_x_offset_m: float = -1.3  # rig origin to bounding box rear (y-z) plane
    aabb_y_offset_m: float = 0.0
    aabb_z_offset_m: float = 0.0


@dataclass
class AABB:
    """
    We effectively duplicate the struct in grpc API because grpc.Message instances are awkward
    to use (fields can't be freely modified, etc)
    """

    x: float
    y: float
    z: float

    def to_grpc(self) -> ProtoAABB:
        return ProtoAABB(size_x=self.x, size_y=self.y, size_z=self.z)


@dataclass
class CameraId:
    logical_name: str
    trajectory_idx: int
    sequence_id: str
    unique_id: str

    @property
    def grpc_name(self) -> str:
        return f"{self.trajectory_idx}@{self.logical_name}"


@dataclass
class Rig:
    sequence_id: str
    trajectory: Trajectory
    camera_ids: list[CameraId]
    world_to_nre: (
        np.ndarray
    )  # needed as long as cuboid tracks are exported in NRE coords
    vehicle_config: Optional[
        VehicleConfig
    ]  # ego configuration if available in usdz checkpoint

    @classmethod
    def load_from_json(cls, json_str: str) -> list[Self]:
        """
        Loads all rig trajectories saved in a `rig_trajectories.json` file created by NRE
        """
        rig_json = json.loads(json_str)
        world_to_nre = np.array(rig_json["world_to_nre"]["matrix"])

        camera_calibrations = rig_json["camera_calibrations"]
        unique_camera_id_to_camera_id = {
            uci: camera_calibrations[uci]["logical_sensor_name"]
            for uci in camera_calibrations
        }

        rigs = []
        for trajectory_idx, trajectory in enumerate(rig_json["rig_trajectories"]):
            sequence_id: str = trajectory["sequence_id"]
            rig_timestamps_us = trajectory["T_rig_world_timestamps_us"]
            rig_poses = trajectory["T_rig_worlds"]

            camera_ids = []
            for unique_camera_id in trajectory["cameras_frame_timestamps_us"].keys():
                id = CameraId(
                    logical_name=unique_camera_id_to_camera_id[unique_camera_id],
                    trajectory_idx=trajectory_idx,
                    sequence_id=sequence_id,
                    unique_id=unique_camera_id,
                )

                camera_ids.append(id)

            # parse vehicle config (bbox and ds_to_aabb transform) if available
            if "rig_bbox" not in trajectory:
                vehicle_config = None
                logger.warning(
                    f"rig_bbox not found in trajectory for {sequence_id=} - likely old artifact format. "
                    "Will apply user override or default."
                )
            elif (rig_bbox := trajectory["rig_bbox"]) is None:
                vehicle_config = None
                logger.info(
                    f"rig_bbox is null for {sequence_id=}. Will apply user override or default."
                )
            else:
                if not all(abs(rot_dim) < 1e-4 for rot_dim in rig_bbox["rot"]):
                    raise ValueError(f"Rig for {sequence_id=} is rotated.")

                off_x, off_y, off_z = rig_bbox["centroid"]
                dim_x, dim_y, dim_z = rig_bbox["dim"]

                # apply the inverse of what NRE does to create the `bbox` field from NV data
                # https://gitlab-master.nvidia.com/nrs/nre/-/commit/5960d03c5cece299dc3bbb9fcb39dc2b9e81ca54#a3188997f9567d6bd647820cff3545f30bc10bb2_233_256
                vehicle_config = VehicleConfig(
                    aabb_x_m=dim_x,
                    aabb_y_m=dim_y,
                    aabb_z_m=dim_z,
                    aabb_x_offset_m=off_x - dim_x / 2,
                    aabb_y_offset_m=off_y,
                    aabb_z_offset_m=off_z - dim_z / 2,
                )

            # Convert SE3 matrices to list of Pose objects
            rig_poses_arr = np.array(rig_poses, dtype=np.float32)
            poses_list = [
                Pose.from_se3(rig_poses_arr[i]) for i in range(rig_poses_arr.shape[0])
            ]
            rigs.append(
                cls(
                    sequence_id=sequence_id,
                    trajectory=Trajectory.from_poses(
                        timestamps=np.array(rig_timestamps_us, dtype=np.uint64),
                        poses=poses_list,
                    ),
                    camera_ids=camera_ids,
                    world_to_nre=world_to_nre,
                    vehicle_config=vehicle_config,
                )
            )

        return rigs


@dataclass
class TrafficObject:
    track_id: str
    aabb: AABB
    trajectory: Trajectory
    is_static: bool
    label_class: str

    def clip_trajectory(self, start_us: int, end_us: int) -> TrafficObject:
        return replace(self, trajectory=self.trajectory.clip(start_us, end_us))


class TrafficObjects(dict[str, TrafficObject]):
    @classmethod
    def load_from_json(cls, json_str: str, smooth=True) -> dict[str, Self]:
        tracks_json = json.loads(json_str)

        objects_per_sequence = {}
        for sequence_id in tracks_json:  # for now there should only be one
            tracks_data = tracks_json[sequence_id]["tracks_data"]
            cuboids_dims = tracks_json[sequence_id]["cuboidtracks_data"]["cuboids_dims"]

            trajectories = {}
            for (
                track_id,
                track_label,
                track_flag,
                aabb_xyz,
                timestamps_us,
                poses_json,
            ) in zip(
                tracks_data["tracks_id"],
                tracks_data["tracks_label_class"],
                tracks_data["tracks_flags"],
                cuboids_dims,
                tracks_data["tracks_timestamps_us"],
                tracks_data["tracks_poses"],
            ):
                poses_np = np.array(poses_json, dtype=np.float32)  # [t, 7]
                positions = poses_np[..., :3]
                quaternions = poses_np[..., 3:]

                is_static = (
                    "CONTROLLABLE" not in track_flag
                )  # there can be multiple flags set

                timestamps_us_arr = np.array(timestamps_us, dtype=np.uint64)

                if smooth:
                    css = csaps.CubicSmoothingSpline(
                        timestamps_us_arr / 1e6,
                        positions.T,  # Expects time in last dimension
                        normalizedsmooth=True,
                    )
                    filtered_positions = css(timestamps_us_arr / 1e6).T

                    max_error = np.max(np.abs(filtered_positions - positions))
                    if max_error > 1.0:
                        logger.warning(
                            f"Max error in cubic spline approximation: {max_error:.6f} m for {track_id=}"
                        )
                    positions = filtered_positions.astype(np.float32)

                # Create list of Pose objects
                poses_list = [
                    Pose(positions[i], quaternions[i])
                    for i in range(len(timestamps_us_arr))
                ]

                trajectory = Trajectory.from_poses(
                    timestamps=timestamps_us_arr,
                    poses=poses_list,
                )

                trajectories[track_id] = TrafficObject(
                    track_id, AABB(*aabb_xyz), trajectory, is_static, track_label
                )

            objects_per_sequence[sequence_id] = cls(trajectories)

        return objects_per_sequence

    def clip_trajectories(
        self, start_us: int, end_us: int, exclude_empty: bool = False
    ) -> TrafficObjects:
        """
        Clips object trajectories to between `start_us` and `end_us` and returns a new TrafficObjects
        with the changes reflected. `exclude_empty` controls whether 0-length trajectories remain in
        the output or not.
        """
        clipped_objects = {}

        for key, traffic_object in self.items():
            clipped = traffic_object.clip_trajectory(start_us, end_us)

            if exclude_empty and clipped.trajectory.is_empty():
                continue

            clipped_objects[key] = clipped

        return TrafficObjects(**clipped_objects)

    def filter_short_trajectories(self, min_duration_us: int) -> TrafficObjects:
        """
        Return a new `TrafficObjects` containing only actors whose
        lifetime is at least `min_duration_us`.

        Args:
            min_duration_us: Required presence duration in micro-seconds.
        """
        filtered: dict[str, TrafficObject] = {}

        logger.info(f"Filtering traffic objects with min duration {min_duration_us} us")
        logger.info(f"Number of traffic objects before filtering: {len(self)}")

        for key, traffic_obj in self.items():
            lifetime_us = (
                traffic_obj.trajectory.time_range_us.stop
                - traffic_obj.trajectory.time_range_us.start
            )

            if lifetime_us >= min_duration_us:
                filtered[key] = traffic_obj

        logger.info(f"Number of traffic objects after filtering: {len(filtered)}")

        return TrafficObjects(**filtered)
