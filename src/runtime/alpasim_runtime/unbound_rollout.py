# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""UnboundRollout — validated rollout metadata created before execution."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    DriverBackendConfig,
    PhysicsUpdateMode,
    RouteGeneratorType,
    RuntimeCameraConfig,
    SimulationConfig,
    VehicleConfig,
)
from alpasim_runtime.services.sensorsim_service import ImageFormat
from alpasim_utils.artifact import Artifact
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import AABB, TrafficObjects
from trajdata.maps import VectorMap

logger = logging.getLogger(__name__)

# A small epsilon is needed to get the last frames of the original clip rendered
ORIGINAL_TRAJECTORY_DURATION_EXTENSION_US = 1000


def get_ds_rig_to_aabb_center_transform(vehicle_config: VehicleConfig) -> Pose:
    """Transforms the ego pose from the DS rig to the center of the AABB.

    The center of the DS rig is the mid bottom rear bbox edge.
    The center of the AABB is the center of the AABB.
    """
    # apply offsets to get to mid bottom rear bbox edge + mid bottom rear bbox edge to bbox center
    ds_rig_to_aabb_center = np.array(
        [
            vehicle_config.aabb_x_offset_m + vehicle_config.aabb_x_m / 2,
            vehicle_config.aabb_y_offset_m,
            vehicle_config.aabb_z_offset_m + vehicle_config.aabb_z_m / 2,
        ],
        dtype=np.float32,
    )

    return Pose(
        ds_rig_to_aabb_center,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


@dataclass
class UnboundRollout:
    """Metadata for a single rollout on a scene.

    Initialized from config in ``UnboundRollout.create``, performs as much set
    up as possible without access to the execution environment.  This
    separation is to perform maximum sanity checking before simulation starts
    (so we don't crash halfway through 10 scenarios because the 5th is
    misconfigured).
    """

    rollout_uuid: str
    scene_id: str
    gt_ego_trajectory: Trajectory
    traffic_objs: TrafficObjects
    version_ids: RolloutMetadata.VersionIds
    n_sim_steps: int
    start_timestamp_us: int
    force_gt_duration_us: int
    physics_update_mode: PhysicsUpdateMode
    save_path_root: str
    time_start_offset_us: int
    control_timestep_us: int
    pose_reporting_interval_us: int
    camera_configs: list[RuntimeCameraConfig]
    control_timestamps_us: list[int]
    force_gt_period: range
    image_format: ImageFormat
    ego_mask_rig_config_id: str
    assert_zero_decision_delay: bool
    transform_ego_coords_ds_to_aabb: Pose
    ego_aabb: AABB
    planner_delay_us: int
    route_generator_type: RouteGeneratorType
    send_recording_ground_truth: bool
    driver_backends: list[DriverBackendConfig]
    observation_cache_size: int
    observation_window_summary_size: int
    nre_runid: str
    nre_version: str
    nre_uuid: str
    vehicle_config: VehicleConfig

    vector_map: Optional[VectorMap] = None
    follow_log: Optional[str] = None

    # Actors filtered out from simulation but still present in USDZ; we keep
    # a lowered-to-ground trajectory so we can override their rendering.
    hidden_traffic_objs: Optional[TrafficObjects] = None

    group_render_requests: bool = False

    @staticmethod
    def create(
        simulation_config: SimulationConfig,
        scene_id: str,
        version_ids: RolloutMetadata.VersionIds,
        available_artifacts: dict[str, Artifact],
        rollouts_dir: str,
    ) -> UnboundRollout:
        artifact = available_artifacts[scene_id]

        camera_configs = list(simulation_config.cameras)

        control_timestamps_us_arr: np.ndarray = (
            artifact.rig.trajectory.time_range_us.start
            + simulation_config.time_start_offset_us
            + np.arange(
                simulation_config.n_sim_steps + 2
            )  # we cut off the first and last interval so +2 here
            * simulation_config.control_timestep_us
        )

        control_timestamps_us = [
            int(min(t, artifact.rig.trajectory.time_range_us.stop - 1))
            for t in control_timestamps_us_arr
            if t
            < artifact.rig.trajectory.time_range_us.stop
            + ORIGINAL_TRAJECTORY_DURATION_EXTENSION_US
        ]

        start_us = control_timestamps_us[0]
        end_us = control_timestamps_us[-1]
        gt_ego_trajectory = artifact.rig.trajectory

        # Filter out objects that are not in the time window
        all_objs_in_window = artifact.traffic_objects.clip_trajectories(
            start_us, end_us + 1, exclude_empty=True
        )

        # Filter out objects that appear for less than the minimum duration.
        traffic_objects = all_objs_in_window.filter_short_trajectories(
            simulation_config.min_traffic_duration_us
        )

        # Objects that were dropped from `traffic_objects` but still exist in
        # the USDZ will re-appear in NRE 3DGUT renders. We override their pose by
        # dropping them far below ground to prevent them from appearing in the renders.
        # NOTE: NRE team is currently working on a fix to this. We will revert this
        # hack once the fix is released.
        hidden_ids = set(all_objs_in_window.keys()) - set(traffic_objects.keys())

        hidden_objs_dict: dict[str, TrafficObjects.TrafficObject] = {}
        if hidden_ids:
            hide_offset = Pose(
                np.array([0.0, 0.0, -100.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )

            for hid in hidden_ids:
                obj = all_objs_in_window[hid]
                lowered_traj = obj.trajectory.transform(hide_offset, is_relative=True)
                hidden_objs_dict[hid] = replace(obj, trajectory=lowered_traj)

        hidden_traffic_objs = (
            TrafficObjects(**hidden_objs_dict) if hidden_objs_dict else None
        )

        # Validate that force_gt_duration is grid-aligned to control_timestep.
        # A non-divisible duration creates ambiguous policy start timing.
        if (
            simulation_config.force_gt_duration_us > 0
            and simulation_config.force_gt_duration_us
            % simulation_config.control_timestep_us
            != 0
        ):
            raise ValueError(
                f"force_gt_duration_us ({simulation_config.force_gt_duration_us}) "
                f"must be a multiple of control_timestep_us "
                f"({simulation_config.control_timestep_us}). "
                f"Non-divisible durations cause ambiguous policy start timing."
            )

        force_gt_period = range(
            control_timestamps_us[0],
            control_timestamps_us[0] + simulation_config.force_gt_duration_us + 1,
        )

        if simulation_config.vehicle is not None:
            vehicle = simulation_config.vehicle
        elif artifact.rig.vehicle_config is not None:
            vehicle = artifact.rig.vehicle_config
        else:
            raise ValueError("No vehicle config provided/found.")

        ego_aabb = AABB(
            x=vehicle.aabb_x_m,
            y=vehicle.aabb_y_m,
            z=vehicle.aabb_z_m,
        )

        return UnboundRollout(
            rollout_uuid=str(uuid.uuid1()),
            scene_id=scene_id,
            gt_ego_trajectory=gt_ego_trajectory,
            traffic_objs=traffic_objects,
            n_sim_steps=simulation_config.n_sim_steps,
            start_timestamp_us=artifact.rig.trajectory.time_range_us.start,
            force_gt_duration_us=simulation_config.force_gt_duration_us,
            control_timestep_us=simulation_config.control_timestep_us,
            follow_log=None,
            save_path_root=os.path.join(rollouts_dir, scene_id),
            time_start_offset_us=simulation_config.time_start_offset_us,
            version_ids=version_ids,
            camera_configs=camera_configs,
            control_timestamps_us=control_timestamps_us,
            force_gt_period=force_gt_period,
            physics_update_mode=simulation_config.physics_update_mode,
            image_format={"jpeg": ImageFormat.JPEG, "png": ImageFormat.PNG}[
                simulation_config.image_format
            ],
            ego_mask_rig_config_id=simulation_config.ego_mask_rig_config_id,
            assert_zero_decision_delay=simulation_config.assert_zero_decision_delay,
            transform_ego_coords_ds_to_aabb=get_ds_rig_to_aabb_center_transform(
                vehicle
            ),
            ego_aabb=ego_aabb,
            nre_runid=str(artifact.metadata.logger.run_id),
            nre_version=artifact.metadata.version_string,
            nre_uuid=str(artifact.metadata.uuid),
            planner_delay_us=simulation_config.planner_delay_us,
            pose_reporting_interval_us=simulation_config.pose_reporting_interval_us,
            route_generator_type=simulation_config.route_generator_type,
            send_recording_ground_truth=simulation_config.send_recording_ground_truth,
            driver_backends=list(simulation_config.driver_backends),
            observation_cache_size=simulation_config.observation_cache_size,
            observation_window_summary_size=simulation_config.observation_window_summary_size,
            vehicle_config=vehicle,
            vector_map=artifact.map,
            hidden_traffic_objs=hidden_traffic_objs,
            group_render_requests=simulation_config.group_render_requests,
        )

    def get_log_metadata(self) -> RolloutMetadata.SessionMetadata:
        return RolloutMetadata.SessionMetadata(
            session_uuid=self.rollout_uuid,
            scene_id=self.scene_id,
            batch_size=1,  # Always 1 since we only have one rollout
            n_sim_steps=self.n_sim_steps,
            start_timestamp_us=self.start_timestamp_us,
            control_timestep_us=self.control_timestep_us,
            nre_runid=self.nre_runid,
            nre_version=self.nre_version,
            nre_uuid=self.nre_uuid,
        )
