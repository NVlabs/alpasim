# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""UnboundRollout — validated rollout metadata created before execution."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, replace

import numpy as np
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    PhysicsUpdateMode,
    RouteGeneratorType,
    RuntimeCameraConfig,
    SimulationConfig,
    VehicleConfig,
)
from alpasim_runtime.services.renderer import RendererService
from alpasim_runtime.services.sensorsim_service import ImageFormat
from alpasim_utils.geometry import Pose, Trajectory
from alpasim_utils.scenario import AABB, TrafficObject, TrafficObjects
from alpasim_utils.scene_data_source import SceneDataSource
from trajdata.maps import VectorMap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutTiming:
    egomotion_context_start_us: int
    render_start_timestamp_us: int
    first_policy_timestamp_us: int
    closed_loop_start_us: int
    end_timestamp_us: int
    n_sim_steps: int
    first_camera_frame_ranges_us: dict[str, range]


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


def _build_rollout_timing(
    simulation_config: SimulationConfig,
    data_source: SceneDataSource,
    camera_configs: list[RuntimeCameraConfig],
    *,
    renderer_service: RendererService,
) -> RolloutTiming:
    camera_logical_ids = [camera_cfg.logical_id for camera_cfg in camera_configs]
    egomotion_context_start_us = data_source.rig.trajectory.time_range_us.start

    # ``first_camera_frame_end_us`` raises through ``first_camera_frame_ranges_us``
    # when no cameras are configured.  Headless rollouts fall back to the GT
    # trajectory start as the render anchor.
    if camera_logical_ids:
        first_camera_frame_ranges_us = data_source.rig.first_camera_frame_ranges_us(
            camera_logical_ids
        )
        render_start_us = data_source.rig.first_camera_frame_end_us(camera_logical_ids)
    else:
        first_camera_frame_ranges_us = {}
        render_start_us = data_source.rig.trajectory.time_range_us.start

    if simulation_config.assert_zero_decision_delay and camera_configs:
        for camera_cfg in camera_configs:
            first_camera_frame_ranges_us[camera_cfg.logical_id] = range(
                render_start_us - camera_cfg.shutter_duration_us,
                render_start_us,
            )

    first_policy_timestamp_us = renderer_service.required_policy_start_timestmap_us(
        render_start_us
    )

    closed_loop_start_us = render_start_us + simulation_config.force_gt_duration_us
    last_valid_gt_timestamp_us = data_source.rig.trajectory.time_range_us.stop - 1
    n_sim_steps_allowed_by_time = max(
        0,
        (last_valid_gt_timestamp_us - first_policy_timestamp_us)
        // simulation_config.control_timestep_us,
    )
    n_sim_steps_actual = min(
        simulation_config.n_sim_steps,
        n_sim_steps_allowed_by_time,
    )
    if n_sim_steps_actual <= 0:
        raise ValueError(
            "No complete policy step fits in the recording after policy start: "
            f"first_policy_timestamp_us={first_policy_timestamp_us}, "
            f"last_valid_gt_timestamp_us={last_valid_gt_timestamp_us}, "
            f"control_timestep_us={simulation_config.control_timestep_us}"
        )
    if n_sim_steps_actual < simulation_config.n_sim_steps:
        logger.info(
            "Clipping n_sim_steps from %d to %d because only %d complete "
            "policy steps fit between first_policy_timestamp_us=%d and scene end=%d",
            simulation_config.n_sim_steps,
            n_sim_steps_actual,
            n_sim_steps_allowed_by_time,
            first_policy_timestamp_us,
            last_valid_gt_timestamp_us,
        )

    return RolloutTiming(
        egomotion_context_start_us=egomotion_context_start_us,
        render_start_timestamp_us=render_start_us,
        first_policy_timestamp_us=first_policy_timestamp_us,
        closed_loop_start_us=closed_loop_start_us,
        end_timestamp_us=first_policy_timestamp_us
        + (n_sim_steps_actual * simulation_config.control_timestep_us),
        n_sim_steps=n_sim_steps_actual,
        first_camera_frame_ranges_us=first_camera_frame_ranges_us,
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
    egomotion_context_start_us: int
    # Timestamp at which the first rendered camera frame's shutter closes —
    # this is also the start of the force-GT period and of the GT/physics
    # blend window when ``physics_update_mode != NONE``.
    render_start_timestamp_us: int
    # First timestamp at which the policy/controller/physics pipeline runs.
    # For sensorsim this is the first rendered frame timestamp. For the video
    # model it is after the short first chunk so later pipeline steps align
    # with regular chunks.
    first_policy_timestamp_us: int
    # Boundary between force-GT and policy-driven control.  This may be after
    # ``end_timestamp_us`` for open-loop/log-replay rollouts.
    closed_loop_start_us: int
    end_timestamp_us: int
    force_gt_duration_us: int
    skip_driver_during_force_gt: bool
    physics_update_mode: PhysicsUpdateMode
    save_path_root: str
    control_timestep_us: int
    pose_reporting_interval_us: int
    camera_configs: list[RuntimeCameraConfig]
    first_camera_frame_ranges_us: dict[str, range]
    force_gt_period: range
    image_format: ImageFormat
    ego_mask_rig_config_id: str
    assert_zero_decision_delay: bool
    transform_ego_coords_ds_to_aabb: Pose
    ego_aabb: AABB
    planner_delay_us: int
    route_generator_type: RouteGeneratorType
    send_recording_ground_truth: bool
    nre_runid: str
    nre_version: str
    nre_uuid: str
    vehicle_config: VehicleConfig

    vector_map: VectorMap | None = None
    follow_log: str | None = None

    # Actors filtered out from simulation but still present in USDZ; we keep
    # a lowered-to-ground trajectory so we can override their rendering.
    hidden_traffic_objs: TrafficObjects | None = None

    group_render_requests: bool = False

    @staticmethod
    def create(
        simulation_config: SimulationConfig,
        scene_id: str,
        version_ids: RolloutMetadata.VersionIds,
        data_source: SceneDataSource,
        rollouts_dir: str,
        renderer_service: RendererService,
        session_uuid: str | None = None,
    ) -> UnboundRollout:
        """Create UnboundRollout from SceneDataSource."""

        camera_configs = list(simulation_config.cameras)
        renderer_service.validate_timing_alignment(simulation_config)
        timing = _build_rollout_timing(
            simulation_config,
            data_source,
            camera_configs,
            renderer_service=renderer_service,
        )
        gt_ego_trajectory = data_source.rig.trajectory

        # Filter out objects that are not in the time window
        all_objs_in_window = data_source.traffic_objects.clip_trajectories(
            timing.egomotion_context_start_us,
            timing.end_timestamp_us + 1,
            exclude_empty=True,
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

        hidden_objs_dict: dict[str, TrafficObject] = {}
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

        force_gt_period = range(
            timing.render_start_timestamp_us,
            timing.closed_loop_start_us + 1,
        )

        if simulation_config.vehicle is not None:
            vehicle = simulation_config.vehicle
        elif data_source.rig.vehicle_config is not None:
            vehicle = data_source.rig.vehicle_config
        else:
            raise ValueError("No vehicle config provided/found.")

        ego_aabb = AABB(
            x=vehicle.aabb_x_m,
            y=vehicle.aabb_y_m,
            z=vehicle.aabb_z_m,
        )

        return UnboundRollout(
            rollout_uuid=session_uuid or str(uuid.uuid1()),
            scene_id=scene_id,
            gt_ego_trajectory=gt_ego_trajectory,
            traffic_objs=traffic_objects,
            n_sim_steps=timing.n_sim_steps,
            egomotion_context_start_us=timing.egomotion_context_start_us,
            render_start_timestamp_us=timing.render_start_timestamp_us,
            first_policy_timestamp_us=timing.first_policy_timestamp_us,
            closed_loop_start_us=timing.closed_loop_start_us,
            end_timestamp_us=timing.end_timestamp_us,
            force_gt_duration_us=simulation_config.force_gt_duration_us,
            skip_driver_during_force_gt=simulation_config.skip_driver_during_force_gt,
            control_timestep_us=simulation_config.control_timestep_us,
            follow_log=None,
            save_path_root=os.path.join(rollouts_dir, scene_id),
            version_ids=version_ids,
            camera_configs=camera_configs,
            first_camera_frame_ranges_us=timing.first_camera_frame_ranges_us,
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
            nre_runid=str(data_source.metadata.logger.run_id),
            nre_version=data_source.metadata.version_string,
            nre_uuid=str(data_source.metadata.uuid),
            planner_delay_us=simulation_config.planner_delay_us,
            pose_reporting_interval_us=simulation_config.pose_reporting_interval_us,
            route_generator_type=simulation_config.route_generator_type,
            send_recording_ground_truth=simulation_config.send_recording_ground_truth,
            vehicle_config=vehicle,
            vector_map=data_source.map,
            hidden_traffic_objs=hidden_traffic_objs,
            group_render_requests=simulation_config.group_render_requests,
        )

    def get_log_metadata(self) -> RolloutMetadata.SessionMetadata:
        return RolloutMetadata.SessionMetadata(
            session_uuid=self.rollout_uuid,
            scene_id=self.scene_id,
            batch_size=1,  # Always 1 since we only have one rollout
            n_sim_steps=self.n_sim_steps,
            start_timestamp_us=self.egomotion_context_start_us,
            control_timestep_us=self.control_timestep_us,
            nre_runid=self.nre_runid,
            nre_version=self.nre_version,
            nre_uuid=self.nre_uuid,
        )
