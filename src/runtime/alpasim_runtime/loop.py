# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Implements the core of the simulation: a batch of rollouts with methods for advancing
their state and logging results.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Optional, Self, Type

import numpy as np
from alpasim_grpc.v0.common_pb2 import DynamicState, PoseAtTime, Vec3
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry, RolloutMetadata
from alpasim_grpc.v0.traffic_pb2 import TrafficReturn
from alpasim_runtime.autoresume import mark_rollout_complete
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import (
    EgomotionNoiseModelConfig,
    PhysicsUpdateMode,
    RouteGeneratorType,
    RuntimeCameraConfig,
    ScenarioConfig,
    UserSimulatorConfig,
    VehicleConfig,
)
from alpasim_runtime.delay_buffer import DelayBuffer
from alpasim_runtime.logs import LogWriterManager
from alpasim_runtime.noise_models import EgomotionNoiseModel
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_runtime.services.controller_service import (
    ControllerService,
    PropagatedPoses,
)
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import ImageFormat, SensorsimService
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.telemetry.telemetry_context import try_get_context
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_utils.artifact import Artifact
from alpasim_utils.logs import LogWriter
from alpasim_utils.qvec import QVec
from alpasim_utils.scenario import AABB, TrafficObjects
from alpasim_utils.trajectory import Trajectory
from trajdata.maps import VectorMap

logger = logging.getLogger(__name__)

# A small epsilon is needed to get the last frames of the original clip rendered
ORIGINAL_TRAJECTORY_DURATION_EXTENSION_US = 1000


def get_ds_rig_to_aabb_center_transform(vehicle_config: VehicleConfig) -> QVec:
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

    return QVec(
        vec3=ds_rig_to_aabb_center,
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


@dataclass
class UnboundRollout:
    """
    Metadata for a single rollout on a scene.
    Initialized from config in `UnboundRollout.create`, performs as much set up
    as possible without access to the execution environment. To run, bind with
    the execution workers via `UnboundRollout.bind(...)` and to obtain a `BoundRollout` which
    has `run`.
    This separation is to perform maximum sanity checking before simulation starts
    (so we don't crash halfway through 10 scenarios because the 5th is misconfigured)
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
    camera_configs: list[RuntimeCameraConfig]
    egopose_clock: Clock
    control_timestamps_us: list[int]
    force_gt_period: range
    random_seed: int
    image_format: ImageFormat
    ego_mask_rig_config_id: str
    assert_zero_decision_delay: bool
    transform_ego_coords_ds_to_aabb: QVec
    ego_aabb: AABB
    planner_delay_us: int
    egomotion_noise: EgomotionNoiseModelConfig
    route_generator_type: RouteGeneratorType
    send_recording_ground_truth: bool
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
        config: UserSimulatorConfig,
        scenario: ScenarioConfig,
        version_ids: RolloutMetadata.VersionIds,
        random_seed: int,
        available_artifacts: dict[str, Artifact],
        asl_dir: str,
    ) -> UnboundRollout:

        # TODO: for now we assume there's a single sequence per NRE scene
        artifact = available_artifacts[scenario.scene_id]

        camera_configs = list(scenario.cameras)

        control_timestamps_us_arr: np.ndarray = (
            artifact.rig.trajectory.time_range_us.start
            + scenario.time_start_offset_us
            + np.arange(
                scenario.n_sim_steps + 2
            )  # we cut off the first and last interval so +2 here
            * scenario.control_timestep_us
        )

        control_timestamps_us = [
            int(min(t, artifact.rig.trajectory.time_range_us.stop - 1))
            for t in control_timestamps_us_arr
            if t
            < artifact.rig.trajectory.time_range_us.stop
            + ORIGINAL_TRAJECTORY_DURATION_EXTENSION_US
        ]

        egopose_clock = Clock(
            interval_us=scenario.egopose_interval_us,
            duration_us=0,
            start_us=control_timestamps_us[0],
        )

        start_us = control_timestamps_us[0]
        end_us = control_timestamps_us[-1]
        gt_ego_trajectory = artifact.rig.trajectory

        # Filter out objects that are not in the time window
        all_objs_in_window = artifact.traffic_objects.clip_trajectories(
            start_us, end_us + 1, exclude_empty=True
        )

        # Filter out objects that appear for less than the minimum duration.
        traffic_objects = all_objs_in_window.filter_short_trajectories(
            scenario.min_traffic_duration_us
        )

        # Objects that were dropped from `traffic_objects` but still exist in
        # the USDZ will re-appear in NRE 3DGUT renders. We override their pose by
        # dropping them far below ground to prevent them from appearing in the renders.
        # NOTE: NRE team is currently working on a fix to this. We will revert this
        # hack once the fix is released.
        hidden_ids = set(all_objs_in_window.keys()) - set(traffic_objects.keys())

        hidden_objs_dict: dict[str, TrafficObjects.TrafficObject] = {}
        if hidden_ids:
            hide_offset = QVec(
                vec3=np.array([0.0, 0.0, -100.0], dtype=np.float32),
                quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )

            for hid in hidden_ids:
                obj = all_objs_in_window[hid]
                lowered_traj = obj.trajectory.transform(hide_offset, is_relative=True)
                hidden_objs_dict[hid] = replace(obj, trajectory=lowered_traj)

        hidden_traffic_objs = (
            TrafficObjects(**hidden_objs_dict) if hidden_objs_dict else None
        )

        force_gt_period = range(
            control_timestamps_us[0],
            control_timestamps_us[0] + scenario.force_gt_duration_us + 1,
        )

        if scenario.vehicle is not None:
            vehicle = scenario.vehicle
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
            scene_id=scenario.scene_id,
            gt_ego_trajectory=gt_ego_trajectory,
            traffic_objs=traffic_objects,
            n_sim_steps=scenario.n_sim_steps,
            start_timestamp_us=artifact.rig.trajectory.time_range_us.start,
            force_gt_duration_us=scenario.force_gt_duration_us,
            control_timestep_us=scenario.control_timestep_us,
            follow_log=None,
            save_path_root=os.path.join(asl_dir, scenario.scene_id),
            time_start_offset_us=scenario.time_start_offset_us,
            version_ids=version_ids,
            camera_configs=camera_configs,
            egopose_clock=egopose_clock,
            control_timestamps_us=control_timestamps_us,
            force_gt_period=force_gt_period,
            physics_update_mode=scenario.physics_update_mode,
            random_seed=random_seed,
            image_format={"jpeg": ImageFormat.JPEG, "png": ImageFormat.PNG}[
                scenario.image_format
            ],
            ego_mask_rig_config_id=scenario.ego_mask_rig_config_id,
            assert_zero_decision_delay=scenario.assert_zero_decision_delay,
            transform_ego_coords_ds_to_aabb=get_ds_rig_to_aabb_center_transform(
                vehicle
            ),
            ego_aabb=ego_aabb,
            nre_runid=str(artifact.metadata.logger.run_id),
            nre_version=artifact.metadata.version_string,
            nre_uuid=str(artifact.metadata.uuid),
            planner_delay_us=scenario.planner_delay_us,
            egomotion_noise=scenario.egomotion_noise,
            route_generator_type=scenario.route_generator_type,
            send_recording_ground_truth=scenario.send_recording_ground_truth,
            vehicle_config=vehicle,
            vector_map=artifact.map,
            hidden_traffic_objs=hidden_traffic_objs,
            group_render_requests=scenario.group_render_requests,
        )

    def bind(
        self,
        driver: DriverService,
        sensorsim: SensorsimService,
        physics: PhysicsService,
        trafficsim: TrafficService,
        controller: ControllerService,
        camera_catalog: CameraCatalog,
    ) -> BoundRollout:
        return BoundRollout(
            self,
            driver,
            sensorsim,
            physics,
            trafficsim,
            controller,
            camera_catalog,
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


@dataclass
class BoundRollout:
    """
    A rollout bound to DriverService, SensorsimService and PhysicsService which will be used to
    simulate it. Provides the runtime logic.
    """

    unbound: UnboundRollout
    driver: DriverService
    sensorsim: SensorsimService
    physics: PhysicsService
    trafficsim: TrafficService
    controller: ControllerService
    camera_catalog: CameraCatalog
    # Sessions will be created dynamically in _loop

    # updated as we go
    ego_trajectory: Trajectory = field(
        init=False
    )  # the true trajectory of the ego vehicle
    ego_trajectory_estimate: Trajectory = field(
        init=False
    )  # the estimated (possibly noisy) trajectory of the ego vehicle
    traffic_objs: TrafficObjects = field(init=False)
    # Latest estimated dynamic state from the controller (velocities, accelerations in rig frame)
    dynamic_state_estimated: DynamicState = field(init=False)

    log_writer: LogWriterManager = field(init=False)
    planner_delay_buffer: DelayBuffer = field(init=False)
    egomotion_noise_model: EgomotionNoiseModel | None = field(init=False)
    route_generator: RouteGenerator = field(init=False)
    data_sensorsim_to_driver: Optional[bytes] = None
    runtime_cameras: list[RuntimeCamera] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # the ego obtains control at prerun_end_us, the prerun period gives us known camera poses to
        # render the initial renders to "seed" the driver
        prerun_start_us = self.unbound.control_timestamps_us[0]
        prerun_end_us = self.unbound.control_timestamps_us[1]

        asl_log_writer = LogWriter(file_path=self.log_path())
        self.log_writer = LogWriterManager(log_writers=[asl_log_writer])

        # populate the rollout trajectory history with the prerun history
        self.traffic_objs = self.unbound.traffic_objs.clip_trajectories(
            prerun_start_us, prerun_end_us + 1
        )
        self.ego_trajectory = self.unbound.gt_ego_trajectory.interpolate_to_timestamps(
            np.array([prerun_start_us, prerun_end_us], dtype=np.uint64)
        )
        self.ego_trajectory_estimate = (
            self.unbound.gt_ego_trajectory.interpolate_to_timestamps(
                np.array([prerun_start_us, prerun_end_us], dtype=np.uint64)
            )
        )
        # Initialize estimated dynamic state

        linear_velocities = self.unbound.gt_ego_trajectory.velocities()
        yaw_rates = self.unbound.gt_ego_trajectory.yaw_rates()

        v_interp = np.array(
            [
                np.interp(
                    prerun_start_us,
                    self.unbound.gt_ego_trajectory.timestamps_us,
                    linear_velocities[:, i],
                )
                for i in range(3)
            ]
        )
        v_grpc = Vec3(x=v_interp[0], y=v_interp[1], z=v_interp[2])
        yaw_rate_grpc = Vec3(
            z=np.interp(
                prerun_start_us, self.unbound.gt_ego_trajectory.timestamps_us, yaw_rates
            )
        )
        self.dynamic_state_estimated = DynamicState(
            angular_velocity=yaw_rate_grpc,
            linear_velocity=v_grpc,
        )

        self.planner_delay_buffer = DelayBuffer(self.unbound.planner_delay_us)
        self.egomotion_noise_model = EgomotionNoiseModel.from_config(
            self.unbound.egomotion_noise
        )
        self.route_generator = RouteGenerator.create(
            self.unbound.gt_ego_trajectory.poses.vec3,
            vector_map=self.unbound.vector_map,
            route_generator_type=self.unbound.route_generator_type,
        )

    def log_path(self) -> str:
        return os.path.join(
            self.unbound.save_path_root, self.unbound.rollout_uuid, "0.asl"
        )

    def rclog_path(self) -> str:
        return os.path.join(
            self.unbound.save_path_root,
            self.unbound.rollout_uuid,
            "0.rclog",
        )

    async def __aenter__(self) -> Self:
        await self.log_writer.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.log_writer.__aexit__(exc_type, exc_value, exc_tb)

    async def log_metadata(
        self,
        session_metadata: RolloutMetadata.SessionMetadata,
        version_ids: RolloutMetadata.VersionIds,
    ) -> None:
        """
        Log rollout metadata at the start of a rollout.
        """
        traffic_actor_aabbs = [
            RolloutMetadata.ActorDefinitions.ActorAABB(
                actor_id=trajectory.track_id,
                aabb=trajectory.aabb.to_grpc(),
                actor_label=trajectory.label_class,
            )
            for trajectory in self.unbound.traffic_objs.values()
        ]
        ego_aabb = RolloutMetadata.ActorDefinitions.ActorAABB(
            actor_id="EGO",
            aabb=self.unbound.ego_aabb.to_grpc(),
        )

        await self.log_writer.log_message(
            LogEntry(
                rollout_metadata=RolloutMetadata(
                    session_metadata=session_metadata,
                    actor_definitions=RolloutMetadata.ActorDefinitions(
                        actor_aabb=[ego_aabb] + traffic_actor_aabbs
                    ),
                    force_gt_duration=self.unbound.force_gt_duration_us,
                    version_ids=version_ids,
                    rollout_index=0,  # Always 0 since we only have one rollout
                    transform_ego_coords_rig_to_aabb=self.unbound.transform_ego_coords_ds_to_aabb.as_grpc_pose(),
                    ego_rig_recorded_ground_truth_trajectory=self.unbound.gt_ego_trajectory.to_grpc(),
                )
            )
        )

    async def render_and_send_image(
        self, camera: RuntimeCamera, trigger: Clock.Trigger, traffic_trajs: dict
    ) -> None:

        image = await self.sensorsim.render(
            ego_trajectory=self.ego_trajectory,
            traffic_trajectories=traffic_trajs,
            trigger=trigger,
            camera=camera,
            scene_id=self.unbound.scene_id,
            image_format=self.unbound.image_format,
            ego_mask_rig_config_id=self.unbound.ego_mask_rig_config_id,
        )

        await self.driver.submit_image(image)

    async def send_egoposes(self, past_us: int, now_us: int) -> None:
        triggers = self.unbound.egopose_clock.triggers_completed_in_range(
            range(past_us, now_us)
        )
        if not triggers:
            return
        ego_trajectory = self.ego_trajectory_estimate.interpolate_to_timestamps(
            np.array([t.time_range_us.start for t in triggers], dtype=np.uint64)
        )
        await self.driver.submit_trajectory(
            ego_trajectory,
            self.dynamic_state_estimated,
        )

    async def send_route(self, timestamp_us: int) -> None:
        if self.ego_trajectory.timestamps_us[-1] != timestamp_us:
            raise ValueError(
                f"Timestamp mismatch: {self.ego_trajectory.timestamps_us[-1]} != {timestamp_us}"
            )

        pose_local_to_rig = self.ego_trajectory.poses[-1]
        route_polyline_in_rig = self.route_generator.generate_route(
            timestamp_us, pose_local_to_rig
        )
        RouteGenerator.prepare_for_policy(route_polyline_in_rig)
        await self.driver.submit_route(timestamp_us, route_polyline_in_rig)

    async def send_recording_ground_truth(self, timestamp_us: int) -> None:
        if self.ego_trajectory.timestamps_us[-1] != timestamp_us:
            raise ValueError(
                f"Timestamp mismatch: {self.ego_trajectory.timestamps_us[-1]} != {timestamp_us}"
            )

        # Get the ground truth trajectory
        trajectory = self.unbound.gt_ego_trajectory

        # Transform the ground truth trajectory into the rig coordinate frame
        pose_local_to_rig = self.ego_trajectory.poses[-1]
        trajectory_in_rig = trajectory.transform(pose_local_to_rig.inverse())

        await self.driver.submit_recording_ground_truth(timestamp_us, trajectory_in_rig)

    async def update_pose(
        self,
        ego_ds_pose_future: QVec,
        local_to_rig_estimate_future: QVec,
        traffic_poses_future: dict[str, QVec],
        future_us: int,
        dynamic_state_estimated: DynamicState,
    ) -> None:
        dt_sec = (future_us - self.ego_trajectory_estimate.time_range_us.stop) / 1e6
        pose_rig_to_noisy_rig = (
            self.egomotion_noise_model.update(dt_sec)
            if self.egomotion_noise_model
            else QVec(
                vec3=np.array([0.0, 0.0, 0.0]),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
            )
        )

        self.ego_trajectory.update_absolute(future_us, ego_ds_pose_future)
        self.ego_trajectory_estimate.update_absolute(
            future_us, local_to_rig_estimate_future @ pose_rig_to_noisy_rig
        )
        # Store the latest estimated dynamic state for use in egomotion observations
        self.dynamic_state_estimated = dynamic_state_estimated
        for obj_id, obj_pose_future in traffic_poses_future.items():
            if obj_id == "EGO":
                continue  # traffic model also models ego trajectory but we don't need that
            self.traffic_objs[obj_id].trajectory.update_absolute(
                future_us, obj_pose_future
            )

        await self._log_actor_poses(future_us)

        egomotion_estimate_error_message = PoseAtTime(
            pose=pose_rig_to_noisy_rig.as_grpc_pose(), timestamp_us=future_us
        )
        await self.log_writer.log_message(
            LogEntry(egomotion_estimate_error=egomotion_estimate_error_message)
        )

    async def _log_actor_poses(self, timestamp_us: int) -> None:
        trajectories = {
            obj_id: obj.trajectory for obj_id, obj in self.traffic_objs.items()
        }
        trajectories["EGO"] = self.ego_trajectory.transform(
            self.unbound.transform_ego_coords_ds_to_aabb,
            is_relative=True,
        )

        actor_poses = []
        for obj_id, trajectory in trajectories.items():
            if timestamp_us not in trajectory.time_range_us:
                if obj_id == "EGO":
                    raise AssertionError("Ego trajectory ended early.")
                continue  # the track has already ended early

            assert trajectory.time_range_us.stop > timestamp_us, (
                trajectory,
                timestamp_us,
            )
            # Interpolate the pose at the requested timestamp
            interpolated_pose = trajectory.interpolate_pose(timestamp_us)
            actor_poses.append(
                ActorPoses.ActorPose(
                    actor_id=obj_id, actor_pose=interpolated_pose.as_grpc_pose()
                )
            )

        poses_message = LogEntry(
            actor_poses=ActorPoses(
                timestamp_us=timestamp_us,
                actor_poses=actor_poses,
            )
        )

        await self.log_writer.log_message(poses_message)

    async def run(self) -> None:
        """
        Run the simulation. Parses random seed sequence to follow (if available), starts at session
        on self.driver and then dispatches to _loop
        """

        # gather stateful objects which need to be opened/closed via `async with`
        async with contextlib.AsyncExitStack() as async_stack:
            # Track rollout timing
            rollout_start_time = time.perf_counter()

            # create rollouts first since they contain LogWriters which need to be active by the time we create sessions
            await async_stack.enter_async_context(self)

            await self.log_metadata(
                session_metadata=self.unbound.get_log_metadata(),
                version_ids=self.unbound.version_ids,
            )

            for service in [self.sensorsim, self.physics, self.controller]:
                await async_stack.enter_async_context(
                    service.session(
                        uuid=str(self.unbound.rollout_uuid), log_writer=self.log_writer
                    )
                )

            sensorsim_cameras = await self.sensorsim.get_available_cameras(
                self.unbound.scene_id
            )
            await self.camera_catalog.merge_local_and_sensorsim_cameras(
                self.unbound.scene_id, sensorsim_cameras
            )

            self.runtime_cameras = []
            rig_start_us = self.unbound.gt_ego_trajectory.time_range_us.start
            for camera_cfg in self.unbound.camera_configs:
                self.camera_catalog.ensure_camera_defined(
                    self.unbound.scene_id, camera_cfg.logical_id
                )
                self.runtime_cameras.append(
                    RuntimeCamera.from_camera_config(
                        camera_cfg, rig_start_us=rig_start_us
                    )
                )

            # Send all cameras that we simulate to the driver
            available_camera_protos = [
                self.camera_catalog.get_camera_definition(
                    self.unbound.scene_id, camera_cfg.logical_id
                ).as_proto()
                for camera_cfg in self.unbound.camera_configs
            ]
            # Create driver session with required parameters
            await async_stack.enter_async_context(
                self.driver.session(
                    uuid=str(self.unbound.rollout_uuid),
                    log_writer=self.log_writer,
                    random_seed=self.unbound.random_seed,
                    sensorsim_cameras=available_camera_protos,
                    scene_id=self.unbound.scene_id,
                )
            )

            # Create traffic session
            # Prepare ground truth trajectory for traffic initialization
            gt_ego_aabb_trajectory = self.unbound.gt_ego_trajectory.clip(
                self.unbound.control_timestamps_us[0],
                self.unbound.control_timestamps_us[-1] + 1,
            ).transform(
                self.unbound.transform_ego_coords_ds_to_aabb,
                is_relative=True,
            )

            await async_stack.enter_async_context(
                self.trafficsim.session(
                    uuid=str(self.unbound.rollout_uuid),
                    log_writer=self.log_writer,
                    traffic_objs=self.unbound.traffic_objs,
                    scene_id=self.unbound.scene_id,
                    ego_aabb=self.unbound.ego_aabb,
                    gt_ego_aabb_trajectory=gt_ego_aabb_trajectory,
                    start_timestamp_us=self.unbound.start_timestamp_us,
                    random_seed=self.unbound.random_seed,
                )
            )

            # Apply physics to the initial trajectory segment if physics is enabled
            if self.unbound.physics_update_mode != PhysicsUpdateMode.NONE:
                self.ego_trajectory = await self.apply_physics_to_trajectory(
                    self.ego_trajectory,
                )

            logger.info(
                "Session STARTING: uuid=%s scene=%s steps=%d seed=%d",
                self.unbound.rollout_uuid,
                self.unbound.scene_id,
                self.unbound.n_sim_steps,
                self.unbound.random_seed,
            )

            await self._loop()

            # Record rollout duration
            rollout_duration = time.perf_counter() - rollout_start_time
            ctx = try_get_context()
            if ctx is not None:
                ctx.rollout_duration.observe(rollout_duration)

            mark_rollout_complete(
                self.unbound.save_path_root, self.unbound.rollout_uuid
            )
            logger.info(
                "Session COMPLETED: uuid=%s scene=%s duration=%.2fs",
                self.unbound.rollout_uuid,
                self.unbound.scene_id,
                rollout_duration,
            )

    async def _send_images(self, past_us: int, now_us: int) -> None:
        camera_triggers = []

        for camera in self.runtime_cameras:
            # skip straddles only on the first control step
            skip_straddles = past_us == self.unbound.control_timestamps_us[0]
            triggers = camera.clock.triggers_completed_in_range(
                range(past_us, now_us), skip_straddles
            )
            camera_triggers.extend([(camera, trigger) for trigger in triggers])

        # Prepare trajectories sent to sensor-sim. Start with normal objects
        traffic_trajs = {
            track_id: traffic_obj.trajectory
            for track_id, traffic_obj in self.traffic_objs.items()
            if not traffic_obj.is_static
        }
        # Add hidden objects with their lowered trajectories so NRE renders nothing.
        if self.unbound.hidden_traffic_objs is not None:
            for hid, hobj in self.unbound.hidden_traffic_objs.items():
                traffic_trajs[hid] = hobj.trajectory

        if self.unbound.group_render_requests:
            # Note: this interface is not tested as no internal services use it!
            images_with_metadata, self.data_sensorsim_to_driver = (
                await self.sensorsim.aggregated_render(
                    camera_triggers,
                    ego_trajectory=self.ego_trajectory,
                    traffic_trajectories=traffic_trajs,
                    scene_id=self.unbound.scene_id,
                    image_format=self.unbound.image_format,
                    ego_mask_rig_config_id=self.unbound.ego_mask_rig_config_id,
                )
            )
            for image in images_with_metadata:
                await self.driver.submit_image(image)
        else:
            await asyncio.gather(
                *[
                    self.render_and_send_image(camera, trigger, traffic_trajs)
                    for camera, trigger in camera_triggers
                ]
            )

    def _assert_cameras_and_egopose_up_to_date(self, now_us: int) -> None:
        # store errors to give the user a comprehensive overview instead of raising the first encountered
        sync_errors = []

        # check if egopose is up to date
        last_trigger = self.unbound.egopose_clock.last_trigger(now_us)
        if (delta_us := last_trigger.time_range_us.stop - now_us) != 0:
            sync_errors.append(
                f"Egopose clock out of sync with planning. Last update at "
                f"{last_trigger.time_range_us.stop=} which is {delta_us=} away from {now_us=}."
            )

        # check if all cameras are up to date
        for camera in self.runtime_cameras:
            last_trigger = camera.clock.last_trigger(now_us)
            if (delta_us := last_trigger.time_range_us.stop - now_us) != 0:
                camera_name = camera.logical_id
                sync_errors.append(
                    f"Camera {camera_name} out of sync with planning. Last started frame finishes at "
                    f"{last_trigger.time_range_us.stop=} which is {delta_us=} away from {now_us=}."
                )

        if sync_errors:  # raise if any issues found
            raise ValueError("\n".join(sync_errors))

    async def _loop(self) -> None:
        """
        Runs the simulation loop with all service sessions initialized.
        """
        # Log initial poses (below we only log them for `future_us` every step)
        # I.e. below we start logging at timestamps_us[2], so we log 0, 1 here.
        t0_us, t1_us = self.unbound.control_timestamps_us[0:2]
        await asyncio.gather(
            self._log_actor_poses(t0_us),
            self._log_actor_poses(t1_us),
            self.send_egoposes(t0_us - self.unbound.control_timestep_us, t0_us),
        )

        # run the actual simulation loop.
        # at each `control_step` we provide the driver with sensor readings from between
        # the previous and current step and ask it to predict position for the next step
        # -> we clip the first and last control intervals from the range.
        for control_step in range(1, len(self.unbound.control_timestamps_us) - 1):
            step_start = time.perf_counter()

            logger.info(f"session {self.unbound.rollout_uuid}, step {control_step}.")

            past_us = self.unbound.control_timestamps_us[control_step - 1]
            now_us = self.unbound.control_timestamps_us[control_step]
            future_us = self.unbound.control_timestamps_us[control_step + 1]

            # Create tasks list for asyncio.gather
            tasks = [
                self._send_images(past_us, now_us),
                self.send_egoposes(past_us, now_us),
                self.send_route(now_us),
            ]

            # Conditionally add ground truth data sending
            if self.unbound.send_recording_ground_truth:
                tasks.append(self.send_recording_ground_truth(now_us))

            await asyncio.gather(*tasks)

            if self.unbound.assert_zero_decision_delay:
                self._assert_cameras_and_egopose_up_to_date(now_us)

            # request driving
            drive_trajectory_noisy = await self.driver.drive(
                time_now_us=now_us,
                time_query_us=future_us,  # is this time actually used?
                renderer_data=self.data_sensorsim_to_driver,
            )
            self.data_sensorsim_to_driver = None  # the data is consumed

            # Note: drive_trajectory is computed in a "noisy" local frame, not
            # the local frame. We must remap (passive transformations) this to the
            # true local frame before sending to the controller.
            # traj = passive_T_rig_est_to_local * points_in_rig
            # so, we need to remap as: traj = passive_T_rig_gt_to_local * inverse(passive_T_rig_est_to_local) * traj
            # or traj = T_local_to_rig_gt * T_rig_est_to_local * traj
            drive_trajectory = (
                drive_trajectory_noisy.transform(
                    self.ego_trajectory_estimate.poses[-1].inverse()
                )
            ).transform(self.ego_trajectory.poses[-1])

            force_gt = future_us in self.unbound.force_gt_period

            propagated_poses = await self.run_controller_and_vehicle_model(
                reference_trajectory_of_rig_in_local=drive_trajectory,
                now_us=now_us,
                future_us=future_us,
                force_gt=force_gt,
            )

            ego_ds_pose_future = await self.apply_physics_to_ego_pose(
                pose_local_to_rig_unconstrained=propagated_poses.pose_local_to_rig,
                now_us=now_us,
                future_us=future_us,
                force_gt=force_gt,
            )

            # note: for the moment, traffic responses for the ego vehicle is in aabb frame
            traffic_response = await self.trafficsim.simulate_traffic(
                ego_aabb_pose_future=ego_ds_pose_future
                @ self.unbound.transform_ego_coords_ds_to_aabb,
                future_us=future_us,
            )

            traffic_poses_future = await self.apply_physics_traffic_poses(
                ego_ds_pose_future=ego_ds_pose_future,
                traffic_response=traffic_response,
                now_us=now_us,
                future_us=future_us,
                force_gt=force_gt,
            )

            await self.update_pose(
                ego_ds_pose_future=ego_ds_pose_future,
                local_to_rig_estimate_future=propagated_poses.pose_local_to_rig_estimate,
                traffic_poses_future=traffic_poses_future,
                future_us=future_us,
                dynamic_state_estimated=propagated_poses.dynamic_state_estimated,
            )

            # Record step duration
            step_duration = time.perf_counter() - step_start
            ctx = try_get_context()
            if ctx is not None:
                ctx.step_duration.observe(step_duration)

        # Send last images so we have them in the logsuntil the rollout end
        if len(self.unbound.control_timestamps_us) >= 2:
            # Use the timestamps corresponding to the last interval processed or intended to be processed
            final_now_us = self.unbound.control_timestamps_us[-2]
            final_future_us = self.unbound.control_timestamps_us[-1]

            final_tasks = [
                self._send_images(final_now_us, final_future_us),
                self.send_egoposes(final_now_us, final_future_us),
                self.send_route(final_future_us),
            ]
            if self.unbound.send_recording_ground_truth:
                final_tasks.append(self.send_recording_ground_truth(final_future_us))

            await asyncio.gather(*final_tasks)

    async def run_controller_and_vehicle_model(
        self,
        reference_trajectory_of_rig_in_local: Trajectory,
        now_us: int,
        future_us: int,
        force_gt: bool,
    ) -> PropagatedPoses:
        """
        Run the controller and vehicle model.
        Args:
            reference_trajectory_of_rig_in_local (Trajectory): The reference trajectory of the rig in local coordinates.
            now_us (int): The current timestamp in microseconds.
            future_us (int): The future timestamp in microseconds.
            force_gt (bool): Whether to use the ground truth trajectory as the reference (controller is still run)
        Returns:
            PropagatedPoses: The future ego pose for the DS/rig coordinate system.
        """
        pose_local_to_rig = self.ego_trajectory.poses[-1]

        # We want to start feeding the controller the 'real' drive trajectory as
        # soon as available to warm it up. So we only replace it if empty.
        # This relies on `force_gt` later replacing the output of this function.
        if force_gt and (len(reference_trajectory_of_rig_in_local.timestamps_us) == 0):
            max_interp_time_us = min(
                now_us + 5e6,  # try to interpolate 5 seconds into the future
                self.unbound.gt_ego_trajectory.time_range_us.stop - 1,
            )
            reference_trajectory_of_rig_in_local = (
                self.unbound.gt_ego_trajectory.interpolate_to_timestamps(
                    np.linspace(now_us, max_interp_time_us, num=51, dtype=np.uint64)
                )
            )

        if force_gt:
            dt = (future_us - now_us) / 1e6
            pose_local_to_rig_t0 = self.unbound.gt_ego_trajectory.interpolate_pose(
                now_us
            )
            pose_local_to_rig_t1 = self.unbound.gt_ego_trajectory.interpolate_pose(
                future_us
            )
            fallback_pose_local_to_rig_future = pose_local_to_rig_t1
        else:
            dt = (
                self.ego_trajectory.timestamps_us[-1]
                - self.ego_trajectory.timestamps_us[-2]
            ) / 1e6
            pose_local_to_rig_t0 = self.ego_trajectory.poses[-2]
            pose_local_to_rig_t1 = self.ego_trajectory.poses[-1]
            fallback_pose_local_to_rig_future = (
                reference_trajectory_of_rig_in_local.interpolate_pose(future_us)
            )

        # model delay
        self.planner_delay_buffer.add(
            reference_trajectory_of_rig_in_local.transform(pose_local_to_rig.inverse()),
            now_us,
        )
        rig_reference_trajectory = self.planner_delay_buffer.at(now_us)

        pose_rig0_to_rig1 = pose_local_to_rig_t0.inverse() @ pose_local_to_rig_t1
        rig_linear_velocity_in_rig = pose_rig0_to_rig1.vec3 / dt
        # using small angle approximation for angular velocity
        rig_angular_velocity_in_rig = 2.0 * pose_rig0_to_rig1.quat[0:3] / dt

        return await self.controller.run_controller_and_vehicle(
            now_us=now_us,
            pose_local_to_rig=pose_local_to_rig,
            rig_linear_velocity_in_rig=rig_linear_velocity_in_rig,
            rig_angular_velocity_in_rig=rig_angular_velocity_in_rig,
            rig_reference_trajectory_in_rig=rig_reference_trajectory,
            future_us=future_us,
            force_gt=force_gt,
            fallback_pose_local_to_rig_future=fallback_pose_local_to_rig_future,
        )

    async def apply_physics_to_ego_pose(
        self,
        pose_local_to_rig_unconstrained: QVec,
        now_us: int,
        future_us: int,
        force_gt: bool,
    ) -> QVec:
        """Post process the ego pose response from the driver.

        Args:
            pose_local_to_rig_unconstrained: The future ego pose, unconstrained by physics.
            now_us: The current timestamp in microseconds.
            future_us: The future timestamp in microseconds.
            force_gt: Whether to force the ground truth (i.e. ignore the
                model response and use the ground truth instead)

        Returns:
            QVec: The future ego pose for the DS/rig coordinate system.
        """
        pose_local_to_rig_future_unconstrained = (
            self.unbound.gt_ego_trajectory.interpolate_pose(future_us)
            if force_gt
            else pose_local_to_rig_unconstrained
        )

        if self.unbound.physics_update_mode == PhysicsUpdateMode.NONE:
            return pose_local_to_rig_future_unconstrained

        # Update the `z` axis and rotation of the ego pose based on ground.
        pose_local_to_aabb_future_constrained, _ = (
            await self.physics.ground_intersection(
                scene_id=self.unbound.scene_id,
                delta_start_us=now_us,
                delta_end_us=future_us,
                # Physics is processed in our aabb coordinates (center of bbox),
                # but the driver returns the pose in the DS coordinate system.
                pose_now=self.ego_trajectory.poses[-1]
                @ self.unbound.transform_ego_coords_ds_to_aabb,
                pose_future=pose_local_to_rig_future_unconstrained
                @ self.unbound.transform_ego_coords_ds_to_aabb,
                traffic_poses={},
                ego_aabb=self.unbound.ego_aabb,
            )
        )

        # Physics is processed in our aabb coordinates (center of bbox), so
        # we need to transform back to DS coordinates for the ego_pose
        return (
            pose_local_to_aabb_future_constrained
            @ self.unbound.transform_ego_coords_ds_to_aabb.inverse()
        )

    async def apply_physics_traffic_poses(
        self,
        ego_ds_pose_future: QVec,
        traffic_response: TrafficReturn,
        now_us: int,
        future_us: int,
        force_gt: bool,
    ) -> dict[str, QVec]:
        """Post process the traffic response.

        Args:
            ego_ds_pose_future (QVec): The future ego pose of the rig/dw relative to local/ENU.
            traffic_response (TrafficReturn): The response from the traffic
            model.
            now_us (int): The current timestamp in microseconds.
            future_us (int): The future timestamp in microseconds.
            force_gt (bool): Whether to force the ground truth (i.e. ignore the
                model response and use the ground truth instead)

        Returns:
            QVec: The corrected poses of objects at `future_us`, all defined
                for their AABB coordinate system (this includes EGO).
        """
        ego_ds_pose_now = self.ego_trajectory.poses[-1]
        ego_aabb_pose_now = (
            ego_ds_pose_now @ self.unbound.transform_ego_coords_ds_to_aabb
        )
        ego_aabb_pose_future = (
            ego_ds_pose_future @ self.unbound.transform_ego_coords_ds_to_aabb
        )

        traffic_poses_future: dict[str, QVec] = {}

        if force_gt:  # override model responses with gt
            for key, traffic_obj in self.unbound.traffic_objs.items():
                if future_us not in traffic_obj.trajectory.time_range_us:
                    continue
                traffic_poses_future[key] = traffic_obj.trajectory.interpolate_pose(
                    future_us
                )
        else:
            for object_trajectory_update in traffic_response.object_trajectory_updates:
                object_trajectory = Trajectory.from_grpc(
                    object_trajectory_update.trajectory
                )

                assert (
                    len(object_trajectory.poses.batch_size) == 1
                ), "only time dimension should be present"
                if future_us not in object_trajectory.time_range_us:
                    continue

                traffic_poses_future[object_trajectory_update.object_id] = (
                    object_trajectory.interpolate_pose(future_us)
                )

        if self.unbound.physics_update_mode == PhysicsUpdateMode.ALL_ACTORS:
            _, traffic_poses_future = await self.physics.ground_intersection(
                scene_id=self.unbound.scene_id,
                delta_start_us=now_us,
                delta_end_us=future_us,
                pose_now=ego_aabb_pose_now,
                pose_future=ego_aabb_pose_future,
                traffic_poses=traffic_poses_future,
                ego_aabb=self.unbound.ego_aabb,
            )
        return traffic_poses_future

    async def apply_physics_to_trajectory(
        self,
        trajectory: Trajectory,
    ) -> Trajectory:
        """Apply physics to a trajectory.

        Args:
            trajectory: The trajectory to apply physics to

        Returns:
            Trajectory: The trajectory with physics applied
        """
        if self.unbound.physics_update_mode == PhysicsUpdateMode.NONE:
            return trajectory

        async def process_pose(i: int) -> QVec:
            # Unused by physics, don't worry about it.
            now_us = int(trajectory.timestamps_us[i]) - self.unbound.control_timestep_us
            # Physics is applied to the "future" pose, but we want to apply it
            # to the "now" pose, so we just call it future. Bad API.
            future_us = int(trajectory.timestamps_us[i])

            pose_now = trajectory.poses[i]  # unused by physics
            pose_future = trajectory.poses[i]

            # Transform to AABB coordinates for physics
            pose_now_aabb = pose_now @ self.unbound.transform_ego_coords_ds_to_aabb
            pose_future_aabb = (
                pose_future @ self.unbound.transform_ego_coords_ds_to_aabb
            )

            # Apply physics
            pose_future_aabb_constrained, _ = await self.physics.ground_intersection(
                scene_id=self.unbound.scene_id,
                delta_start_us=now_us,
                delta_end_us=future_us,
                pose_now=pose_now_aabb,
                pose_future=pose_future_aabb,
                traffic_poses={},
                ego_aabb=self.unbound.ego_aabb,
            )

            # Transform back to DS coordinates
            return (
                pose_future_aabb_constrained
                @ self.unbound.transform_ego_coords_ds_to_aabb.inverse()
            )

        # Process all poses concurrently
        processed_poses = await asyncio.gather(
            *[process_pose(i) for i in range(len(trajectory.timestamps_us))]
        )

        return Trajectory(
            timestamps_us=trajectory.timestamps_us,
            poses=QVec.stack(processed_poses),
        )
