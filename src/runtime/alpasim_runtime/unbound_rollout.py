# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""UnboundRollout — validated rollout metadata created before execution."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, replace

import numpy as np
from alpasim_grpc.v0 import common_pb2
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    PhysicsUpdateMode,
    RenderBundling,
    RouteGeneratorType,
    RuntimeCameraConfig,
    SimulationConfig,
    VehicleConfig,
)
from alpasim_runtime.ego_masks import EgoMaskOverride
from alpasim_runtime.errors import InvalidSceneError
from alpasim_runtime.nav_route import NavRouteLibrary, RouteNavigator
from alpasim_runtime.services.renderer import RendererService
from alpasim_runtime.services.sensorsim_service import ImageFormat
from alpasim_utils.geometry import Pose, Trajectory, trajectory_from_grpc
from alpasim_utils.scenario import AABB, Rig, TrafficObject, TrafficObjects
from alpasim_utils.scene_data_source import SceneDataSource
from alpasim_utils.scene_timing import derive_clip_start_offset_us
from alpasim_utils.types import T0_AUTO, Scene, scene_to_proto
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


def _synchronized_first_camera_frame_ranges(
    camera_configs: list[RuntimeCameraConfig],
    recorded_ranges: dict[str, range],
    render_start_us: int,
) -> dict[str, range]:
    """Place each first exposure at one render anchor."""
    synchronized_ranges: dict[str, range] = {}
    for camera_cfg in camera_configs:
        recorded_range = recorded_ranges[camera_cfg.logical_id]
        duration_us = camera_cfg.shutter_duration_us or (
            recorded_range.stop - recorded_range.start
        )
        synchronized_ranges[camera_cfg.logical_id] = range(
            render_start_us - duration_us,
            render_start_us,
        )
    return synchronized_ranges


def _build_rollout_timing(
    simulation_config: SimulationConfig,
    rig: Rig,
    camera_configs: list[RuntimeCameraConfig],
    *,
    renderer_service: RendererService,
    explicit_render_start_us: int | None = None,
) -> RolloutTiming:
    """Derive the rollout timing anchors from the (possibly clipped) rig.

    Anchors the render start on the first camera frame (or the trajectory
    start when headless), derives the first policy timestamp and force-GT
    boundary, and clamps ``n_sim_steps`` to however many complete control
    steps fit before the recording ends. Raises ``ValueError`` when not even
    one step fits.
    """
    camera_logical_ids = [camera_cfg.logical_id for camera_cfg in camera_configs]
    egomotion_context_start_us = rig.trajectory.time_range_us.start

    # ``first_camera_frame_end_us`` raises through ``first_camera_frame_ranges_us``
    # when no cameras are configured.  Headless rollouts fall back to the GT
    # trajectory start as the render anchor.
    if camera_logical_ids:
        first_camera_frame_ranges_us = rig.first_camera_frame_ranges_us(
            camera_logical_ids
        )
        render_start_us = (
            explicit_render_start_us
            if explicit_render_start_us is not None
            else rig.first_camera_frame_end_us(camera_logical_ids)
        )
    else:
        first_camera_frame_ranges_us = {}
        render_start_us = (
            explicit_render_start_us
            if explicit_render_start_us is not None
            else rig.trajectory.time_range_us.start
        )

    if explicit_render_start_us is not None and camera_configs:
        first_camera_frame_ranges_us = _synchronized_first_camera_frame_ranges(
            camera_configs,
            first_camera_frame_ranges_us,
            render_start_us,
        )
    elif simulation_config.assert_zero_decision_delay and camera_configs:
        for camera_cfg in camera_configs:
            first_camera_frame_ranges_us[camera_cfg.logical_id] = range(
                render_start_us - camera_cfg.shutter_duration_us,
                render_start_us,
            )

    first_policy_timestamp_us = renderer_service.required_policy_start_timestmap_us(
        render_start_us
    )

    closed_loop_start_us = render_start_us + simulation_config.force_gt_duration_us
    last_valid_gt_timestamp_us = rig.trajectory.time_range_us.stop - 1
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


def _prepare_fork_ego_history(
    *,
    ego_history_bytes: bytes,
    timing: RolloutTiming,
    recording_range_us: range,
    control_timestep_us: int,
    scene_key: str,
) -> Trajectory:
    """Parse and align the parent rollout's ego history for a fork."""
    fork_timestamp_us = timing.closed_loop_start_us
    if fork_timestamp_us not in recording_range_us:
        raise InvalidSceneError(
            scene_key,
            "Fork timestamp is outside the recording",
        )

    try:
        history = trajectory_from_grpc(
            common_pb2.Trajectory.FromString(ego_history_bytes)
        )
    except ValueError as exc:
        raise InvalidSceneError(scene_key, "ego_history is invalid") from exc
    if len(history) < 2:
        raise InvalidSceneError(
            scene_key, "ego_history must contain at least two poses"
        )
    if (
        not np.isfinite(history.positions).all()
        or not np.isfinite(history.quaternions).all()
    ):
        raise InvalidSceneError(
            scene_key, "ego_history contains a non-finite pose value"
        )

    history_start_us = int(history.timestamps_us[0])
    history_end_us = int(history.timestamps_us[-1])
    if history_end_us != fork_timestamp_us:
        raise InvalidSceneError(
            scene_key,
            "ego_history must end at the policy start",
        )
    if history_start_us < recording_range_us.start:
        raise InvalidSceneError(
            scene_key,
            "ego_history starts before the recording",
        )
    if history_start_us > timing.egomotion_context_start_us:
        raise InvalidSceneError(
            scene_key,
            "ego_history does not cover the warm-up",
        )
    max_gap_us = int(np.diff(history.timestamps_us).max())
    if max_gap_us > control_timestep_us:
        raise InvalidSceneError(
            scene_key,
            "ego_history is too sparse",
        )

    # Preserve the complete recorded parent history, not only the rendered
    # warm-up window. The renderer needs just ``force_gt_duration_us`` of
    # frames, but Alpamayo's first policy call consumes a longer egomotion
    # sampling grid. Add the exact render boundary without discarding the
    # earlier parent poses that seed that model input.
    timestamps_us = history.timestamps_us
    sample_timestamps_us = np.unique(
        np.concatenate(
            [
                np.array(
                    [timing.egomotion_context_start_us, fork_timestamp_us],
                    dtype=np.uint64,
                ),
                timestamps_us,
            ]
        )
    )
    return history.interpolate(sample_timestamps_us)


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
    scene: Scene
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
    use_cached_frames_during_force_gt: bool
    force_gt_frame_cache_extra_key: str | None
    physics_update_mode: PhysicsUpdateMode
    save_path_root: str
    control_timestep_us: int
    pose_reporting_interval_us: int
    camera_configs: list[RuntimeCameraConfig]
    first_camera_frame_ranges_us: dict[str, range]
    force_gt_period: range
    image_format: ImageFormat
    ego_mask_rig_config_id: str | None
    assert_zero_decision_delay: bool
    transform_ego_coords_ds_to_aabb: Pose
    ego_aabb: AABB
    planner_delay_us: int
    route_generator_type: RouteGeneratorType
    route_start_offset_m: float
    send_recording_ground_truth: bool
    send_vectorized_observation: bool
    write_asl: bool
    nre_runid: str
    nre_version: str
    nre_uuid: str
    vehicle_config: VehicleConfig

    # Executed ego poses from a parent rollout, covering renderer/policy warm-up
    # through the fork point. None identifies an ordinary recording-GT rollout.
    fork_ego_history: Trajectory | None = None

    vector_map: VectorMap | None = None
    follow_log: str | None = None

    # Recorded (position-based) navigation instructions; None unless the
    # navigation source is ``recorded``.
    route_navigator: RouteNavigator | None = None
    # Send route-geometry navigation text (source ``geometry``); needs a route
    # generator, otherwise the driver runs without navigation.
    geometry_navigation: bool = False

    # Actors filtered out from simulation but still present in USDZ; we keep
    # a lowered-to-ground trajectory so we can override their rendering.
    hidden_traffic_objs: TrafficObjects | None = None

    render_bundling: RenderBundling = RenderBundling.NONE
    # None preserves global bank behavior; a list is a validated per-clip selection.
    ego_mask_camera_logical_ids: list[str] | None = None

    # Session seed shared by this rollout's driver and traffic sessions;
    # 0 ⇒ each service picks a random one (today's behaviour).
    session_seed: int = 0

    @staticmethod
    def create(
        simulation_config: SimulationConfig,
        scene: Scene,
        version_ids: RolloutMetadata.VersionIds,
        data_source: SceneDataSource,
        rollouts_dir: str,
        renderer_service: RendererService,
        session_uuid: str | None = None,
        session_seed: int = 0,
        ego_history_bytes: bytes | None = None,
        nav_routes: NavRouteLibrary | None = None,
        ego_mask_overrides: dict[str, EgoMaskOverride] | None = None,
    ) -> UnboundRollout:
        """Create an unbound rollout from a scene and its loaded data source.

        A numeric ``policy_start_us`` sets the render anchor to the recording start
        plus ``policy_start_us - force_gt_duration_us``. ``AUTO`` uses the rig's
        first-frame end for the configured cameras, or the trajectory start when no
        cameras are configured.
        """
        scene_key = scene.scene_key
        ego_mask_rig_config_id = simulation_config.ego_mask_rig_config_id
        ego_mask_camera_logical_ids = None
        if (
            simulation_config.ego_mask_overrides_file is not None
            and ego_mask_overrides is None
        ):
            raise ValueError(
                "Ego mask overrides must be loaded before creating rollouts"
            )
        if ego_mask_overrides is not None:
            override = ego_mask_overrides.get(scene.source.source_id)
            if override is not None:
                configured_cameras = {
                    cam.logical_id for cam in simulation_config.cameras
                }
                ego_mask_rig_config_id = override.rig_config_id
                ego_mask_camera_logical_ids = [
                    camera
                    for camera in override.camera_logical_ids
                    if camera in configured_cameras
                ]
                if not ego_mask_camera_logical_ids:
                    logger.warning(
                        "Ego mask override for %s (bank %s) applies no patch: "
                        "listed cameras %s do not intersect configured cameras %s",
                        scene.source.source_id,
                        override.rig_config_id,
                        override.camera_logical_ids,
                        sorted(configured_cameras),
                    )
        recording_start_us = data_source.rig.trajectory.time_range_us.start
        camera_configs = list(simulation_config.cameras)
        renderer_service.validate_timing_alignment(simulation_config)
        try:
            clip_start_offset_us = derive_clip_start_offset_us(
                scene.policy_start_us,
                simulation_config.force_gt_duration_us,
            )
            explicit_render_start_us: int | None = None
            if clip_start_offset_us is None:
                rig = data_source.rig
            else:
                explicit_render_start_us = recording_start_us + clip_start_offset_us
                context_start_us = explicit_render_start_us
                if camera_configs:
                    recorded_ranges = data_source.rig.first_camera_frame_ranges_us(
                        camera_cfg.logical_id for camera_cfg in camera_configs
                    )
                    explicit_ranges = _synchronized_first_camera_frame_ranges(
                        camera_configs,
                        recorded_ranges,
                        explicit_render_start_us,
                    )
                    context_start_us = min(
                        frame_range.start for frame_range in explicit_ranges.values()
                    )
                if context_start_us < recording_start_us:
                    raise InvalidSceneError(
                        scene_key,
                        "Explicit-T0 scene does not provide enough camera exposure "
                        "history: "
                        f"policy_start_us={scene.policy_start_us}, "
                        f"force_gt_duration_us={simulation_config.force_gt_duration_us}, "
                        f"required_context_start_us={context_start_us}, "
                        f"recording_start_us={recording_start_us}",
                    )
                trajectory = data_source.rig.trajectory.clip(
                    context_start_us,
                    data_source.rig.trajectory.time_range_us.stop,
                )
                rig = replace(data_source.rig, trajectory=trajectory)
        except ValueError as exc:
            raise InvalidSceneError(scene_key, str(exc)) from exc
        # A clip that is too short — natively or after the start offset — cannot
        # host a rollout. Surface it as a deterministic (non-retryable) scene
        # error rather than the raw ValueError from timing construction.
        try:
            timing = _build_rollout_timing(
                simulation_config,
                rig,
                camera_configs,
                renderer_service=renderer_service,
                explicit_render_start_us=explicit_render_start_us,
            )
        except ValueError as exc:
            detail = str(exc)
            if scene.policy_start_us != T0_AUTO:
                detail = (
                    f"{detail} (policy_start_us={scene.policy_start_us}, "
                    f"clip_start_offset_us={clip_start_offset_us})"
                )
            raise InvalidSceneError(scene_key, detail) from exc
        # A fork may start too late in the recording for the configured horizon.
        # It then runs the steps that ``_build_rollout_timing`` fits before the
        # recording end; the post-handoff check below still requires one.
        if (
            scene.policy_start_us != T0_AUTO
            and ego_history_bytes is None
            and timing.n_sim_steps != simulation_config.n_sim_steps
        ):
            raise InvalidSceneError(
                scene_key,
                "Explicit-T0 scene requires the full configured horizon: "
                f"expected {simulation_config.n_sim_steps} steps, got "
                f"{timing.n_sim_steps}",
            )
        gt_ego_trajectory = rig.trajectory
        if (
            ego_history_bytes is not None
            and not simulation_config.skip_driver_during_force_gt
        ):
            raise InvalidSceneError(
                scene_key,
                "ego_history requires skip_driver_during_force_gt=true",
            )
        if (
            ego_history_bytes is not None
            and timing.end_timestamp_us <= timing.closed_loop_start_us
        ):
            raise InvalidSceneError(
                scene_key,
                "ego_history requires at least one post-handoff control step",
            )
        fork_ego_history = (
            _prepare_fork_ego_history(
                ego_history_bytes=ego_history_bytes,
                timing=timing,
                recording_range_us=data_source.rig.trajectory.time_range_us,
                control_timestep_us=simulation_config.control_timestep_us,
                scene_key=scene_key,
            )
            if ego_history_bytes is not None
            else None
        )

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
        elif rig.vehicle_config is not None:
            vehicle = rig.vehicle_config
        else:
            raise InvalidSceneError(
                scene_key,
                "no vehicle config in simulation config or rig",
            )

        ego_aabb = AABB(
            x=vehicle.aabb_x_m,
            y=vehicle.aabb_y_m,
            z=vehicle.aabb_z_m,
        )

        try:
            vector_map = data_source.map
        except (OSError, MemoryError):
            raise
        except Exception as exc:
            raise InvalidSceneError(
                scene_key, f"failed to load vector map: {exc}"
            ) from exc

        route_navigator = None
        if nav_routes is not None:
            try:
                # The whole recording, so heading look-backs reach before an
                # explicit T0's cropped trajectory.
                route_navigator = RouteNavigator(
                    nav_routes.route_index_for_clip(scene.source.source_id),
                    data_source.rig.trajectory,
                )
            except (KeyError, ValueError) as exc:
                raise InvalidSceneError(
                    scene_key, f"navigation instructions unavailable: {exc}"
                ) from exc

        return UnboundRollout(
            rollout_uuid=session_uuid or str(uuid.uuid1()),
            session_seed=session_seed,
            scene=scene,
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
            use_cached_frames_during_force_gt=(
                simulation_config.force_gt_frame_cache.enabled
                and fork_ego_history is None
            ),
            force_gt_frame_cache_extra_key=(
                simulation_config.force_gt_frame_cache.extra_key
            ),
            control_timestep_us=simulation_config.control_timestep_us,
            follow_log=None,
            save_path_root=os.path.join(rollouts_dir, scene_key),
            version_ids=version_ids,
            camera_configs=camera_configs,
            first_camera_frame_ranges_us=timing.first_camera_frame_ranges_us,
            force_gt_period=force_gt_period,
            physics_update_mode=simulation_config.physics_update_mode,
            image_format={"jpeg": ImageFormat.JPEG, "png": ImageFormat.PNG}[
                simulation_config.image_format
            ],
            ego_mask_rig_config_id=ego_mask_rig_config_id,
            ego_mask_camera_logical_ids=ego_mask_camera_logical_ids,
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
            route_start_offset_m=simulation_config.route_start_offset_m,
            send_recording_ground_truth=simulation_config.send_recording_ground_truth,
            send_vectorized_observation=simulation_config.send_vectorized_observation,
            write_asl=simulation_config.write_asl,
            vehicle_config=vehicle,
            fork_ego_history=fork_ego_history,
            vector_map=vector_map,
            hidden_traffic_objs=hidden_traffic_objs,
            render_bundling=simulation_config.render_bundling,
            route_navigator=route_navigator,
            geometry_navigation=(
                simulation_config.nav_instruction.source == "geometry"
            ),
        )

    def get_log_metadata(self) -> RolloutMetadata.SessionMetadata:
        return RolloutMetadata.SessionMetadata(
            session_uuid=self.rollout_uuid,
            n_sim_steps=self.n_sim_steps,
            start_timestamp_us=self.egomotion_context_start_us,
            render_start_timestamp_us=self.render_start_timestamp_us,
            control_timestep_us=self.control_timestep_us,
            nre_runid=self.nre_runid,
            nre_version=self.nre_version,
            nre_uuid=self.nre_uuid,
            scene=scene_to_proto(self.scene),
        )
