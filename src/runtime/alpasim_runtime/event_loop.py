# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Event-based simulation loop.

Components are modelled as self-scheduling events processed from a priority
queue, allowing each one to run at its own cadence.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from alpasim_grpc.v0.logging_pb2 import LogEntry, RolloutMetadata
from alpasim_runtime.autoresume import mark_rollout_complete
from alpasim_runtime.broadcaster import MessageBroadcaster, MessageHandler
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.delay_buffer import DelayBuffer
from alpasim_runtime.events.base import (
    EndSimulationException,
    EventQueue,
    SimulationEndEvent,
)
from alpasim_runtime.events.controller import ControllerEvent
from alpasim_runtime.events.force_gt_utils import (
    FORCE_GT_REFERENCE_HORIZON_US,
    force_gt_dynamic_trajectory,
)
from alpasim_runtime.events.physics import PhysicsEvent, PhysicsTarget
from alpasim_runtime.events.policy import PolicyEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_runtime.events.step import InitialStepEvent, StepEvent
from alpasim_runtime.events.traffic import TrafficEvent
from alpasim_runtime.force_gt_blend import (
    force_gt_physics_blend_alpha,
    force_gt_physics_blend_hold_end_us,
)
from alpasim_runtime.force_gt_cache_signature import build_force_gt_render_signature
from alpasim_runtime.force_gt_frame_cache import ForceGtFrameCache
from alpasim_runtime.nav_route import GeometryNavigator
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.frame_quality_service import FrameQualityService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.renderer import RendererService
from alpasim_runtime.services.session_configs import (
    DriverSessionConfig,
    FrameQualitySessionConfig,
    RendererSessionConfig,
    TrafficSessionConfig,
)
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.telemetry.telemetry_context import try_get_context
from alpasim_runtime.types import RuntimeCamera
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_utils import geometry
from alpasim_utils.logs import LogWriter
from alpasim_utils.scenario import TrafficObjects
from alpasim_utils.scene_data_source import SceneDataSource

from eval.runtime_evaluator import RuntimeEvaluator
from eval.scenario_evaluator import ScenarioEvalResult
from eval.schema import EvalConfig

logger = logging.getLogger(__name__)


def _build_traffic_session_trajectory(unbound: UnboundRollout) -> geometry.Trajectory:
    """Build the ego AABB trajectory used for traffic session initialization."""
    ego_source = unbound.fork_ego_history
    if ego_source is None:
        ego_source = unbound.gt_ego_trajectory.clip(
            unbound.egomotion_context_start_us,
            unbound.end_timestamp_us + 1,
        )
    return ego_source.transform(
        unbound.transform_ego_coords_ds_to_aabb,
        is_relative=True,
    )


def _simulated_duration_us(unbound: UnboundRollout) -> int:
    """Return the effective simulated span covered by this rollout."""
    return unbound.end_timestamp_us - unbound.egomotion_context_start_us


def _build_initial_ego_trajectory(
    unbound: UnboundRollout,
) -> geometry.DynamicTrajectory:
    """Seed ego poses and dynamics from fork history or recording GT."""
    ego_source = (
        unbound.fork_ego_history
        if unbound.fork_ego_history is not None
        else unbound.gt_ego_trajectory
    )
    # A fork carries parent poses from before the short render warm-up so the
    # first driver update can populate the policy's complete egomotion grid.
    # Ordinary recording-GT rollouts still start at the render context boundary.
    initial_history_start_us = (
        ego_source.time_range_us.start
        if unbound.fork_ego_history is not None
        else unbound.egomotion_context_start_us
    )
    ego_trajectory = ego_source.clip(
        initial_history_start_us,
        unbound.first_policy_timestamp_us + 1,
    )

    # Estimate dynamics over the full source trajectory with the same helper the
    # force-GT controller coercion uses, so the seed and the coerced dynamics
    # agree at shared timestamps (dynamics are rig-frame, see DynamicState).
    initial_dynamics = force_gt_dynamic_trajectory(ego_source).interpolate_dynamics(
        ego_trajectory.timestamps_us
    )

    return geometry.DynamicTrajectory.from_trajectory_and_dynamics(
        ego_trajectory, initial_dynamics
    )


@dataclass
class EventBasedRollout:
    """Event-based simulation loop implementation.

    Processes events sequentially in timestamp order, with priority
    determining order at the same timestamp.

    The renderer is pluggable: ``renderer_service`` is the active renderer's
    service client (built-in sensorsim/video_model or a plugin service). Core is
    renderer-agnostic at the rollout level; all renderer-specific behavior
    lives in the service and the render event subclass.
    """

    unbound: UnboundRollout
    data_source: SceneDataSource
    driver: DriverService
    renderer_service: RendererService
    physics: PhysicsService
    trafficsim: TrafficService
    controller: ControllerService
    frame_quality: FrameQualityService
    camera_catalog: CameraCatalog
    eval_config: EvalConfig
    eval_executor: ProcessPoolExecutor
    # Globally shared on-disk cache reused across rollouts/runs to skip
    # re-rendering force-GT frames. ``None`` disables force-GT render caching.
    force_gt_frame_cache: ForceGtFrameCache | None = None

    # Mutable state (initialized in __post_init__). ``scene_uuid`` and
    # ``render_signature`` are the two force-GT cache directory levels; both are
    # set together once camera definitions resolve, or both stay ``None``.
    force_gt_scene_uuid: str | None = field(init=False, default=None)
    force_gt_render_signature: str | None = field(init=False, default=None)
    ego_trajectory: geometry.DynamicTrajectory = field(init=False)
    ego_trajectory_estimate: geometry.DynamicTrajectory = field(init=False)
    force_gt_ego_trajectory: geometry.Trajectory | None = field(
        init=False, default=None
    )
    traffic_objs: TrafficObjects = field(init=False)

    broadcaster: MessageBroadcaster = field(init=False)
    planner_delay_buffer: DelayBuffer = field(init=False)
    route_generator: RouteGenerator | None = field(init=False)
    runtime_cameras: list[RuntimeCamera] = field(init=False, default_factory=list)

    _runtime_evaluator: RuntimeEvaluator = field(init=False)
    _frame_quality_enabled: bool = field(init=False)

    def __post_init__(self) -> None:
        """Initialize mutable state."""
        context_start_us = self.unbound.egomotion_context_start_us
        first_policy_timestamp_us = self.unbound.first_policy_timestamp_us

        # Seed the selected ego context through the first policy decision.
        # The first policy call then receives dense egomotion history rather
        # than a synthetic two-point shortcut.
        self.traffic_objs = self.unbound.traffic_objs.clip_trajectories(
            context_start_us, first_policy_timestamp_us + 1
        )

        self.ego_trajectory = _build_initial_ego_trajectory(self.unbound)
        self.ego_trajectory_estimate = self.ego_trajectory.clone()
        self.force_gt_ego_trajectory = self.unbound.fork_ego_history

        self.planner_delay_buffer = DelayBuffer(self.unbound.planner_delay_us)
        self.route_generator = RouteGenerator.create(
            self.unbound.gt_ego_trajectory.positions,
            vector_map=self.unbound.vector_map,
            route_generator_type=self.unbound.route_generator_type,
            route_start_offset_m=self.unbound.route_start_offset_m,
            recorded_quaternions_in_local=self.unbound.gt_ego_trajectory.quaternions,
        )

        self._runtime_evaluator = RuntimeEvaluator(
            eval_config=self.eval_config,
            rollout_uuid=self.unbound.rollout_uuid,
            save_path_root=self.unbound.save_path_root,
            vector_map=self.unbound.vector_map,
        )

        handlers: list[MessageHandler] = [self._runtime_evaluator]
        self._frame_quality_enabled = not self.frame_quality.skip and (
            self.eval_config.enabled or self.unbound.write_asl
        )
        if self.unbound.write_asl:
            handlers.insert(0, LogWriter(file_path=self._asl_log_path()))
        self.broadcaster = MessageBroadcaster(handlers=handlers)

    def _rollout_dir(self) -> str:
        return os.path.join(self.unbound.save_path_root, self.unbound.rollout_uuid)

    def _asl_log_path(self) -> str:
        return os.path.join(self._rollout_dir(), "rollout.asl")

    async def _log_metadata(
        self,
        session_metadata: RolloutMetadata.SessionMetadata,
        version_ids: RolloutMetadata.VersionIds,
    ) -> None:
        """Log rollout metadata at the start of a rollout."""
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

        await self.broadcaster.broadcast(
            LogEntry(
                rollout_metadata=RolloutMetadata(
                    session_metadata=session_metadata,
                    actor_definitions=RolloutMetadata.ActorDefinitions(
                        actor_aabb=[ego_aabb, *traffic_actor_aabbs]
                    ),
                    force_gt_duration=self.unbound.force_gt_duration_us,
                    version_ids=version_ids,
                    rollout_index=0,
                    transform_ego_coords_rig_to_aabb=geometry.pose_to_grpc(
                        self.unbound.transform_ego_coords_ds_to_aabb
                    ),
                    ego_rig_recorded_ground_truth_trajectory=geometry.trajectory_to_grpc(
                        self.unbound.gt_ego_trajectory
                    ),
                )
            )
        )

    async def _apply_physics_to_trajectory(
        self,
        trajectory: geometry.Trajectory,
    ) -> geometry.Trajectory:
        """Apply physics ground-correction to every pose in *trajectory*."""
        if self.unbound.physics_update_mode == PhysicsUpdateMode.NONE:
            return trajectory

        ds_to_aabb = self.unbound.transform_ego_coords_ds_to_aabb
        aabb_to_ds = ds_to_aabb.inverse()

        traj_aabb = trajectory.transform(ds_to_aabb, is_relative=True)

        delta_start_us = (
            int(trajectory.timestamps_us[0]) - self.unbound.control_timestep_us
        )
        delta_end_us = int(trajectory.timestamps_us[-1])

        corrected_aabb, _ = await self.physics.ground_intersection(
            clip_id=self.unbound.scene.source.source_id,
            delta_start_us=delta_start_us,
            delta_end_us=delta_end_us,
            ego_trajectory_aabb=traj_aabb,
            traffic_poses={},
            ego_aabb=self.unbound.ego_aabb,
        )

        return corrected_aabb.transform(aabb_to_ds, is_relative=True)

    async def _build_force_gt_physics_blend_trajectory(self) -> geometry.Trajectory:
        """Build force-GT fallback and the post-handover controller lookahead."""
        unbound = self.unbound
        gt_hold_end_us = force_gt_physics_blend_hold_end_us(unbound)
        configured_blend_end_us = (
            unbound.render_start_timestamp_us + unbound.force_gt_duration_us
        )
        blend_end_us = min(configured_blend_end_us, unbound.end_timestamp_us)
        reference_end_us = min(
            blend_end_us + FORCE_GT_REFERENCE_HORIZON_US,
            unbound.end_timestamp_us,
            unbound.gt_ego_trajectory.time_range_us.stop - 1,
        )

        if blend_end_us <= gt_hold_end_us:
            return unbound.gt_ego_trajectory.clip(
                unbound.egomotion_context_start_us,
                min(gt_hold_end_us, unbound.end_timestamp_us) + 1,
            )

        dt_us = unbound.control_timestep_us
        clipped_gt_timestamps = unbound.gt_ego_trajectory.clip(
            unbound.egomotion_context_start_us,
            reference_end_us + 1,
        ).timestamps_us
        timestamps = list(
            range(unbound.first_policy_timestamp_us, reference_end_us + 1, dt_us)
        )
        if not timestamps or timestamps[-1] != reference_end_us:
            timestamps.append(reference_end_us)
        timestamps = sorted(
            set(
                [
                    unbound.egomotion_context_start_us,
                    gt_hold_end_us,
                    blend_end_us,
                    reference_end_us,
                    *clipped_gt_timestamps.tolist(),
                    *timestamps,
                ]
            )
        )
        timestamps_us = np.array(timestamps, dtype=np.uint64)

        gt_trajectory = unbound.gt_ego_trajectory.interpolate(timestamps_us)
        physics_trajectory = await self._apply_physics_to_trajectory(gt_trajectory)

        alphas = np.array(
            [
                force_gt_physics_blend_alpha(unbound, timestamp_us)
                for timestamp_us in timestamps
            ],
            dtype=np.float32,
        )

        return gt_trajectory.blend(physics_trajectory, alphas)

    def _apply_force_gt_blend_to_seeded_ego_trajectory(self) -> None:
        """Replace seeded GT poses with the blended force-GT poses."""
        if self.force_gt_ego_trajectory is None:
            return

        timestamps_us = self.ego_trajectory.timestamps_us
        blended_seed = self.force_gt_ego_trajectory.interpolate(timestamps_us)
        self.ego_trajectory = geometry.DynamicTrajectory.from_trajectory_and_dynamics(
            blended_seed,
            self.ego_trajectory.dynamics,
        )
        self.ego_trajectory_estimate = (
            geometry.DynamicTrajectory.from_trajectory_and_dynamics(
                blended_seed,
                self.ego_trajectory_estimate.dynamics,
            )
        )

    def _create_rollout_state(self) -> RolloutState:
        """Create the RolloutState from the current rollout."""
        force_gt_trajectory = self.force_gt_ego_trajectory
        if force_gt_trajectory is None:
            force_gt_trajectory = self.unbound.gt_ego_trajectory
        return RolloutState(
            unbound=self.unbound,
            ego_trajectory=self.ego_trajectory,
            ego_trajectory_estimate=self.ego_trajectory_estimate,
            traffic_objs=self.traffic_objs,
            force_gt_ego_trajectory=self.force_gt_ego_trajectory,
            force_gt_dynamics=force_gt_dynamic_trajectory(force_gt_trajectory),
            force_gt_frame_cache=self.force_gt_frame_cache,
            force_gt_scene_uuid=self.force_gt_scene_uuid,
            force_gt_render_signature=self.force_gt_render_signature,
            step_context=StepContext(),
            last_egopose_update_us=None,
        )

    def _resolve_force_gt_cache_coordinates(self) -> None:
        """Set the per-rollout force-GT cache directory levels on ``self``.

        Leaves both ``None`` when caching is disabled (feature off or no cache
        configured). When caching is enabled a cache key must be derivable, so a
        missing USDZ path or empty camera set raises rather than silently
        disabling the cache.
        """
        if (
            self.force_gt_frame_cache is None
            or not self.unbound.use_cached_frames_during_force_gt
        ):
            return

        camera_definitions = {
            camera.logical_id: self.camera_catalog.get_camera_definition(
                self.unbound.scene.source.source_id, camera.logical_id
            )
            for camera in self.runtime_cameras
        }
        # Key the cache by the stable dataset UUID rather than the USDZ
        # filename: the two are not always the same (e.g. for the public
        # dataset), while the UUID uniquely identifies the scene contents.
        scene_uuid = self.unbound.nre_uuid
        if not scene_uuid or scene_uuid == "None" or not camera_definitions:
            raise ValueError(
                "Cannot build a force-GT cache key: "
                f"scene_uuid={scene_uuid!r}, cameras={sorted(camera_definitions)}"
            )

        self.force_gt_scene_uuid = scene_uuid
        self.force_gt_render_signature = build_force_gt_render_signature(
            unbound=self.unbound,
            camera_definitions=camera_definitions,
            smooth_trajectories=getattr(self.data_source, "_smooth_trajectories", None),
        )

    def _create_service_bundle(self) -> ServiceBundle:
        """Create a ServiceBundle from the rollout's service handles."""
        return ServiceBundle(
            driver=self.driver,
            controller=self.controller,
            physics=self.physics,
            trafficsim=self.trafficsim,
            broadcaster=self.broadcaster,
            planner_delay_buffer=self.planner_delay_buffer,
        )

    def _create_initial_events(self) -> EventQueue:
        """Create and schedule the initial set of events.

        The rig start seeds/logs context state. Rendering starts at the
        centralized first-camera frame anchor, while the policy/control
        pipeline starts on the renderer-aligned force-GT grid.
        """
        unbound = self.unbound
        queue = EventQueue()
        services = self._create_service_bundle()

        scene_start_us = unbound.egomotion_context_start_us
        simulation_end_us = unbound.end_timestamp_us
        closed_loop_start_us = unbound.closed_loop_start_us
        first_policy_timestamp_us = unbound.first_policy_timestamp_us

        camera_ids = [cam.logical_id for cam in self.runtime_cameras]

        # === Render event (selected by the active renderer service) ===
        render_events = self.renderer_service.make_initial_render_event(
            scene_start_us=scene_start_us,
            render_start_timestamp_us=unbound.render_start_timestamp_us,
            closed_loop_start_us=closed_loop_start_us,
            simulation_end_us=simulation_end_us,
            control_timestep_us=unbound.control_timestep_us,
            runtime_cameras=list(self.runtime_cameras),
            driver=self.driver,
            frame_quality=self.frame_quality,
            broadcaster=self.broadcaster,
        )
        if isinstance(render_events, list):
            for event in render_events:
                queue.submit(event)
        else:
            queue.submit(render_events)

        # === Pipeline events — all start at first_policy_timestamp_us ===
        dt = unbound.control_timestep_us

        queue.submit(InitialStepEvent(timestamp_us=scene_start_us, services=services))

        queue.submit(
            PolicyEvent(
                timestamp_us=first_policy_timestamp_us,
                policy_timestep_us=dt,
                services=services,
                camera_ids=camera_ids,
                route_generator=self.route_generator,
                navigation=unbound.route_navigator
                or (
                    GeometryNavigator()
                    if unbound.geometry_navigation and self.route_generator is not None
                    else None
                ),
                send_recording_ground_truth=unbound.send_recording_ground_truth,
                send_vectorized_observation=unbound.send_vectorized_observation,
            )
        )
        queue.submit(
            ControllerEvent(
                timestamp_us=first_policy_timestamp_us,
                control_timestep_us=dt,
                services=services,
            )
        )
        queue.submit(
            PhysicsEvent(
                timestamp_us=first_policy_timestamp_us,
                control_timestep_us=dt,
                services=services,
                target=PhysicsTarget.EGO,
            )
        )
        queue.submit(
            TrafficEvent(
                timestamp_us=first_policy_timestamp_us,
                control_timestep_us=dt,
                services=services,
            )
        )
        queue.submit(
            PhysicsEvent(
                timestamp_us=first_policy_timestamp_us,
                control_timestep_us=dt,
                services=services,
                target=PhysicsTarget.TRAFFIC,
            )
        )
        queue.submit(
            StepEvent(
                timestamp_us=first_policy_timestamp_us,
                control_timestep_us=dt,
                services=services,
            )
        )

        # === Simulation end ===
        queue.submit(SimulationEndEvent(timestamp_us=simulation_end_us))

        return queue

    async def run(self) -> ScenarioEvalResult | None:
        """Run the event-based simulation loop.

        Returns:
            ScenarioEvalResult if in-runtime evaluation is enabled, None otherwise.
        """
        async with contextlib.AsyncExitStack() as async_stack:
            rollout_start_time = time.perf_counter()

            # Enter broadcaster context
            await async_stack.enter_async_context(self.broadcaster)

            await self._log_metadata(
                session_metadata=self.unbound.get_log_metadata(),
                version_ids=self.unbound.version_ids,
            )

            # Build runtime cameras from user config (no camera-catalog
            # dependency at this point — the renderer's session init below is
            # responsible for populating camera_catalog for this scene).
            self.runtime_cameras = [
                RuntimeCamera.from_camera_config(
                    camera_cfg,
                    first_frame_range_us=self.unbound.first_camera_frame_ranges_us[
                        camera_cfg.logical_id
                    ],
                )
                for camera_cfg in self.unbound.camera_configs
            ]

            # Enter the renderer's session: owns scene-specific camera
            # registration and any session-bootstrap work. The video model
            # opens a remote session with hdmap + initial frames here.
            await async_stack.enter_async_context(
                self.renderer_service.rollout_session(
                    uuid=str(self.unbound.rollout_uuid),
                    broadcaster=self.broadcaster,
                    session_config=RendererSessionConfig(
                        data_source=self.data_source,
                        runtime_cameras=self.runtime_cameras,
                        gt_ego_trajectory=self.unbound.gt_ego_trajectory,
                        image_format=self.unbound.image_format,
                        ego_mask_rig_config_id=self.unbound.ego_mask_rig_config_id,
                        ego_mask_camera_logical_ids=self.unbound.ego_mask_camera_logical_ids,
                    ),
                )
            )

            for service in [self.physics, self.controller]:
                await async_stack.enter_async_context(
                    service.rollout_session(
                        uuid=str(self.unbound.rollout_uuid),
                        broadcaster=self.broadcaster,
                    )
                )

            if self._frame_quality_enabled:
                quality_config = self.eval_config.scorers.frame_quality
                await async_stack.enter_async_context(
                    self.frame_quality.rollout_session(
                        uuid=str(self.unbound.rollout_uuid),
                        broadcaster=self.broadcaster,
                        session_config=FrameQualitySessionConfig(
                            camera_logical_id=quality_config.camera_logical_id,
                            rpc_timeout_s=quality_config.rpc_timeout_s,
                            enabled=True,
                        ),
                    )
                )

            # camera_catalog is now populated by the renderer's session init.
            available_camera_protos = [
                self.camera_catalog.get_camera_definition(
                    self.unbound.scene.source.source_id, camera_cfg.logical_id
                ).as_proto()
                for camera_cfg in self.unbound.camera_configs
            ]

            # Resolve the force-GT cache coordinates now that camera definitions
            # are available (depends on the resolved intrinsics/extrinsics).
            self._resolve_force_gt_cache_coordinates()

            await async_stack.enter_async_context(
                self.driver.rollout_session(
                    uuid=str(self.unbound.rollout_uuid),
                    broadcaster=self.broadcaster,
                    session_config=DriverSessionConfig(
                        sensorsim_cameras=available_camera_protos,
                        ego_aabb=self.unbound.ego_aabb,
                        rig_to_ego_aabb=(self.unbound.transform_ego_coords_ds_to_aabb),
                        scene_key=self.unbound.scene.scene_key,
                        random_seed=self.unbound.session_seed or None,
                    ),
                )
            )

            # Create traffic session
            await async_stack.enter_async_context(
                self.trafficsim.rollout_session(
                    uuid=str(self.unbound.rollout_uuid),
                    broadcaster=self.broadcaster,
                    session_config=TrafficSessionConfig(
                        traffic_objs=self.unbound.traffic_objs,
                        clip_id=self.unbound.scene.source.source_id,
                        ego_aabb=self.unbound.ego_aabb,
                        gt_ego_aabb_trajectory=_build_traffic_session_trajectory(
                            self.unbound
                        ),
                        start_timestamp_us=self.unbound.egomotion_context_start_us,
                        force_gt_duration_us=self.unbound.force_gt_duration_us,
                        control_timestep_us=self.unbound.control_timestep_us,
                        random_seed=self.unbound.session_seed or None,
                    ),
                )
            )

            if (
                self.unbound.physics_update_mode != PhysicsUpdateMode.NONE
                and self.unbound.fork_ego_history is None
            ):
                self.force_gt_ego_trajectory = (
                    await self._build_force_gt_physics_blend_trajectory()
                )
                self._apply_force_gt_blend_to_seeded_ego_trajectory()

            logger.info(
                "Session STARTING: uuid=%s scene=%s steps=%d",
                self.unbound.rollout_uuid,
                self.unbound.scene.scene_key,
                self.unbound.n_sim_steps,
            )

            # Start timing the main loop
            loop_start_time = time.perf_counter()
            logger.info("Event-based simulation loop timer started")

            # Create state and initial events
            state = self._create_rollout_state()
            event_queue = self._create_initial_events()

            # Main event loop
            try:
                while event_queue:
                    event = event_queue.pop()
                    logger.info(
                        f"sim_time {event.timestamp_us:_}us: {event.description()}"
                    )
                    await event.handle(state, event_queue)
            except EndSimulationException:
                logger.info("Simulation ended via SimulationEndEvent")
            except Exception:
                logger.exception(
                    "Error during event handling. Pending events in queue (%d):",
                    len(event_queue),
                )
                for event_desc in event_queue.pending_events_summary():
                    logger.error("  - %s", event_desc)
                raise

            await self._drain_runtime_work(state)

            # Record timing
            loop_duration = time.perf_counter() - loop_start_time
            logger.info("Event-based simulation loop timer stopped")

            rollout_duration = time.perf_counter() - rollout_start_time
            ctx = try_get_context()
            if ctx is not None:
                ctx.record_rollout_duration(rollout_duration)

        eval_result = await self._runtime_evaluator.run_evaluation(self.eval_executor)
        mark_rollout_complete(self.unbound.save_path_root, self.unbound.rollout_uuid)

        simulated_duration_us = _simulated_duration_us(self.unbound)
        simulated_duration_s = simulated_duration_us / 1e6
        realtime_ratio = simulated_duration_s / loop_duration

        logger.info(
            "Session COMPLETED: uuid=%s scene=%s "
            "simulated %.2f sim seconds in %.2f wall clock seconds for %.2fx real time "
            "(total rollout %.2fs incl. setup/warmup)",
            self.unbound.rollout_uuid,
            self.unbound.scene.scene_key,
            simulated_duration_s,
            loop_duration,
            realtime_ratio,
            rollout_duration,
        )

        return eval_result

    async def _drain_runtime_work(self, state: RolloutState) -> None:
        """Finish driver-path and quality tasks before evaluation starts."""
        if state.step_context is not None:
            await state.step_context.drain_outstanding_tasks()
        if self._frame_quality_enabled:
            await self.frame_quality.drain()


def create_event_rollout(
    *,
    unbound: UnboundRollout,
    data_source: SceneDataSource,
    driver: DriverService,
    renderer_service: RendererService,
    physics: PhysicsService,
    trafficsim: TrafficService,
    controller: ControllerService,
    frame_quality: FrameQualityService,
    camera_catalog: CameraCatalog,
    eval_config: EvalConfig,
    eval_executor: ProcessPoolExecutor,
    force_gt_frame_cache: ForceGtFrameCache | None = None,
) -> EventBasedRollout:
    """Construct :class:`EventBasedRollout` with the active renderer service."""
    return EventBasedRollout(
        unbound=unbound,
        data_source=data_source,
        driver=driver,
        renderer_service=renderer_service,
        physics=physics,
        trafficsim=trafficsim,
        controller=controller,
        frame_quality=frame_quality,
        camera_catalog=camera_catalog,
        eval_config=eval_config,
        eval_executor=eval_executor,
        force_gt_frame_cache=force_gt_frame_cache,
    )
