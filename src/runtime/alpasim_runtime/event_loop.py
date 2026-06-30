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
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import PhysicsUpdateMode
from alpasim_runtime.delay_buffer import DelayBuffer
from alpasim_runtime.events.base import (
    EndSimulationException,
    EventQueue,
    SimulationEndEvent,
)
from alpasim_runtime.events.controller import ControllerEvent
from alpasim_runtime.events.physics import PhysicsEvent, PhysicsTarget
from alpasim_runtime.events.policy import PolicyEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_runtime.events.step import InitialStepEvent, StepEvent
from alpasim_runtime.events.traffic import TrafficEvent
from alpasim_runtime.force_gt_blend import (
    force_gt_physics_blend_alpha,
    force_gt_physics_blend_hold_end_us,
)
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.renderer import RendererService
from alpasim_runtime.services.session_configs import (
    DriverSessionConfig,
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
    return unbound.gt_ego_trajectory.clip(
        unbound.egomotion_context_start_us,
        unbound.end_timestamp_us + 1,
    ).transform(
        unbound.transform_ego_coords_ds_to_aabb,
        is_relative=True,
    )


def _simulated_duration_us(unbound: UnboundRollout) -> int:
    """Return the effective simulated span covered by this rollout."""
    return unbound.end_timestamp_us - unbound.egomotion_context_start_us


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
    camera_catalog: CameraCatalog
    eval_config: EvalConfig
    eval_executor: ProcessPoolExecutor

    # Mutable state (initialized in __post_init__)
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

    def __post_init__(self) -> None:
        """Initialize mutable state."""
        context_start_us = self.unbound.egomotion_context_start_us
        first_policy_timestamp_us = self.unbound.first_policy_timestamp_us

        asl_log_writer = LogWriter(file_path=self._asl_log_path())

        # Seed all recorded ego context through the first policy decision.
        # The first policy call then receives dense egomotion history rather
        # than a synthetic two-point shortcut.
        self.traffic_objs = self.unbound.traffic_objs.clip_trajectories(
            context_start_us, first_policy_timestamp_us + 1
        )

        gt = self.unbound.gt_ego_trajectory
        ego_traj = gt.clip(context_start_us, first_policy_timestamp_us + 1)
        context_timestamps = ego_traj.timestamps_us

        # Build initial dynamics from GT derivatives at each context timestamp.
        gt_velocities = gt.velocities()
        gt_yaw_rates = gt.yaw_rates()
        gt_ts = gt.timestamps_us

        n_context = len(context_timestamps)
        initial_dynamics = np.zeros((n_context, 12), dtype=np.float64)
        for i in range(3):
            initial_dynamics[:, i] = np.interp(
                context_timestamps, gt_ts, gt_velocities[:, i]
            )
        initial_dynamics[:, 5] = np.interp(context_timestamps, gt_ts, gt_yaw_rates)

        self.ego_trajectory = geometry.DynamicTrajectory.from_trajectory_and_dynamics(
            ego_traj, initial_dynamics
        )
        self.ego_trajectory_estimate = self.ego_trajectory.clone()

        self.planner_delay_buffer = DelayBuffer(self.unbound.planner_delay_us)
        self.route_generator = RouteGenerator.create(
            self.unbound.gt_ego_trajectory.positions,
            vector_map=self.unbound.vector_map,
            route_generator_type=self.unbound.route_generator_type,
            route_start_offset_m=self.unbound.route_start_offset_m,
        )

        self._runtime_evaluator = RuntimeEvaluator(
            eval_config=self.eval_config,
            rollout_uuid=self.unbound.rollout_uuid,
            scene_id=self.unbound.scene_id,
            save_path_root=self.unbound.save_path_root,
            vector_map=self.unbound.vector_map,
        )

        self.broadcaster = MessageBroadcaster(
            handlers=[asl_log_writer, self._runtime_evaluator],
        )

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
            scene_id=self.unbound.scene_id,
            delta_start_us=delta_start_us,
            delta_end_us=delta_end_us,
            ego_trajectory_aabb=traj_aabb,
            traffic_poses={},
            ego_aabb=self.unbound.ego_aabb,
        )

        return corrected_aabb.transform(aabb_to_ds, is_relative=True)

    async def _build_force_gt_physics_blend_trajectory(self) -> geometry.Trajectory:
        """Build force-GT fallback that eases from recorded GT to physics."""
        unbound = self.unbound
        gt_hold_end_us = force_gt_physics_blend_hold_end_us(unbound)
        configured_blend_end_us = (
            unbound.render_start_timestamp_us + unbound.force_gt_duration_us
        )
        blend_end_us = min(configured_blend_end_us, unbound.end_timestamp_us)

        if blend_end_us <= gt_hold_end_us:
            return unbound.gt_ego_trajectory.clip(
                unbound.egomotion_context_start_us,
                min(gt_hold_end_us, unbound.end_timestamp_us) + 1,
            )

        dt_us = unbound.control_timestep_us
        clipped_gt_timestamps = unbound.gt_ego_trajectory.clip(
            unbound.egomotion_context_start_us,
            blend_end_us + 1,
        ).timestamps_us
        timestamps = list(
            range(unbound.first_policy_timestamp_us, blend_end_us + 1, dt_us)
        )
        if not timestamps or timestamps[-1] != blend_end_us:
            timestamps.append(blend_end_us)
        timestamps = sorted(
            set(
                [
                    unbound.egomotion_context_start_us,
                    gt_hold_end_us,
                    blend_end_us,
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
        return RolloutState(
            unbound=self.unbound,
            ego_trajectory=self.ego_trajectory,
            ego_trajectory_estimate=self.ego_trajectory_estimate,
            traffic_objs=self.traffic_objs,
            force_gt_ego_trajectory=self.force_gt_ego_trajectory,
            step_context=StepContext(),
            last_egopose_update_us=None,
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
                send_recording_ground_truth=unbound.send_recording_ground_truth,
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

            # camera_catalog is now populated by the renderer's session init.
            available_camera_protos = [
                self.camera_catalog.get_camera_definition(
                    self.unbound.scene_id, camera_cfg.logical_id
                ).as_proto()
                for camera_cfg in self.unbound.camera_configs
            ]

            await async_stack.enter_async_context(
                self.driver.rollout_session(
                    uuid=str(self.unbound.rollout_uuid),
                    broadcaster=self.broadcaster,
                    session_config=DriverSessionConfig(
                        sensorsim_cameras=available_camera_protos,
                        scene_id=self.unbound.scene_id,
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
                        scene_id=self.unbound.scene_id,
                        ego_aabb=self.unbound.ego_aabb,
                        gt_ego_aabb_trajectory=_build_traffic_session_trajectory(
                            self.unbound
                        ),
                        start_timestamp_us=self.unbound.egomotion_context_start_us,
                        force_gt_duration_us=self.unbound.force_gt_duration_us,
                        control_timestep_us=self.unbound.control_timestep_us,
                    ),
                )
            )

            if self.unbound.physics_update_mode != PhysicsUpdateMode.NONE:
                self.force_gt_ego_trajectory = (
                    await self._build_force_gt_physics_blend_trajectory()
                )
                self._apply_force_gt_blend_to_seeded_ego_trajectory()

            logger.info(
                "Session STARTING: uuid=%s scene=%s steps=%d",
                self.unbound.rollout_uuid,
                self.unbound.scene_id,
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

            if state.step_context is not None:
                await state.step_context.drain_outstanding_tasks()

            # Record timing
            loop_duration = time.perf_counter() - loop_start_time
            logger.info("Event-based simulation loop timer stopped")

            rollout_duration = time.perf_counter() - rollout_start_time
            ctx = try_get_context()
            if ctx is not None:
                ctx.rollout_duration.observe(rollout_duration)

            eval_result = await self._runtime_evaluator.run_evaluation(
                self.eval_executor
            )

            mark_rollout_complete(
                self.unbound.save_path_root, self.unbound.rollout_uuid
            )

            # Calculate realtime ratio
            simulated_duration_us = _simulated_duration_us(self.unbound)
            simulated_duration_s = simulated_duration_us / 1e6
            realtime_ratio = simulated_duration_s / loop_duration

            logger.info(
                "Session COMPLETED: uuid=%s scene=%s "
                "simulated %.2f sim seconds in %.2f wall clock seconds for %.2fx real time "
                "(total rollout %.2fs incl. setup/warmup)",
                self.unbound.rollout_uuid,
                self.unbound.scene_id,
                simulated_duration_s,
                loop_duration,
                realtime_ratio,
                rollout_duration,
            )

        return eval_result


def create_event_rollout(
    *,
    unbound: UnboundRollout,
    data_source: SceneDataSource,
    driver: DriverService,
    renderer_service: RendererService,
    physics: PhysicsService,
    trafficsim: TrafficService,
    controller: ControllerService,
    camera_catalog: CameraCatalog,
    eval_config: EvalConfig,
    eval_executor: ProcessPoolExecutor,
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
        camera_catalog=camera_catalog,
        eval_config=eval_config,
        eval_executor=eval_executor,
    )
