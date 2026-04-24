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
from alpasim_runtime.events.camera import GroupedRenderEvent
from alpasim_runtime.events.controller import ControllerEvent
from alpasim_runtime.events.physics import PhysicsEvent, PhysicsTarget
from alpasim_runtime.events.policy import PolicyEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_runtime.events.step import StepEvent
from alpasim_runtime.events.traffic import TrafficEvent
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.session_configs import (
    DriverSessionConfig,
    TrafficSessionConfig,
)
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.telemetry.telemetry_context import tag_telemetry, try_get_context
from alpasim_runtime.types import Clock, RuntimeCamera
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_utils import geometry
from alpasim_utils.logs import LogWriter
from alpasim_utils.scenario import TrafficObjects

from eval.runtime_evaluator import RuntimeEvaluator
from eval.scenario_evaluator import ScenarioEvalResult
from eval.schema import EvalConfig

logger = logging.getLogger(__name__)


def _build_traffic_session_trajectory(unbound: UnboundRollout) -> geometry.Trajectory:
    """Build the ego AABB trajectory used for traffic session initialization."""
    return unbound.gt_ego_trajectory.clip(
        unbound.control_timestamps_us[0],
        unbound.control_timestamps_us[-1] + 1,
    ).transform(
        unbound.transform_ego_coords_ds_to_aabb,
        is_relative=True,
    )


def _simulated_duration_us(unbound: UnboundRollout) -> int:
    """Return the effective simulated span covered by control timestamps."""
    return unbound.control_timestamps_us[-1] - unbound.control_timestamps_us[0]


@dataclass
class EventBasedRollout:
    """Event-based simulation loop implementation.

    Processes events sequentially in timestamp order, with priority
    determining order at the same timestamp.
    """

    unbound: UnboundRollout
    driver: DriverService
    sensorsim: SensorsimService
    physics: PhysicsService
    trafficsim: TrafficService
    controller: ControllerService
    camera_catalog: CameraCatalog
    eval_config: EvalConfig
    eval_executor: ProcessPoolExecutor

    # Mutable state (initialized in __post_init__)
    ego_trajectory: geometry.DynamicTrajectory = field(init=False)
    ego_trajectory_estimate: geometry.DynamicTrajectory = field(init=False)
    traffic_objs: TrafficObjects = field(init=False)

    broadcaster: MessageBroadcaster = field(init=False)
    planner_delay_buffer: DelayBuffer = field(init=False)
    route_generator: RouteGenerator | None = field(init=False)
    runtime_cameras: list[RuntimeCamera] = field(init=False, default_factory=list)

    _runtime_evaluator: RuntimeEvaluator = field(init=False)

    def __post_init__(self) -> None:
        """Initialize mutable state."""
        prerun_start_us = self.unbound.control_timestamps_us[0]
        prerun_end_us = self.unbound.control_timestamps_us[1]

        asl_log_writer = LogWriter(file_path=self._asl_log_path())

        # Match legacy loop behavior: start with prerun [t0, t1] and extend as
        # each policy step produces future traffic updates.
        self.traffic_objs = self.unbound.traffic_objs.clip_trajectories(
            prerun_start_us, prerun_end_us + 1
        )

        # Initialize ego trajectories with prerun history [t0, t1], matching the
        # original loop contract used by sensorsim/driver warm-up.
        gt = self.unbound.gt_ego_trajectory
        prerun_timestamps = np.array([prerun_start_us, prerun_end_us], dtype=np.uint64)
        ego_traj = gt.interpolate(prerun_timestamps)

        # Build initial dynamics from GT derivatives at each prerun timestamp.
        gt_velocities = gt.velocities()
        gt_yaw_rates = gt.yaw_rates()
        gt_ts = gt.timestamps_us

        n_prerun = len(prerun_timestamps)
        initial_dynamics = np.zeros((n_prerun, 12), dtype=np.float64)
        for i in range(3):
            initial_dynamics[:, i] = np.interp(
                prerun_timestamps, gt_ts, gt_velocities[:, i]
            )
        initial_dynamics[:, 5] = np.interp(prerun_timestamps, gt_ts, gt_yaw_rates)

        self.ego_trajectory = geometry.DynamicTrajectory.from_trajectory_and_dynamics(
            ego_traj, initial_dynamics
        )
        self.ego_trajectory_estimate = self.ego_trajectory.clone()

        self.planner_delay_buffer = DelayBuffer(self.unbound.planner_delay_us)
        self.route_generator = RouteGenerator.create(
            self.unbound.gt_ego_trajectory.positions,
            vector_map=self.unbound.vector_map,
            route_generator_type=self.unbound.route_generator_type,
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

    async def _warmup_sensorsim(self) -> None:
        """Send a single render request to warm up sensorsim."""
        camera = self.runtime_cameras[0]
        gt_traj = self.unbound.gt_ego_trajectory
        traj_range = gt_traj.time_range_us

        trigger_start_us = traj_range.start
        trigger_end_us = min(
            traj_range.start + camera.clock.duration_us, traj_range.stop - 1
        )
        trigger = Clock.Trigger(
            time_range_us=range(trigger_start_us, trigger_end_us + 1),
            sequential_idx=-1,
        )

        traffic_trajs: dict[str, geometry.Trajectory] = {}

        logger.info("Warming up sensorsim with initial render request...")
        warmup_start = time.perf_counter()

        with tag_telemetry("warmup"):
            await self.sensorsim.render(
                ego_trajectory=gt_traj,
                traffic_trajectories=traffic_trajs,
                camera=camera,
                trigger=trigger,
                scene_id=self.unbound.scene_id,
                image_format=self.unbound.image_format,
                ego_mask_rig_config_id=self.unbound.ego_mask_rig_config_id,
            )

        warmup_duration = time.perf_counter() - warmup_start
        logger.info(f"Sensorsim warmup complete in {warmup_duration:.3f}s")

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

    def _create_rollout_state(self) -> RolloutState:
        """Create the RolloutState from the current rollout."""
        return RolloutState(
            unbound=self.unbound,
            ego_trajectory=self.ego_trajectory,
            ego_trajectory_estimate=self.ego_trajectory_estimate,
            traffic_objs=self.traffic_objs,
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
        """Create and schedule the initial set of events."""
        unbound = self.unbound
        queue = EventQueue()
        services = self._create_service_bundle()

        scene_start_us = unbound.control_timestamps_us[0]
        simulation_end_us = unbound.control_timestamps_us[-1]
        first_control_us = unbound.control_timestamps_us[1]

        camera_ids = [cam.logical_id for cam in self.runtime_cameras]

        # === Camera events ===
        queue.submit(
            GroupedRenderEvent(
                timestamp_us=scene_start_us + unbound.control_timestep_us,
                control_timestep_us=unbound.control_timestep_us,
                cameras=list(self.runtime_cameras),
                sensorsim=self.sensorsim,
                driver=self.driver,
                scene_start_us=scene_start_us,
                use_aggregated_render=unbound.group_render_requests,
            )
        )

        # === Pipeline events — all start at first_control_us ===
        dt = unbound.control_timestep_us

        queue.submit(
            PolicyEvent(
                timestamp_us=first_control_us,
                policy_timestep_us=dt,
                services=services,
                camera_ids=camera_ids,
                route_generator=self.route_generator,
                send_recording_ground_truth=unbound.send_recording_ground_truth,
            )
        )
        queue.submit(
            ControllerEvent(
                timestamp_us=first_control_us,
                control_timestep_us=dt,
                services=services,
            )
        )
        queue.submit(
            PhysicsEvent(
                timestamp_us=first_control_us,
                control_timestep_us=dt,
                services=services,
                target=PhysicsTarget.EGO,
            )
        )
        queue.submit(
            TrafficEvent(
                timestamp_us=first_control_us,
                control_timestep_us=dt,
                services=services,
            )
        )
        queue.submit(
            PhysicsEvent(
                timestamp_us=first_control_us,
                control_timestep_us=dt,
                services=services,
                target=PhysicsTarget.TRAFFIC,
            )
        )
        queue.submit(
            StepEvent(
                timestamp_us=scene_start_us,
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

            # Initialize service sessions
            for service in [self.sensorsim, self.physics, self.controller]:
                await async_stack.enter_async_context(
                    service.rollout_session(
                        uuid=str(self.unbound.rollout_uuid),
                        broadcaster=self.broadcaster,
                    )
                )

            # Get available cameras and merge
            sensorsim_cameras = await self.sensorsim.get_available_cameras(
                self.unbound.scene_id
            )
            await self.camera_catalog.merge_local_and_sensorsim_cameras(
                self.unbound.scene_id, sensorsim_cameras
            )

            # Build runtime cameras
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

            # Send cameras to driver
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
            gt_ego_aabb_trajectory = _build_traffic_session_trajectory(self.unbound)

            await async_stack.enter_async_context(
                self.trafficsim.rollout_session(
                    uuid=str(self.unbound.rollout_uuid),
                    broadcaster=self.broadcaster,
                    session_config=TrafficSessionConfig(
                        traffic_objs=self.unbound.traffic_objs,
                        scene_id=self.unbound.scene_id,
                        ego_aabb=self.unbound.ego_aabb,
                        gt_ego_aabb_trajectory=gt_ego_aabb_trajectory,
                        start_timestamp_us=self.unbound.start_timestamp_us,
                    ),
                )
            )

            # Apply physics to initial trajectory (corrects poses, keeps dynamics)
            if self.unbound.physics_update_mode != PhysicsUpdateMode.NONE:
                corrected_traj = await self._apply_physics_to_trajectory(
                    self.ego_trajectory.trajectory()
                )
                self.ego_trajectory = (
                    geometry.DynamicTrajectory.from_trajectory_and_dynamics(
                        corrected_traj, self.ego_trajectory.dynamics
                    )
                )

            logger.info(
                "Session STARTING: uuid=%s scene=%s steps=%d",
                self.unbound.rollout_uuid,
                self.unbound.scene_id,
                self.unbound.n_sim_steps,
            )

            # Warmup sensorsim
            if self.runtime_cameras and not self.sensorsim.skip:
                await self._warmup_sensorsim()

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
