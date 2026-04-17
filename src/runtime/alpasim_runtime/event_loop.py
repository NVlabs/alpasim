# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Event-based simulation loop.

Components are modelled as self-scheduling events processed from a priority
queue, allowing each one to run at its own cadence.
"""

from __future__ import annotations

import copy
import contextlib
import logging
import os
import time
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from alpasim_grpc.v0.logging_pb2 import LogEntry, RolloutMetadata
from alpasim_runtime.decision import (
    CandidateDecision,
    DecisionBundle,
    DecisionSnapshot,
    DriverBackendRegistry,
    DriverServiceBackendAdapter,
    MultiBackendDriverOrchestrator,
)
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
from alpasim_runtime.observation_cache import ObservationCache
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
from alpasim_utils.types import ImageWithMetadata

from eval.runtime_evaluator import RuntimeEvaluator
from eval.scenario_evaluator import ScenarioEvalResult
from eval.schema import EvalConfig

logger = logging.getLogger(__name__)


def _clone_trajectory(
    trajectory: geometry.Trajectory | None,
) -> geometry.Trajectory | None:
    if trajectory is None:
        return None
    time_range = trajectory.time_range_us
    return trajectory.clip(time_range.start, time_range.stop)


def _clone_dynamic_trajectory(
    trajectory: geometry.DynamicTrajectory | None,
) -> geometry.DynamicTrajectory | None:
    if trajectory is None:
        return None
    return trajectory.clone()


def _clone_image_with_metadata(image: ImageWithMetadata) -> ImageWithMetadata:
    return ImageWithMetadata(
        start_timestamp_us=image.start_timestamp_us,
        end_timestamp_us=image.end_timestamp_us,
        image_bytes=image.image_bytes,
        camera_logical_id=image.camera_logical_id,
    )


def _clone_decision_snapshot(snapshot: DecisionSnapshot) -> DecisionSnapshot:
    return DecisionSnapshot(
        step_id=snapshot.step_id,
        input_snapshot_id=snapshot.input_snapshot_id,
        time_now_us=snapshot.time_now_us,
        time_query_us=snapshot.time_query_us,
        ego_pose_history_timestamps_us=list(snapshot.ego_pose_history_timestamps_us),
        traffic_actor_ids=list(snapshot.traffic_actor_ids),
        route_waypoints_in_rig=[list(waypoint) for waypoint in snapshot.route_waypoints_in_rig],
        planner_context=copy.deepcopy(snapshot.planner_context),
        renderer_data=snapshot.renderer_data,
        camera_frame_timestamps_us=dict(snapshot.camera_frame_timestamps_us),
    )


def _clone_candidate_decision(candidate: CandidateDecision) -> CandidateDecision:
    return CandidateDecision(
        candidate_id=candidate.candidate_id,
        step_id=candidate.step_id,
        input_snapshot_id=candidate.input_snapshot_id,
        backend_id=candidate.backend_id,
        status=candidate.status,
        trajectory=_clone_trajectory(candidate.trajectory),
        diagnostics=copy.deepcopy(candidate.diagnostics),
        generated_at_us=candidate.generated_at_us,
        recompute_count=candidate.recompute_count,
        error=candidate.error,
    )


def _clone_decision_bundle(bundle: DecisionBundle | None) -> DecisionBundle | None:
    if bundle is None:
        return None
    return DecisionBundle(
        snapshot=_clone_decision_snapshot(bundle.snapshot),
        candidates=[_clone_candidate_decision(candidate) for candidate in bundle.candidates],
        selected_candidate_id=bundle.selected_candidate_id,
        arbitration_reason=bundle.arbitration_reason,
    )


def _clone_rollout_state(state: RolloutState) -> RolloutState:
    cloned_step_context = None
    if state.step_context is not None:
        if state.step_context.outstanding_tasks:
            raise RuntimeError("Cannot checkpoint with outstanding step tasks")
        cloned_step_context = replace(
            state.step_context,
            decision_bundle=_clone_decision_bundle(state.step_context.decision_bundle),
            driver_trajectory=_clone_trajectory(state.step_context.driver_trajectory),
            ego_true=_clone_dynamic_trajectory(state.step_context.ego_true),
            ego_estimated=_clone_dynamic_trajectory(state.step_context.ego_estimated),
            corrected_ego_trajectory=_clone_trajectory(
                state.step_context.corrected_ego_trajectory
            ),
            traffic_response=copy.deepcopy(state.step_context.traffic_response),
            traffic_trajectories={
                object_id: _clone_trajectory(trajectory)
                for object_id, trajectory in state.step_context.traffic_trajectories.items()
            },
            outstanding_tasks=[],
        )

    observation_cache = None
    if state.observation_cache is not None:
        observation_cache = ObservationCache(max_frames=state.observation_cache._max_frames)
        observation_cache.restore(state.observation_cache.checkpoint())

    return RolloutState(
        unbound=state.unbound,
        ego_trajectory=state.ego_trajectory.clone(),
        ego_trajectory_estimate=state.ego_trajectory_estimate.clone(),
        traffic_objs=state.traffic_objs.clip_trajectories(
            state.unbound.control_timestamps_us[0],
            state.unbound.control_timestamps_us[-1] + 1,
        ),
        last_egopose_update_us=state.last_egopose_update_us,
        current_front_steering_angle_rad=state.current_front_steering_angle_rad,
        last_camera_frame_us=dict(state.last_camera_frame_us),
        last_decision_step_id=state.last_decision_step_id,
        last_committed_decision_bundle=_clone_decision_bundle(
            state.last_committed_decision_bundle
        ),
        available_driver_backend_ids=list(state.available_driver_backend_ids),
        active_driver_backend_ids=(
            list(state.active_driver_backend_ids)
            if state.active_driver_backend_ids is not None
            else None
        ),
        observation_cache=observation_cache,
        data_sensorsim_to_driver=state.data_sensorsim_to_driver,
        last_rendered_images={
            camera_id: _clone_image_with_metadata(image)
            for camera_id, image in state.last_rendered_images.items()
        },
        rendered_images_handler=state.rendered_images_handler,
        step_wall_start=state.step_wall_start,
        step_context=cloned_step_context,
    )


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


@dataclass(frozen=True)
class EventLoopAdvanceResult:
    """Result of advancing the event loop.

    `step_committed` becomes true after a `StepEvent` has finished mutating
    state. `simulation_finished` becomes true once the loop reaches the end
    condition, either via `SimulationEndEvent` or an empty queue.
    """

    step_committed: bool = False
    simulation_finished: bool = False
    committed_step_timestamp_us: int | None = None


@dataclass(frozen=True)
class RuntimeCheckpoint:
    """Deep-copied runtime state captured at a committed step boundary."""

    state: RolloutState
    event_queue: EventQueue
    backend_checkpoint: dict[str, Any] = field(default_factory=dict)
    unsupported_backend_ids: list[str] = field(default_factory=list)


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

    # Mutable state (initialized in __post_init__)
    ego_trajectory: geometry.DynamicTrajectory = field(init=False)
    ego_trajectory_estimate: geometry.DynamicTrajectory = field(init=False)
    traffic_objs: TrafficObjects = field(init=False)

    broadcaster: MessageBroadcaster = field(init=False)
    planner_delay_buffer: DelayBuffer = field(init=False)
    route_generator: Optional[RouteGenerator] = field(init=False)
    runtime_cameras: list[RuntimeCamera] = field(init=False, default_factory=list)

    _runtime_evaluator: RuntimeEvaluator = field(init=False)
    _async_stack: contextlib.AsyncExitStack | None = field(init=False, default=None)
    _state: RolloutState | None = field(init=False, default=None)
    _event_queue: EventQueue | None = field(init=False, default=None)
    _rollout_start_time: float = field(init=False, default=0.0)
    _loop_start_time: float = field(init=False, default=0.0)
    _initialized: bool = field(init=False, default=False)
    _closed: bool = field(init=False, default=False)
    _default_driver_orchestrator: Any = field(init=False, default=None)

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
        available_backend_ids = (
            list(self._default_driver_orchestrator.backend_ids)
            if self._build_default_driver_orchestrator() is not None
            else ["default_driver"]
        )
        return RolloutState(
            unbound=self.unbound,
            ego_trajectory=self.ego_trajectory,
            ego_trajectory_estimate=self.ego_trajectory_estimate,
            traffic_objs=self.traffic_objs,
            available_driver_backend_ids=available_backend_ids,
            active_driver_backend_ids=list(available_backend_ids),
            observation_cache=ObservationCache(
                max_frames=self.unbound.observation_cache_size
            ),
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
            driver_orchestrator=self._build_default_driver_orchestrator(),
        )

    def _build_default_driver_orchestrator(self):
        """Build the default per-rollout driver orchestrator from runtime config."""
        cached_orchestrator = getattr(self, "_default_driver_orchestrator", None)
        if cached_orchestrator is not None:
            return cached_orchestrator

        backend_cfgs = getattr(self.unbound, "driver_backends", []) or []
        if not backend_cfgs:
            self._default_driver_orchestrator = None
            return None

        def _observation_window_summary(
            input_snapshot_id: str,
            window_size: int,
        ) -> dict[str, Any]:
            if self._state is None or self._state.observation_cache is None:
                return {
                    "anchor_snapshot_id": input_snapshot_id,
                    "window_size": window_size,
                    "available_frames": 0,
                    "frames": [],
                }
            from alpasim_runtime.observation_cache import ObservationCacheReader

            return ObservationCacheReader(
                self._state.observation_cache
            ).build_window_summary(input_snapshot_id, window_size)

        adapters = [
            DriverServiceBackendAdapter(
                self.driver,
                backend_id=backend.backend_id,
                backend_type=backend.backend_type,
                model_type_override=backend.model_type,
                supports_parallel=backend.supports_parallel,
                supports_hot_switch=backend.supports_hot_switch,
                priority=backend.priority,
                observation_window_summary_getter=_observation_window_summary,
                observation_window_summary_size=self.unbound.observation_window_summary_size,
            )
            for backend in backend_cfgs
        ]
        self._default_driver_orchestrator = MultiBackendDriverOrchestrator(
            DriverBackendRegistry(adapters)
        )
        return self._default_driver_orchestrator

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

    def _require_initialized(self) -> tuple[RolloutState, EventQueue]:
        """Return the active rollout state and queue after `initialize()`."""
        if not self._initialized or self._state is None or self._event_queue is None:
            raise RuntimeError("EventBasedRollout must be initialized before stepping")
        return (self._state, self._event_queue)

    @property
    def current_state(self) -> RolloutState:
        """Expose the active rollout state after initialization."""
        state, _ = self._require_initialized()
        return state

    def capture_runtime_checkpoint(self) -> RuntimeCheckpoint:
        """Capture a runtime-only checkpoint at a committed step boundary."""
        state, event_queue = self._require_initialized()
        cloned_state = _clone_rollout_state(state)
        cloned_events = [copy.copy(event) for event in event_queue.queue]
        cloned_queue = EventQueue.init_from_sequence(cloned_events)
        orchestrator = self._build_default_driver_orchestrator()
        backend_checkpoint: dict[str, Any] = {}
        unsupported_backend_ids: list[str] = []
        if orchestrator is not None and hasattr(orchestrator, "capture_backend_checkpoint"):
            backend_checkpoint, unsupported_backend_ids = (
                orchestrator.capture_backend_checkpoint()
            )
        return RuntimeCheckpoint(
            state=cloned_state,
            event_queue=cloned_queue,
            backend_checkpoint=backend_checkpoint,
            unsupported_backend_ids=unsupported_backend_ids,
        )

    async def restore_runtime_checkpoint(self, checkpoint: RuntimeCheckpoint) -> None:
        """Restore a previously captured runtime checkpoint."""
        self._require_initialized()
        restored_state = _clone_rollout_state(checkpoint.state)
        restored_state.rendered_images_handler = self.current_state.rendered_images_handler
        restored_queue = EventQueue.init_from_sequence(
            [copy.copy(event) for event in checkpoint.event_queue.queue]
        )
        orchestrator = self._build_default_driver_orchestrator()
        if (
            orchestrator is not None
            and checkpoint.backend_checkpoint
            and hasattr(orchestrator, "restore_backend_checkpoint")
        ):
            await orchestrator.restore_backend_checkpoint(checkpoint.backend_checkpoint)
        self._state = restored_state
        self._event_queue = restored_queue

    async def initialize(self) -> None:
        """Initialize services and create the event loop state.

        This splits setup from execution so callers can either run the rollout
        to completion or stop after a committed `StepEvent`.
        """
        if self._initialized:
            return
        if self._closed:
            raise RuntimeError("EventBasedRollout cannot be re-initialized after close")

        stack = contextlib.AsyncExitStack()
        await stack.__aenter__()
        self._async_stack = stack
        self._rollout_start_time = time.perf_counter()

        try:
            await stack.enter_async_context(self.broadcaster)

            await self._log_metadata(
                session_metadata=self.unbound.get_log_metadata(),
                version_ids=self.unbound.version_ids,
            )

            for service in [self.sensorsim, self.physics, self.controller]:
                await stack.enter_async_context(
                    service.rollout_session(
                        uuid=str(self.unbound.rollout_uuid),
                        broadcaster=self.broadcaster,
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

            available_camera_protos = [
                self.camera_catalog.get_camera_definition(
                    self.unbound.scene_id, camera_cfg.logical_id
                ).as_proto()
                for camera_cfg in self.unbound.camera_configs
            ]

            await stack.enter_async_context(
                self.driver.rollout_session(
                    uuid=str(self.unbound.rollout_uuid),
                    broadcaster=self.broadcaster,
                    session_config=DriverSessionConfig(
                        sensorsim_cameras=available_camera_protos,
                        scene_id=self.unbound.scene_id,
                    ),
                )
            )

            gt_ego_aabb_trajectory = _build_traffic_session_trajectory(self.unbound)

            await stack.enter_async_context(
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

            if self.runtime_cameras and not self.sensorsim.skip:
                await self._warmup_sensorsim()

            self._state = self._create_rollout_state()
            self._event_queue = self._create_initial_events()
            self._loop_start_time = time.perf_counter()
            self._initialized = True
            logger.info("Event-based simulation loop timer started")
        except BaseException:
            await stack.aclose()
            self._async_stack = None
            self._state = None
            self._event_queue = None
            self._rollout_start_time = 0.0
            self._loop_start_time = 0.0
            raise

    async def _advance_one_event(self) -> EventLoopAdvanceResult:
        """Execute the next queued event and summarize the outcome."""
        state, event_queue = self._require_initialized()

        if not event_queue:
            return EventLoopAdvanceResult(simulation_finished=True)

        event = event_queue.pop()
        event_timestamp_us = event.timestamp_us
        logger.info(f"sim_time {event_timestamp_us:_}us: {event.description()}")

        try:
            await event.handle(state, event_queue)
        except EndSimulationException:
            logger.info("Simulation ended via SimulationEndEvent")
            return EventLoopAdvanceResult(simulation_finished=True)
        except Exception:
            logger.exception(
                "Error during event handling. Pending events in queue (%d):",
                len(event_queue),
            )
            for event_desc in event_queue.pending_events_summary():
                logger.error("  - %s", event_desc)
            raise

        step_committed = isinstance(event, StepEvent)
        return EventLoopAdvanceResult(
            step_committed=step_committed,
            simulation_finished=not bool(event_queue),
            committed_step_timestamp_us=(
                event_timestamp_us if step_committed else None
            ),
        )

    async def run_until_step_commit(self) -> EventLoopAdvanceResult:
        """Advance the loop until one `StepEvent` has committed state."""
        while True:
            result = await self._advance_one_event()
            if result.step_committed or result.simulation_finished:
                return result

    async def run_until_complete(self) -> None:
        """Run the event loop until completion."""
        while True:
            result = await self._advance_one_event()
            if result.simulation_finished:
                return

    async def aclose(
        self,
        *,
        mark_complete: bool = False,
        run_evaluation: bool = False,
    ) -> Optional[ScenarioEvalResult]:
        """Drain outstanding work, optionally evaluate, and close sessions."""
        if self._closed:
            return None

        eval_result: Optional[ScenarioEvalResult] = None
        try:
            if self._state is not None and self._state.step_context is not None:
                await self._state.step_context.drain_outstanding_tasks()

            loop_duration = 0.0
            if self._loop_start_time > 0.0:
                loop_duration = time.perf_counter() - self._loop_start_time
                logger.info("Event-based simulation loop timer stopped")

            rollout_duration = 0.0
            if self._rollout_start_time > 0.0:
                rollout_duration = time.perf_counter() - self._rollout_start_time
                ctx = try_get_context()
                if ctx is not None:
                    ctx.rollout_duration.observe(rollout_duration)

            if run_evaluation:
                eval_result = self._runtime_evaluator.run_evaluation()

            if mark_complete:
                mark_rollout_complete(
                    self.unbound.save_path_root, self.unbound.rollout_uuid
                )

                simulated_duration_us = _simulated_duration_us(self.unbound)
                simulated_duration_s = simulated_duration_us / 1e6
                realtime_ratio = (
                    simulated_duration_s / loop_duration if loop_duration > 0 else 0.0
                )

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
        finally:
            if self._async_stack is not None:
                await self._async_stack.aclose()
            self._async_stack = None
            self._state = None
            self._event_queue = None
            self._initialized = False
            self._closed = True
            self._rollout_start_time = 0.0
            self._loop_start_time = 0.0

    async def run(self) -> Optional[ScenarioEvalResult]:
        """Run the event-based simulation loop.

        Returns:
            ScenarioEvalResult if in-runtime evaluation is enabled, None otherwise.
        """
        await self.initialize()
        try:
            await self.run_until_complete()
        except BaseException:
            await self.aclose()
            raise
        return await self.aclose(mark_complete=True, run_evaluation=True)
