# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Rollout state and service bundle for event-based simulation loop.

RolloutState holds the mutable simulation-world state shared across events.
ServiceBundle groups the service handles and setup objects that events need,
replacing the many individual constructor parameters with a single reference.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Coroutine, Optional

from alpasim_grpc.v0.traffic_pb2 import TrafficReturn
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.decision import DecisionBundle, DriverOrchestrator
from alpasim_runtime.delay_buffer import DelayBuffer
from alpasim_runtime.observation_cache import ObservationCache
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_utils import geometry
from alpasim_utils.scenario import TrafficObjects
from alpasim_utils.types import ImageWithMetadata

logger = logging.getLogger(__name__)


@dataclass
class ServiceBundle:
    """Immutable bundle of service handles shared across events.

    Groups all service references and setup objects that events need,
    replacing the many individual constructor parameters with a single reference.
    """

    driver: DriverService
    controller: ControllerService
    physics: PhysicsService
    trafficsim: TrafficService
    broadcaster: MessageBroadcaster
    planner_delay_buffer: DelayBuffer
    driver_orchestrator: DriverOrchestrator | None = None


@dataclass(slots=True)
class StepContext:
    """Per-driver-cycle in-flight data shared across pipeline events.

    Created by StepEvent at the end of each cycle (or at simulation start
    for the initial context).  Timing fields are filled by PolicyEvent before
    any pipeline event reads them.  Cleared implicitly when StepEvent
    replaces the context with a fresh one.
    """

    # Timing and mode — filled by PolicyEvent, read by pipeline events.
    # Defaults are placeholders; PolicyEvent overwrites before any consumer reads.
    step_start_us: int = 0
    target_time_us: int = 0
    force_gt: bool = False

    # PolicyEvent → ControllerEvent
    # Candidate set and selected output for the current policy step.
    decision_bundle: DecisionBundle | None = None
    # Driver output transformed to the true local frame.
    driver_trajectory: Optional[geometry.Trajectory] = None

    # ControllerEvent → PhysicsEvent(EGO) + StepEvent
    # True ego state from controller (poses + dynamics).
    ego_true: Optional[geometry.DynamicTrajectory] = None
    # Estimated ego state from controller (poses + dynamics).
    ego_estimated: Optional[geometry.DynamicTrajectory] = None

    # PhysicsEvent(EGO) → TrafficEvent + PhysicsEvent(TRAFFIC) + StepEvent
    # Physics-corrected ego poses (poses only — dynamics are unchanged).
    corrected_ego_trajectory: Optional[geometry.Trajectory] = None

    # TrafficEvent → PhysicsEvent(TRAFFIC) (transient, overwritten each round)
    # Raw response from the traffic simulation service for the current round.
    traffic_response: Optional[TrafficReturn] = None

    # PhysicsEvent(TRAFFIC) → StepEvent (accumulated across rounds)
    # Per-object trajectory of physics-corrected poses, grown each traffic round.
    traffic_trajectories: dict[str, geometry.Trajectory] = field(default_factory=dict)

    # Async observation tasks tracked between GroupedRenderEvent and PolicyEvent.
    outstanding_tasks: list[asyncio.Task[None]] = field(default_factory=list)

    def track_task(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Dispatch a coroutine as a fire-and-forget task."""
        self.outstanding_tasks.append(asyncio.create_task(coroutine))

    async def drain_outstanding_tasks(self) -> None:
        """Await all outstanding tasks and clear the list."""
        if not self.outstanding_tasks:
            return
        logger.info("Draining %d outstanding tasks", len(self.outstanding_tasks))
        tasks = self.outstanding_tasks
        self.outstanding_tasks = []
        await asyncio.gather(*tasks)


@dataclass
class RolloutState:
    """Mutable simulation-world state shared across all events.

    Only contains data that genuinely changes during the simulation (trajectories,
    traffic objects, dynamic state) plus the immutable configuration reference.
    """

    # === Immutable configuration ===
    unbound: UnboundRollout

    # === Mutable trajectory state ===
    ego_trajectory: geometry.DynamicTrajectory
    ego_trajectory_estimate: geometry.DynamicTrajectory
    traffic_objs: TrafficObjects

    # === Assertion tracking (for assert_zero_decision_delay) ===
    last_egopose_update_us: int = 0
    current_front_steering_angle_rad: float = 0.0
    last_camera_frame_us: dict[str, int] = field(default_factory=dict)
    last_decision_step_id: int = 0
    last_committed_decision_bundle: DecisionBundle | None = None
    available_driver_backend_ids: list[str] = field(default_factory=list)
    active_driver_backend_ids: list[str] | None = None
    observation_cache: ObservationCache | None = None

    # === Inter-event data ===
    data_sensorsim_to_driver: Optional[bytes] = None
    last_rendered_images: dict[str, ImageWithMetadata] = field(default_factory=dict)
    rendered_images_handler: (
        Callable[[list[ImageWithMetadata]], Awaitable[None]] | None
    ) = None

    # === Step timing (for step_duration telemetry) ===
    step_wall_start: float = 0.0

    # === Step context (pipeline inter-event data) ===
    step_context: Optional[StepContext] = None
