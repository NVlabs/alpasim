# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Event base classes and queue for the event-based simulation loop.

Events are ordered by (timestamp_us, priority) where lower priority values
are processed first at the same timestamp. The main loop pops events
sequentially until SimulationEndEvent terminates it.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from alpasim_runtime.events.state import RolloutState


class EventPriority:
    """Execution order for events sharing the same timestamp.

    Lower values run first.  Gaps between values leave room for future
    events without renumbering.

    ===  ==================  ==========================================
    Pri  Event               Role
    ===  ==================  ==========================================
    10   GroupedRenderEvent   Render camera images, dispatch to driver
    20   PolicyEvent          Gather observations, query driver
    30   SimulationEndEvent   Terminate loop (final timestamp only)
    40   ControllerEvent      Run controller + vehicle model
    50   PhysicsEvent(EGO)    Ground-correct ego poses
    60   TrafficEvent         Run traffic simulation
    70   PhysicsEvent(TRAFFIC)Ground-correct traffic poses
    80   StepEvent            Commit state, log poses, open next step
    ===  ==================  ==========================================
    """

    CAMERA = 10
    POLICY = 20
    SIMULATION_END = 30
    CONTROLLER = 40
    PHYSICS_EGO = 50
    TRAFFIC = 60
    PHYSICS_TRAFFIC = 70
    STEP = 80


class EndSimulationException(Exception):
    """Raised by SimulationEndEvent to terminate the event loop."""


@dataclass
class EventQueue:
    """Min-heap of events, ordered by (timestamp_us, priority)."""

    queue: list[Event] = field(default_factory=list, init=False)

    @classmethod
    def init_from_sequence(cls, events: Sequence[Event]) -> EventQueue:
        queue = cls()
        for event in events:
            queue.submit(event)
        return queue

    def submit(self, item: Event) -> None:
        heapq.heappush(self.queue, item)

    def pop(self) -> Event:
        return heapq.heappop(self.queue)

    def peek(self) -> Event:
        return self.queue[0]

    def __bool__(self) -> bool:
        return bool(self.queue)

    def __len__(self) -> int:
        return len(self.queue)

    def pending_events_summary(self) -> list[str]:
        """Return descriptions of all pending events in order.

        Useful for debugging when an error occurs — shows what events
        were queued and in what order they would have been processed.
        """
        sorted_events = sorted(self.queue)
        return [event.description() for event in sorted_events]


class Event(ABC):
    """Base class for simulation events.

    Events are ordered by (timestamp_us, priority) where lower priority
    values are processed first at the same timestamp.
    """

    # Default priority — subclasses should override.
    priority: int = 50

    def __init__(self, timestamp_us: int):
        self.timestamp_us = timestamp_us

    def __lt__(self, other: Event) -> bool:
        """Order by (timestamp_us, priority)."""
        if not isinstance(other, Event):
            raise TypeError(f"Cannot compare Event with {type(other)}")
        if self.timestamp_us != other.timestamp_us:
            return self.timestamp_us < other.timestamp_us
        return self.priority < other.priority

    def description(self) -> str:
        return f"{self.__class__.__name__} @ {self.timestamp_us:_}us"

    @abstractmethod
    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        """Handle the event, reading and writing to the rollout state.

        Args:
            rollout_state: The rollout state to read and write to.
            queue: The event queue to submit new events to.
        """
        ...


class RecurringEvent(Event):
    """Event that automatically reschedules itself after handling.

    Subclasses implement ``run()`` instead of ``handle()``.  After ``run()``
    completes, the event advances its own timestamp and resubmits itself to
    the queue.

    Override ``_next_timestamp()`` for non-uniform intervals (e.g.
    PolicyEvent where the step size is computed from other fields).
    """

    interval_us: int  # Set by subclass __init__

    def _next_timestamp(self) -> int:
        """Return the timestamp for the next occurrence.

        Default: advance by ``interval_us``.  Override for non-uniform steps.
        """
        return self.timestamp_us + self.interval_us

    @abstractmethod
    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        """Execute the event logic (without rescheduling)."""
        ...

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        """Execute ``run()`` then advance timestamp and resubmit to the queue.

        The same object is reused across recurrences — once popped from the
        heap it is not referenced elsewhere, so mutating in place is safe.
        """
        await self.run(rollout_state, queue)
        self.timestamp_us = self._next_timestamp()
        queue.submit(self)


class SimulationEndEvent(Event):
    """Terminates the simulation by raising EndSimulationException.

    Fires after PolicyEvent but before ControllerEvent at the final
    timestamp.
    """

    priority: int = EventPriority.SIMULATION_END

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        """Raise EndSimulationException to terminate the event loop."""
        raise EndSimulationException()
