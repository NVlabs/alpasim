# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for event base classes: Event, RecurringEvent, EventQueue, SimulationEndEvent."""

from __future__ import annotations

import pytest
from alpasim_runtime.events.base import (
    EndSimulationException,
    Event,
    EventQueue,
    RecurringEvent,
    SimulationEndEvent,
)
from alpasim_runtime.events.state import RolloutState

# ---------------------------------------------------------------------------
# Concrete test implementations
# ---------------------------------------------------------------------------


class DummyEvent(Event):
    """Concrete event for testing."""

    priority: int = 50

    def __init__(self, timestamp_us: int, priority: int = 50):
        super().__init__(timestamp_us)
        self.priority = priority
        self.handled = False

    async def handle(self, rollout_state: RolloutState, queue: EventQueue) -> None:
        self.handled = True


class DummyRecurringEvent(RecurringEvent):
    """Concrete recurring event for testing."""

    priority: int = 50

    def __init__(self, timestamp_us: int, interval_us: int, priority: int = 50):
        super().__init__(timestamp_us)
        self.interval_us = interval_us
        self.priority = priority
        self.run_count = 0

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        self.run_count += 1


# ---------------------------------------------------------------------------
# EventQueue tests
# ---------------------------------------------------------------------------


class TestEventQueue:
    def test_empty_queue_is_falsy(self):
        q = EventQueue()
        assert not q
        assert len(q) == 0

    def test_submit_and_pop_single(self):
        q = EventQueue()
        event = DummyEvent(timestamp_us=1000)
        q.submit(event)
        assert len(q) == 1
        assert q
        popped = q.pop()
        assert popped is event
        assert not q

    def test_ordering_by_timestamp(self):
        q = EventQueue()
        e1 = DummyEvent(timestamp_us=300)
        e2 = DummyEvent(timestamp_us=100)
        e3 = DummyEvent(timestamp_us=200)
        q.submit(e1)
        q.submit(e2)
        q.submit(e3)

        assert q.pop().timestamp_us == 100
        assert q.pop().timestamp_us == 200
        assert q.pop().timestamp_us == 300

    def test_ordering_by_priority_at_same_timestamp(self):
        q = EventQueue()
        e_low = DummyEvent(timestamp_us=100, priority=90)
        e_high = DummyEvent(timestamp_us=100, priority=10)
        e_mid = DummyEvent(timestamp_us=100, priority=50)

        q.submit(e_low)
        q.submit(e_high)
        q.submit(e_mid)

        assert q.pop().priority == 10  # highest priority (lowest number) first
        assert q.pop().priority == 50
        assert q.pop().priority == 90

    def test_timestamp_takes_precedence_over_priority(self):
        q = EventQueue()
        e_early_low_prio = DummyEvent(timestamp_us=100, priority=90)
        e_late_high_prio = DummyEvent(timestamp_us=200, priority=10)

        q.submit(e_late_high_prio)
        q.submit(e_early_low_prio)

        # Earlier timestamp wins regardless of priority
        assert q.pop().timestamp_us == 100
        assert q.pop().timestamp_us == 200

    def test_peek_does_not_remove(self):
        q = EventQueue()
        event = DummyEvent(timestamp_us=100)
        q.submit(event)
        peeked = q.peek()
        assert peeked is event
        assert len(q) == 1

    def test_init_from_sequence(self):
        events = [
            DummyEvent(timestamp_us=300),
            DummyEvent(timestamp_us=100),
            DummyEvent(timestamp_us=200),
        ]
        q = EventQueue.init_from_sequence(events)
        assert len(q) == 3
        assert q.pop().timestamp_us == 100

    def test_pending_events_summary(self):
        q = EventQueue()
        q.submit(DummyEvent(timestamp_us=200))
        q.submit(DummyEvent(timestamp_us=100))

        summary = q.pending_events_summary()
        assert len(summary) == 2
        # Summary should be in sorted order
        assert "100" in summary[0]
        assert "200" in summary[1]


# ---------------------------------------------------------------------------
# Event comparison tests
# ---------------------------------------------------------------------------


class TestEventComparison:
    def test_lt_by_timestamp(self):
        e1 = DummyEvent(timestamp_us=100)
        e2 = DummyEvent(timestamp_us=200)
        assert e1 < e2
        assert not e2 < e1

    def test_lt_by_priority_at_same_timestamp(self):
        e1 = DummyEvent(timestamp_us=100, priority=10)
        e2 = DummyEvent(timestamp_us=100, priority=50)
        assert e1 < e2
        assert not e2 < e1

    def test_lt_equal_events(self):
        e1 = DummyEvent(timestamp_us=100, priority=50)
        e2 = DummyEvent(timestamp_us=100, priority=50)
        assert not e1 < e2
        assert not e2 < e1

    def test_description(self):
        event = DummyEvent(timestamp_us=1_000_000)
        desc = event.description()
        assert "DummyEvent" in desc
        assert "1_000_000" in desc


# ---------------------------------------------------------------------------
# RecurringEvent tests
# ---------------------------------------------------------------------------


class TestRecurringEvent:
    @pytest.mark.asyncio
    async def test_handle_runs_and_reschedules(self, rollout_state: RolloutState):
        q = EventQueue()
        event = DummyRecurringEvent(timestamp_us=100, interval_us=50)

        await event.handle(rollout_state, q)

        assert event.run_count == 1
        assert event.timestamp_us == 150  # Advanced by interval
        assert len(q) == 1
        assert q.pop() is event  # Same object resubmitted

    @pytest.mark.asyncio
    async def test_multiple_recurrences(self, rollout_state: RolloutState):
        q = EventQueue()
        event = DummyRecurringEvent(timestamp_us=0, interval_us=100)

        # Simulate 3 recurrences
        for expected_ts in [0, 100, 200]:
            assert event.timestamp_us == expected_ts
            await event.handle(rollout_state, q)
            # Pop the resubmitted event (same object)
            popped = q.pop()
            assert popped is event

        assert event.run_count == 3
        assert event.timestamp_us == 300


# ---------------------------------------------------------------------------
# SimulationEndEvent tests
# ---------------------------------------------------------------------------


class TestSimulationEndEvent:
    def test_priority_after_policy_before_controller(self):
        event = SimulationEndEvent(timestamp_us=1000)
        # Priority 30: fires after PolicyEvent (20) so the final observations
        # and drive() call happen naturally, but before ControllerEvent (40).
        assert event.priority == 30

    @pytest.mark.asyncio
    async def test_raises_end_simulation_exception(
        self,
        rollout_state: RolloutState,
    ):
        q = EventQueue()
        event = SimulationEndEvent(timestamp_us=1000)

        with pytest.raises(EndSimulationException):
            await event.handle(rollout_state, q)

    def test_end_event_ordering_at_final_timestamp(self):
        q = EventQueue()
        camera = DummyEvent(timestamp_us=1000, priority=10)
        policy = DummyEvent(timestamp_us=1000, priority=20)
        end = SimulationEndEvent(timestamp_us=1000)
        controller = DummyEvent(timestamp_us=1000, priority=40)

        q.submit(controller)
        q.submit(end)
        q.submit(policy)
        q.submit(camera)

        # Camera fires first, then policy submits final observations + drive(),
        # then SimulationEndEvent terminates before controller runs.
        assert q.pop().priority == 10  # camera
        assert q.pop().priority == 20  # policy
        assert isinstance(q.pop(), SimulationEndEvent)  # end (priority 30)
        assert q.pop().priority == 40  # controller (never reached in practice)
