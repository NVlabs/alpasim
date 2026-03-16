# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for StepContext async task tracking."""

from __future__ import annotations

import asyncio

import pytest
from alpasim_runtime.events.state import StepContext


class TestStepContextOutstandingTasks:
    def test_default_construction_has_empty_tasks(self) -> None:
        ctx = StepContext()
        assert ctx.outstanding_tasks == []

    @pytest.mark.asyncio
    async def test_track_task_adds_to_list(self) -> None:
        ctx = StepContext()

        async def work() -> None:
            await asyncio.sleep(0)

        ctx.track_task(work())

        assert len(ctx.outstanding_tasks) == 1

    @pytest.mark.asyncio
    async def test_drain_waits_and_clears(self) -> None:
        ctx = StepContext()
        results: list[int] = []

        async def work(n: int) -> None:
            results.append(n)

        ctx.track_task(work(1))
        ctx.track_task(work(2))

        await ctx.drain_outstanding_tasks()

        assert sorted(results) == [1, 2]
        assert ctx.outstanding_tasks == []

    @pytest.mark.asyncio
    async def test_drain_noop_when_empty(self) -> None:
        ctx = StepContext()
        await ctx.drain_outstanding_tasks()
        assert ctx.outstanding_tasks == []

    @pytest.mark.asyncio
    async def test_drain_propagates_exception(self) -> None:
        ctx = StepContext()

        async def failing() -> None:
            raise RuntimeError("boom")

        ctx.track_task(failing())

        with pytest.raises(RuntimeError, match="boom"):
            await ctx.drain_outstanding_tasks()
