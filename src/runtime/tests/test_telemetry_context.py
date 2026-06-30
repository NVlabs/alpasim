# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import asyncio

import alpasim_runtime.telemetry.rpc_wrapper as rpc_wrapper
import alpasim_runtime.telemetry.telemetry_context as telemetry_context
import pytest
from alpasim_runtime.telemetry.telemetry_context import TelemetryContext
from prometheus_client import generate_latest


def test_record_rollout_complete_updates_simulation_summary_metrics() -> None:
    ctx = TelemetryContext(worker_id=0)
    ctx._simulation_started_at = 10.0
    now = 25.0

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(telemetry_context, "perf_counter", lambda: now)
        ctx.record_rollout_complete()

    metrics = generate_latest(ctx.registry).decode("utf-8")
    assert ('alpasim_simulation_rollouts_completed_total{worker_id="0"} 1.0') in metrics
    assert ('alpasim_simulation_elapsed_seconds{worker_id="0"} 15.0') in metrics


def test_refresh_gauges_snapshots_event_loop_and_gc_stats(monkeypatch) -> None:
    ctx = TelemetryContext(worker_id=0)
    monkeypatch.setattr(
        telemetry_context,
        "get_event_loop_idle_stats",
        lambda: {
            "idle_seconds": 1.0,
            "poll_seconds": 2.0,
            "work_seconds": 3.0,
            "select_calls": 4,
        },
    )
    monkeypatch.setattr(
        telemetry_context,
        "get_gc_pressure_stats",
        lambda: {
            "total_duration_s": 4.0,
            "max_duration_s": 5.0,
            "collection_count": 6,
            "collected_total": 7,
            "gen0_count": 8,
            "gen1_count": 9,
            "gen2_count": 10,
        },
    )

    ctx.refresh_gauges()

    metrics = generate_latest(ctx.registry).decode("utf-8")
    assert ('alpasim_event_loop_idle_seconds_total{worker_id="0"} 1.0') in metrics
    assert ('alpasim_event_loop_poll_seconds_total{worker_id="0"} 2.0') in metrics
    assert ('alpasim_event_loop_work_seconds_total{worker_id="0"} 3.0') in metrics
    assert ('alpasim_gc_total_duration_seconds{worker_id="0"} 4.0') in metrics
    assert ('alpasim_gc_max_duration_seconds{worker_id="0"} 5.0') in metrics
    assert ('alpasim_gc_collection_count_total{worker_id="0"} 6.0') in metrics


@pytest.mark.asyncio
async def test_profiled_rpc_call_records_latest_queue_depth_gauge(
    monkeypatch,
) -> None:
    ctx = TelemetryContext(worker_id=3)
    monkeypatch.setattr(rpc_wrapper, "try_get_context", lambda: ctx)

    first_done = asyncio.Event()
    second_done = asyncio.Event()
    first_task = asyncio.create_task(
        rpc_wrapper.profiled_rpc_call(
            "render_rgb",
            "sensorsim",
            lambda: asyncio.create_task(first_done.wait()),
        )
    )
    await asyncio.sleep(0)

    second_task = asyncio.create_task(
        rpc_wrapper.profiled_rpc_call(
            "render_rgb",
            "sensorsim",
            lambda: asyncio.create_task(second_done.wait()),
        )
    )
    await asyncio.sleep(0)
    second_done.set()
    await second_task

    metrics = generate_latest(ctx.registry).decode("utf-8")
    latest_metric = (
        'alpasim_rpc_queue_depth_at_start_latest{service="sensorsim",'
        'tag="default",worker_id="3"}'
    )
    assert f"{latest_metric} 1.0" in metrics

    first_done.set()
    await first_task

    metrics = generate_latest(ctx.registry).decode("utf-8")
    assert f"{latest_metric} 0.0" in metrics
