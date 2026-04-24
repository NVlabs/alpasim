# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Background resource sampling with summary statistics.

Uses a class-based design for proper encapsulation and lifecycle management.
Samples CPU (high-utilization processes) and GPU utilization every interval,
then exports min/max/mean/p50/p95/p99 as Prometheus gauges at shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil
import pynvml
from prometheus_client import CollectorRegistry, Gauge

logger = logging.getLogger(__name__)

# Threshold: once a process exceeds this CPU utilization threshold,
# it becomes "tracked" for the rest of the run.
# This is to automatically track high-utilization processes without manually
# having to configure the jobs (which ppl might forget if they add new ones).
TRACK_THRESHOLD_PCT = 50.0

# Processes to ignore (e.g., nvidia-smi appears due to pynvml queries)
IGNORED_PROCESSES = {"nvidia-smi"}


# How often to sample resources in seconds
SAMPLE_INTERVAL_DEFAULT = 5.0


@dataclass
class GPUMetrics:
    """Structured GPU metric samples."""

    gpu_id: int
    utilization: list[float] = field(default_factory=list)
    memory_used_bytes: list[float] = field(default_factory=list)
    memory_total_bytes: float | None = None  # Constant per GPU, set once


class ResourceSampler:
    """
    Encapsulated resource sampler with proper lifecycle management.

    Usage:
        sampler = ResourceSampler()
        await sampler.start(interval=1.0)
        # ... run simulation ...
        await sampler.stop()
        sampler.export_to(registry)
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._nvml_initialized: bool = False

        # Process tracking: name -> list of CPU% samples
        # Once a process exceeds TRACK_THRESHOLD_PCT, it's tracked for the rest of the run
        self._process_samples: dict[str, list[float]] = defaultdict(list)

        # Persistent process objects for accurate CPU measurement
        # pid -> (psutil.Process, name)
        # We need to keep the same Process objects across samples because
        # cpu_percent(interval=None) requires the same object to compute deltas
        self._process_cache: dict[int, tuple[psutil.Process, str]] = {}

        # GPU tracking: gpu_id -> GPUMetrics
        self._gpu_metrics: dict[int, GPUMetrics] = {}

    def _get_process_name(self, proc: psutil.Process) -> str:
        """Extract a readable process name from a process object."""
        try:
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            cmdline = []

        if not cmdline:
            try:
                return proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return f"pid_{proc.pid}"

        exe = Path(cmdline[0]).name
        # For python scripts, use the script/module name
        if exe in ("python", "python3") and len(cmdline) > 1:
            # Handle "python -m module_name" invocation
            if cmdline[1] == "-m" and len(cmdline) > 2:
                # Keep full module path (e.g., "alpasim_runtime.worker.main")
                return cmdline[2]
            return Path(cmdline[1]).stem
        return exe

    def _sample_processes(self) -> None:
        """Sample CPU for tracked processes + discover new high-utilization ones."""
        # Get current PIDs to detect terminated processes
        current_pids = set()

        for proc in psutil.process_iter():
            try:
                pid = proc.pid
                current_pids.add(pid)

                # Check if we have a cached process object for this PID
                if pid in self._process_cache:
                    cached_proc, name = self._process_cache[pid]
                    try:
                        cpu = cached_proc.cpu_percent(interval=None)
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.ZombieProcess,
                    ):
                        # Process died, remove from cache
                        del self._process_cache[pid]
                        continue

                else:
                    # New process - create a fresh Process object and cache it
                    # Use the proc from iteration to get a stable object
                    name = self._get_process_name(proc)
                    # Prime it (first call returns 0)
                    try:
                        proc.cpu_percent(interval=None)
                    except (
                        psutil.NoSuchProcess,
                        psutil.AccessDenied,
                        psutil.ZombieProcess,
                    ):
                        continue
                    # Cache for future samples (first cpu_percent call is priming only)
                    self._process_cache[pid] = (proc, name)
                    continue  # Skip this sample, wait for next iteration

                if cpu is None or name in IGNORED_PROCESSES:
                    continue

                # Record sample if process exceeds threshold or is already tracked
                if cpu >= TRACK_THRESHOLD_PCT or name in self._process_samples:
                    self._process_samples[name].append(cpu)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # Clean up cache for terminated processes
        terminated_pids = set(self._process_cache.keys()) - current_pids
        for pid in terminated_pids:
            del self._process_cache[pid]

    def _sample_gpus(self) -> None:
        """Sample GPU utilization and memory."""
        if not self._nvml_initialized:
            return

        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                if i not in self._gpu_metrics:
                    self._gpu_metrics[i] = GPUMetrics(gpu_id=i)

                self._gpu_metrics[i].utilization.append(util.gpu)
                self._gpu_metrics[i].memory_used_bytes.append(mem.used)
                # Total memory is constant, only set once
                if self._gpu_metrics[i].memory_total_bytes is None:
                    self._gpu_metrics[i].memory_total_bytes = float(mem.total)
        except pynvml.NVMLError:
            pass

    async def _sample_loop(self, interval: float) -> None:
        """Background task that samples resources at regular intervals."""
        # Initial sample to populate the process cache and prime cpu_percent()
        # The first sample for each process won't record data (priming phase)
        self._sample_processes()
        await asyncio.sleep(interval)

        while True:
            self._sample_processes()
            self._sample_gpus()
            await asyncio.sleep(interval)

    async def start(self, interval: float = SAMPLE_INTERVAL_DEFAULT) -> None:
        """Start background resource sampling."""
        # Initialize NVML if available
        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            self._nvml_initialized = True
            logger.info(f"GPU monitoring initialized: {gpu_count} GPU(s) detected")
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to initialize NVML for GPU monitoring: {e}")

        self._task = asyncio.create_task(self._sample_loop(interval))

    async def stop(self) -> None:
        """Stop background sampling."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Shutdown NVML
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._nvml_initialized = False

    @staticmethod
    def _compute_summary_stats(samples: list[float]) -> dict[str, float] | None:
        """Compute summary statistics. Returns None if samples is empty."""
        if not samples:
            return None
        arr = np.array(samples)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def export_to(self, registry: CollectorRegistry) -> None:
        """Export summary statistics as Prometheus gauges to the given registry."""
        # CPU gauge: single gauge with name + stat labels
        if self._process_samples:
            cpu_gauge = Gauge(
                "process_cpu_utilization",
                "CPU utilization summary statistics",
                ["name", "stat"],
                registry=registry,
            )

            for proc_name, samples in self._process_samples.items():
                stats = self._compute_summary_stats(samples)
                if stats is None:
                    continue
                for stat_name, stat_value in stats.items():
                    cpu_gauge.labels(name=proc_name, stat=stat_name).set(stat_value)

            logger.debug(
                f"Exported CPU metrics for {len(self._process_samples)} processes"
            )
        else:
            logger.debug("No CPU metrics to export")

        # GPU gauges: utilization and memory with gpu + stat labels
        if self._gpu_metrics:
            logger.info(f"Exporting GPU metrics for {len(self._gpu_metrics)} GPU(s)")
            gpu_util_gauge = Gauge(
                "gpu_utilization",
                "GPU utilization summary statistics",
                ["gpu", "stat"],
                registry=registry,
            )
            gpu_mem_gauge = Gauge(
                "gpu_memory_used_bytes",
                "GPU memory usage summary statistics",
                ["gpu", "stat"],
                registry=registry,
            )
            gpu_mem_total_gauge = Gauge(
                "gpu_memory_total_bytes",
                "Total GPU memory available",
                ["gpu"],
                registry=registry,
            )

            for gpu_id, metrics in self._gpu_metrics.items():
                gpu_label = str(gpu_id)

                util_stats = self._compute_summary_stats(metrics.utilization)
                if util_stats:
                    for stat_name, stat_value in util_stats.items():
                        gpu_util_gauge.labels(gpu=gpu_label, stat=stat_name).set(
                            stat_value
                        )

                mem_stats = self._compute_summary_stats(metrics.memory_used_bytes)
                if mem_stats:
                    for stat_name, stat_value in mem_stats.items():
                        gpu_mem_gauge.labels(gpu=gpu_label, stat=stat_name).set(
                            stat_value
                        )

                if metrics.memory_total_bytes is not None:
                    gpu_mem_total_gauge.labels(gpu=gpu_label).set(
                        metrics.memory_total_bytes
                    )
        else:
            if self._nvml_initialized:
                logger.warning(
                    "No GPU samples collected despite NVML being initialized"
                )
            else:
                logger.debug("No GPU metrics to export (NVML not initialized)")
