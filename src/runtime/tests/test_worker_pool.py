# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import asyncio
from multiprocessing import Queue
from typing import cast
from unittest.mock import MagicMock

import pytest
from alpasim_runtime.address_pool import AddressPool, ServiceAddress
from alpasim_runtime.runtime_context import (
    ALL_SKIP_PER_WORKER_CONCURRENCY,
    compute_max_in_flight,
    compute_num_consumers_per_worker,
    create_address_pools,
)
from alpasim_runtime.worker.ipc import (
    AssignedRolloutJob,
    JobResult,
    ServiceEndpoints,
    _ShutdownSentinel,
)
from alpasim_runtime.worker.runtime import WorkerRuntime


def _make_pools(
    driver_cap: int = 2,
    sensorsim_cap: int = 2,
    physics_cap: int = 2,
    trafficsim_cap: int = 2,
    controller_cap: int = 2,
    driver_skip: bool = False,
    sensorsim_skip: bool = False,
    physics_skip: bool = False,
    trafficsim_skip: bool = False,
    controller_skip: bool = False,
) -> dict[str, AddressPool]:
    return {
        "driver": AddressPool(["driver:50051"], driver_cap, skip=driver_skip),
        "sensorsim": AddressPool(
            ["sensorsim:50052"], sensorsim_cap, skip=sensorsim_skip
        ),
        "physics": AddressPool(["physics:50053"], physics_cap, skip=physics_skip),
        "trafficsim": AddressPool(
            ["trafficsim:50054"], trafficsim_cap, skip=trafficsim_skip
        ),
        "controller": AddressPool(
            ["controller:50055"], controller_cap, skip=controller_skip
        ),
    }


def _make_config_mock(nr_workers: int = 1) -> MagicMock:
    config = MagicMock()
    config.user.nr_workers = nr_workers
    config.user.smooth_trajectories = True
    config.user.endpoints.driver.skip = False
    config.user.endpoints.driver.n_concurrent_rollouts = 2
    config.user.endpoints.sensorsim.skip = False
    config.user.endpoints.sensorsim.n_concurrent_rollouts = 2
    config.user.endpoints.physics.skip = False
    config.user.endpoints.physics.n_concurrent_rollouts = 2
    config.user.endpoints.trafficsim.skip = False
    config.user.endpoints.trafficsim.n_concurrent_rollouts = 2
    config.user.endpoints.controller.skip = False
    config.user.endpoints.controller.n_concurrent_rollouts = 2

    config.network.driver.addresses = ["driver:50051"]
    config.network.sensorsim.addresses = ["sensorsim:50052"]
    config.network.physics.addresses = ["physics:50053"]
    config.network.trafficsim.addresses = ["trafficsim:50054"]
    config.network.controller.addresses = ["controller:50055"]

    return config


class TestComputeMaxInFlight:
    def test_bottleneck_is_min_capacity(self) -> None:
        pools = _make_pools(
            driver_cap=4,
            sensorsim_cap=2,
            physics_cap=6,
            trafficsim_cap=3,
            controller_cap=5,
        )
        assert compute_max_in_flight(pools, _make_config_mock()) == 2

    def test_mixed_skip_and_non_skip(self) -> None:
        pools = _make_pools(
            driver_cap=4,
            sensorsim_cap=0,
            physics_cap=6,
            trafficsim_cap=0,
            controller_cap=0,
            sensorsim_skip=True,
            trafficsim_skip=True,
            controller_skip=True,
        )
        assert compute_max_in_flight(pools, _make_config_mock()) == 4

    def test_all_skip_uses_fixed_per_worker_cap(self) -> None:
        pools = _make_pools(
            driver_cap=0,
            sensorsim_cap=0,
            physics_cap=0,
            trafficsim_cap=0,
            controller_cap=0,
            driver_skip=True,
            sensorsim_skip=True,
            physics_skip=True,
            trafficsim_skip=True,
            controller_skip=True,
        )
        assert compute_max_in_flight(pools, _make_config_mock(nr_workers=3)) == (
            3 * ALL_SKIP_PER_WORKER_CONCURRENCY
        )

    def test_zero_capacity_raises(self) -> None:
        pools = _make_pools(driver_cap=0, physics_cap=2)
        with pytest.raises(ValueError, match="Service 'driver' has zero capacity"):
            compute_max_in_flight(pools, _make_config_mock())


class TestComputeNumConsumersPerWorker:
    def test_uses_job_count_cap_for_batch(self) -> None:
        consumers = compute_num_consumers_per_worker(
            max_in_flight=8,
            nr_workers=4,
            job_count=2,
        )
        assert consumers == 1

    def test_uses_full_max_when_job_count_not_set(self) -> None:
        consumers = compute_num_consumers_per_worker(
            max_in_flight=8,
            nr_workers=4,
            job_count=None,
        )
        assert consumers == 2

    def test_invalid_worker_count_raises(self) -> None:
        with pytest.raises(ValueError, match="nr_workers"):
            compute_num_consumers_per_worker(
                max_in_flight=1,
                nr_workers=0,
                job_count=1,
            )


def test_create_address_pools_builds_expected_service_pools() -> None:
    config = _make_config_mock(nr_workers=2)
    config.user.endpoints.driver.skip = True
    config.user.endpoints.driver.n_concurrent_rollouts = 7

    pools = create_address_pools(config)

    assert set(pools.keys()) == {
        "driver",
        "sensorsim",
        "physics",
        "trafficsim",
        "controller",
    }
    assert pools["driver"].skip is True
    assert pools["driver"].total_capacity is None
    assert pools["physics"].skip is False
    assert pools["physics"].total_capacity == 2


def _make_endpoints() -> ServiceEndpoints:
    return ServiceEndpoints(
        driver=ServiceAddress("driver:50051", skip=False),
        sensorsim=ServiceAddress("sensorsim:50052", skip=False),
        physics=ServiceAddress("physics:50053", skip=False),
        trafficsim=ServiceAddress("trafficsim:50054", skip=False),
        controller=ServiceAddress("controller:50055", skip=False),
    )


def _make_assigned_job(request_id: str, job_id: str) -> AssignedRolloutJob:
    return AssignedRolloutJob(
        request_id=request_id,
        job_id=job_id,
        scene_id="scene-A",
        rollout_spec_index=0,
        artifact_path="/tmp/scene-A.usdz",
        endpoints=_make_endpoints(),
    )


@pytest.mark.asyncio
async def test_worker_runtime_lifecycle_inline() -> None:
    job_queue: Queue = Queue()
    result_queue: Queue = Queue()

    async def _fake_worker() -> None:
        loop = asyncio.get_running_loop()
        while True:
            queued = await loop.run_in_executor(None, job_queue.get)
            if isinstance(queued, _ShutdownSentinel):
                break
            assert isinstance(queued, AssignedRolloutJob)
            result_queue.put(
                JobResult(
                    request_id=queued.request_id,
                    job_id=queued.job_id,
                    rollout_spec_index=queued.rollout_spec_index,
                    success=True,
                    error=None,
                    error_traceback=None,
                    rollout_uuid=f"uuid-{queued.job_id}",
                )
            )

    runtime = WorkerRuntime(
        job_queue=job_queue,
        result_queue=result_queue,
        worker_args=[],
        inline_main=_fake_worker,
    )
    runtime.submit_assigned_job(_make_assigned_job("req-1", "job-1"))

    result = await runtime.poll_result()
    assert result is not None
    assert result.request_id == "req-1"
    assert result.job_id == "job-1"

    await runtime.stop()


def test_worker_runtime_check_for_crashes_raises() -> None:
    class _DeadProcess:
        exitcode = 1

        @staticmethod
        def is_alive() -> bool:
            return False

    runtime = WorkerRuntime(
        job_queue=Queue(),
        result_queue=Queue(),
        worker_args=[],
    )
    runtime._workers = cast(list, [_DeadProcess()])

    with pytest.raises(RuntimeError, match="crashed"):
        runtime.check_for_crashes()
