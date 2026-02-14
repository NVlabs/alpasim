# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from __future__ import annotations

"""
Implements a Dispatcher type which manages a pool of available egodriver endpoints
and sensor simulation endpoints and assigns them to simulation rollouts as they
come available.
"""

import asyncio
import functools
import logging
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import alpasim_runtime
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import ScenarioConfig, UserSimulatorConfig
from alpasim_runtime.loop import UnboundRollout
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.service_pool import ServicePool
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.worker.ipc import JobResult, RolloutJob, ServiceAllocations
from alpasim_utils.artifact import Artifact
from alpasim_utils.data_source_loader import load_data_sources
from alpasim_utils.scene_data_source import SceneDataSource

logger = logging.getLogger(__name__)


@dataclass
class Dispatcher:
    """
    Keeps track of contention of each microservice and assigns tasks as they come available.
    Preserves named pool attributes for type safety.
    """

    driver_pool: ServicePool[DriverService]
    sensorsim_pool: ServicePool[SensorsimService]
    physics_pool: ServicePool[PhysicsService]
    trafficsim_pool: ServicePool[TrafficService]
    controller_pool: ServicePool[ControllerService]

    camera_catalog: CameraCatalog

    user_config: UserSimulatorConfig
    artifacts: dict[str, SceneDataSource]
    version_ids: RolloutMetadata.VersionIds
    asl_dir: str

    async def find_scenario_incompatibilities(
        self, scenario: ScenarioConfig
    ) -> list[str]:
        results = await asyncio.gather(
            self.driver_pool.find_scenario_incompatibilities(scenario),
            self.sensorsim_pool.find_scenario_incompatibilities(scenario),
            self.physics_pool.find_scenario_incompatibilities(scenario),
            self.trafficsim_pool.find_scenario_incompatibilities(scenario),
            self.controller_pool.find_scenario_incompatibilities(scenario),
        )

        return [item for sublist in results for item in sublist]

    @staticmethod
    async def create(
        user_config: UserSimulatorConfig,
        allocations: ServiceAllocations,
        usdz_glob: str | None = None,
        asl_dir: str = "",
        trajdata_config_path: str | None = None,
    ) -> Dispatcher:
        """
        Initialize dispatcher: discover artifacts, build pools from allocations.
        
        Args:
            user_config: User simulator configuration
            allocations: Service allocations
            usdz_glob: Glob pattern for USDZ files (mutually exclusive with trajdata_config_path)
            asl_dir: Directory for ASL output files
            trajdata_config_path: Path to JSON/YAML file containing trajdata configuration
        """
        camera_catalog = CameraCatalog(user_config.extra_cameras)

        # NOTE: In multi-worker mode, each worker re-discovers artifacts independently.
        artifacts = load_data_sources(
            usdz_glob=usdz_glob,
            trajdata_config_path=trajdata_config_path,
            smooth_trajectories=user_config.smooth_trajectories,
        )

        endpoints = user_config.endpoints
        timeout = endpoints.startup_timeout_s

        logger.info("Acquiring physics connections: %s", allocations.physics)
        physics = await ServicePool.create_from_allocation(
            PhysicsService,
            allocations.physics,
            skip=endpoints.physics.skip,
            connection_timeout_s=timeout,
        )

        logger.info("Acquiring controller connections: %s", allocations.controller)
        controller = await ServicePool.create_from_allocation(
            ControllerService,
            allocations.controller,
            skip=endpoints.controller.skip,
            connection_timeout_s=timeout,
        )

        logger.info("Acquiring traffic connections: %s", allocations.trafficsim)
        traffic = await ServicePool.create_from_allocation(
            TrafficService,
            allocations.trafficsim,
            skip=endpoints.trafficsim.skip,
            connection_timeout_s=timeout,
        )

        logger.info("Acquiring sensorsim connections: %s", allocations.sensorsim)
        sensorsim = await ServicePool.create_from_allocation(
            SensorsimService,
            allocations.sensorsim,
            skip=endpoints.sensorsim.skip,
            connection_timeout_s=timeout,
            camera_catalog=camera_catalog,
        )

        logger.info("Acquiring driver connections: %s", allocations.driver)
        driver = await ServicePool.create_from_allocation(
            DriverService,
            allocations.driver,
            skip=endpoints.driver.skip,
            connection_timeout_s=timeout,
        )

        # Gather version info from each service pool
        version_ids = await _gather_versions_from_pools(
            driver=driver,
            sensorsim=sensorsim,
            physics=physics,
            trafficsim=traffic,
            controller=controller,
        )

        logger.info("Dispatcher ready.")
        return Dispatcher(
            driver_pool=driver,
            sensorsim_pool=sensorsim,
            physics_pool=physics,
            trafficsim_pool=traffic,
            controller_pool=controller,
            camera_catalog=camera_catalog,
            user_config=user_config,
            artifacts=artifacts,
            version_ids=version_ids,
            asl_dir=asl_dir,
        )

    def get_pool_capacity(self) -> int:
        """Return how many concurrent rollouts this dispatcher can handle."""
        return min(
            self.driver_pool.get_number_of_services(),
            self.sensorsim_pool.get_number_of_services(),
            self.physics_pool.get_number_of_services(),
            self.trafficsim_pool.get_number_of_services(),
            self.controller_pool.get_number_of_services(),
        )

    @asynccontextmanager
    async def acquire_all_services(
        self,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Atomically acquire all services, releasing on failure.

        This context manager ensures that if any service acquisition fails,
        all previously acquired services are returned to their pools before
        the exception propagates. On successful acquisition, all services
        are released when the context exits (whether normally or via exception).

        Yields:
            dict with keys: 'driver', 'sensorsim', 'physics', 'trafficsim', 'controller'
        """
        services: dict[str, Any] = {}
        pools: list[tuple[str, ServicePool]] = [
            ("driver", self.driver_pool),
            ("sensorsim", self.sensorsim_pool),
            ("physics", self.physics_pool),
            ("trafficsim", self.trafficsim_pool),
            ("controller", self.controller_pool),
        ]

        try:
            for name, pool in pools:
                services[name] = await pool.get()
            yield services
        finally:
            # Release all acquired services back to their pools
            for name, pool in pools:
                service = services.get(name)
                if service is not None:
                    await pool.put_back(service)

    async def run_job(self, job: RolloutJob) -> JobResult:
        """Execute a single rollout job."""
        rollout: UnboundRollout | None = None

        try:
            # Offload CPU-bound rollout preparation to thread to keep event loop responsive.
            loop = asyncio.get_running_loop()
            rollout = await loop.run_in_executor(
                None,
                functools.partial(
                    UnboundRollout.create,
                    config=self.user_config,
                    scenario=job.scenario,
                    version_ids=self.version_ids,
                    random_seed=job.seed,
                    available_artifacts=self.artifacts,
                    asl_dir=self.asl_dir,
                ),
            )

            # Acquire all services atomically with automatic cleanup
            async with self.acquire_all_services() as services:
                await rollout.bind(
                    services["driver"],
                    services["sensorsim"],
                    services["physics"],
                    services["trafficsim"],
                    services["controller"],
                    self.camera_catalog,
                ).run()

            return JobResult(
                job_id=job.job_id,
                success=True,
                error=None,
                error_traceback=None,
                rollout_uuid=rollout.rollout_uuid,
            )

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.warning(
                "Rollout FAILED: job=%s scene=%s uuid=%s error=%s\n%s",
                job.job_id,
                rollout.scene_id if rollout else "N/A",
                rollout.rollout_uuid if rollout else "N/A",
                exc,
                tb,
            )
            return JobResult(
                job_id=job.job_id,
                success=False,
                error=str(exc),
                error_traceback=tb,
                rollout_uuid=rollout.rollout_uuid if rollout else None,
            )


async def _gather_versions_from_pools(
    driver: ServicePool[DriverService],
    sensorsim: ServicePool[SensorsimService],
    physics: ServicePool[PhysicsService],
    trafficsim: ServicePool[TrafficService],
    controller: ServicePool[ControllerService],
) -> RolloutMetadata.VersionIds:
    """
    Gather version info from each service pool.
    """
    versions: dict[str, VersionId] = {}

    # Query version from first service in each pool
    versions["driver"] = await driver.services[0].get_version()
    versions["sensorsim"] = await sensorsim.services[0].get_version()
    versions["physics"] = await physics.services[0].get_version()
    versions["trafficsim"] = await trafficsim.services[0].get_version()
    versions["controller"] = await controller.services[0].get_version()

    return RolloutMetadata.VersionIds(
        runtime_version=alpasim_runtime.VERSION_MESSAGE,
        egodriver_version=versions.get("driver"),
        sensorsim_version=versions.get("sensorsim"),
        physics_version=versions.get("physics"),
        traffic_version=versions.get("trafficsim"),
    )
