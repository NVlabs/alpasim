# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Pre-flight validation for simulation runs."""

import asyncio
import logging
import os
import time
from collections import defaultdict
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any, TypeVar

import alpasim_runtime
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0.common_pb2 import Empty, VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    NetworkSimulatorConfig,
    PhysicsUpdateMode,
    RendererKind,
    SceneConfig,
    SimulatorConfig,
    UserEndpointConfig,
)
from alpasim_runtime.endpoints import get_service_endpoints

import grpc

logger = logging.getLogger(__name__)

T = TypeVar("T")

_PENDING_PROBE_LOG_INTERVAL_S = 10.0

# Sentinel VersionId for skipped services.
_SKIP_VERSION = VersionId(
    version_id="<skip>",
    git_hash="<skip>",
    grpc_api_version=API_VERSION_MESSAGE,
)


async def _log_awaitable_progress(
    coroutine: Coroutine[Any, Any, T],
    *,
    label: str,
    log_interval_s: float = _PENDING_PROBE_LOG_INTERVAL_S,
) -> T:
    """
    Wraps a coroutine in another coroutine which periodically logs if it remains pending.

    Useful for detecting stuck/slow operations.
    """
    task = asyncio.create_task(coroutine)
    start_time = time.monotonic()

    try:
        while True:
            done, _ = await asyncio.wait(
                {task},
                timeout=log_interval_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                return task.result()

            logger.info(
                "%s still waiting after %.0fs",
                label,
                time.monotonic() - start_time,
            )
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        raise


async def _probe_version_for_address(
    svc_name: str,
    stub_class: type,
    address: str,
    timeout_s: int,
) -> tuple[str, str, VersionId]:
    """Probe a single service address for its version.

    Returns:
        (service_name, address, VersionId) tuple.
    """
    logger.info("Probing version for %s at %s...", svc_name, address)
    channel = grpc.aio.insecure_channel(address)
    try:
        stub = stub_class(channel)
        if svc_name == "trafficsim":
            metadata = await stub.get_metadata(
                Empty(), wait_for_ready=True, timeout=timeout_s
            )
            version = metadata.version_id
        else:
            version = await stub.get_version(
                Empty(), wait_for_ready=True, timeout=timeout_s
            )
        logger.info("Connected to %s at %s: %s", svc_name, address, version)
        return svc_name, address, version
    finally:
        await channel.close()


def _assert_consistent_versions(
    results: list[tuple[str, str, VersionId]],
) -> dict[str, VersionId]:
    """Assert that all addresses for the same service returned the same version.

    Args:
        results: List of (service_name, address, VersionId) from probing.

    Returns:
        Dict mapping service_name -> canonical VersionId (one per service).

    Raises:
        AssertionError: If any service has mixed versions across its addresses.
    """
    by_service: dict[str, list[tuple[str, VersionId]]] = defaultdict(list)
    for svc_name, address, version in results:
        by_service[svc_name].append((address, version))

    canonical: dict[str, VersionId] = {}
    for svc_name, addr_versions in by_service.items():
        identities_seen = {
            (
                version.version_id,
                version.git_hash,
                version.grpc_api_version.SerializeToString(deterministic=True),
            )
            for _, version in addr_versions
        }
        if len(identities_seen) > 1:
            details = ", ".join(
                f"{addr}=(version_id={version.version_id}, git_hash={version.git_hash}, "
                f"grpc_api_version={version.grpc_api_version})"
                for addr, version in addr_versions
            )
            raise AssertionError(
                f"Service '{svc_name}' has mixed versions across addresses: {details}"
            )
        canonical[svc_name] = addr_versions[0][1]

    return canonical


async def gather_versions_from_addresses(
    network_config: NetworkSimulatorConfig,
    user_endpoints: UserEndpointConfig,
    renderer_kind: RendererKind,
    timeout_s: int = 30,
) -> RolloutMetadata.VersionIds:
    """Probe all service addresses and return canonical version IDs.

    This is the single source of truth for version IDs in the runtime.
    Workers receive the result; they never probe versions themselves.

    Skipped services get a synthetic '<skip>' VersionId without network calls.

    Args:
        network_config: Network configuration with service addresses.
        user_endpoints: User endpoint config (for skip flags).
        timeout_s: gRPC probe timeout per address.

    Returns:
        Fully populated RolloutMetadata.VersionIds.

    Raises:
        AssertionError: If any service has mixed versions across its addresses.
    """
    runtime_version = alpasim_runtime.VERSION_MESSAGE
    logger.info("runtime: %s", runtime_version)

    # Determine which services to skip based on user config.
    skip_flags = {
        "driver": user_endpoints.driver.skip,
        "renderer": user_endpoints.renderer.skip,
        "physics": user_endpoints.physics.skip,
        "trafficsim": user_endpoints.trafficsim.skip,
        "controller": user_endpoints.controller.skip,
    }

    endpoint_stubs = get_service_endpoints(
        network_config,
        renderer_kind=renderer_kind,
    )

    # Build probe tasks for non-skip services (probe ALL addresses per service).
    tasks = []
    for svc_name, (stub_class, addresses) in endpoint_stubs.items():
        if skip_flags[svc_name]:
            continue
        if not addresses:
            raise AssertionError(
                f"Service '{svc_name}' has no addresses configured but skip=False"
            )
        for address in addresses:
            tasks.append(
                _log_awaitable_progress(
                    _probe_version_for_address(
                        svc_name,
                        stub_class,
                        address,
                        timeout_s,
                    ),
                    label=f"Service version probe for {svc_name} at {address}",
                )
            )

    results = await asyncio.gather(*tasks)
    canonical = _assert_consistent_versions(list(results))

    renderer_version_field = (
        "video_model_version"
        if renderer_kind == RendererKind.video_model
        else "sensorsim_version"
    )
    service_to_version_field = {
        "driver": "egodriver_version",
        "renderer": renderer_version_field,
        "physics": "physics_version",
        "trafficsim": "traffic_version",
        "controller": "controller_version",
    }

    version_kwargs = {
        "runtime_version": runtime_version,
        "egodriver_version": _SKIP_VERSION,
        "sensorsim_version": _SKIP_VERSION,
        "video_model_version": _SKIP_VERSION,
        "physics_version": _SKIP_VERSION,
        "traffic_version": _SKIP_VERSION,
        "controller_version": _SKIP_VERSION,
    }
    for svc_name, field_name in service_to_version_field.items():
        if not skip_flags[svc_name]:
            version_kwargs[field_name] = canonical[svc_name]
    return RolloutMetadata.VersionIds(**version_kwargs)


async def validate_scenarios(config: SimulatorConfig) -> None:
    """
    Validate all scenarios before building job list.

    Uses lightweight probes to check scene availability without creating full pools.
    This ensures we fail fast in the parent if any scenario is invalid.
    """
    simulation_config = config.user.simulation_config

    if (
        config.user.endpoints.physics.skip
        and simulation_config.physics_update_mode != PhysicsUpdateMode.NONE
    ):
        raise AssertionError("Physics update mode requires the physics service to run.")
    if (
        not config.user.endpoints.physics.skip
        and simulation_config.physics_update_mode == PhysicsUpdateMode.NONE
    ):
        raise AssertionError(
            "Physics is disabled in simulation config but physics service is not skipped."
        )

    # driver and controller return wildcard (work with any scene), no need to probe
    services_to_probe = ["physics", "trafficsim"]
    if config.user.renderer.kind == RendererKind.sensorsim:
        services_to_probe.append("renderer")
    skip_flags = {
        "renderer": config.user.endpoints.renderer.skip,
        "physics": config.user.endpoints.physics.skip,
        "trafficsim": config.user.endpoints.trafficsim.skip,
    }
    services_to_probe = [s for s in services_to_probe if not skip_flags[s]]
    service_endpoints = get_service_endpoints(
        config.network,
        services=services_to_probe,
        renderer_kind=config.user.renderer.kind,
    )

    tasks = []
    for svc_name, (stub_class, addresses) in service_endpoints.items():
        if not addresses:
            continue
        address = addresses[0]
        tasks.append(
            _log_awaitable_progress(
                _probe_scenario_compatibility(
                    svc_name,
                    stub_class,
                    address,
                    config.user.scenes,
                    timeout_s=config.user.endpoints.startup_timeout_s,
                ),
                label=f"Scenario validation probe for {svc_name} at {address}",
            )
        )

    results = await asyncio.gather(*tasks)
    error_messages = [msg for errors in results for msg in errors]

    if error_messages:
        raise AssertionError("\n".join(error_messages))


async def _probe_scenario_compatibility(
    svc_name: str,
    stub_class: type,
    address: str,
    scenes: list[SceneConfig],
    timeout_s: int = 30,
) -> list[str]:
    """Probe a service address to validate scenario compatibility without creating pools."""
    incompatibilities = []

    logger.info("Validating scenarios on %s at %s...", svc_name, address)
    channel = grpc.aio.insecure_channel(address)
    try:
        stub = stub_class(channel)
        response = await stub.get_available_scenes(
            Empty(), wait_for_ready=True, timeout=timeout_s
        )
        available_scenes = set(response.scene_ids)

        for scene in scenes:
            if scene.scene_id not in available_scenes and "*" not in available_scenes:
                incompatibilities.append(
                    f"Scene {scene.scene_id} not available at {address}. "
                    f"Available: {sorted(available_scenes)}"
                )

        if incompatibilities:
            logger.error(
                "Scenario validation failed on %s: %d issue(s)",
                svc_name,
                len(incompatibilities),
            )
        else:
            logger.info("Scenario validation passed on %s", svc_name)
    finally:
        await channel.close()

    return incompatibilities


def validate_array_job_config(array_job_dir: str | None) -> None:
    """Validate array_job_dir is provided when running as SLURM array job.

    Args:
        array_job_dir: The --array-job-dir argument value (or None if not provided).

    Raises:
        ValueError: If running as SLURM array job without explicit array_job_dir.
    """
    slurm_array_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "0"))
    if slurm_array_count > 0 and array_job_dir is None:
        raise ValueError("Running as SLURM array job but --array-job-dir not provided.")
