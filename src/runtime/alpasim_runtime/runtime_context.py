# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.config import (
    BASE_SERVICE_NAMES,
    CORE_SERVICE_NAMES,
    NetworkSimulatorConfig,
    RenderBundling,
    RendererKind,
    SimulatorConfig,
    UserSimulatorConfig,
)
from alpasim_runtime.endpoints import get_endpoint_addresses
from alpasim_runtime.scene_loader import SceneLoader, build_scene_loader
from alpasim_runtime.validation import (
    gather_versions_from_addresses,
    validate_scenarios,
)
from alpasim_utils.yaml_utils import typed_parse_config
from omegaconf import MISSING
from omegaconf.errors import MissingMandatoryValue

from eval.schema import EvalConfig

ALL_SKIP_PER_WORKER_CONCURRENCY = 16


def create_address_pools(config: SimulatorConfig) -> dict[str, AddressPool]:
    """Create one AddressPool per service type from the simulator config."""
    endpoints = config.user.endpoints
    network = config.network

    pools = {}
    for service_name in CORE_SERVICE_NAMES:
        network_endpoint = getattr(network, service_name)
        user_endpoint = getattr(endpoints, service_name)
        addresses = get_endpoint_addresses(network_endpoint)
        if not user_endpoint.skip and not addresses:
            raise ValueError(
                f"Service {service_name!r} has no endpoint addresses in "
                f"network.{service_name} and "
                f"runtime.endpoints.{service_name}.skip=False"
            )
        pools[service_name] = AddressPool(
            addresses,
            user_endpoint.n_concurrent_rollouts,
            skip=user_endpoint.skip,
        )

    return pools


def compute_max_in_flight(
    pools: dict[str, AddressPool],
    config: SimulatorConfig,
) -> int:
    """
    Compute the maximum number of jobs that can be in flight at once.

    For non-skip pools, the limit is the minimum total_capacity.
    If all pools are skip, use a fixed per-worker cap so dispatch does not
    become unbounded.
    """
    num_workers = config.user.nr_workers
    required_names = (*BASE_SERVICE_NAMES, "renderer")
    service_caps = {name: pools[name].total_capacity for name in required_names}

    for name in required_names:
        pool = pools[name]
        if not pool.skip and service_caps[name] == 0:
            raise ValueError(f"Service '{name}' has zero capacity")

    limiting_caps = [cap for cap in service_caps.values() if cap is not None]

    if limiting_caps:
        return min(limiting_caps)

    return max(1, num_workers * ALL_SKIP_PER_WORKER_CONCURRENCY)


def compute_num_consumers_per_worker(
    *,
    max_in_flight: int,
    nr_workers: int,
    job_count: int | None = None,
) -> int:
    """Compute how many concurrent consumer tasks each worker should run.

    Divides the effective in-flight limit evenly across workers (rounded up).
    When *job_count* is provided, caps effective in-flight to avoid
    over-provisioning for small batches.

    Args:
        max_in_flight: Maximum concurrent jobs across all workers.
        nr_workers: Number of worker processes.
        job_count: If given, cap concurrency to this many jobs.

    Returns:
        Number of consumer tasks per worker (at least 1).
    """
    if nr_workers < 1:
        raise ValueError(f"nr_workers must be >= 1, got {nr_workers}")

    effective_in_flight = max_in_flight
    if job_count is not None:
        effective_in_flight = min(max_in_flight, max(1, job_count))

    return math.ceil(effective_in_flight / nr_workers)


def validate_renderer_config(config: SimulatorConfig) -> None:
    """Ensure renderer-specific options match the selected renderer."""
    try:
        renderer = config.user.renderer
        renderer_kind = renderer.kind
    except MissingMandatoryValue as exc:
        raise ValueError("runtime.renderer.kind is required") from exc

    if renderer == MISSING or renderer_kind == MISSING:
        raise ValueError("runtime.renderer.kind is required")

    if (
        renderer_kind == RendererKind.sensorsim
        and renderer.video_model_config is not None
    ):
        raise ValueError(
            "runtime.renderer.video_model_config is only valid when "
            "runtime.renderer.kind=video_model"
        )
    if (
        renderer_kind == RendererKind.video_model
        and renderer.video_model_config is None
    ):
        raise ValueError(
            "runtime.renderer.video_model_config is required when "
            "runtime.renderer.kind=video_model"
        )
    if not isinstance(renderer_kind, RendererKind):
        raise ValueError(f"Unknown runtime.renderer.kind: {renderer_kind!r}")


def validate_scene_affinity_config(config: SimulatorConfig) -> None:
    """Validate scene-affine dispatch bounds and renderer compatibility."""
    affine = config.user.scene_affine_dispatch
    if affine.enabled and config.user.renderer.kind == RendererKind.video_model:
        raise ValueError(
            "runtime.scene_affine_dispatch.enabled=true is not supported with "
            "runtime.renderer.kind=video_model (no per-scene GPU cache); set "
            "runtime.scene_affine_dispatch.enabled=false"
        )
    render_bundling = config.user.simulation_config.render_bundling
    if affine.enabled and render_bundling != RenderBundling.BATCH_RENDER_RGB:
        raise ValueError(
            "runtime.scene_affine_dispatch.enabled=true requires "
            "runtime.simulation_config.render_bundling=BATCH_RENDER_RGB "
            f"(got {render_bundling.name}); scene-affine dispatch assumes the "
            "NRE batched-render scene cache"
        )
    if affine.max_renderers_per_scene <= 0:
        raise ValueError(
            "runtime.scene_affine_dispatch.max_renderers_per_scene must be positive"
        )

    if (
        affine.max_scenes_per_renderer is not None
        and affine.max_scenes_per_renderer <= 0
    ):
        raise ValueError(
            "runtime.scene_affine_dispatch.max_scenes_per_renderer must be "
            "positive or null"
        )


@dataclass(frozen=True)
class RuntimeContext:
    """Immutable snapshot of all runtime state needed to dispatch simulation jobs.

    Built once during startup by ``build_runtime_context`` after config parsing,
    service version probing, scenario validation, scene loader creation, and address pool creation.
    """

    config: SimulatorConfig
    eval_config: EvalConfig
    version_ids: RolloutMetadata.VersionIds
    scene_loader: SceneLoader
    pools: dict[str, AddressPool]
    max_in_flight: int


def parse_simulator_config(
    user_config_path: str,
    network_config_path: str,
) -> SimulatorConfig:
    """Parse user and network YAML configs into a unified SimulatorConfig."""
    user_config = typed_parse_config(user_config_path, UserSimulatorConfig)
    network_config = typed_parse_config(network_config_path, NetworkSimulatorConfig)
    config = SimulatorConfig(user=user_config, network=network_config)
    validate_renderer_config(config)
    validate_scene_affinity_config(config)
    return config


async def build_runtime_context(
    *,
    user_config_path: str,
    network_config_path: str,
    eval_config_path: str,
    validate_config_scenes: bool = True,
) -> RuntimeContext:
    """Build the RuntimeContext by parsing configs, probing services, and validating scenarios.

    Steps:
        1. Parse user and network configs.
        2. Probe all service addresses for version IDs.
        3. Validate scenario compatibility (unless *validate_config_scenes* is False).
        4. Build the scene loader from scene_provider config.
        5. Create address pools and compute max in-flight concurrency.

    Args:
        user_config_path: Path to user YAML config.
        network_config_path: Path to network YAML config.
        eval_config_path: Path to evaluation YAML config.
        validate_config_scenes: If False, skip scene compatibility checks
            (useful for daemon mode where scenes come from requests).
    """
    config = parse_simulator_config(user_config_path, network_config_path)
    eval_config = typed_parse_config(eval_config_path, EvalConfig)

    # Validate configuration
    version_ids = await gather_versions_from_addresses(
        config.network,
        config.user.endpoints,
        renderer_kind=config.user.renderer.kind,
        timeout_s=config.user.endpoints.startup_timeout_s,
    )
    config_for_validation = config
    if not validate_config_scenes:
        user_no_scenes = copy.deepcopy(config.user)
        user_no_scenes.scenes = []
        config_for_validation = SimulatorConfig(
            user=user_no_scenes,
            network=config.network,
        )
    await validate_scenarios(config_for_validation)

    scene_loader = build_scene_loader(config.user)

    pools = create_address_pools(config)
    max_in_flight = compute_max_in_flight(pools, config)

    return RuntimeContext(
        config=config,
        eval_config=eval_config,
        version_ids=version_ids,
        scene_loader=scene_loader,
        pools=pools,
        max_in_flight=max_in_flight,
    )
