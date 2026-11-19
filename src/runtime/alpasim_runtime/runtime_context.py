# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass

from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.config import (
    CORE_SERVICE_NAMES,
    ROLLOUT_SERVICE_NAMES,
    NavInstructionConfig,
    NetworkSimulatorConfig,
    RenderBundling,
    RendererKind,
    RouteGeneratorType,
    SimulatorConfig,
    UserSimulatorConfig,
)
from alpasim_runtime.data_source_loader import (
    DataSourceLoader,
    build_data_source_loader,
)
from alpasim_runtime.ego_masks import EgoMaskOverride, load_ego_mask_overrides
from alpasim_runtime.endpoints import get_endpoint_addresses
from alpasim_runtime.nav_route import NavRouteLibrary
from alpasim_runtime.scene_registry import SceneRegistry, scene_from_config
from alpasim_runtime.validation import (
    gather_versions_from_addresses,
    validate_scenarios,
)
from alpasim_utils.yaml_utils import typed_parse_config
from omegaconf import MISSING
from omegaconf.errors import MissingMandatoryValue

from eval.schema import EvalConfig

logger = logging.getLogger(__name__)

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
        pool = AddressPool(
            addresses,
            user_endpoint.n_concurrent_rollouts,
            skip=user_endpoint.skip,
        )
        if not pool.skip and pool.total_capacity == 0:
            raise ValueError(f"Service {service_name!r} has zero capacity")
        pools[service_name] = pool

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
    # Optional post-rollout services must not reduce simulation concurrency.
    service_caps = {name: pools[name].total_capacity for name in ROLLOUT_SERVICE_NAMES}

    for name in ROLLOUT_SERVICE_NAMES:
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
    """Ensure renderer options and force-GT cache flags are consistent."""
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

    force_gt_frame_cache = config.user.simulation_config.force_gt_frame_cache
    if (
        config.user.simulation_config.ego_mask_overrides_file is not None
        and renderer_kind != RendererKind.sensorsim
    ):
        raise ValueError("ego_mask_overrides_file requires the sensorsim renderer")
    if force_gt_frame_cache.enabled and renderer_kind != RendererKind.sensorsim:
        raise ValueError(
            "force_gt_frame_cache.enabled is only supported by the sensorsim "
            "renderer; the video model renders chunks statefully and cannot replay "
            "cached force-GT frames."
        )

    # NOTE: trafficsim does not need to be skipped for caching. During force-GT,
    # PhysicsEvent ignores the trafficsim response entirely and derives traffic
    # poses directly from the logged trajectories in ``state.unbound.traffic_objs``
    # (see events/physics.py, the ``if force_gt:`` branch). Independently, the
    # traffic service is handed ``handover_time_us`` (set past the last force-GT
    # frame, see TrafficService in services/traffic_service.py) so it stays on the
    # logged, deterministic traffic state throughout that window. Frames rendered
    # during force-GT therefore do not depend on the trafficsim backend or its
    # configuration, and remain safe to cache and replay across rollouts.

    if (
        force_gt_frame_cache.enabled
        and not config.user.simulation_config.skip_driver_during_force_gt
    ):
        raise ValueError(
            "force_gt_frame_cache.enabled requires "
            "skip_driver_during_force_gt=true. Bundled rendering returns a "
            "renderer payload passed to driver.drive(renderer_data=...) that the "
            "frame cache does not store, so cache hits cannot reproduce it. "
            "Skipping the driver query during force-GT avoids feeding the driver "
            "a different payload than a non-cached run."
        )

    if force_gt_frame_cache.enabled and not force_gt_frame_cache.cache_dir:
        raise ValueError(
            "force_gt_frame_cache.enabled requires "
            "simulation_config.force_gt_frame_cache.cache_dir to be set; point it "
            "at shared writable storage (e.g. the Lustre artifact mount)."
        )


def validate_nav_instruction_config(
    nav_config: NavInstructionConfig,
    route_generator_type: RouteGeneratorType = RouteGeneratorType.MAP,
) -> None:
    """Ensure each navigation source has what it reads from: a route generator for
    ``geometry``, a navigation version directory for ``recorded``."""
    if nav_config.source not in ("none", "geometry", "recorded"):
        raise ValueError(
            "runtime.simulation_config.nav_instruction.source must be "
            f"'none', 'geometry' or 'recorded' (got {nav_config.source!r})"
        )
    if (
        nav_config.source == "geometry"
        and route_generator_type == RouteGeneratorType.NONE
    ):
        raise ValueError(
            "runtime.simulation_config.nav_instruction.source='geometry' classifies the "
            "generated route, so it needs route_generator_type other than NONE"
        )
    if nav_config.source != "recorded":
        return
    if not nav_config.route_index_dir or not os.path.isdir(nav_config.route_index_dir):
        raise ValueError(
            "runtime.simulation_config.nav_instruction.route_index_dir must be an "
            f"existing directory when source is recorded (got "
            f"{nav_config.route_index_dir!r})"
        )


def validate_clip_affinity_config(config: SimulatorConfig) -> None:
    """Validate clip-affine dispatch bounds and renderer compatibility."""
    affine = config.user.clip_affine_dispatch
    if affine.enabled and config.user.renderer.kind == RendererKind.video_model:
        raise ValueError(
            "runtime.clip_affine_dispatch.enabled=true is not supported with "
            "runtime.renderer.kind=video_model (no per-clip GPU cache); set "
            "runtime.clip_affine_dispatch.enabled=false"
        )
    render_bundling = config.user.simulation_config.render_bundling
    if affine.enabled and render_bundling != RenderBundling.BATCH_RENDER_RGB:
        raise ValueError(
            "runtime.clip_affine_dispatch.enabled=true requires "
            "runtime.simulation_config.render_bundling=BATCH_RENDER_RGB "
            f"(got {render_bundling.name}); clip-affine dispatch assumes the "
            "NRE batched-render clip cache"
        )
    if affine.max_renderers_per_clip <= 0:
        raise ValueError(
            "runtime.clip_affine_dispatch.max_renderers_per_clip must be positive"
        )

    if affine.max_clips_per_renderer is not None and affine.max_clips_per_renderer <= 0:
        raise ValueError(
            "runtime.clip_affine_dispatch.max_clips_per_renderer must be "
            "positive or null"
        )


@dataclass(frozen=True)
class RuntimeContext:
    """Immutable snapshot of all runtime state needed to dispatch simulation jobs.

    Built once during startup by ``build_runtime_context`` after config parsing,
    service version probing, scenario validation, data-source loading, scene registration,
    and address pool creation.
    """

    config: SimulatorConfig
    eval_config: EvalConfig
    version_ids: RolloutMetadata.VersionIds
    data_source_loader: DataSourceLoader
    scene_registry: SceneRegistry
    pools: dict[str, AddressPool]
    max_in_flight: int
    # Route indices for navigation instructions; None when geometry navigation is selected.
    nav_routes: NavRouteLibrary | None = None
    ego_mask_overrides: dict[str, EgoMaskOverride] | None = None


def parse_simulator_config(
    user_config_path: str,
    network_config_path: str,
) -> SimulatorConfig:
    """Parse user and network YAML configs into a unified SimulatorConfig."""
    user_config = typed_parse_config(user_config_path, UserSimulatorConfig)
    network_config = typed_parse_config(network_config_path, NetworkSimulatorConfig)
    config = SimulatorConfig(user=user_config, network=network_config)
    validate_renderer_config(config)
    validate_clip_affinity_config(config)
    validate_nav_instruction_config(
        config.user.simulation_config.nav_instruction,
        config.user.simulation_config.route_generator_type,
    )
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
        4. Build the data-source loader from data_source_provider config.
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
    overrides_path = config.user.simulation_config.ego_mask_overrides_file
    ego_mask_overrides = (
        load_ego_mask_overrides(overrides_path) if overrides_path is not None else None
    )
    if ego_mask_overrides is not None:
        source_ids = {
            scene_from_config(scene).source.source_id for scene in config.user.scenes
        }
        covered = source_ids.intersection(ego_mask_overrides)
        logger.info(
            "Loaded %d ego mask overrides from %s; %d/%d configured source clips covered",
            len(ego_mask_overrides),
            overrides_path,
            len(covered),
            len(source_ids),
        )
        if source_ids and ego_mask_overrides and not covered:
            logger.warning(
                "No configured source clips match ego mask overrides in %s; "
                "check exact source IDs (ClipGT IDs use clipgt-<uuid>)",
                overrides_path,
            )

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

    data_source_loader = build_data_source_loader(config.user)
    scene_registry = SceneRegistry(
        scenes=config.user.scenes,
        data_source_infos=data_source_loader.data_source_infos,
        simulation_config=config.user.simulation_config,
    )

    pools = create_address_pools(config)
    max_in_flight = compute_max_in_flight(pools, config)

    nav_config = config.user.simulation_config.nav_instruction
    nav_routes = None
    if nav_config.source == "recorded":
        # validate_nav_instruction_config has checked the directory exists.
        assert nav_config.route_index_dir is not None
        nav_routes = NavRouteLibrary.build(
            route_index_dir=nav_config.route_index_dir,
            clip_ids=data_source_loader.source_ids,
        )

    return RuntimeContext(
        config=config,
        eval_config=eval_config,
        version_ids=version_ids,
        data_source_loader=data_source_loader,
        scene_registry=scene_registry,
        pools=pools,
        max_in_flight=max_in_flight,
        ego_mask_overrides=ego_mask_overrides,
        nav_routes=nav_routes,
    )
