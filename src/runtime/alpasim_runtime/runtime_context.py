# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
import copy
import math
from dataclasses import dataclass

from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.address_pool import AddressPool
from alpasim_runtime.config import (
    NetworkSimulatorConfig,
    SimulatorConfig,
    UserSimulatorConfig,
    typed_parse_config,
)
from alpasim_runtime.validation import (
    gather_versions_from_addresses,
    validate_scenarios,
)
from trajdata.dataset import UnifiedDataset

from eval.schema import EvalConfig

logger = logging.getLogger(__name__)

ALL_SKIP_PER_WORKER_CONCURRENCY = 16


def create_address_pools(config: SimulatorConfig) -> dict[str, AddressPool]:
    """Create one AddressPool per service type from the simulator config."""
    endpoints = config.user.endpoints
    network = config.network

    return {
        "driver": AddressPool(
            network.driver.addresses,
            endpoints.driver.n_concurrent_rollouts,
            skip=endpoints.driver.skip,
        ),
        "sensorsim": AddressPool(
            network.sensorsim.addresses,
            endpoints.sensorsim.n_concurrent_rollouts,
            skip=endpoints.sensorsim.skip,
        ),
        "physics": AddressPool(
            network.physics.addresses,
            endpoints.physics.n_concurrent_rollouts,
            skip=endpoints.physics.skip,
        ),
        "trafficsim": AddressPool(
            network.trafficsim.addresses,
            endpoints.trafficsim.n_concurrent_rollouts,
            skip=endpoints.trafficsim.skip,
        ),
        "controller": AddressPool(
            network.controller.addresses,
            endpoints.controller.n_concurrent_rollouts,
            skip=endpoints.controller.skip,
        ),
    }


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
    service_caps = {name: pool.total_capacity for name, pool in pools.items()}

    for name, pool in pools.items():
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


@dataclass(frozen=True)
class RuntimeContext:
    """Immutable snapshot of all runtime state needed to dispatch simulation jobs.

    Built once during startup by ``build_runtime_context`` after config parsing,
    service version probing, scenario validation, and scene mapping creation.
    """

    config: SimulatorConfig
    eval_config: EvalConfig
    version_ids: RolloutMetadata.VersionIds
    scene_id_to_idx: dict[str, int]
    dataset: UnifiedDataset
    pools: dict[str, AddressPool]
    max_in_flight: int


def parse_simulator_config(
    user_config_path: str,
    network_config_path: str,
) -> SimulatorConfig:
    """Parse user and network YAML configs into a unified SimulatorConfig."""
    user_config = typed_parse_config(user_config_path, UserSimulatorConfig)
    network_config = typed_parse_config(network_config_path, NetworkSimulatorConfig)
    return SimulatorConfig(user=user_config, network=network_config)


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
        4. Create UnifiedDataset from data_source config and build scene data sources.
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

    # Create UnifiedDataset and build scene_id to data source mapping
    logger.info("Creating UnifiedDataset from config")

    data_source_config = config.user.data_source
    trajdata_params = data_source_config.to_trajdata_params()
    dataset = UnifiedDataset(**trajdata_params)
    logger.info(
        f"Created UnifiedDataset with {dataset.num_scenes()} scenes, "
        f"desired_data={trajdata_params['desired_data']}"
    )

    # Build scene_id to index mapping (once, in main process)
    scene_id_to_idx = {}
    num_scenes = dataset.num_scenes()
    for idx in range(num_scenes):
        try:
            scene = dataset.get_scene(idx)
            scene_id_to_idx[scene.name] = idx
        except Exception as e:
            logger.warning(f"Failed to get scene at index {idx}: {e}")
            continue
    logger.info(f"Built scene_id mapping for {len(scene_id_to_idx)} scenes")

    pools = create_address_pools(config)
    max_in_flight = compute_max_in_flight(pools, config)

    return RuntimeContext(
        config=config,
        eval_config=eval_config,
        version_ids=version_ids,
        scene_id_to_idx=scene_id_to_idx,
        dataset=dataset,
        pools=pools,
        max_in_flight=max_in_flight,
    )
