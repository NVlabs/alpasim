# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Context management for AlpasimWizard."""

from __future__ import annotations

import logging
import os
import posixpath
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterator, cast

from alpasim_utils.types import ClipDataSource, Scene

from .ego_hoods import generate_ego_hood_banks
from .scenes import (
    SceneMaterialization,
    USDZManager,
    materialize_scene_selection,
    resolve_scene_groups,
)
from .scenes.sceneset import scan_local_usdz_directory
from .schema import AlpasimConfig, ContainerConfig, RunMethod

logger = logging.getLogger(__name__)

_SCENE_SERVICE_NAMES = ("renderer", "physics", "trafficsim", "runtime")


def _split_volume_spec(volume: str) -> tuple[str, str, str | None]:
    parts = volume.split(":", maxsplit=2)
    if len(parts) < 2:
        raise ValueError(f"Invalid service volume mount: {volume!r}")
    container = str(PurePosixPath(posixpath.normpath(parts[1])))
    options = parts[2] if len(parts) == 3 else None
    return parts[0], container, options


def _container_mounts_overlap(first: str, second: str) -> bool:
    first_path = PurePosixPath(first)
    second_path = PurePosixPath(second)
    return first_path.is_relative_to(second_path) or second_path.is_relative_to(
        first_path
    )


def _is_read_only(options: str | None) -> bool:
    return options is not None and "ro" in options.split(",")


def _mount_external_scene_artifacts(
    cfg: AlpasimConfig,
    manager: USDZManager,
    uuids: list[str],
) -> list[str]:
    """Expose selected USDZ symlink targets to scene-consuming services."""
    reeval_container_cache = "/mnt/nre-data"
    external_mounts_by_container_cache = {
        reeval_container_cache: manager.external_artifact_mounts(
            uuids,
            reeval_container_cache,
        )
    }
    scene_cache = os.path.abspath(str(cfg.scenes.scene_cache))
    services_config = getattr(cfg, "services", None)
    if services_config is None:
        return [
            f"{host}:{container}:ro"
            for host, container in external_mounts_by_container_cache[
                reeval_container_cache
            ]
        ]

    services_by_container_cache: dict[str, list[ContainerConfig]] = {}
    for service_name in _SCENE_SERVICE_NAMES:
        service = getattr(services_config, service_name)
        if service is None:
            continue
        container_caches = {
            container
            for volume in service.volumes
            for host, container, _ in [_split_volume_spec(volume)]
            if os.path.abspath(host) == scene_cache
        }
        if len(container_caches) > 1:
            raise ValueError(
                f"Service {service_name!r} mounts the scene cache more than once: "
                f"{sorted(container_caches)}"
            )
        for container_cache in container_caches:
            services_by_container_cache.setdefault(container_cache, []).append(service)

    for container_cache, services in services_by_container_cache.items():
        if container_cache not in external_mounts_by_container_cache:
            external_mounts_by_container_cache[container_cache] = (
                manager.external_artifact_mounts(uuids, container_cache)
            )
        external_mounts = external_mounts_by_container_cache[container_cache]
        for service in services:
            existing_mounts = [_split_volume_spec(volume) for volume in service.volumes]
            for host, container in external_mounts:
                for existing_host, existing_container, options in existing_mounts:
                    if existing_container == container:
                        if Path(existing_host).resolve() != Path(host).resolve():
                            raise ValueError(
                                "USDZ symlink target mount conflicts with an existing "
                                f"service mount at {container}: {existing_host} versus "
                                f"{host}"
                            )
                        if not _is_read_only(options):
                            raise ValueError(
                                "Existing USDZ symlink target mount must be read-only: "
                                f"{existing_host}:{existing_container}"
                            )
                        break
                    if _container_mounts_overlap(existing_container, container):
                        raise ValueError(
                            "USDZ symlink target mount overlaps an existing service "
                            f"mount: {existing_container} and {container}"
                        )
                else:
                    service.volumes.append(f"{host}:{container}:ro")
                    existing_mounts.append((host, container, "ro"))
                    logger.info(
                        "Mounted external USDZ directory read-only: %s -> %s",
                        host,
                        container,
                    )

    return [
        f"{host}:{container}:ro"
        for host, container in external_mounts_by_container_cache[
            reeval_container_cache
        ]
    ]


def fetch_scenes(
    cfg: AlpasimConfig,
) -> tuple[SceneMaterialization, dict[str, tuple[Scene, ...]]]:
    """Materialize config-feasible scenes and stage accepted artifacts once."""
    Path(cfg.scenes.scene_cache).mkdir(parents=True, exist_ok=True)
    manager = USDZManager.from_cfg(cfg.scenes)
    scenes, requested_groups = resolve_scene_groups(cfg.scenes, manager)
    simulation_config = cfg.runtime.simulation_config
    materialization = materialize_scene_selection(
        scenes,
        manager,
        force_gt_duration_us=int(simulation_config.force_gt_duration_us),
        n_sim_steps=int(simulation_config.n_sim_steps),
        control_timestep_us=int(simulation_config.control_timestep_us),
    )
    logger.info(
        "Scene materialization: requested=%d accepted=%d rejected=%d "
        "invalid_pct=%.2f%%",
        materialization.requested_count,
        materialization.accepted_count,
        materialization.rejected_count,
        materialization.invalid_percent,
    )
    logger.info(
        "Scene materialization rejections: insufficient_history=%d "
        "insufficient_horizon=%d",
        materialization.rejection_counts["insufficient_history"],
        materialization.rejection_counts["insufficient_horizon"],
    )
    logger.info(
        "Accepted scene population: sha256=%s",
        materialization.accepted_sha256,
    )
    if materialization.requested_count > 0 and materialization.accepted_count == 0:
        raise ValueError(
            "All selected scenes are invalid for the simulation timing config: "
            f"requested={materialization.requested_count}, "
            f"rejections={materialization.rejection_counts}"
        )

    accepted_scene_keys = {scene.scene_key for scene in materialization.accepted_scenes}
    accepted_groups = {
        name: tuple(scene for scene in group if scene.scene_key in accepted_scene_keys)
        for name, group in requested_groups.items()
    }
    for name, group in accepted_groups.items():
        if not group:
            raise ValueError(f"Scene suite group {name!r} has no valid scenes")

    cfg.scenes.external_artifact_mounts = []

    # Create sceneset directory if not using local USDZ directory
    if cfg.scenes.local_usdz_dir is not None:
        sceneset_dir_abs_path = os.path.abspath(str(cfg.scenes.local_usdz_dir))
        sceneset_dir_relative_path = "."
        # Note: for local USDZ directories, override the scene cache directory for proper mounting
        cfg.scenes.scene_cache = sceneset_dir_abs_path
        logger.info(
            f"Using local files--overriding scene_cache to: {sceneset_dir_abs_path}"
        )
    else:
        accepted_uuids = sorted(
            {
                cast(ClipDataSource, scene.source).usdz_uuid
                for scene in materialization.accepted_scenes
            }
        )
        sceneset_dir_abs_path = manager.create_sceneset_directory(accepted_uuids)
        sceneset_dir_relative_path = str(
            Path(sceneset_dir_abs_path).relative_to(cfg.scenes.scene_cache)
        )
        logger.info(f"Relative sceneset path: {sceneset_dir_relative_path}")
        cfg.scenes.external_artifact_mounts = _mount_external_scene_artifacts(
            cfg,
            manager,
            accepted_uuids,
        )

    cfg.scenes.sceneset_path = sceneset_dir_relative_path
    return materialization, accepted_groups


def detect_gpus() -> int:
    """Detect number of GPUs on the system."""
    try:
        num_gpus = int(
            subprocess.check_output(
                "nvidia-smi -i 0 --query-gpu=count --format=csv,noheader",
                shell=True,
            )
        )
        logger.debug(f"Found {num_gpus} GPUs on system.")
    except subprocess.CalledProcessError as exc:
        logger.warning(
            f"Failed to determine GPU count via 'nvidia-smi' (code {exc.returncode}). "
            "Defaulting to 0 GPUs."
        )
        return 0
    return num_gpus


def discover_gpu_allocation(run_method: RunMethod) -> tuple[int, ...]:
    """Return the ordered devices available to this Wizard invocation."""
    if run_method.is_slurm:
        # srun exposes a step allocation; direct sbatch execution only exposes the job.
        devices = os.environ.get("SLURM_STEP_GPUS") or os.environ["SLURM_JOB_GPUS"]
        allocation = tuple(int(device) for device in devices.split(","))
    else:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES cannot select GPUs for a local Wizard run; "
                "unset it and use services.*.gpus with host device slots"
            )
        allocation = tuple(range(detect_gpus()))

    logger.info("GPUs available to Wizard: %s", allocation)
    return allocation


def setup_directories(cfg: AlpasimConfig) -> None:
    """Create necessary directories and symlinks."""
    log_dir = Path(cfg.wizard.log_dir)

    logger.debug(f"Creating log directory at path: {log_dir}")

    # Create subdirectories
    for subdir in (
        "rollouts",
        "txt-logs",
        "controller",
        "prometheus",
    ):
        subdir_path = log_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True, mode=0o777)
        os.chmod(subdir_path, 0o777)


@dataclass
class TelemetryPorts:
    """Ports allocated together for telemetry services."""

    workers: tuple[int, ...]
    prometheus: int
    node_exporter: int
    process_exporter: int
    dcgm_exporter: int

    def prometheus_service_ports(self) -> dict[str, int]:
        return {
            "prometheus": self.prometheus,
            "node_exporter": self.node_exporter,
            "process_exporter": self.process_exporter,
            "dcgm_exporter": self.dcgm_exporter,
        }

    def runtime_worker_ports(self) -> dict[str, int]:
        return {f"runtime_worker_{idx}": port for idx, port in enumerate(self.workers)}


@dataclass
class WizardContext:
    """Unified context for all wizard operations.

    Combines configuration access with runtime state,
    eliminating the need for a separate GlobalContext.
    """

    cfg: AlpasimConfig
    port_assigner: Iterator[int]
    telemetry_ports: TelemetryPorts
    gpu_allocation: tuple[int, ...]

    # Expensive operations (only loaded when needed for actual execution)
    scene_list: list[Scene] = field(default_factory=list)
    scene_materialization: SceneMaterialization | None = None
    scene_groups: dict[str, tuple[Scene, ...]] = field(default_factory=dict)

    def resolve_gpu_slot(self, slot: int) -> int:
        """Resolve an allocation-relative topology slot to a device ID."""
        if not 0 <= slot < len(self.gpu_allocation):
            raise RuntimeError(
                f"GPU slot {slot} is outside the Wizard's "
                f"{len(self.gpu_allocation)}-GPU allocation"
            )
        return self.gpu_allocation[slot]

    @property
    def all_services_to_run(self) -> list[str]:
        """Get all services that should be run."""
        return list(self.cfg.wizard.run_sim_services or [])

    @staticmethod
    def create(cfg: AlpasimConfig) -> WizardContext:
        """Build context."""

        gpu_allocation = discover_gpu_allocation(cfg.wizard.run_method)
        # Always set these basic attributes
        scene_materialization, scene_groups = fetch_scenes(cfg)
        port_assigner = create_port_assigner(cfg.wizard.baseport)
        nr_workers = int(cfg.runtime.nr_workers)
        # We preallocate them so they are consistent across call sites
        telemetry_ports = TelemetryPorts(
            workers=tuple(next(port_assigner) for _ in range(nr_workers)),
            prometheus=next(port_assigner),
            node_exporter=next(port_assigner),
            process_exporter=next(port_assigner),
            dcgm_exporter=next(port_assigner),
        )
        context = WizardContext(
            cfg=cfg,
            port_assigner=port_assigner,
            telemetry_ports=telemetry_ports,
            gpu_allocation=gpu_allocation,
            scene_list=list(scene_materialization.accepted_scenes),
            scene_materialization=scene_materialization,
            scene_groups=scene_groups,
        )
        if cfg.wizard.prometheus.start_prometheus:
            logger.info(
                "Prometheus UI: http://localhost:%d",
                telemetry_ports.prometheus,
            )
        logger.info(
            "Prometheus file-SD dir: %s",
            cfg.wizard.prometheus.file_sd_dir,
        )
        setup_directories(cfg)
        if cfg.ego_hoods.carlines is not None:
            generate_ego_hood_banks(
                list(cfg.ego_hoods.transforms),
                cfg.ego_hoods.carlines,
                cfg.ego_hoods.banks,
                _usdz_paths(cfg, slurm_array_task_scenes(context.scene_list)),
                os.path.join(cfg.wizard.log_dir, "ego_mask_overrides.yaml"),
            )

        return context


def slurm_array_task_scenes(scenes: list[Scene]) -> list[Scene]:
    """The scenes this Slurm array task runs: all scenes, sorted, dealt round-robin to the tasks."""
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0))
    if task_count <= 1:
        return scenes
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) - int(
        os.environ.get("SLURM_ARRAY_TASK_MIN", 0)
    )
    ordered = sorted(
        scenes,
        key=lambda scene: (
            cast(ClipDataSource, scene.source).clip_id,
            scene.policy_start_us == "AUTO",
            0 if scene.policy_start_us == "AUTO" else scene.policy_start_us,
        ),
    )
    return ordered[task::task_count]


def _usdz_paths(cfg: AlpasimConfig, scenes: list[Scene]) -> dict[str, str]:
    """Clip ID -> local USDZ path for the accepted scenes (after fetch_scenes staged them)."""
    uuids = {
        cast(ClipDataSource, s.source).clip_id: cast(ClipDataSource, s.source).usdz_uuid
        for s in scenes
    }
    if cfg.scenes.local_usdz_dir is not None:
        catalog = scan_local_usdz_directory(cfg.scenes.local_usdz_dir)
        path = dict(zip(catalog["uuid"].to_list(), catalog["path"].to_list()))
        return {clip: path[uuid] for clip, uuid in uuids.items()}
    return {
        clip: os.path.join(
            cfg.scenes.scene_cache, USDZManager.ALL_USDZ_DIR_NAME, f"{uuid}.usdz"
        )
        for clip, uuid in uuids.items()
    }


def _is_port_open(port: int) -> bool:
    """Check if a port is available (not in use)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def _find_next_open_port(start_port: int, max_search: int = 1000) -> int:
    """Find the next available port starting from start_port.

    Args:
        start_port: Port number to start searching from
        max_search: Maximum number of ports to check

    Returns:
        The next available port number

    Raises:
        RuntimeError: If no available port is found within the search range
    """
    for port in range(start_port, start_port + max_search):
        if _is_port_open(port):
            logger.debug(f"Found available port: {port}")
            return port
    raise RuntimeError(
        f"Could not find an available port in range {start_port}-{start_port + max_search - 1}"
    )


def create_port_assigner(baseport: int) -> Iterator[int]:
    """Create an iterator over port numbers starting from the first available port >= baseport."""

    def port_assigner() -> Iterator[int]:
        ports_assigned = 0
        max_ports = 1000
        # Start from the first available port >= baseport
        next_port = _find_next_open_port(baseport)

        while ports_assigned < max_ports:
            yield next_port
            ports_assigned += 1
            # Find the next available port after the one we just yielded
            next_port = _find_next_open_port(next_port + 1)

        raise AssertionError(
            f"Handed out {max_ports} different port numbers - something's fishy."
        )

    return port_assigner()
