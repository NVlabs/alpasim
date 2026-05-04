# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class DebugFlags:
    """Flags or settings purely for developing or debugging.

    All must be entirely optional and cannot be used in production.
    Even the existence of `debug_flags` in the config should be optional!
    """

    # Use `localhost` in `generated-network-config.yaml` and adds
    # `network_mode: host` to the docker-compose.yaml.
    # This allows combining running services and/or the runtime on the host and
    # in containers. Very helpful for debugging.
    use_localhost: bool = False


@dataclass
class AlpasimConfig:
    defines: dict[str, str] = MISSING
    wizard: WizardConfig = MISSING
    scenes: ScenesConfig = MISSING
    services: ServicesConfig = MISSING
    runtime: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    trafficsim: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    eval: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    driver: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    controller: DictConfig = field(default_factory=lambda: OmegaConf.create({}))


@dataclass
class ScenesConfig:
    # Selection method (exactly one must be set)
    scene_ids: list[str] | None = None
    test_suite_id: str | None = None

    # Limit the number of scenes to run (0 or negative means no limit)
    limit_to_first_n: int = 0

    # Paths
    scene_cache: str = MISSING
    scenes_csv: list[str] = MISSING
    suites_csv: list[str] = MISSING

    # Relative path within scene_cache to the sceneset directory for this run.
    # Set automatically by the wizard; used by reeval to locate the correct USDZs.
    sceneset_path: str | None = None

    # Optional: path to a local directory containing *.usdz files.
    # When set, the wizard will scan this directory to generate in-memory
    # sim_scenes/sim_suites data, bypassing the CSV files. A test suite (called "local")
    # is created automatically containing all discovered scenes.
    # If local_usdz_dir is provided and neither scene_ids nor test_suite_id is set,
    # all scenes in the directory will be simulated.
    local_usdz_dir: str | None = None

    # used to override services.sensorsim.image for the USDZ database service if NRE is not enabled
    nre_version_string: str | None = None


class RunMethod(Enum):
    SLURM = "slurm"
    DOCKER_COMPOSE = "docker_compose"
    NONE = "none"


class RunMode(Enum):
    """Runtime lifecycle mode.

    ONESHOT starts the runtime to execute the generated simulation once and
    then exit. SERVER starts a long-running runtime daemon that serves
    simulation requests over gRPC.
    """

    ONESHOT = "oneshot"
    SERVER = "server"


@dataclass
class WizardConfig:
    # Name of the run, used to identify the run in the databases.
    run_name: str | None = None
    run_method: RunMethod = MISSING
    run_mode: RunMode = MISSING

    # Global log level for all alpasim services (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    description: str | None = None  # TODO(mwatson): is this redundant to run_name?
    submitter: str | None = None

    latest_symlink: bool = MISSING
    log_dir: str = "."
    array_job_dir: str | None = None
    dry_run: bool = MISSING
    baseport: int = MISSING
    runtime_server_port: int | None = None
    validate_mount_points: bool = MISSING

    # If set, the wizard will pull the driver code from the specified hash into
    # `${wizard.log_dir}/driver_code`. Can be useful for mounting into the
    # driver container for debugging.
    driver_code_hash: str | None = None

    # Used if `driver_code_hash` is set. Requires configured ssh keys for
    # pulling from gitlab, but can also point towards a local repo!
    driver_code_repo: str | None = None

    helper: str = MISSING
    vscode: str = MISSING

    sqshcaches: list[str] = MISSING

    slurm_job_id: int | None = MISSING
    timeout: int = MISSING
    nr_retries: int = 3
    run_sim_services: list[str] | None = MISSING
    debug_flags: DebugFlags = field(default_factory=DebugFlags)

    # When True, add --cpu-bind=none to srun --overlap steps.  Required on
    # SLURM nodes where the batch step binds all CPUs (e.g. non-exclusive
    # allocations on CI nodes), otherwise overlapping steps are killed.
    slurm_cpu_bind_none: bool = False

    # External service addresses for services running outside the deployment.
    # Maps service name to list of addresses (e.g., {"driver": ["localhost:6789"]}).
    # These addresses are added to generated-network-config.yaml so the runtime
    # can connect to services running externally (e.g., on developer's machine).
    external_services: dict[str, list[str]] | None = None


@dataclass
class ServicesConfig:
    driver: ServiceConfig | None = MISSING
    sensorsim: ServiceConfig | None = MISSING
    physics: ServiceConfig | None = MISSING
    trafficsim: ServiceConfig | None = MISSING
    controller: ServiceConfig | None = MISSING
    runtime: RuntimeServiceConfig = MISSING


@dataclass
class ServiceConfig:
    volumes: list[str] = field(default_factory=list)
    image: str = MISSING
    # Images that don't correspond to a service in the repo.
    # No Dockerfile path is added to the docker-compose.yaml.
    external_image: bool = False
    command: list[str] = MISSING
    # Number of service replicas to run per container.
    # If gpus is None or empty, creates a single container with this many replicas.
    # If gpus is specified, creates one container per GPU, each with this many replicas.
    replicas_per_container: int = MISSING
    gpus: list[int] | None = MISSING

    environments: list[str] = field(default_factory=list)
    workdir: str | None = None
    remap_root: bool = False


@dataclass
class RuntimeServiceConfig(ServiceConfig):
    depends_on: list[str] = MISSING
