# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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

    # See avmf/src/alpasim_avmf/schema.py for the actual schema.
    avmf: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    eval: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    driver: DictConfig = field(default_factory=lambda: OmegaConf.create({}))


@dataclass
class ScenesConfig:
    # Selection method (exactly one must be set)
    scene_ids: Optional[list[str]] = None
    test_suite_id: Optional[str] = None

    # Paths
    scene_cache: str = MISSING
    scenes_csv: str = MISSING
    suites_csv: str = MISSING

    # Optional: path to a local directory containing *.usdz files.
    # When set, the wizard will scan this directory to generate in-memory
    # sim_scenes/sim_suites data, bypassing the CSV files. A test suite (called "local")
    # is created automatically containing all discovered scenes.
    # If local_usdz_dir is provided and neither scene_ids nor test_suite_id is set,
    # all scenes in the directory will be simulated.
    local_usdz_dir: Optional[str] = None

    # used to override services.sensorsim.image for the USDZ database service if NRE is not enabled
    nre_version_string: Optional[str] = None

    # maps (nre_version, artifact_version) -> is_compatible
    # 1. versions are version strings with dots replaced by _, i.e. 0_2_335-deadbeef
    #    underscores will be replaced by dots when used as keys in the config
    # 2. all versions are assumed to be compatible with themselves. Including
    #    `artifact_compatibility_matrix.0_2_335-deadbeef.0_2_335-deadbeef` is an error
    artifact_compatibility_matrix: dict[str, dict[str, bool]] = MISSING


class RunMethod(Enum):
    SLURM = "slurm"
    DOCKER_COMPOSE = "docker_compose"
    NONE = "none"


class RunMode(Enum):
    # TODO: delete in favor of hydra overrides for debugging
    BATCH = "batch"
    ATTACH_BASH = "attach_bash"
    ATTACH_VSCODE = "attach_vscode"


@dataclass
class WizardConfig:
    # Name of the run, used to identify the run in the databases.
    run_name: Optional[str] = None
    run_method: RunMethod = MISSING
    run_mode: RunMode = MISSING

    # Global log level for all alpasim services (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    description: Optional[str] = None  # TODO(mwatson): is this redundant to run_name?
    submitter: Optional[str] = None

    latest_symlink: bool = MISSING
    log_dir: str = "."
    array_job_dir: Optional[str] = None
    dry_run: bool = MISSING
    baseport: int = MISSING
    validate_mount_points: bool = MISSING

    # If set, the wizard will pull the driver code from the specified hash into
    # `${wizard.log_dir}/driver_code`. Can be useful for mounting into the
    # driver container for debugging.
    driver_code_hash: Optional[str] = None

    # Used if `driver_code_hash` is set. Requires configured ssh keys for
    # pulling from gitlab, but can also point towards a local repo!
    driver_code_repo: Optional[str] = None

    helper: str = MISSING
    vscode: str = MISSING

    sqshcaches: list[str] = MISSING

    slurm_job_id: Optional[int] = MISSING
    timeout: int = MISSING
    nr_retries: int = 3
    run_sim_services: Optional[list[str]] = MISSING
    run_eval_services: Optional[list[str]] = MISSING
    run_aggregation_services: Optional[list[str]] = MISSING
    debug_flags: DebugFlags = field(default_factory=DebugFlags)

    # Set GPU & CPU partition preferences for slurm runs.
    # Usually, one will not need to set these, but CI jobs do not
    # want to use interactive partitions.
    # For GPU partition, use SLURM_JOB_ACCOUNT if `None`
    slurm_gpu_partition: Optional[str] = None
    slurm_cpu_partition: str = "cpu_short"


@dataclass
class ServicesConfig:
    driver: Optional[ServiceConfig] = MISSING
    sensorsim: Optional[ServiceConfig] = MISSING
    physics: Optional[ServiceConfig] = MISSING
    trafficsim: Optional[ServiceConfig] = MISSING
    controller: Optional[ServiceConfig] = MISSING
    runtime: RuntimeServiceConfig = MISSING
    eval: Optional[ServiceConfig] = None
    post_eval_aggregation: Optional[ServiceConfig] = None


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
    gpus: Optional[list[int]] = MISSING

    environments: list[str] = field(default_factory=list)
    workdir: Optional[str] = None
    remap_root: bool = False


@dataclass
class RuntimeServiceConfig(ServiceConfig):
    depends_on: list[str] = MISSING
