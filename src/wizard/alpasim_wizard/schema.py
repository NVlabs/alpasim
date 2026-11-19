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
class EgoHoodsConfig:
    """Ego hood banks the renderer mounts, and optional per-scene re-rendering (see ego_hoods.py)."""

    # Directory of banks, mounted at /mnt/ego-hoods
    banks: str = MISSING
    # Clip ID -> carline bank map; when set, the wizard writes re-rendered banks to `banks`
    carlines: str | None = None
    # Transform specs (source image and camera per carline, plus stock banks), combined
    transforms: list[str] = field(default_factory=list)


@dataclass
class AlpasimConfig:
    defines: dict[str, str] = MISSING
    ego_hoods: EgoHoodsConfig = MISSING
    wizard: WizardConfig = MISSING
    scenes: ScenesConfig = MISSING
    services: ServicesConfig = MISSING
    runtime: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    trafficsim: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    eval: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    driver: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    controller: DictConfig = field(default_factory=lambda: OmegaConf.create({}))


@dataclass
class SceneSuiteConfig:
    """One configured scene-suite CSV."""

    csv_path: str = MISSING
    t0_offset_us: int = 0


@dataclass
class ScenesConfig:
    # Selection method. Suites are exclusive; local mode may use clip_ids as a filter.
    clip_ids: list[str] | None = None
    suite_ids: list[str] | None = None
    suite_configs: list[SceneSuiteConfig] | None = None
    suite_groups: dict[str, list[SceneSuiteConfig]] = field(default_factory=dict)

    # Clip IDs to remove from the selected suites or explicit clip list
    exclude_clip_ids: list[str] = field(default_factory=list)

    # Limit the number of scenes to run (0 or negative means no limit)
    limit_to_first_n: int = 0

    # Paths
    scene_cache: str = MISSING
    clip_catalog_csvs: list[str] = MISSING
    scene_suite_dirs: list[str] = MISSING

    # Relative path within scene_cache to the sceneset directory for this run.
    # Set automatically by the wizard; used by reeval to locate the correct USDZs.
    sceneset_path: str | None = None

    # Read-only mounts needed to resolve external USDZ symlink targets during reeval.
    external_artifact_mounts: list[str] = field(default_factory=list)

    # Optional: path to a local directory containing *.usdz files.
    # When set, the wizard will scan this directory to generate in-memory
    # clip catalog data, bypassing the configured catalog CSVs. If local_usdz_dir
    # is provided without clip_ids, all discovered clips are selected.
    local_usdz_dir: str | None = None

    # Used to override services.renderer.image for the USDZ database service if NRE is not enabled.
    nre_version_string: str | None = None


class RunMethod(Enum):
    SLURM = "slurm"
    SLURM_ENROOT = "slurm_enroot"
    DOCKER_COMPOSE = "docker_compose"
    NONE = "none"

    @property
    def is_slurm(self) -> bool:
        return self in (RunMethod.SLURM, RunMethod.SLURM_ENROOT)


class RunMode(Enum):
    """Runtime lifecycle mode.

    ONESHOT starts the runtime to execute the generated simulation once and
    then exit. SERVER starts a long-running runtime daemon that serves
    simulation requests over gRPC.
    """

    ONESHOT = "oneshot"
    SERVER = "server"


@dataclass
class WizardPrometheusConfig:
    """Prometheus ownership, identity, and discovery settings."""

    scrape_interval: str = "5s"
    file_sd_dir: str | None = None
    start_prometheus: bool = True
    run_uuid: str | None = None


@dataclass
class WizardConfig:
    # Name of the run, used to identify the run in the databases.
    run_name: str | None = None
    run_method: RunMethod = MISSING
    run_mode: RunMode = MISSING
    # Optional slots assigned to this Wizard from the GPUs available to its process.
    gpu_slots: list[int] | None = None

    # Global log level for all alpasim services (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    prometheus: WizardPrometheusConfig = field(default_factory=WizardPrometheusConfig)
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

    # Root directory where the wizard installs direct-Enroot FUSE tools.
    # Required with run_method=SLURM_ENROOT.
    fuse_dir: str | None = None

    # Optional persistent node-local cache for direct-Enroot squash images.
    node_local_sqsh_cache_dir: str | None = None
    node_local_sqsh_cache_max_gib: int = 500

    # Run GPU services through CUDA MPS so kernels from co-located processes
    # execute concurrently instead of time-slicing between CUDA contexts.
    # Starts a per-job MPS control daemon on the node as the Wizard starts.
    # Slurm run methods only.
    enable_mps: bool = False
    # Pipe directory of that daemon; defaults to /tmp/nvidia-mps-<job id>. Set it
    # when processes outside the Wizard join the daemon. Unix socket paths are
    # capped at ~108 characters, so keep it short and node-local.
    mps_pipe_dir: str | None = None

    # External service addresses for services running outside the deployment.
    # Maps service name to list of addresses (e.g., {"driver": ["localhost:6789"]}).
    # These addresses are added to generated-network-config.yaml so the runtime
    # can connect to services running externally (e.g., on developer's machine).
    external_services: dict[str, list[str]] | None = None


@dataclass
class ServicesConfig:
    driver: ServiceConfig | None = MISSING
    renderer: ServiceConfig | None = MISSING
    physics: ServiceConfig | None = MISSING
    trafficsim: ServiceConfig | None = MISSING
    controller: ServiceConfig | None = MISSING
    frame_quality: ServiceConfig | None = None
    runtime: RuntimeServiceConfig = MISSING
    prometheus: ContainerConfig = MISSING


@dataclass
class ContainerConfig:
    volumes: list[str] = field(default_factory=list)
    image: str = MISSING
    # Images that don't correspond to a service in the repo.
    # No Dockerfile path is added to the docker-compose.yaml.
    external_image: bool = False
    pull_policy: str = "missing"
    environments: list[str] = field(default_factory=list)
    workdir: str | None = None
    remap_root: bool = False


@dataclass
class ServiceConfig(ContainerConfig):
    command: list[str] = MISSING
    # Number of service replicas to run per container.
    # If gpus is None or empty, creates a single container with this many replicas.
    # If gpus is specified, creates one container per entry, each with this many replicas.
    replicas_per_container: int = MISSING
    # GPU slots relative to the Wizard's allocation (one container per entry).
    # A slot may appear multiple times to run multiple containers on the same GPU,
    # e.g. gpus: [0, 0, 1, 1] creates four containers, two on each GPU.
    gpus: list[int] | None = MISSING


@dataclass
class RuntimeServiceConfig(ServiceConfig):
    depends_on: list[str] = MISSING
