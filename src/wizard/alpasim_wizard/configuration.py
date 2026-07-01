# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Configuration manager for unified config generation."""

from __future__ import annotations

import datetime
import logging
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, List, cast

from alpasim_wizard.context import WizardContext
from alpasim_wizard.schema import AlpasimConfig, RunMode
from alpasim_wizard.telemetry.prometheus import generate_prometheus_configs
from omegaconf import OmegaConf

from .services import ContainerDefinition, ContainerSet
from .utils import read_yaml, save_loadable_wizard_config, write_yaml

logger = logging.getLogger(__name__)

CORE_SERVICE_NAMES = (
    "driver",
    "renderer",
    "physics",
    "trafficsim",
    "controller",
)


class ConfigurationManager:
    """Manages all configuration generation and writing."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self._central_file_sd_path: Path | None = None

    def generate_all(self, container_set: ContainerSet, context: WizardContext) -> None:
        """Generate all required configurations.

        Args:
            container_set: ContainerSet with all services
            context: WizardContext containing configuration and artifacts
        """
        logger.info("Generating all configurations...")

        cfg = context.cfg
        artifact_list = context.artifact_list

        run_metadata = self._load_or_create_run_metadata(cfg)
        self._write_config("run_metadata.yaml", run_metadata)

        # Generate each configuration
        self._generate_runtime_config(cfg, artifact_list, context)

        # Get sim containers from service_manager for network config
        sim_containers = container_set.sim
        self._generate_network_config(sim_containers, cfg)
        self._generate_runtime_server_config(container_set, cfg)

        self._generate_trafficsim_config(cfg)
        self._generate_eval_config(cfg)
        self._central_file_sd_path = generate_prometheus_configs(
            self.log_dir,
            run_metadata,
            context,
        )
        self._generate_driver_config(cfg)
        self._generate_controller_config(cfg)

        # Save wizard config
        self._save_wizard_config(cfg)

        logger.info("Generated configuration files")

    def _generate_runtime_config(
        self,
        cfg: Any,
        artifact_list: List[Any],
        context: WizardContext,
    ) -> str | None:
        """Generate runtime configuration."""
        runtime_config = OmegaConf.to_container(cfg.runtime, resolve=True)
        runtime_config = self._remove_none_values(runtime_config)
        assert isinstance(runtime_config, dict)

        sceneset_path = getattr(getattr(cfg, "scenes", None), "sceneset_path", None)
        if sceneset_path is not None:
            scene_provider = runtime_config.get("scene_provider")
            if (
                isinstance(scene_provider, dict)
                and scene_provider.get("kind") == "usdz"
                and isinstance(scene_provider.get("usdz"), dict)
            ):
                scene_provider["usdz"]["data_dir"] = (
                    "/mnt/nre-data"
                    if sceneset_path == "."
                    else f"/mnt/nre-data/{sceneset_path}"
                )

        # Write simulation params directly (was: fan out per scene)
        simulation_config = runtime_config.pop("simulation_config", {})
        telemetry_ports = context.telemetry_ports
        prometheus_host = (
            "localhost"
            if cfg.wizard.debug_flags.use_localhost
            or cfg.wizard.run_method.name == "SLURM"
            else "prometheus-0"
        )
        runtime_config["simulation_config"] = simulation_config
        runtime_config["prometheus"] = {
            "worker_ports": list(telemetry_ports.workers),
            "url": f"http://{prometheus_host}:{telemetry_ports.prometheus}",
        }

        # Write flat scene list
        runtime_config["scenes"] = [{"scene_id": s.scene_id} for s in artifact_list]

        runtime_config = self._maybe_split_user_config_for_slurm_array(runtime_config)

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        filename = f"generated-user-config-{task_id}.yaml"
        self._write_config(filename, runtime_config)

        logger.debug(f"Generated runtime config: {filename}")
        return filename

    def _generate_network_config(
        self,
        service_containers: List[ContainerDefinition],
        cfg: AlpasimConfig,
    ) -> None:
        """Generate network configuration for service discovery.

        Args:
            service_containers: List of container definitions from which to extract addresses.
            cfg: AlpasimConfig containing wizard settings including external_services.
        """

        network_config: Dict[str, Any] = {
            "driver": {"endpoints": []},
            "physics": {"endpoints": []},
            "renderer": {"endpoints": []},
            "trafficsim": {"endpoints": []},
            "controller": {"endpoints": []},
        }

        for c in service_containers:
            for inst in c.service_instances:
                # A special configuration has been requested, where the renderer and
                # physics service exist in the same process/at the same port. This logical
                # branch handles that mapping.
                if c.name == "physics" and inst.service_config.image == "*renderer*":
                    logger.info("Mapping the physics service to renderer addresses")
                    renderer_containers = [
                        sc for sc in service_containers if sc.name == "renderer"
                    ]
                    if (len(renderer_containers) != 1) or (
                        len(renderer_containers[0].get_all_addresses()) != 1
                    ):
                        raise ValueError(
                            "Expected exactly one renderer container/address"
                        )
                    renderer_address = renderer_containers[0].get_all_addresses()[0]
                    if inst.address is None:
                        raise ValueError("Physics service must have an address defined")
                    inst.address.host = renderer_address.host
                    inst.address.port = renderer_address.port
                    logger.info("Mapped physics to renderer at %s", inst.address)

                elif inst.address is None:
                    continue

                network_service_name = c.name
                if network_service_name in network_config:
                    address = str(inst.address)
                    network_config[network_service_name]["endpoints"].append(
                        {"address": address, "managed": True}
                    )

        # Add external service addresses (for services running outside the deployment).
        # Unknown service names are routed into extra_services so plugin-owned
        # endpoints stay out of the core public schema.
        external_services_raw: Any = cfg.wizard.external_services
        if external_services_raw is not None and OmegaConf.is_config(
            external_services_raw
        ):
            external_services_raw = OmegaConf.to_container(
                external_services_raw,
                resolve=True,
            )
        if external_services_raw is not None:
            external_services = cast(dict[str, list[str]], external_services_raw)
            for service_name, addresses in external_services.items():
                if service_name not in CORE_SERVICE_NAMES:
                    raise ValueError(
                        f"Unknown external service {service_name!r}; expected one of "
                        f"{CORE_SERVICE_NAMES}"
                    )
                target = network_config[service_name]
                target["endpoints"].extend(
                    {"address": address, "managed": False} for address in addresses
                )
                logger.info("Added external %s addresses: %s", service_name, addresses)

        self._write_config("generated-network-config.yaml", network_config)
        logger.debug("Generated network config")

    def _generate_runtime_server_config(
        self,
        container_set: ContainerSet,
        cfg: AlpasimConfig,
    ) -> None:
        """Generate client-facing runtime daemon endpoint metadata.

        A wildcard runtime address means the runtime binds on the deployment
        node, so this process' hostname is used as the client endpoint. Other
        current backends store internal service names here and publish the
        runtime on localhost.
        """
        if cfg.wizard.run_mode != RunMode.SERVER:
            return

        runtime_container = container_set.runtime
        if runtime_container is None:
            raise ValueError(
                "Server mode requires `runtime` in wizard.run_sim_services"
            )

        runtime_addresses = runtime_container.get_all_addresses()
        if not runtime_addresses:
            raise ValueError("Runtime server mode requires a runtime address")

        runtime_address = runtime_addresses[0]
        client_host = (
            socket.gethostname() if runtime_address.host == "0.0.0.0" else "localhost"
        )
        endpoint = {
            "host": client_host,
            "port": runtime_address.port,
        }
        self._write_config("generated-runtime-server.yaml", endpoint)
        logger.debug("Generated runtime server endpoint")

    def _generate_trafficsim_config(self, cfg: Any) -> None:
        """Generate traffic simulation configuration."""
        if not hasattr(cfg, "trafficsim"):
            return

        trafficsim_config = OmegaConf.to_container(cfg.trafficsim, resolve=True)
        assert isinstance(trafficsim_config, dict)

        self._write_config("trafficsim-config.yaml", trafficsim_config)
        logger.debug("Generated trafficsim config")

    def _generate_eval_config(self, cfg: Any) -> None:
        """Generate evaluation configuration."""
        if not hasattr(cfg, "eval"):
            return

        eval_config = OmegaConf.to_container(cfg.eval, resolve=True)
        assert isinstance(eval_config, dict)

        self._write_config("eval-config.yaml", eval_config)
        logger.debug("Generated eval config")

    def _generate_driver_config(self, cfg: Any) -> None:
        """Generate driver configuration."""
        if not hasattr(cfg, "driver"):
            return

        driver_config = OmegaConf.to_container(cfg.driver, resolve=True)
        assert isinstance(driver_config, dict)

        self._write_config("driver-config.yaml", driver_config)
        logger.debug("Generated driver config")

    def _generate_controller_config(self, cfg: Any) -> None:
        """Generate controller configuration."""
        if not hasattr(cfg, "controller"):
            return

        controller_config = OmegaConf.to_container(cfg.controller, resolve=True)
        assert isinstance(controller_config, dict)

        self._write_config("controller-config.yaml", controller_config)
        logger.debug("Generated controller config")

    def _load_or_create_run_metadata(self, cfg: Any) -> dict[str, Any]:
        """Load existing run metadata, or create it for a new run.

        Makes sure e.g. run_uuid stays the same across restarts of the same run.
        """
        metadata_path = self.log_dir / "run_metadata.yaml"
        if metadata_path.exists():
            run_metadata = read_yaml(str(metadata_path))
            run_metadata.setdefault("run_uuid", str(uuid.uuid4()))
            return run_metadata

        run_uuid = uuid.uuid4()
        run_name = (
            cfg.wizard.run_name
            or os.environ.get("SLURM_JOB_NAME", None)
            or f"LR-{run_uuid}"
        )

        run_metadata = {
            "run_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": run_name,
            "run_uuid": str(run_uuid),
            "slurm_job_id": cfg.wizard.slurm_job_id,
            "run_user": str(os.environ.get("USER", "unknownUser")),
            "run_dir": str(os.environ.get("PWD", "unknownDir")),
            "run_args": str(os.environ.get("SLURM_JOB_ARGS", "unknownArgs")),
            "submitter": (
                cfg.wizard.submitter if hasattr(cfg.wizard, "submitter") else None
            ),
            "description": (
                cfg.wizard.description if hasattr(cfg.wizard, "description") else None
            ),
            "test_suite_id": (
                cfg.scenes.test_suite_id
                if hasattr(cfg.scenes, "test_suite_id")
                else None
            ),
        }
        return run_metadata

    def _save_wizard_config(self, cfg: Any) -> None:
        """Save the complete wizard configuration."""
        # Save resolved config
        wizard_config_path = self.log_dir / "wizard-config.yaml"
        with open(wizard_config_path, "w") as cfg_file:
            OmegaConf.save(cfg, f=cfg_file, resolve=True)

        # Save loadable config
        wizard_config_path_loadable = self.log_dir / "wizard-config-loadable.yaml"
        save_loadable_wizard_config(cfg, str(wizard_config_path_loadable))

        logger.debug("Saved wizard configurations")

    def _write_config(self, filename: str, data: Dict) -> Path:
        """Write configuration to file."""
        filepath = self.log_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        write_yaml(data, str(filepath))
        return filepath

    def cleanup_central_file_sd(self) -> None:
        """Remove this run's central file-SD publication after deployment exits."""
        if self._central_file_sd_path is None:
            return
        try:
            self._central_file_sd_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to remove Prometheus file-SD %s: %s",
                self._central_file_sd_path,
                exc,
            )

    def _remove_none_values(self, d: Any) -> Any:
        """Recursively remove all keys with None values from the dictionary."""
        if not isinstance(d, dict):
            return d
        return {k: self._remove_none_values(v) for k, v in d.items() if v is not None}

    def _maybe_split_user_config_for_slurm_array(self, user_config: Any) -> Any:
        """Split scenes for SLURM array jobs."""
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0))

        if task_count <= 1:
            return user_config

        logger.info(
            f"Detected SLURM_ARRAY_TASK_COUNT = {task_count}, splitting user-config"
        )
        user_config = user_config.copy()

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        min_task_id = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))

        all_scenes = user_config["scenes"]
        # Sort for deterministic distribution
        all_scenes = sorted(all_scenes, key=lambda x: (x.get("scene_id", ""), str(x)))

        # Distribute scenes across array tasks (round-robin)
        split_scenes: List[List[Any]] = [[] for _ in range(task_count)]
        for idx, scene in enumerate(all_scenes):
            split_scenes[idx % task_count].append(scene)

        user_config["scenes"] = split_scenes[task_id - min_task_id]
        return user_config

    def get_runtime_config_name(self) -> str:
        """Get the runtime configuration filename."""
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        return f"generated-user-config-{task_id}.yaml"
