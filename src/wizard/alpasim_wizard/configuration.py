# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Configuration manager for unified config generation."""

from __future__ import annotations

import datetime
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from alpasim_wizard.context import WizardContext
from alpasim_wizard.schema import AlpasimConfig
from omegaconf import OmegaConf

from .services import ContainerDefinition, ContainerSet
from .utils import save_loadable_wizard_config, write_yaml

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages all configuration generation and writing."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.generated_configs: Dict[str, Path] = {}

    def generate_all(
        self, container_set: ContainerSet, context: WizardContext
    ) -> Dict[str, Path]:
        """Generate all required configurations.

        Args:
            container_set: ContainerSet with all services
            context: WizardContext containing configuration and artifacts
        """
        logger.info("Generating all configurations...")

        cfg = context.cfg
        artifact_list = context.get_artifacts()

        # Generate each configuration
        self._generate_runtime_config(cfg, artifact_list)

        # Get sim containers from service_manager for network config
        sim_containers = container_set.sim
        self._generate_network_config(sim_containers, cfg)

        self._generate_trafficsim_config(cfg)
        self._generate_eval_config(cfg)
        self._generate_run_metadata(cfg)
        self._generate_driver_config(cfg)

        # Save wizard config
        self._save_wizard_config(cfg)

        logger.info(f"Generated {len(self.generated_configs)} configuration files")
        return self.generated_configs

    def _generate_runtime_config(
        self, cfg: Any, artifact_list: List[Any]
    ) -> Optional[str]:
        """Generate runtime configuration."""
        runtime_config = OmegaConf.to_container(cfg.runtime, resolve=True)
        runtime_config = self._remove_none_values(runtime_config)

        # Get default_scenario_parameters as a regular dict
        default_params = runtime_config.get("default_scenario_parameters", {})

        runtime_config = self._create_scenarios(
            runtime_config, [s.scene_id for s in artifact_list], default_params
        )
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
            "driver": {"addresses": []},
            "physics": {"addresses": []},
            "sensorsim": {"addresses": []},
            "trafficsim": {"addresses": []},
            "controller": {"addresses": []},
        }

        for c in service_containers:
            for inst in c.service_instances:
                # A special configuration has been requested, where the sensorsim and
                # physics service exist in the same process/at the same port. This logical
                # branch handles that mapping.
                if c.name == "physics" and inst.service_config.image == "*sensorsim*":
                    logger.info("Mapping the physics service to sensorsim addresses")
                    # get the sensorsim container
                    sensorsim_containers = [
                        sc for sc in service_containers if sc.name == "sensorsim"
                    ]
                    if (len(sensorsim_containers) != 1) or (
                        len(sensorsim_containers[0].get_all_addresses()) != 1
                    ):
                        raise ValueError(
                            "Expected exactly one sensorsim container/address"
                        )
                    sensorsim_address = sensorsim_containers[0].get_all_addresses()[0]
                    if inst.address is None:
                        raise ValueError("Physics service must have an address defined")
                    inst.address.port = sensorsim_address.port
                    logger.info("Mapped physics to sensorsim at %s", inst.address)

                elif inst.address is None:
                    continue

                if c.name in network_config:
                    network_config[c.name]["addresses"].append(str(inst.address))

        # Add external service addresses (for services running outside the deployment)
        external_services = cfg.wizard.external_services
        if external_services is not None:
            for service_name, addresses in external_services.items():
                if service_name in network_config:
                    network_config[service_name]["addresses"].extend(addresses)
                    logger.info(
                        "Added external %s addresses: %s", service_name, addresses
                    )
                else:
                    logger.warning(
                        "Unknown external service '%s', skipping", service_name
                    )

        self._write_config("generated-network-config.yaml", network_config)
        logger.debug("Generated network config")

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

    def _generate_run_metadata(self, cfg: Any) -> None:
        """Generate run metadata."""
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

        self._write_config("run_metadata.yaml", run_metadata)
        logger.debug("Generated run metadata")

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
        write_yaml(data, str(filepath))
        self.generated_configs[filename] = filepath
        return filepath

    def _remove_none_values(self, d: Any) -> Any:
        """Recursively remove all keys with None values from the dictionary."""
        if not isinstance(d, dict):
            return d
        return {k: self._remove_none_values(v) for k, v in d.items() if v is not None}

    def _create_scenarios(
        self, runtime_config: Dict, scene_id_list: List[str], default_params: Dict
    ) -> Dict:
        """Create scenarios from scene IDs."""
        new_scenarios = []

        for scene_id in scene_id_list:
            # Create a new scenario based on default params
            scenario = default_params.copy() if default_params else {}
            scenario["scene_id"] = scene_id
            new_scenarios.append(scenario)

        runtime_config["scenarios"] = new_scenarios
        runtime_config.pop("default_scenario_parameters", None)

        return runtime_config

    def _maybe_split_user_config_for_slurm_array(self, user_config: Any) -> Any:
        """Split scenarios for SLURM array jobs."""
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0))

        if task_count <= 1:
            return user_config

        logger.info(
            f"Detected SLURM_ARRAY_TASK_COUNT = {task_count}, splitting user-config"
        )
        user_config = user_config.copy()

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        min_task_id = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))

        all_scenarios = user_config["scenarios"]
        # Sort for deterministic distribution
        all_scenarios = sorted(
            all_scenarios, key=lambda x: (x.get("scene_id", ""), str(x))
        )

        # Distribute scenarios across array tasks
        split_scenarios: List[List[Any]] = [[] for _ in range(task_count)]
        for scenario_id, scenario in enumerate(all_scenarios):
            split_scenarios[scenario_id % task_count].append(all_scenarios[scenario_id])

        user_config["scenarios"] = split_scenarios[task_id - min_task_id]
        return user_config

    def get_runtime_config_name(self) -> str:
        """Get the runtime configuration filename."""
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        return f"generated-user-config-{task_id}.yaml"
