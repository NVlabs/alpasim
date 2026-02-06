# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

# mypy: disable-error-code=no-untyped-def
"""Main entry point for Alpasim wizard."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import cast

import git

# Import new refactored components
from .configuration import ConfigurationManager
from .context import WizardContext
from .deployment import DockerComposeDeployment, SlurmDeployment
from .schema import AlpasimConfig, RunMethod

logger = logging.getLogger("alpasim_wizard")


@dataclass
class AlpasimWizard:
    """Main entry point for Alpasim wizard.

    The AlpasimWizard class serves as the main entry point for orchestrating the
    Alpasim simulation workflow. It manages configurations, services, and deployment
    logic. This class coordinates the initialization of core components, handles
    optional driver code cloning, and exposes methods for executing the full pipeline.
    """

    context: WizardContext

    @staticmethod
    def create(cfg: AlpasimConfig) -> AlpasimWizard:
        """Factory method to create AlpasimWizard instance."""

        if cfg.wizard.slurm_job_id is None:
            cfg.wizard.slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))

        context = WizardContext.create(cfg)
        # Return wizard with context
        return AlpasimWizard(
            context=context,
        )

    def maybe_clone_driver_code(self) -> None:
        """Clone driver code if configured."""
        if self.context.cfg.wizard.driver_code_hash is None:
            return
        code_repo = cast(str, self.context.cfg.wizard.driver_code_repo)

        logger.info(
            "Cloning driver code from %s to %s/driver_code",
            code_repo,
            self.context.cfg.wizard.log_dir,
        )
        repo = git.Repo.clone_from(
            code_repo,
            os.path.join(self.context.cfg.wizard.log_dir, "driver_code"),
        )
        logger.info(
            "Checking out driver code hash %s", self.context.cfg.wizard.driver_code_hash
        )
        repo.git.checkout(self.context.cfg.wizard.driver_code_hash)

    def cast(self) -> None:
        """Main execution method - simplified with refactored components."""

        # Normal execution path
        self.maybe_clone_driver_code()

        # Do this for all run methods to generate docker compose config files.
        docker_compose_deployment = DockerComposeDeployment(self.context)
        docker_compose_deployment.generate_docker_compose_and_run_script()
        slurm_deployment = SlurmDeployment(self.context)
        # Use docker compose container set for DOCKER_COMPOSE and NONE
        # (NONE generates docker-compose files that will be run manually)
        container_set = (
            docker_compose_deployment.container_set
            if self.context.cfg.wizard.run_method
            in (RunMethod.DOCKER_COMPOSE, RunMethod.NONE)
            else slurm_deployment.container_set
        )

        # With the container set, we can now generate the configs and save them.
        config_manager = ConfigurationManager(self.context.cfg.wizard.log_dir)
        config_manager.generate_all(container_set, self.context)

        # Handle different run methods
        if self.context.cfg.wizard.run_method == RunMethod.SLURM:
            slurm_deployment.deploy_all_services()
        elif self.context.cfg.wizard.run_method == RunMethod.DOCKER_COMPOSE:
            docker_compose_deployment.deploy_all_services()
        elif self.context.cfg.wizard.run_method == RunMethod.NONE:
            logger.info(
                "Scripts generated but not executed. Run %s/run.sh to start the simulation",
                self.context.cfg.wizard.log_dir,
            )

        logger.info("Alpasim finished")
