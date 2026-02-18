# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Docker Compose deployment strategy."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from ..context import WizardContext
from ..services import ContainerDefinition, build_container_set
from ..utils import LiteralStr, write_yaml

logger = logging.getLogger(__name__)


class DockerComposeDeployment:
    """Deployment strategy using Docker Compose."""

    def __init__(self, context: WizardContext):
        """Initialize with context and build container set.

        Args:
            context: The wizard context
        """
        self.context = context
        self.container_set = build_container_set(context, use_address_string="uuid")

    def generate_docker_compose_and_run_script(self) -> None:
        """Generates the docker-compose.yaml file and run.sh script.

        Note: This does not actually start the service. This can be done using
        ```bash
        docker compose --profile sim up
        docker compose --profile eval up
        docker compose --profile aggregation up
        ```

        """
        # Generate Docker Compose configuration
        docker_compose_filepath = self.generate_docker_compose_yaml(self.container_set)
        self.generate_run_script(docker_compose_filepath)

        logger.info(
            "Docker Compose configuration generated in %s",
            self.context.cfg.wizard.log_dir,
        )
        logger.info("Generated run script: %s/run.sh", self.context.cfg.wizard.log_dir)

    def deploy_all_services(self) -> None:
        """Execute the generated run.sh script to deploy all services."""
        run_script_path = Path(self.context.cfg.wizard.log_dir) / "run.sh"
        logger.info("Executing docker compose run script: %s", run_script_path)

        try:
            # Execute the run script
            subprocess.run(
                [str(run_script_path)], check=True, cwd=self.context.cfg.wizard.log_dir
            )
            logger.info("Docker Compose deployment completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(
                "Docker Compose deployment failed with return code: %s", e.returncode
            )
            raise

    def generate_run_script(self, docker_compose_filepath: str) -> None:
        """Generate run.sh script to execute Docker Compose profiles sequentially.

        Args:
            docker_compose_filepath: Path to the docker-compose.yaml file
        """
        # Build script dynamically based on which services are configured
        script_lines = [
            "#!/bin/bash",
            "set -e",
            "",
        ]

        # Always run simulation phase (required)
        script_lines.extend(
            [
                "# Run simulation profile",
                'echo "Starting simulation phase..."',
                f"docker compose -f {docker_compose_filepath} --profile sim up",
                "",
            ]
        )

        # Only run evaluation phase if services are configured
        if self.container_set.eval:
            script_lines.extend(
                [
                    "# Run evaluation profile",
                    'echo "Starting evaluation phase..."',
                    f"docker compose -f {docker_compose_filepath} --profile eval up",
                    "",
                ]
            )
        else:
            logger.info("Skipping evaluation phase (no eval services configured)")

        # Only run aggregation phase if services are configured
        if self.container_set.agg:
            script_lines.extend(
                [
                    "# Run aggregation profile",
                    'echo "Starting aggregation phase..."',
                    f"docker compose -f {docker_compose_filepath} --profile aggregation up",
                    "",
                ]
            )
        else:
            logger.info(
                "Skipping aggregation phase (no aggregation services configured)"
            )

        script_lines.append('echo "All phases completed successfully!"')
        script_content = "\n".join(script_lines) + "\n"

        run_script_path = Path(self.context.cfg.wizard.log_dir) / "run.sh"
        with open(run_script_path, "w") as f:
            f.write(script_content)

        # Make the script executable
        os.chmod(run_script_path, 0o755)
        logger.info("Generated run script: %s", run_script_path)

    def _to_docker_compose_service(
        self, container: ContainerDefinition
    ) -> dict[str, Any]:
        """Convert container to Docker Compose service definition.

        Args:
            container: ContainerDefinition instance

        Returns:
            Docker Compose service configuration dict
        """
        ret: dict[str, Any] = {}
        if self.context.cfg.wizard.debug_flags.use_localhost:
            # Tell Docker to use the host network
            ret["network_mode"] = "host"
        else:
            ret["networks"] = ["microservices_network"]
        ret["volumes"] = [v.to_str() for v in container.volumes]
        ret["pull_policy"] = "missing"
        ret["image"] = container.service_config.image

        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()

        if not container.service_config.external_image:
            ret["build"] = {
                "context": repo_root,
                "dockerfile": "Dockerfile",
                "tags": [container.service_config.image],
            }

        if container.command:
            ret["entrypoint"] = "bash"
            command = container.command
            # Escaping:
            # We use \$ to declare fields that should not be interpreted by
            # 'our' OmegaConf parser, but by downstream parsers in the service.
            # Furhtermore, for docker-compose, we need to escape $ as $$
            command = command.replace(r"\$", "$$")
            # Use literal scalar string for multi-line commands to get | format in YAML
            if "\n" in command:
                command = LiteralStr(command)
            ret["command"] = ["-c", command]
        if container.workdir:
            ret["working_dir"] = container.workdir
        if container.environments:
            ret["environment"] = container.environments

        addresses = container.get_all_addresses()
        if addresses:
            ports = [f"{addr.port}:{addr.port}" for addr in addresses]
            ret["ports"] = ports

        if container.gpu is not None:
            ret["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "capabilities": ["gpu"],
                                "device_ids": [str(container.gpu)],
                            }
                        ]
                    }
                }
            }
        return ret

    def generate_docker_compose_yaml(self, container_set: Any) -> str:
        """Generate docker-compose.yaml with profiles, services sorted by execution order.

        Args:
            container_set: ContainerSet instance with sim, eval, agg, and runtime containers

        Returns:
            Filename of the generated docker-compose.yaml
        """
        # Build services in execution order
        services = {}

        # Phase 1: Simulation services (runtime should start last in sim phase)
        for c in container_set.sim or []:
            if c.command == "noop":
                # Special logic to support sensorsim/physics combined process
                continue
            service = self._to_docker_compose_service(c)
            service["profiles"] = ["sim"]
            services[c.uuid] = service

        # Add runtime services last in sim phase
        for c in container_set.runtime or []:
            service = self._to_docker_compose_service(c)
            service["profiles"] = ["sim"]
            # Runtime needs host PID namespace for process monitoring
            service["pid"] = "host"
            # Runtime needs access to all GPUs for telemetry/resource monitoring
            service["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "count": "all",
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            }
            services[c.uuid] = service

        # Phase 2: Evaluation services
        for c in container_set.eval or []:
            service = self._to_docker_compose_service(c)
            service["profiles"] = ["eval"]
            # Note: We cannot add depends_on to services in other profiles
            # Docker Compose will handle the ordering when both profiles are active
            services[c.uuid] = service

        # Phase 3: Aggregation services
        for c in container_set.agg or []:
            service = self._to_docker_compose_service(c)
            service["profiles"] = ["aggregation"]
            # Note: We cannot add depends_on to services in other profiles
            services[c.uuid] = service

        # Create compose structure with ordered services
        compose = {
            "networks": {"microservices_network": {"driver": "bridge"}},
            "services": services,  # Services maintain insertion order in Python 3.7+
        }

        # Write to file
        filename = "docker-compose.yaml"
        log_dir = Path(self.context.cfg.wizard.log_dir)
        logger.info("Writing docker compose YAML to %s/%s", log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
        write_yaml(compose, str(log_dir / filename))
        return filename
