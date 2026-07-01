# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Docker Compose deployment strategy."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from alpasim_utils.paths import find_repo_root

from ..context import WizardContext
from ..schema import RunMode
from ..services import ContainerDefinition, build_container_set
from ..utils import LiteralStr, write_yaml

logger = logging.getLogger(__name__)


def _netrc_secret_file() -> Path | None:
    netrc_path = Path.home() / ".netrc"
    return netrc_path if netrc_path.is_file() else None


class DockerComposeDeployment:
    """Deployment strategy using Docker Compose."""

    def __init__(self, context: WizardContext):
        """Initialize with context and build container set.

        Args:
            context: The wizard context
        """
        self.context = context
        self.container_set = build_container_set(context, use_address_string="uuid")

    def generate_docker_compose(self) -> None:
        """Generates the docker-compose.yaml file.

        Note: This does not actually start the services. This can be done using
        ```bash
        docker compose up --exit-code-from runtime-0
        ```
        """
        self.docker_compose_filepath = self.generate_docker_compose_yaml(
            self.container_set
        )
        logger.info(
            "Docker Compose configuration generated in %s",
            self.context.cfg.wizard.log_dir,
        )

    def deploy_all_services(self) -> None:
        """Run docker compose up to deploy all services."""
        log_dir = self.context.cfg.wizard.log_dir
        compose_file = Path(log_dir) / self.docker_compose_filepath
        command = [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "up",
        ]
        if self.container_set.runtime:
            command.extend(
                [
                    "--remove-orphans",
                    "--exit-code-from",
                    "runtime-0",
                ]
            )

        if self.context.cfg.wizard.dry_run:
            logger.info("[DRY-RUN] Would execute: %s", shlex.join(command))
            return

        logger.info("Running docker compose: %s", compose_file)
        try:
            subprocess.run(
                command,
                check=True,
                cwd=log_dir,
            )
            logger.info("Docker Compose deployment completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(
                "Docker Compose deployment failed with return code: %s", e.returncode
            )
            raise

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
        service_config = container.service_config
        use_host_network = self.context.cfg.wizard.debug_flags.use_localhost
        if use_host_network:
            # Tell Docker to use the host network
            ret["network_mode"] = "host"
        else:
            ret["networks"] = ["microservices_network"]
        ret["volumes"] = [v.to_str() for v in container.volumes]
        ret["pull_policy"] = service_config.pull_policy
        ret["image"] = service_config.image

        repo_root = str(find_repo_root(__file__))

        if not service_config.external_image:
            build_config: dict[str, Any] = {
                "context": repo_root,
                "dockerfile": "Dockerfile",
                "tags": [service_config.image],
            }
            if _netrc_secret_file() is not None:
                build_config["secrets"] = ["netrc"]
            ret["build"] = build_config

        if container.command:
            ret["entrypoint"] = "bash"
            command = container.command
            # Escaping:
            # We use \$ to declare fields that should not be interpreted by
            # 'our' OmegaConf parser, but by downstream parsers in the service.
            # Furhtermore, for docker-compose, we need to escape $ as $$
            command = command.replace("$", "$$")
            # Set permissive umask so files written to bind-mounted volumes
            # are accessible by the host user (containers run as root).
            command = "umask 0000\n" + command
            # Use literal scalar string for multi-line commands to get | format in YAML
            if "\n" in command:
                command = LiteralStr(command)
            ret["command"] = ["-c", command]
        if container.workdir:
            ret["working_dir"] = container.workdir
        if container.environments:
            ret["environment"] = container.environments

        addresses = container.get_all_addresses()
        publish_runtime_server_port = (
            container.name == "runtime"
            and self.context.cfg.wizard.run_mode == RunMode.SERVER
        )
        ports: list[str] = []
        if not use_host_network and container.published_ports:
            ports.extend(
                f"{port}:{port}" for port in container.published_ports.values()
            )
        if addresses and (use_host_network or publish_runtime_server_port):
            ports.extend(f"{addr.port}:{addr.port}" for addr in addresses)
        if ports:
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
        elif container.name == "prometheus" and self.context.num_gpus > 0:
            ret["deploy"] = {
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
        return ret

    def generate_docker_compose_yaml(self, container_set: Any) -> str:
        """Generate docker-compose.yaml with services sorted by execution order.

        Args:
            container_set: ContainerSet instance with sim and runtime containers

        Returns:
            Filename of the generated docker-compose.yaml
        """
        # Build services in execution order
        services = {}

        # Simulation services (runtime should start last)
        for c in container_set.sim or []:
            if c.command == "noop":
                # Special logic to support renderer/physics combined process.
                continue
            service = self._to_docker_compose_service(c)
            services[c.uuid] = service

        service = self._to_docker_compose_service(container_set.prometheus)
        services[container_set.prometheus.uuid] = service

        # Add runtime service last
        if container_set.runtime is not None:
            service = self._to_docker_compose_service(container_set.runtime)
            services[container_set.runtime.uuid] = service

        # Create compose structure with ordered services
        compose: dict[str, Any] = {
            "networks": {"microservices_network": {"driver": "bridge"}},
            "services": services,  # Services maintain insertion order in Python 3.7+
        }
        if _netrc_secret_file() is not None:
            compose["secrets"] = {"netrc": {"file": "${HOME}/.netrc"}}

        # Write to file
        filename = "docker-compose.yaml"
        log_dir = Path(self.context.cfg.wizard.log_dir)
        logger.info("Writing docker compose YAML to %s/%s", log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
        write_yaml(compose, str(log_dir / filename))
        return filename
