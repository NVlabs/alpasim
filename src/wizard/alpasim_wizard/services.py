# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Service manager for organizing and building services."""

from __future__ import annotations

import itertools
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Literal

from .context import WizardContext
from .schema import ServiceConfig

logger = logging.getLogger(__name__)


@dataclass
class Address:
    host: str
    port: int

    def __repr__(self) -> str:
        return f"{self.host}:{self.port}"

    def is_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0


@dataclass
class VolumeMount:
    host: str
    container: str

    @staticmethod
    def from_str(string: str) -> VolumeMount:
        try:
            host, container = string.split(":")
        except ValueError as e:
            raise ValueError(f"Failed to convert {string=} to VolumeMount") from e
        return VolumeMount(host, container)

    def to_str(self) -> str:
        return f"{self.host}:{self.container}"

    def host_exists(self) -> bool:
        return os.path.exists(self.host)


@dataclass
class ContainerDefinition:
    """
    Unified container definition supporting single or multiple service instances.
    """

    @dataclass
    class ServiceInstance:
        """Represents a single service instance within a container."""

        replica_idx: int
        command: str
        address: Address | None
        parent_container_definition: ContainerDefinition | None = field(default=None)

        @property
        def port(self) -> int | None:
            """Get port from address if available, otherwise None."""
            return self.address.port if self.address is not None else None

        @property
        def service_config(self) -> ServiceConfig:
            """Get service config from parent container definition."""
            if self.parent_container_definition is None:
                raise ValueError("Parent container definition is not set")
            return self.parent_container_definition.service_config

    uuid: str  # Format: {container_name == service_name}-{container_idx}
    name: str  # Name of the service
    service_config: ServiceConfig
    service_instances: list[ServiceInstance]
    gpu: int | None
    context: WizardContext
    workdir: str | None
    environments: list[str]
    volumes: list[VolumeMount]

    @property
    def command(self) -> str:
        """Get command string."""
        if not self.service_instances:
            return ""
        if len(self.service_instances) == 1:
            return self.service_instances[0].command
        else:
            # Build command that captures exit codes and propagates errors
            # Each command runs in background, we capture PIDs, wait for each,
            # and exit with non-zero if any command fails
            commands = [inst.command for inst in self.service_instances]
            pid_vars = []

            # Build the script with proper formatting:
            # - Each command on its own line with & at the end
            # - PID assignment on the next line
            # - Trap to kill all processes on TERM/INT
            # - Wait loop that captures exit codes
            script_lines = []
            for i, cmd in enumerate(commands):
                pid_var = f"PID{i}"
                pid_vars.append(pid_var)
                script_lines.append(f"{cmd} &")
                script_lines.append(f"{pid_var}=\\$!")
                script_lines.append("")

            # Add trap to kill all processes
            pid_list = " ".join(f'"\\${pid}"' for pid in pid_vars)
            script_lines.append(f"trap 'kill {pid_list} 2>/dev/null' TERM INT")
            script_lines.append("")

            # Add wait loop
            script_lines.append("EXIT_CODE=0")
            pid_list_wait = " ".join(f'"\\${pid}"' for pid in pid_vars)
            script_lines.append(f"for pid in {pid_list_wait}; do")
            script_lines.append('    wait "\\$pid" || EXIT_CODE=\\$?')
            script_lines.append("done")
            script_lines.append('exit "\\$EXIT_CODE"')

            return "\n".join(script_lines)

    def get_all_addresses(self) -> list[Address]:
        """Get all addresses from service instances."""
        return [
            inst.address for inst in self.service_instances if inst.address is not None
        ]

    @staticmethod
    def create(
        name: str,
        service_instances: list[ServiceInstance],
        gpu: int | None,
        service_config: ServiceConfig,
        context: WizardContext,
    ) -> ContainerDefinition:
        """Create a container definition with one or more service instances.

        Args:
            name: Name of the service
            service_instances: List of service instances to run in this container
            gpu: GPU ID to assign to this container
            service_config: ServiceConfig for the service instances
            context: WizardContext containing configuration

        Returns:
            ContainerDefinition instance with the service instances
        """
        if not service_instances:
            raise ValueError("Must provide at least one service instance")

        # Note: all service instances share the same ServiceConfig, volumes and environments
        first_instance = service_instances[0]

        workdir = getattr(service_config, "workdir", None)
        environments = list(service_config.environments)
        volumes: list[VolumeMount] = []
        for volume_str in service_config.volumes:
            volumes.append(VolumeMount.from_str(volume_str))

        if getattr(context.cfg.wizard, "validate_mount_points", False):
            for volume in volumes:
                if not volume.host_exists():
                    raise FileNotFoundError(
                        f"Mount point does not exist: {volume.host}"
                    )

        # Generate container UUID from first service instance
        container_idx = (
            first_instance.replica_idx // service_config.replicas_per_container
        )
        uuid = f"{name}-{container_idx}"

        container_definition = ContainerDefinition(
            name=name,
            uuid=uuid,
            service_instances=service_instances,
            gpu=gpu,
            service_config=service_config,
            context=context,
            workdir=workdir,
            environments=environments,
            volumes=volumes,
        )
        for instance in service_instances:
            instance.parent_container_definition = container_definition
        return container_definition

    @staticmethod
    def _build_command(
        service_config: ServiceConfig,
        port: int | None,
        context: WizardContext,
        service_name: str,
    ) -> str:
        # Build command with all replacements
        command = " ".join(service_config.command)

        assert (
            "{port}" not in command or port is not None
        ), f"Port is required for {service_name}"
        # Apply all variable replacements
        command = command.replace("{port}", str(port))
        command = command.replace("{sceneset}", context.sceneset_path or "None")

        # Runtime config name replacement
        runtime_config_name = f"generated-user-config-{int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))}.yaml"
        command = command.replace("{runtime_config_name}", runtime_config_name)

        return command

    @staticmethod
    def _build_address(
        port: int | None,
        uuid: str,
        use_address_string: Literal["localhost", "0.0.0.0", "uuid"],
    ) -> Address | None:
        """Get the address of the container.

        Args:
            use_localhost: Whether to use localhost for the address. If False,
            the address will be the container UUID (for docker compose).

        Returns:
            The address of the container
        """
        if port is None:
            return None

        if use_address_string == "localhost":
            return Address(host="localhost", port=port)
        elif use_address_string == "0.0.0.0":
            return Address(host="0.0.0.0", port=port)
        elif use_address_string == "uuid":
            return Address(host=uuid, port=port)
        else:
            raise ValueError(f"Invalid address string: {use_address_string}")


@dataclass
class ContainerSet:
    """Container organization for deployment strategies."""

    sim: list[ContainerDefinition] = field(default_factory=list)
    runtime: list[ContainerDefinition] = field(default_factory=list)


def create_gpu_assigner(gpu_ids: List[int] | None) -> Iterator[int | None]:
    """Create an iterator for GPU assignment."""

    def gpu_assigner() -> Iterator[int | None]:
        if gpu_ids is None:
            yield from itertools.repeat(None)
        else:
            yield from itertools.cycle(gpu_ids)

    return gpu_assigner()


def build_container_set(
    context: WizardContext, use_address_string: Literal["localhost", "0.0.0.0", "uuid"]
) -> ContainerSet:
    """Build container set from configuration.

    Args:
        context: WizardContext containing configuration and state

    Returns:
        ContainerSet populated with containers for all configured services
    """
    cfg = context.cfg
    num_gpus = context.get_num_gpus()

    # Overwrite from config
    use_address_string = (
        "localhost"
        if context.cfg.wizard.debug_flags.use_localhost
        else use_address_string
    )

    def build_service_containers(
        service_name: str,
        service_config: ServiceConfig,
        runtime_cfg: Any | None = None,
    ) -> List[ContainerDefinition]:
        """Build containers for a single service."""

        # Skip if not in services_to_run
        if service_name not in context.all_services_to_run:
            return []

        # Check if service should be skipped (skip: true in runtime config)
        if runtime_cfg:
            endpoints = getattr(runtime_cfg, "endpoints", {})
            service_endpoint = endpoints.get(service_name, {})
            if service_endpoint.get("skip", False):
                logger.debug(f"Skipping service {service_name} (marked as skip)")
                return []

        # Validate replicas_per_container
        replicas_per_container = service_config.replicas_per_container
        if replicas_per_container < 1:
            raise ValueError(
                f"replicas_per_container must be >= 1, got {replicas_per_container}"
            )

        # Validate GPU configuration
        if (
            service_config.gpus is not None
            and len(service_config.gpus) > 0
            and num_gpus > 0
            and not all(gpu_id < num_gpus for gpu_id in service_config.gpus)
        ):
            raise RuntimeError(
                f"Service {service_name} requested GPUs {service_config.gpus} "
                f"but only 0 .. {num_gpus - 1} are available."
            )

        # Determine number of containers
        # If no GPUs specified, create a single container
        # Otherwise, create one container per GPU
        if service_config.gpus is None or len(service_config.gpus) == 0:
            num_containers = 1
            gpu_assigner = create_gpu_assigner(None)  # No GPUs
        else:
            num_containers = len(service_config.gpus)
            gpu_assigner = create_gpu_assigner(service_config.gpus)

        containers: List[ContainerDefinition] = []
        replica_idx = 0

        for container_idx in range(num_containers):
            # Build service instances for this container
            service_instances = []
            gpu = next(gpu_assigner)
            uuid = service_name + "-" + str(container_idx)

            for _ in range(replicas_per_container):
                port = next(context.port_assigner)

                # Build command for this service instance
                command = ContainerDefinition._build_command(
                    service_config, port, context, service_name
                )

                # Build addresses for all service instances using the container UUID
                address = ContainerDefinition._build_address(
                    port, uuid, use_address_string
                )

                # Create service instance
                service_instance = ContainerDefinition.ServiceInstance(
                    replica_idx=replica_idx,
                    command=command,
                    address=address,
                )

                service_instances.append(service_instance)
                replica_idx += 1

            # Create container with service instances
            containers.append(
                ContainerDefinition.create(
                    name=service_name,
                    service_instances=service_instances,
                    gpu=gpu,
                    service_config=service_config,
                    context=context,
                )
            )

        return containers

    # Build containers for each service type
    sim_containers = []
    runtime_containers = []

    # Simulation services
    for name in cfg.wizard.run_sim_services or []:
        if name == "runtime":
            # Runtime handled separately
            runtime_config = cfg.services.runtime
            command = ContainerDefinition._build_command(
                runtime_config, None, context, "runtime"
            )
            runtime_instance = ContainerDefinition.ServiceInstance(
                replica_idx=0,
                command=command,
                address=None,  # Unused for runtime
            )
            runtime_containers = [
                ContainerDefinition.create(
                    name="runtime",
                    service_instances=[runtime_instance],
                    service_config=cfg.services.runtime,
                    gpu=None,
                    context=context,
                )
            ]
        else:
            if config := getattr(cfg.services, name, None):
                sim_containers.extend(
                    build_service_containers(name, config, cfg.runtime)
                )

    logger.info("Built %d simulation containers", len(sim_containers))

    return ContainerSet(
        sim=sim_containers,
        runtime=runtime_containers,
    )
