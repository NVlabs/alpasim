# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""SLURM deployment strategy."""

from __future__ import annotations

import logging
import os
import shlex
import socket
import time
from pathlib import Path
from typing import Any, List

from ..context import WizardContext
from ..schema import RunMode
from ..services import ContainerDefinition, build_container_set
from ..utils import ensure_sqsh_path
from .dispatcher import dispatch_command

logger = logging.getLogger(__name__)


class SlurmDeployment:
    """Deployment strategy using SLURM."""

    def __init__(self, context: WizardContext):
        """Initialize with context and build container set.

        Args:
            context: The wizard context
        """
        self.context = context
        self.container_set = build_container_set(context, use_address_string="0.0.0.0")

    def deploy_all_services(self) -> None:
        """Deploy simulation services (including runtime) on SLURM."""
        logger.info("Running simulation services")
        containers_to_start_last = (
            [self.container_set.runtime] if self.container_set.runtime else []
        )
        containers = list(self.container_set.sim)
        containers.append(self.container_set.prometheus)
        self.deploy(
            containers=containers,
            containers_to_start_last=containers_to_start_last,
        )

    def deploy(
        self,
        containers: List[Any],
        containers_to_start_last: List[Any] | None = None,
    ) -> None:
        """Deploy containers using SLURM."""
        launched_containers: list[ContainerDefinition] = []
        if containers_to_start_last:
            assert (
                self.context.cfg.wizard.timeout is not None
            ), "Timeout must be set if container_to_start_last is set"

        def _wait_for_containers_running() -> bool:
            return (
                containers_to_start_last is not None
                and not self.context.cfg.wizard.dry_run
            )

        # Only do this if we're waiting for the last container to start
        nr_retries = (
            (self.context.cfg.wizard.nr_retries or 1)
            if _wait_for_containers_running()
            else 1
        )

        logger.info(
            "Starting %d containers with %d retries and %d timeout",
            len(containers),
            nr_retries,
            self.context.cfg.wizard.timeout or -1,
        )

        # Deploy containers with retries
        for retry in range(nr_retries):
            missing_containers = self.get_missing_containers(containers)
            if not missing_containers:
                break

            for c in missing_containers:
                dispatch_command(
                    self._get_slurm_dispatch_command(
                        c,
                        self.context.cfg.wizard.run_mode.name,
                    ),
                    log_dir=Path(self.context.cfg.wizard.log_dir),
                    dry_run=self.context.cfg.wizard.dry_run,
                    blocking=False,
                )
                launched_containers.append(c)

            # Wait for containers if needed
            if _wait_for_containers_running():
                self.wait_for_containers(
                    containers,
                    timeout=self.context.cfg.wizard.timeout,
                    raise_on_timeout=(retry == nr_retries - 1),
                )

        # Deploy containers that should start last
        if containers_to_start_last:
            try:
                for c in containers_to_start_last:
                    dispatch_command(
                        self._get_slurm_dispatch_command(
                            c,
                            self.context.cfg.wizard.run_mode.name,
                        ),
                        log_dir=Path(self.context.cfg.wizard.log_dir),
                        dry_run=self.context.cfg.wizard.dry_run,
                        blocking=True,
                    )
            finally:
                self._cleanup_launched_service_steps(launched_containers)

    def _cleanup_launched_service_steps(
        self, containers: list[ContainerDefinition]
    ) -> None:
        """Cancel non-runtime service steps after the blocking runtime exits."""
        if self.context.cfg.wizard.dry_run:
            return

        seen_step_names: set[str] = set()
        for container in containers:
            step_name = self._get_slurm_step_name(container)
            if step_name in seen_step_names:
                continue
            seen_step_names.add(step_name)
            try:
                dispatch_command(
                    self._get_slurm_cleanup_command(container),
                    log_dir=Path(self.context.cfg.wizard.log_dir),
                    dry_run=False,
                    blocking=True,
                )
            except Exception as e:
                logger.warning(
                    "Failed to clean up SLURM step for %s: %s",
                    container.uuid,
                    e,
                )

    def _get_slurm_step_name(self, container: ContainerDefinition) -> str:
        """Return a unique SLURM job-step name for a launched container."""
        return f"alpasim-{self.context.cfg.wizard.slurm_job_id}-{container.uuid}"

    def _get_slurm_cleanup_command(self, container: ContainerDefinition) -> str:
        """Generate the host-side cleanup command for a launched service step."""
        return f"scancel --signal=TERM --name={self._get_slurm_step_name(container)}"

    def _to_slurm_run(
        self,
        container: ContainerDefinition,
        mode: RunMode,
    ) -> str:
        """Generate SLURM srun command for a container.

        Args:
            container: ContainerDefinition instance
            mode: RunMode (ONESHOT or SERVER)

        Returns:
            SLURM srun command string
        """
        assert (
            container.context.cfg.wizard.slurm_job_id is not None
        ), "SLURM environment not detected"
        slurm_job_id = container.context.cfg.wizard.slurm_job_id

        s_log = (
            f"{container.context.cfg.wizard.log_dir}/txt-logs/"
            f"out-{slurm_job_id}-{container.uuid}-log.txt"
        )

        sqsh = ensure_sqsh_path(
            container.service_config.image,
            list(container.context.cfg.wizard.sqshcaches),
        )

        # Note that we cannot use --export=CUDA_VISIBLE_DEVICES=... with srun because SLURM
        # overrides CUDA_VISIBLE_DEVICES even when exported as an environment variable.
        # Instead we set it in the command line. Use export to allow chaining commands with &&.
        s_gpu = (
            f"export CUDA_VISIBLE_DEVICES={container.gpu};"
            if container.gpu is not None
            else ""
        )
        # Separate environment variables:
        #  - 'VAR=value' format to export in bash. The value will be logged, not secure for secrets.
        #  - 'VAR' format pass-through from host. The value will not be logged, secure for secrets.
        env_export_set = []  # VAR=value format
        env_passthrough_set = ["SLURM_JOB_ID"]  # VAR only format
        for e in container.environments or []:
            if "=" in e:
                env_export_set.append(e)
            elif e not in env_passthrough_set:
                env_passthrough_set.append(e)

        # Construct environment variable arguments
        # Export VAR=value vars inside bash command (more reliable than --container-env)
        s_env_exports = (
            " ".join(f"export {e};" for e in env_export_set) + " "
            if env_export_set
            else ""
        )
        # Slurm exports the submit environment by default. Keep container steps
        # isolated, while preserving the job id needed for Slurm-scoped telemetry.
        s_env_export_arg = f"--export={','.join(env_passthrough_set)} "

        s_mnt = ",".join([v.to_str() for v in container.volumes])

        # Pin child srun steps to the wizard's node so services are co-located
        # and reachable via localhost.  Without --nodelist, SLURM may schedule
        # them on other nodes in a multi-node allocation.
        # Prefer SLURMD_NODENAME which is guaranteed to match SLURM's node
        # naming; fall back to socket.gethostname() for local testing.
        current_node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()

        cmd = r"srun --verbose --overlap "
        cmd += f"--job-name={self._get_slurm_step_name(container)} "
        cmd += f"--nodes=1 --ntasks=1 --nodelist={current_node} "
        if container.context.cfg.wizard.slurm_cpu_bind_none:
            cmd += "--cpu-bind=none "
        cmd += f" --container-image={sqsh} "
        cmd += " --container-writable "
        cmd += f" --container-mounts={s_mnt} "

        if container.workdir is not None:
            cmd += f" --container-workdir={container.workdir} "

        if not container.service_config.remap_root:
            cmd += " --no-container-remap-root "

        expanded_command = container.command.replace("$$", "$")
        bash_command = shlex.quote(f"{s_gpu}{s_env_exports}{expanded_command}")

        if mode in (RunMode.ONESHOT, RunMode.SERVER):
            cmd += f"--output={s_log} --error={s_log} {s_env_export_arg}"
            cmd += f"bash -c {bash_command}"
        else:
            raise ValueError(f"Unknown run mode: {mode}")
        return cmd

    def _get_slurm_dispatch_command(
        self,
        container: ContainerDefinition,
        mode: str,
    ) -> str:
        """Get the full SLURM dispatch command.

        Args:
            container: ContainerDefinition instance
            mode: Run mode string

        Returns:
            Complete SLURM command string
        """
        # Convert mode string to RunMode enum
        run_mode = RunMode[mode.upper()] if isinstance(mode, str) else mode

        logger.info(f"Launch {container.uuid} in {run_mode.name}")
        return self._to_slurm_run(container, mode=run_mode)

    def wait_for_containers(
        self,
        containers: List[ContainerDefinition],
        timeout: int | None = None,
        raise_on_timeout: bool = True,
    ) -> bool:
        """Wait for containers to be ready."""
        logger.info("Waiting for addresses:")
        for container in containers:
            for service_instance in container.service_instances:
                logger.info(
                    "  %s:%s",
                    container.name,
                    service_instance.address,
                )

        s_waited = 0
        for container in containers:
            for service_instance in container.service_instances:
                if service_instance.address is None:
                    continue
                while not service_instance.address.is_open():
                    time.sleep(1)
                    s_waited += 1
                    if timeout is not None and s_waited > timeout:
                        if raise_on_timeout:
                            raise TimeoutError(
                                f"Address {service_instance.address} of "
                                f"{container.name} "
                                "did not open in time"
                            )
                        else:
                            logger.info(
                                "  %s of %s not open yet after %d seconds.",
                                service_instance.address,
                                container.name,
                                s_waited,
                            )
                            return False
                logger.info("  %s found.", service_instance.address)

        logger.info("  All addresses open.")
        return True

    def get_missing_containers(
        self, containers: List[ContainerDefinition]
    ) -> List[ContainerDefinition]:
        """Get containers that are not yet running."""
        missing: List[ContainerDefinition] = []
        for container in containers:
            # Check if any service instance is not ready
            for inst in container.service_instances:
                if inst.address is not None and not inst.address.is_open():
                    missing.append(container)
                    break
        return missing
