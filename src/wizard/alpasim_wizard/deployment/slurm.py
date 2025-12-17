# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""SLURM deployment strategy."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, List, Optional

from ..context import WizardContext
from ..schema import RunMode
from ..services import ContainerDefinition, build_container_set
from ..utils import image_url_to_sqsh_filename
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
        """Deploy all services (simulation, evaluation, aggregation)."""
        self.deploy_simulation()

        # Deploy evaluation, then aggregation with dependencies
        job_ids = self.deploy_evaluation()
        logger.info(f"Evaluation services submitted job ids: {job_ids}")
        self.deploy_aggregation(job_ids)

    def deploy_simulation(self) -> List[int]:
        """Deploy simulation services (including runtime) on SLURM.

        Returns:
            List of job IDs for submitted jobs
        """
        logger.info("Running simulation services")
        containers_to_start_last = (
            self.container_set.runtime if self.container_set.runtime else []
        )
        return self.deploy(
            containers=self.container_set.sim,
            containers_to_start_last=containers_to_start_last,
            requires_sbatch=False,
        )

    def deploy_evaluation(self, dependencies: Optional[List[int]] = None) -> List[int]:
        """Deploy evaluation services on SLURM.

        Args:
            dependencies: Job IDs to depend on

        Returns:
            List of job IDs for submitted jobs
        """

        logger.info(
            "Running evaluation services with dependencies: %s",
            dependencies,
        )
        return self.deploy(
            containers=self.container_set.eval,
            requires_sbatch=True,
            dependencies=dependencies,
        )

    def deploy_aggregation(self, dependencies: Optional[List[int]] = None) -> List[int]:
        """Deploy aggregation services on SLURM.

        Args:
            dependencies: Job IDs to depend on

        Returns:
            List of job IDs for submitted jobs
        """
        logger.info(
            "Running aggregation services with dependencies: %s",
            dependencies,
        )

        return self.deploy(
            containers=self.container_set.agg,
            requires_sbatch=True,
            dependencies=dependencies,
        )

    def deploy(
        self,
        containers: List[Any],
        containers_to_start_last: Optional[List[Any]] = None,
        requires_sbatch: bool = False,
        dependencies: Optional[List[int]] = None,
    ) -> List[int]:
        """Deploy containers using SLURM."""
        job_ids = []

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
                slurm_output = dispatch_command(
                    self._get_slurm_dispatch_command(
                        c,
                        self.context.cfg.wizard.run_mode.name,
                        requires_sbatch=requires_sbatch,
                        dependencies=dependencies or [],
                    ),
                    log_dir=Path(self.context.cfg.wizard.log_dir),
                    dry_run=self.context.cfg.wizard.dry_run,
                    blocking=requires_sbatch,
                )

                # Extract job IDs from sbatch output
                if (
                    requires_sbatch
                    and not self.context.cfg.wizard.dry_run
                    and slurm_output
                ):
                    match = re.search(r"Submitted batch job (\d+)", slurm_output)
                    if match:
                        job_ids.append(int(match.group(1)))

            # Wait for containers if needed
            if _wait_for_containers_running():
                self.wait_for_containers(
                    containers,
                    timeout=self.context.cfg.wizard.timeout,
                    raise_on_timeout=(retry == nr_retries - 1),
                )

        # Deploy containers that should start last
        if containers_to_start_last:
            for c in containers_to_start_last:
                slurm_output = dispatch_command(
                    self._get_slurm_dispatch_command(
                        c,
                        self.context.cfg.wizard.run_mode.name,
                        requires_sbatch=requires_sbatch,
                        dependencies=dependencies or [],
                    ),
                    log_dir=Path(self.context.cfg.wizard.log_dir),
                    dry_run=self.context.cfg.wizard.dry_run,
                    blocking=True,
                )

                if requires_sbatch and slurm_output:
                    match = re.search(r"Submitted batch job (\d+)", slurm_output)
                    if match:
                        job_ids.append(int(match.group(1)))

        return job_ids

    def _to_slurm_run(
        self,
        container: ContainerDefinition,
        mode: RunMode,
    ) -> str:
        """Generate SLURM srun command for a container.

        Args:
            container: ContainerDefinition instance
            mode: RunMode (BATCH, ATTACH_BASH, or ATTACH_VSCODE)

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

        sqsh = image_url_to_sqsh_filename(
            container.service_config.image, container.context.cfg.wizard.sqshcaches
        )

        # Note that we cannot use --export=CUDA_VISIBLE_DEVICES=... with srun because SLURM
        # overrides CUDA_VISIBLE_DEVICES even when exported as an environment variable.
        # Instead we set it in the command line. Use export to allow chaining commands with &&.
        s_gpu = f"export CUDA_VISIBLE_DEVICES={container.gpu};" if container.gpu else ""

        # Separate environment variables:
        #  - 'VAR=value' format to export in bash. The value will be logged, not secure for secrets.
        #  - 'VAR' format pass-through from host. The value will not be logged, secure for secrets.
        env_export_set = []  # VAR=value format
        env_passthrough_set = []  # VAR only format
        for e in container.environments or []:
            if "=" in e:
                env_export_set.append(e)
            else:
                env_passthrough_set.append(e)

        # Construct environment variable arguments
        # Export VAR=value vars inside bash command (more reliable than --container-env)
        s_env_exports = (
            " ".join(f"export {e};" for e in env_export_set) + " "
            if env_export_set
            else ""
        )
        # Use --export for pass-through variables from host environment
        s_env_passthrough = (
            f"--export={','.join(env_passthrough_set)} " if env_passthrough_set else ""
        )

        s_mnt = ",".join([v.to_str() for v in container.volumes])

        cmd = r"srun --verbose --overlap "
        cmd += f" --container-image={sqsh} "
        cmd += f" --container-mounts={s_mnt} "

        if container.workdir is not None:
            cmd += f" --container-workdir={container.workdir} "

        if not container.service_config.remap_root:
            cmd += " --no-container-remap-root "

        escaped_command = container.command.replace("$$", r"\$")

        if mode == RunMode.BATCH:
            cmd += f"--output={s_log} --error={s_log} {s_env_passthrough}"
            cmd += f'bash -c "{s_gpu}{s_env_exports}{escaped_command}"'
        elif mode == RunMode.ATTACH_BASH:
            cmd += "--pty bash"
        elif mode == RunMode.ATTACH_VSCODE:
            cmd += "/mnt/helper/launch-vscode-auto-port.sh"
        else:
            raise ValueError(f"Unknown run mode: {mode}")
        return cmd

    def _wrap_in_sbatch_script(
        self,
        container: ContainerDefinition,
        srun_command_str: str,
        dependencies: list[int] | None = None,
    ) -> str:
        """Wrap srun command in sbatch script.

        Args:
            container: ContainerDefinition instance
            srun_command_str: The srun command to wrap
            dependencies: List of job IDs to depend on

        Returns:
            sbatch command string
        """
        if dependencies is None:
            dependencies = []

        if srun_command_str.endswith(" &"):
            srun_command_str = srun_command_str[:-2]

        if (
            container.context.cfg.wizard.slurm_gpu_partition is None
            and os.environ.get("SLURM_JOB_PARTITION") is None
        ):
            raise ValueError(
                "SLURM GPU partition must be set in wizard config (wizard.slurm_gpu_partition) "
                "or available via SLURM_JOB_PARTITION environment variable for sbatch submission."
            )

        if container.gpu is not None:
            if container.gpu != 0:
                raise ValueError(
                    "Jobs dispatched via wrap_in_sbatch_script should request gpu 0 or none at all."
                )
            gpu_partition = container.context.cfg.wizard.slurm_gpu_partition
            if gpu_partition is None:
                gpu_partition = os.environ.get("SLURM_JOB_PARTITION", "")
            bash_script_template = build_sbatch(
                srun_command_str,
                partition=gpu_partition,
                gpus=1,
            )
        else:
            bash_script_template = build_sbatch(
                srun_command_str,
                partition=container.context.cfg.wizard.slurm_cpu_partition,
                gpus=0,
            )

        i = 0
        while True:
            script_name = f"generated_submit_{container.name}_{i}.sh"
            script_path = os.path.join(
                container.context.cfg.wizard.log_dir, script_name
            )
            if not os.path.exists(script_path):
                break
            i += 1

        script_path = os.path.join(container.context.cfg.wizard.log_dir, script_name)
        with open(script_path, "w") as script_file:
            script_file.write(bash_script_template)

        cmd = (
            f"unset SLURM_CPU_BIND; sbatch "
            f"--output={container.context.cfg.wizard.log_dir}/txt-logs/slurm_{container.uuid}.log "
            f"--job-name='{container.context.cfg.wizard.slurm_job_id}-{container.uuid}'"
        )
        if dependencies:
            cmd += f" --dependency=afterany:{':'.join(map(str, dependencies))}"

        return f"({cmd} {script_path})"

    def _get_slurm_dispatch_command(
        self,
        container: ContainerDefinition,
        mode: str,
        requires_sbatch: bool = False,
        dependencies: list[int] | None = None,
    ) -> str:
        """Get the full SLURM dispatch command.

        Args:
            container: ContainerDefinition instance
            mode: Run mode string
            requires_sbatch: Whether to wrap in sbatch
            dependencies: List of job IDs to depend on

        Returns:
            Complete SLURM command string
        """
        # Convert mode string to RunMode enum
        run_mode = RunMode[mode.upper()] if isinstance(mode, str) else mode

        logger.info(f"Launch {container.uuid} in {run_mode.name}")
        srun_command_str = self._to_slurm_run(container, mode=run_mode)
        if requires_sbatch:
            return self._wrap_in_sbatch_script(
                container, srun_command_str=srun_command_str, dependencies=dependencies
            )
        else:
            return srun_command_str

    def wait_for_containers(
        self,
        containers: List[ContainerDefinition],
        timeout: Optional[int] = None,
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
                                f"Address {service_instance.address} of {container.name} "
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


def build_sbatch(
    srun_command_str: str,
    partition: str,
    gpus: int = 0,
    args: Optional[str] = "",
) -> str:
    # reflect the correct slurm account name in the submit job
    slurm_account_name = os.environ["SLURM_JOB_ACCOUNT"]

    bash_script_template = f"""
    #!/bin/bash
    #SBATCH --account {slurm_account_name}
    #SBATCH --partition {partition}
    #SBATCH --time 03:59:00
    #SBATCH --gpus {gpus}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task 8
    #SBATCH --mem 32gb

    {args}

    """

    bash_script_template = dedent(bash_script_template).strip()
    bash_script_template += "\n\n" + srun_command_str

    return bash_script_template
