# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""SLURM deployment strategy."""

from __future__ import annotations

import logging
import os
import shlex
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, List

from ..context import WizardContext
from ..schema import RunMode, WizardConfig
from ..services import ContainerDefinition, build_container_set
from ..utils import ensure_sqsh_path
from .dispatcher import dispatch_background, dispatch_command, terminate_process

logger = logging.getLogger(__name__)

_HOST_EXPORTER_START_TIMEOUT_SECONDS = 10


@contextmanager
def mps_daemon(cfg: WizardConfig) -> Iterator[None]:
    """Run a CUDA MPS control daemon on this node at ``cfg.mps_pipe_dir``.

    A stale pipe directory (e.g. from a requeued job) is removed first so the
    daemon does not fail on leftover sockets. On exit the daemon is told to quit
    and the pipe directory is removed, even when the shutdown command fails.
    """
    pipe_dir = cfg.mps_pipe_dir
    log_dir = Path(cfg.log_dir)
    dispatch_command(
        f"rm -rf {pipe_dir} && mkdir -p {pipe_dir} {log_dir}/mps && "
        f"CUDA_MPS_PIPE_DIRECTORY={pipe_dir} CUDA_MPS_LOG_DIRECTORY={log_dir}/mps "
        "nvidia-cuda-mps-control -d",
        log_dir=log_dir,
        dry_run=cfg.dry_run,
    )
    try:
        yield
    finally:
        try:
            try:
                dispatch_command(
                    f"echo quit | CUDA_MPS_PIPE_DIRECTORY={pipe_dir} "
                    "nvidia-cuda-mps-control",
                    log_dir=log_dir,
                    dry_run=cfg.dry_run,
                )
            finally:
                dispatch_command(
                    f"rm -rf {pipe_dir}",
                    log_dir=log_dir,
                    dry_run=cfg.dry_run,
                )
        except Exception as e:
            logger.warning("Failed to stop MPS daemon: %s", e)


class SlurmDeployment:
    """Deploy services as containerized Slurm job steps."""

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
        with ExitStack() as stack:
            self._prepare_deployment(stack)
            self.deploy(
                containers=containers,
                containers_to_start_last=containers_to_start_last,
            )

    def _prepare_deployment(self, stack: ExitStack) -> None:
        """Register launcher-specific resources with the deployment lifecycle."""
        stack.enter_context(self._host_process_exporter())

    @contextmanager
    def _host_process_exporter(self) -> Iterator[None]:
        """Run process telemetry on the host so it can inspect host process IDs."""
        cfg = self.context.cfg.wizard
        job_id = cfg.slurm_job_id
        assert job_id is not None, "SLURM environment not detected"
        containers = [*self.container_set.sim, self.container_set.prometheus]
        if self.container_set.runtime is not None:
            containers.append(self.container_set.runtime)
        gpu_indices = sorted(
            {
                self.context.resolve_gpu_slot(container.gpu)
                for container in containers
                if container.gpu is not None
            }
        )
        command = shlex.join(
            [
                sys.executable,
                "-m",
                "alpasim_wizard.telemetry.slurm_process_exporter",
                f"--job-id={job_id}",
                f"--port={self.context.telemetry_ports.process_exporter}",
                "--procfs=/proc",
                "--cgroupfs=/sys/fs/cgroup",
                *(f"--gpu-index={gpu_index}" for gpu_index in gpu_indices),
            ]
        )
        process = dispatch_background(
            command,
            log_dir=Path(cfg.log_dir),
            dry_run=cfg.dry_run,
        )
        try:
            if process is not None:
                self._wait_for_host_exporter(process)
            yield
        finally:
            if process is not None:
                terminate_process(process)

    def _wait_for_host_exporter(self, process: subprocess.Popen[str]) -> None:
        port = self.context.telemetry_ports.process_exporter
        deadline = time.monotonic() + _HOST_EXPORTER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Host process exporter exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError(f"Host process exporter did not open port {port}")

    def _service_log_path(self, container: ContainerDefinition) -> str:
        job_id = container.context.cfg.wizard.slurm_job_id
        assert job_id is not None, "SLURM environment not detected"
        restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
        return (
            f"{container.context.cfg.wizard.log_dir}/txt-logs/"
            f"out-{job_id}-attempt-{restart_count}-{container.uuid}-log.txt"
        )

    def _mps_pipe_dir(self, container: ContainerDefinition) -> str | None:
        """Return the MPS pipe directory a GPU service joins, or None without MPS."""
        wizard_cfg = container.context.cfg.wizard
        if not wizard_cfg.enable_mps or container.gpu is None:
            return None
        return wizard_cfg.mps_pipe_dir

    def _sqsh_path(self, container: ContainerDefinition) -> str:
        return ensure_sqsh_path(
            container.service_config.image,
            list(container.context.cfg.wizard.sqshcaches),
        )

    def deploy(
        self,
        containers: List[Any],
        containers_to_start_last: List[Any] | None = None,
    ) -> None:
        """Deploy containers using SLURM."""
        launched: dict[
            str,
            tuple[ContainerDefinition, subprocess.Popen[str] | None],
        ] = {}
        launch_attempts: dict[str, int] = {}
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

        try:
            # Deploy containers with retries
            readiness_waits = 0
            while True:
                missing_containers = self.get_missing_containers(containers)
                if not missing_containers:
                    break

                for c in missing_containers:
                    previous_launch = launched.get(c.uuid)
                    running_process = (
                        previous_launch[1] if previous_launch is not None else None
                    )
                    if running_process is not None and running_process.poll() is None:
                        logger.info(
                            "Service %s is not ready but its launcher is still running",
                            c.uuid,
                        )
                        continue

                    attempts = launch_attempts.get(c.uuid, 0)
                    if attempts >= nr_retries:
                        continue

                    process = dispatch_background(
                        self._get_dispatch_command(
                            c,
                            self.context.cfg.wizard.run_mode.name,
                        ),
                        log_dir=Path(self.context.cfg.wizard.log_dir),
                        dry_run=self.context.cfg.wizard.dry_run,
                    )
                    launch_attempts[c.uuid] = attempts + 1
                    launched[c.uuid] = (c, process)

                # Wait for containers if needed
                if _wait_for_containers_running():
                    try:
                        if self.wait_for_containers(
                            containers,
                            timeout=self.context.cfg.wizard.timeout,
                            raise_on_timeout=(readiness_waits >= nr_retries - 1),
                        ):
                            break
                    except TimeoutError:
                        relaunchable_container_found = False
                        for c in self.get_missing_containers(containers):
                            previous_launch = launched.get(c.uuid)
                            process = (
                                previous_launch[1]
                                if previous_launch is not None
                                else None
                            )
                            if (
                                process is not None
                                and process.poll() is not None
                                and launch_attempts.get(c.uuid, 0) < nr_retries
                            ):
                                launched.pop(c.uuid)
                                relaunchable_container_found = True
                        if not relaunchable_container_found:
                            raise
                    else:
                        readiness_waits += 1
                else:
                    break

            # Deploy containers that should start last
            if containers_to_start_last:
                for c in containers_to_start_last:
                    dispatch_command(
                        self._get_dispatch_command(
                            c,
                            self.context.cfg.wizard.run_mode.name,
                        ),
                        log_dir=Path(self.context.cfg.wizard.log_dir),
                        dry_run=self.context.cfg.wizard.dry_run,
                    )
        finally:
            if containers_to_start_last:
                self._cleanup_launched_services(launched)

    def _cleanup_launched_services(
        self,
        launched: dict[
            str,
            tuple[ContainerDefinition, subprocess.Popen[str] | None],
        ],
    ) -> None:
        """Stop launched services after the blocking runtime exits."""
        self._cleanup_launched_service_steps(
            [container for container, _ in launched.values()]
        )

    def _cleanup_launched_service_steps(
        self, containers: list[ContainerDefinition]
    ) -> None:
        """Cancel non-runtime service steps after the blocking runtime exits."""
        if self.context.cfg.wizard.dry_run:
            return

        for container in containers:
            try:
                dispatch_command(
                    self._get_slurm_cleanup_command(container),
                    log_dir=Path(self.context.cfg.wizard.log_dir),
                    dry_run=False,
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
        s_log = self._service_log_path(container)
        sqsh = self._sqsh_path(container)

        # Note that we cannot use --export=CUDA_VISIBLE_DEVICES=... with srun because SLURM
        # overrides CUDA_VISIBLE_DEVICES even when exported as an environment variable.
        # Instead we set it in the command line. Use export to allow chaining commands with &&.
        mps_pipe_dir = self._mps_pipe_dir(container)
        s_gpu = ""
        if container.gpu is not None:
            # Topology GPUs are allocation-relative slots, but nested service srun steps use
            # job-global GPU IDs; map through the Wizard step (or its direct batch job).
            device = self.context.resolve_gpu_slot(container.gpu)
            logger.info(
                "Container %s maps GPU slot %d to device %d",
                container.uuid,
                container.gpu,
                device,
            )
            s_gpu = f"export CUDA_VISIBLE_DEVICES={device};"
        if mps_pipe_dir is not None:
            s_gpu += f"export CUDA_MPS_PIPE_DIRECTORY={mps_pipe_dir};"
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

        mounts = [v.to_str() for v in container.volumes]
        if mps_pipe_dir is not None:
            # enroot containers have an isolated rootfs; the MPS pipe directory
            # must be mounted explicitly for clients to reach the daemon.
            mounts.append(f"{mps_pipe_dir}:{mps_pipe_dir}")
        s_mnt = ",".join(mounts)

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

    def _get_dispatch_command(
        self,
        container: ContainerDefinition,
        mode: str | RunMode,
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

        logger.info("Launch %s in %s", container.uuid, run_mode.name)
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
