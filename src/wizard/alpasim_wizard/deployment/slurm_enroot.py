# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Direct Enroot deployment inside a Slurm allocation."""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from filelock import FileLock

from ..context import WizardContext
from ..schema import RunMode
from ..services import ContainerDefinition, VolumeMount
from .dispatcher import dispatch_background, terminate_process
from .fuse_tools import ensure_fuse_tools, fuse_tool_paths
from .slurm import SlurmDeployment

logger = logging.getLogger(__name__)

# Enroot cannot bind these through `--mount` with `x-create=auto`; they need a
# late hook that creates the destination inside the private overlay first.
_ENROOT_HOOK_MOUNT_PATHS = frozenset({"/", "/proc", "/sys"})
_HOST_EXPORTER_START_TIMEOUT_SECONDS = 10


class SlurmEnrootDeployment(SlurmDeployment):
    """Launch services directly with Enroot inside a Slurm allocation."""

    def __init__(self, context: WizardContext):
        if not context.cfg.wizard.fuse_dir:
            raise ValueError("wizard.fuse_dir is required for SLURM_ENROOT")
        super().__init__(context)
        self._staged_sqsh_paths: dict[str, str] = {}

    def _prepare_deployment(self, stack: ExitStack) -> None:
        cache_config = self._node_local_cache_config()
        if not self.context.cfg.wizard.dry_run:
            ensure_fuse_tools(self._fuse_dir())
            if cache_config is not None:
                self._stage_sqsh_images(*cache_config)
        stack.enter_context(self._host_process_exporter())
        stack.callback(self._cleanup_enroot_runtime_paths)

    def _node_local_cache_config(self) -> tuple[Path, float] | None:
        cache_dir_value = self.context.cfg.wizard.node_local_sqsh_cache_dir
        if not cache_dir_value:
            return None
        cache_limit = self.context.cfg.wizard.node_local_sqsh_cache_max_gib * 1024**3
        if cache_limit <= 0:
            raise ValueError("wizard.node_local_sqsh_cache_max_gib must be positive")
        return Path(cache_dir_value).resolve(), cache_limit

    def _stage_sqsh_images(self, cache_dir: Path, cache_limit: float) -> None:
        """Copy service images into the configured node-local cache."""
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        cache_dir.chmod(0o700)

        containers = [*self.container_set.sim, self.container_set.prometheus]
        if self.container_set.runtime is not None:
            containers.append(self.container_set.runtime)

        source_paths = set()
        for container in containers:
            source_paths.add(super()._sqsh_path(container))
        digests = {
            source_path: self._sqsh_digest(Path(source_path))
            for source_path in source_paths
        }
        destinations = {
            source_path: self._cache_destination(
                cache_dir,
                Path(source_path),
                digests[source_path],
            )
            for source_path in source_paths
        }
        protected_names = {path.name for path in destinations.values()}
        with FileLock(cache_dir / ".cache.lock"):
            for partial in cache_dir.glob(".*.partial"):
                partial.unlink()

            for source_path in sorted(source_paths):
                source = Path(source_path)
                destination = destinations[source_path]
                source_size = source.stat().st_size
                with FileLock(str(destination) + ".lock"):
                    cache_hit = (
                        destination.is_file()
                        and destination.stat().st_size == source_size
                        and self._sqsh_digest(destination) == digests[source_path]
                    )
                    if cache_hit:
                        destination.touch()
                        logger.info("Using node-local squash image %s", destination)
                    else:
                        destination.unlink(missing_ok=True)
                        self._evict_sqsh_images(
                            cache_dir=cache_dir,
                            additional_bytes=source_size,
                            cache_limit=cache_limit,
                            protected_names=protected_names,
                        )
                        file_descriptor, partial_path = tempfile.mkstemp(
                            dir=cache_dir,
                            prefix=f".{source.name}.",
                            suffix=".partial",
                        )
                        os.close(file_descriptor)
                        partial = Path(partial_path)
                        started_at = time.monotonic()
                        try:
                            subprocess.run(
                                [
                                    "cp",
                                    "--reflink=never",
                                    "--",
                                    str(source),
                                    str(partial),
                                ],
                                check=True,
                            )
                            if partial.stat().st_size != source_size:
                                raise OSError(
                                    f"Staged squash image has wrong size: {partial}"
                                )
                            if self._sqsh_digest(partial) != digests[source_path]:
                                raise OSError(
                                    f"Staged squash image has wrong digest: {partial}"
                                )
                            os.replace(partial, destination)
                        finally:
                            partial.unlink(missing_ok=True)
                        logger.info(
                            "Staged squash image: source=%s destination=%s seconds=%.2f",
                            source,
                            destination,
                            time.monotonic() - started_at,
                        )
                self._staged_sqsh_paths[source_path] = str(destination)

    @staticmethod
    def _sqsh_digest(source: Path) -> str:
        with source.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _cache_destination(
        cache_dir: Path, source: Path, digest: str | None = None
    ) -> Path:
        digest = digest or SlurmEnrootDeployment._sqsh_digest(source)
        return cache_dir / f"{source.stem}-{digest}{source.suffix}"

    @staticmethod
    def _evict_sqsh_images(
        cache_dir: Path,
        additional_bytes: int,
        cache_limit: float,
        protected_names: set[str],
    ) -> None:
        """Evict least-recently-used images until the new image fits."""
        cached_images = list(cache_dir.glob("*.sqsh"))
        projected_size = sum(path.stat().st_size for path in cached_images)
        projected_size += additional_bytes
        for image in sorted(cached_images, key=lambda path: path.stat().st_mtime):
            if projected_size <= cache_limit:
                return
            if image.name in protected_names:
                continue
            image_size = image.stat().st_size
            image.unlink()
            projected_size -= image_size
            logger.info("Evicted node-local squash image %s", image)
        if projected_size > cache_limit:
            raise OSError(
                f"Node-local squash images require {projected_size} bytes, "
                f"exceeding the {int(cache_limit)}-byte cache limit"
            )

    def _sqsh_path(self, container: ContainerDefinition) -> str:
        """Return the node-local image path when it has been staged."""
        source_path = super()._sqsh_path(container)
        return self._staged_sqsh_paths.get(source_path, source_path)

    def _get_dispatch_command(
        self,
        container: ContainerDefinition,
        mode: str | RunMode,
    ) -> str:
        run_mode = RunMode[mode.upper()] if isinstance(mode, str) else mode
        logger.info("Launch %s in %s", container.uuid, run_mode.name)
        return self._to_enroot_run(container, mode=run_mode)

    def _cleanup_launched_services(
        self,
        launched: dict[
            str,
            tuple[ContainerDefinition, subprocess.Popen[str] | None],
        ],
    ) -> None:
        for container, process in launched.values():
            if process is None:
                continue
            try:
                terminate_process(process)
            except Exception as e:
                logger.warning(
                    "Failed to clean up Enroot process for %s: %s",
                    container.uuid,
                    e,
                )

    def _fuse_dir(self) -> str:
        fuse_dir = self.context.cfg.wizard.fuse_dir
        assert fuse_dir is not None, "wizard.fuse_dir is required for SLURM_ENROOT"
        return fuse_dir

    def _enroot_runtime_root(self) -> Path:
        job_id = self.context.cfg.wizard.slurm_job_id
        assert job_id is not None, "SLURM environment not detected"
        return Path(f"/tmp/alpasim-enroot-runtime-{job_id}")

    def _enroot_runtime_path(self, container: ContainerDefinition) -> Path:
        return self._enroot_runtime_root() / container.uuid

    def _enroot_environment_prefix(self) -> str:
        fuse_bin_dir, fuse_lib_dir = fuse_tool_paths(self._fuse_dir())
        return (
            "ENROOT_MOUNT_HOME=no "
            f"PATH={shlex.quote(str(fuse_bin_dir))}:$PATH "
            f"LD_LIBRARY_PATH={shlex.quote(str(fuse_lib_dir))}"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        )

    def _write_enroot_config(
        self,
        container: ContainerDefinition,
    ) -> Path | None:
        """Create late hook mounts for host filesystems that need mount targets."""
        special_volumes = [
            volume
            for volume in container.volumes
            if volume.host in _ENROOT_HOOK_MOUNT_PATHS
        ]
        if not special_volumes:
            return None

        config_dir = Path(self.context.cfg.wizard.log_dir) / "enroot-configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{container.uuid}.sh"

        lines = ["hooks() {"]
        for volume in sorted(special_volumes, key=lambda item: item.container):
            destination = volume.container
            options = volume.options or "rw"
            mount = f"{volume.host} {destination} none rbind,{options}"
            lines.extend(
                [
                    f'    mkdir -p -- "${{ENROOT_ROOTFS}}"{shlex.quote(destination)}',
                    f"    printf '%s\\n' {shlex.quote(mount)} " '>> "${ENROOT_MOUNTS}"',
                ]
            )
        lines.append("}")
        config_path.write_text("\n".join(lines) + "\n")
        return config_path

    @staticmethod
    def _to_enroot_mount(volume: VolumeMount) -> str:
        options = volume.options or "rw"
        return f"{volume.host}:{volume.container}:none:x-create=auto,bind,{options}"

    def _cleanup_enroot_runtime_paths(self) -> None:
        """Remove job-local per-service Enroot runtime directories."""
        if self.context.cfg.wizard.dry_run:
            return
        runtime_root = self._enroot_runtime_root()
        if not runtime_root.exists():
            return
        try:
            shutil.rmtree(runtime_root)
        except OSError as e:
            logger.warning(
                "Failed to clean up Enroot runtime path %s: %s",
                runtime_root,
                e,
            )

    @contextmanager
    def _host_process_exporter(self) -> Iterator[None]:
        """Run process telemetry outside Enroot to retain host procfs access."""
        cfg = self.context.cfg.wizard
        job_id = cfg.slurm_job_id
        assert job_id is not None, "SLURM environment not detected"
        command = shlex.join(
            [
                sys.executable,
                "-m",
                "alpasim_wizard.telemetry.slurm_process_exporter",
                f"--job-id={job_id}",
                f"--port={self.context.telemetry_ports.process_exporter}",
                "--procfs=/proc",
                "--cgroupfs=/sys/fs/cgroup",
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

    def _to_enroot_run(
        self,
        container: ContainerDefinition,
        mode: RunMode,
    ) -> str:
        """Generate a direct Enroot command for a container."""
        log_path = self._service_log_path(container)
        sqsh_path = self._sqsh_path(container)
        environments = list(container.environments or [])
        environments.append("SLURM_JOB_ID")
        mps_enabled = self._mps_enabled(container)
        if container.gpu is not None:
            environments.extend(
                [
                    f"CUDA_VISIBLE_DEVICES={container.gpu}",
                    "NVIDIA_VISIBLE_DEVICES=all",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                ]
            )
        if mps_enabled:
            environments.append(f"CUDA_MPS_PIPE_DIRECTORY={self._mps_pipe_dir()}")

        mounts = [
            self._to_enroot_mount(volume)
            for volume in container.volumes
            if volume.host not in _ENROOT_HOOK_MOUNT_PATHS
        ]
        if mps_enabled:
            mounts.append(
                self._to_enroot_mount(
                    VolumeMount(self._mps_pipe_dir(), self._mps_pipe_dir())
                )
            )

        command = container.command.replace("$$", "$")
        if container.workdir is not None:
            command = f"cd {shlex.quote(container.workdir)} && {command}"

        if mode not in (RunMode.ONESHOT, RunMode.SERVER):
            raise ValueError(f"Unknown run mode: {mode}")

        config_path = self._write_enroot_config(container)
        parts = [
            self._enroot_environment_prefix(),
            f"ENROOT_RUNTIME_PATH={shlex.quote(str(self._enroot_runtime_path(container)))}",
            "enroot",
            "start",
            "--rw",
        ]
        if config_path is not None:
            parts.append(f"--conf={shlex.quote(str(config_path))}")
        if container.service_config.remap_root:
            parts.append("--root")
        parts.extend(f"--mount={shlex.quote(mount)}" for mount in mounts)
        parts.extend(
            f"--env={shlex.quote(environment)}" for environment in environments
        )
        parts.extend(
            [
                shlex.quote(sqsh_path),
                "bash",
                "-c",
                shlex.quote(command),
                ">",
                shlex.quote(log_path),
                "2>&1",
            ]
        )
        return " ".join(parts)
