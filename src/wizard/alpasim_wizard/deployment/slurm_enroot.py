# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Direct Enroot deployment inside a Slurm allocation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock

from ..context import WizardContext
from ..schema import RunMode
from ..services import ContainerDefinition, VolumeMount
from .dispatcher import terminate_process
from .fuse_tools import ensure_fuse_tools, fuse_tool_paths
from .slurm import SlurmDeployment

logger = logging.getLogger(__name__)

# Enroot cannot bind these through `--mount` with `x-create=auto`; they need a
# late hook that creates the destination inside the private overlay first.
_ENROOT_HOOK_MOUNT_PATHS = frozenset({"/", "/proc", "/sys"})


def _resolve_enroot_gpu_uuids(
    allocated_gpus: tuple[int, ...],
) -> tuple[str, ...]:
    """Return allocated GPU UUIDs in topology-slot order."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    visible_gpus: dict[int, str] = {}
    for line in result.stdout.splitlines():
        index, uuid = line.split(",", maxsplit=1)
        visible_gpus[int(index)] = uuid.strip()

    try:
        return tuple(visible_gpus[device] for device in allocated_gpus)
    except KeyError as error:
        if tuple(visible_gpus) == tuple(range(len(allocated_gpus))):
            # ConstrainDevices renumbers allocated GRES to 0..N-1 in allocation order.
            return tuple(visible_gpus.values())
        raise RuntimeError(
            "Cannot reconcile Slurm GPU allocation "
            f"{allocated_gpus} with nvidia-smi indices {tuple(visible_gpus)}"
        ) from error


class SlurmEnrootDeployment(SlurmDeployment):
    """Launch services directly with Enroot inside a Slurm allocation."""

    def __init__(self, context: WizardContext):
        if not context.cfg.wizard.fuse_dir:
            raise ValueError("wizard.fuse_dir is required for SLURM_ENROOT")
        super().__init__(context)
        self._staged_sqsh_paths: dict[str, str] = {}
        self._gpu_uuids: tuple[str, ...] | None = None
        self._command_rc_path: Path | None = None

    def _prepare_deployment(self, stack: ExitStack) -> None:
        cache_config = self._node_local_cache_config()
        if not self.context.cfg.wizard.dry_run:
            ensure_fuse_tools(self._fuse_dir())
            if cache_config is not None:
                self._stage_sqsh_images(*cache_config)
        super()._prepare_deployment(stack)
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
        with FileLock(cache_dir / ".cache.lock"):
            digest_path = cache_dir / ".digests.json"
            records = (
                json.loads(digest_path.read_text()) if digest_path.exists() else {}
            )
            digests = {
                source_path: self._cached_sqsh_digest(Path(source_path), records)
                for source_path in source_paths
            }
            destinations = {
                source_path: self._cache_destination(
                    cache_dir, Path(source_path), digests[source_path]
                )
                for source_path in source_paths
            }
            protected_names = {path.name for path in destinations.values()}
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
            # The cache lock covers both image publication and its digest metadata.
            metadata_partial = cache_dir / ".digests.json.partial"
            metadata_partial.write_text(json.dumps(records))
            os.replace(metadata_partial, digest_path)

    @staticmethod
    def _sqsh_fingerprint(source: Path) -> list[int]:
        """Identify normal file replacements and writes, even with restored mtime."""
        stat = source.stat()
        return [
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ]

    @classmethod
    def _cached_sqsh_digest(cls, source: Path, records: dict) -> str:
        """Reuse a prior full hash only while the file's identity is unchanged."""
        fingerprint = cls._sqsh_fingerprint(source)
        record = records.get(str(source))
        if (
            record is not None
            and record[0] == fingerprint
            and record[2] - fingerprint[4] >= 2_000_000_000
        ):
            logger.info("Startup timing: phase=squash_hash_reused path=%s", source)
            return record[1]
        # Shared filesystems can expose only whole-second ctime. A hash taken
        # in the same timestamp window cannot certify later same-size writes.
        started_ns = time.time_ns()
        digest = cls._sqsh_digest(source)
        if cls._sqsh_fingerprint(source) != fingerprint:
            raise OSError(f"Squash image changed while hashing: {source}")
        records[str(source)] = [fingerprint, digest, started_ns]
        return digest

    @staticmethod
    def _sqsh_digest(source: Path) -> str:
        started = time.monotonic()
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        logger.info(
            "Startup timing: phase=squash_hash path=%s bytes=%d seconds=%.3f",
            source,
            source.stat().st_size,
            time.monotonic() - started,
        )
        return digest

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

    def _command_rc(self) -> Path:
        """Return the start script that runs a service command without the image entrypoint.

        Enroot 4 runs an image's entrypoint with the given command as its arguments
        (``/app/run bash -c ...`` for the NRE renderer). The script keeps the image's
        working directory and /etc/rc.local, then runs the command itself. It is written
        once per deployment: rewriting it could truncate it while an earlier service's
        ``enroot start`` is reading it.
        """
        if self._command_rc_path is not None:
            return self._command_rc_path
        config_dir = Path(self.context.cfg.wizard.log_dir) / "enroot-configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        rc_path = config_dir / "command-rc.sh"
        rc_path.write_text(
            r"""workdir=$(sed -n 's/^cd "\(.*\)" && unset OLDPWD.*$/\1/p' /etc/rc)
cd "${workdir:-/}" && unset OLDPWD || exit 1
if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi
exec "$@"
"""
        )
        self._command_rc_path = rc_path
        return rc_path

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
        mps_pipe_dir = self._mps_pipe_dir(container)
        if container.gpu is not None:
            device = self.context.resolve_gpu_slot(container.gpu)
            logger.info(
                "Container %s maps GPU slot %d to device %d",
                container.uuid,
                container.gpu,
                device,
            )
            if self._gpu_uuids is None:
                self._gpu_uuids = _resolve_enroot_gpu_uuids(self.context.gpu_allocation)
            gpu_uuid = self._gpu_uuids[container.gpu]
            environments.extend(
                [
                    # Enroot may retain multiple device nodes from the Slurm step.
                    f"CUDA_VISIBLE_DEVICES={gpu_uuid}",
                    f"NVIDIA_VISIBLE_DEVICES={gpu_uuid}",
                    "NVIDIA_DRIVER_CAPABILITIES=all",
                ]
            )
        if mps_pipe_dir is not None:
            environments.append(f"CUDA_MPS_PIPE_DIRECTORY={mps_pipe_dir}")

        mounts = [
            self._to_enroot_mount(volume)
            for volume in container.volumes
            if volume.host not in _ENROOT_HOOK_MOUNT_PATHS
        ]
        if mps_pipe_dir is not None:
            mounts.append(
                self._to_enroot_mount(VolumeMount(mps_pipe_dir, mps_pipe_dir))
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
            f"--rc={shlex.quote(str(self._command_rc()))}",
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
