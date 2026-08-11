# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpasim_wizard.context import TelemetryPorts, WizardContext
from alpasim_wizard.deployment.slurm_enroot import SlurmEnrootDeployment
from alpasim_wizard.schema import RunMode
from alpasim_wizard.services import VolumeMount


def _context(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    enable_mps: bool = False,
    node_local_sqsh_cache_dir: str | None = None,
) -> WizardContext:
    cfg = SimpleNamespace(
        wizard=SimpleNamespace(
            log_dir=str(tmp_path),
            dry_run=dry_run,
            timeout=1,
            nr_retries=1,
            run_mode=RunMode.ONESHOT,
            slurm_job_id=123,
            sqshcaches=[],
            slurm_cpu_bind_none=False,
            enable_mps=enable_mps,
            fuse_dir="/cache/fuse-tools",
            node_local_sqsh_cache_dir=node_local_sqsh_cache_dir,
            node_local_sqsh_cache_max_gib=500,
        )
    )
    return WizardContext(
        cfg=cfg,
        port_assigner=iter(()),
        telemetry_ports=TelemetryPorts(
            workers=(),
            prometheus=6100,
            node_exporter=6101,
            process_exporter=6102,
            dcgm_exporter=6103,
        ),
        artifact_list=[],
        num_gpus=0,
    )


def _deployment(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    enable_mps: bool = False,
    node_local_sqsh_cache_dir: str | None = None,
) -> SlurmEnrootDeployment:
    deployment = SlurmEnrootDeployment.__new__(SlurmEnrootDeployment)
    deployment.context = _context(
        tmp_path,
        dry_run=dry_run,
        enable_mps=enable_mps,
        node_local_sqsh_cache_dir=node_local_sqsh_cache_dir,
    )
    deployment._staged_sqsh_paths = {}
    return deployment


def _container(
    deployment: SlurmEnrootDeployment,
    gpu: int | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name="driver",
        uuid="driver-0",
        context=deployment.context,
        service_config=SimpleNamespace(image="driver-image", remap_root=False),
        gpu=gpu,
        environments=[],
        volumes=[],
        workdir=None,
        command="echo ok",
    )


def test_enroot_run_uses_sqsh_and_preserves_container_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path, enable_mps=True)
    container = _container(deployment, gpu=2)
    container.service_config.remap_root = True
    container.environments = ["HF_TOKEN", "HOME=/tmp"]
    container.volumes = [VolumeMount("/host/data", "/container/data")]
    container.workdir = "/workspace"
    container.command = "echo $$HOME"
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: "/cache/driver.sqsh",
    )

    command = deployment._to_enroot_run(container, RunMode.ONESHOT)

    assert command.startswith(
        "ENROOT_MOUNT_HOME=no "
        "PATH=/cache/fuse-tools/bin:$PATH "
        "LD_LIBRARY_PATH=/cache/fuse-tools/lib"
        "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "
        "ENROOT_RUNTIME_PATH=/tmp/alpasim-enroot-runtime-123/driver-0 "
        "enroot start --rw"
    )
    assert " --root " in command
    assert "srun" not in command
    assert "ENROOT_DATA_PATH" not in command
    assert "--conf=" not in command
    assert "--mount=/host/data:/container/data:none:x-create=auto,bind,rw" in command
    assert (
        "--mount=/tmp/nvidia-mps-123:/tmp/nvidia-mps-123:none:"
        "x-create=auto,bind,rw" in command
    )
    assert "--env=HF_TOKEN" in command
    assert "--env=HOME=/tmp" in command
    assert "--env=SLURM_JOB_ID" in command
    assert "--env=CUDA_VISIBLE_DEVICES=2" in command
    assert "--env=NVIDIA_VISIBLE_DEVICES=all" in command
    assert "--env=CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-123" in command
    assert "cd /workspace && echo $HOME" in command
    assert "/cache/driver.sqsh" in command
    assert f"> {tmp_path}/txt-logs/out-123-attempt-0-driver-0-log.txt 2>&1" in command


def test_enroot_runtime_paths_are_isolated_per_service(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)

    first = _container(deployment, gpu=None)
    second = _container(deployment, gpu=None)
    second.uuid = "driver-1"

    assert deployment._enroot_runtime_path(first) == Path(
        "/tmp/alpasim-enroot-runtime-123/driver-0"
    )
    assert deployment._enroot_runtime_path(second) == Path(
        "/tmp/alpasim-enroot-runtime-123/driver-1"
    )


def test_enroot_runtime_cleanup_removes_job_scoped_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    runtime_root = tmp_path / "runtime"
    (runtime_root / "driver-0").mkdir(parents=True)
    monkeypatch.setattr(deployment, "_enroot_runtime_root", lambda: runtime_root)

    deployment._cleanup_enroot_runtime_paths()

    assert not runtime_root.exists()


def test_enroot_starts_process_exporter_on_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    process = MagicMock(spec=subprocess.Popen)
    process.poll.return_value = None
    commands = []

    def fake_dispatch(command, *, log_dir, dry_run):
        commands.append((command, log_dir, dry_run))
        return process

    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm_enroot.dispatch_background",
        fake_dispatch,
    )
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm_enroot.terminate_process",
        lambda candidate: commands.append(("stop", candidate)),
    )
    monkeypatch.setattr(deployment, "_wait_for_host_exporter", MagicMock())

    with deployment._host_process_exporter():
        pass

    command, log_dir, dry_run = commands[0]
    assert "alpasim_wizard.telemetry.slurm_process_exporter" in command
    assert "--job-id=123" in command
    assert "--port=6102" in command
    assert "--procfs=/proc" in command
    assert "--cgroupfs=/sys/fs/cgroup" in command
    assert Path(log_dir) == tmp_path
    assert not dry_run
    deployment._wait_for_host_exporter.assert_called_once_with(process)
    assert commands[1] == ("stop", process)


def test_enroot_rejects_host_process_exporter_startup_failure(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    process = MagicMock(spec=subprocess.Popen)
    process.poll.return_value = 1

    with pytest.raises(RuntimeError, match="exited during startup"):
        deployment._wait_for_host_exporter(process)


def test_enroot_cleanup_terminates_service_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    container = _container(deployment, gpu=None)
    process = MagicMock(spec=subprocess.Popen)
    terminated = []
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm_enroot.terminate_process",
        terminated.append,
    )

    deployment._cleanup_launched_services(
        {container.uuid: (container, process)},
    )

    assert terminated == [process]


def test_enroot_prepares_launcher_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    calls = []
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm_enroot.ensure_fuse_tools",
        lambda path: calls.append(path),
    )
    monkeypatch.setattr(deployment, "_host_process_exporter", MagicMock())
    monkeypatch.setattr(deployment, "_cleanup_enroot_runtime_paths", MagicMock())

    with ExitStack() as stack:
        deployment._prepare_deployment(stack)

    assert calls == ["/cache/fuse-tools"]
    deployment._host_process_exporter.assert_called_once_with()
    deployment._cleanup_enroot_runtime_paths.assert_called_once_with()


def test_enroot_dry_run_validates_cache_without_provisioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    deployment = _deployment(
        tmp_path,
        dry_run=True,
        node_local_sqsh_cache_dir=str(cache_dir),
    )
    ensure_tools = MagicMock()
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm_enroot.ensure_fuse_tools",
        ensure_tools,
    )
    monkeypatch.setattr(deployment, "_host_process_exporter", MagicMock())

    with ExitStack() as stack:
        deployment._prepare_deployment(stack)

    ensure_tools.assert_not_called()
    assert not cache_dir.exists()

    deployment.context.cfg.wizard.node_local_sqsh_cache_max_gib = 0
    with pytest.raises(ValueError, match="must be positive"):
        with ExitStack() as stack:
            deployment._prepare_deployment(stack)


def test_enroot_stages_sqsh_in_node_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source" / "alpasim_abc123.sqsh"
    source.parent.mkdir()
    source.write_bytes(b"immutable squash image")
    cache_dir = tmp_path / "cache"
    deployment = _deployment(
        tmp_path,
        node_local_sqsh_cache_dir=str(cache_dir),
    )
    container = _container(deployment, gpu=None)
    deployment.container_set = SimpleNamespace(
        sim=[container],
        prometheus=container,
        runtime=None,
    )
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: str(source),
    )

    cache_config = deployment._node_local_cache_config()
    assert cache_config is not None
    deployment._stage_sqsh_images(*cache_config)

    cached_image = deployment._cache_destination(cache_dir, source)
    assert cached_image.read_bytes() == source.read_bytes()
    assert deployment._sqsh_path(container) == str(cached_image)
    assert cache_dir.stat().st_mode & 0o777 == 0o700
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm_enroot.subprocess.run",
        MagicMock(side_effect=AssertionError("cache hit must not copy")),
    )

    deployment._stage_sqsh_images(*cache_config)


def test_enroot_replaces_incomplete_cached_sqsh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source" / "alpasim_abc123.sqsh"
    source.parent.mkdir()
    source.write_bytes(b"complete squash image")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    deployment = _deployment(
        tmp_path,
        node_local_sqsh_cache_dir=str(cache_dir),
    )
    cached_image = deployment._cache_destination(cache_dir, source)
    cached_image.write_bytes(b"incomplete")
    container = _container(deployment, gpu=None)
    deployment.container_set = SimpleNamespace(
        sim=[container],
        prometheus=container,
        runtime=None,
    )
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: str(source),
    )

    cache_config = deployment._node_local_cache_config()
    assert cache_config is not None
    deployment._stage_sqsh_images(*cache_config)

    assert cached_image.read_bytes() == source.read_bytes()
    assert not list(cache_dir.glob("*.partial"))


def test_enroot_evicts_old_sqsh_when_cache_is_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source" / "current.sqsh"
    source.parent.mkdir()
    source.write_bytes(b"new")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old_image = cache_dir / "old.sqsh"
    old_image.write_bytes(b"old")
    deployment = _deployment(
        tmp_path,
        node_local_sqsh_cache_dir=str(cache_dir),
    )
    deployment.context.cfg.wizard.node_local_sqsh_cache_max_gib = 5 / 1024**3
    container = _container(deployment, gpu=None)
    deployment.container_set = SimpleNamespace(
        sim=[container],
        prometheus=container,
        runtime=None,
    )
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: str(source),
    )

    cache_config = deployment._node_local_cache_config()
    assert cache_config is not None
    deployment._stage_sqsh_images(*cache_config)

    assert not old_image.exists()
    assert (
        deployment._cache_destination(cache_dir, source).read_bytes()
        == source.read_bytes()
    )


def test_enroot_cache_keys_same_basename_sources_by_content(tmp_path: Path) -> None:
    first = tmp_path / "first" / "image.sqsh"
    second = tmp_path / "second" / "image.sqsh"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    cache_dir = tmp_path / "cache"

    first_destination = SlurmEnrootDeployment._cache_destination(cache_dir, first)
    second_destination = SlurmEnrootDeployment._cache_destination(cache_dir, second)
    first.write_bytes(b"other")
    replacement_destination = SlurmEnrootDeployment._cache_destination(cache_dir, first)

    assert first_destination != second_destination
    assert replacement_destination != first_destination


def test_enroot_cache_reclaims_abandoned_partial_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source" / "current.sqsh"
    source.parent.mkdir()
    source.write_bytes(b"new")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    abandoned = cache_dir / ".current.sqsh.dead.partial"
    abandoned.write_bytes(b"abandoned")
    deployment = _deployment(
        tmp_path,
        node_local_sqsh_cache_dir=str(cache_dir),
    )
    container = _container(deployment, gpu=None)
    deployment.container_set = SimpleNamespace(
        sim=[container],
        prometheus=container,
        runtime=None,
    )
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: str(source),
    )

    cache_config = deployment._node_local_cache_config()
    assert cache_config is not None
    deployment._stage_sqsh_images(*cache_config)

    assert not abandoned.exists()


def test_enroot_cache_evicts_stale_requested_image_before_other_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_a = tmp_path / "z-source" / "a.sqsh"
    source_b = tmp_path / "a-source" / "b.sqsh"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_a.write_bytes(b"aa")
    source_b.write_bytes(b"bb")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale_a = cache_dir / "a-stale.sqsh"
    stale_a.write_bytes(b"12345678")
    deployment = _deployment(tmp_path, node_local_sqsh_cache_dir=str(cache_dir))
    deployment.context.cfg.wizard.node_local_sqsh_cache_max_gib = 8 / 1024**3
    container_a = _container(deployment, gpu=None)
    container_a.service_config.image = "a"
    container_b = _container(deployment, gpu=None)
    container_b.service_config.image = "b"
    deployment.container_set = SimpleNamespace(
        sim=[container_a, container_b],
        prometheus=container_a,
        runtime=None,
    )
    sources = {"a": str(source_a), "b": str(source_b)}
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: sources[image],
    )

    cache_config = deployment._node_local_cache_config()
    assert cache_config is not None
    deployment._stage_sqsh_images(*cache_config)

    assert not stale_a.exists()
    assert len(list(cache_dir.glob("*.sqsh"))) == 2


def test_enroot_direct_squash_creates_mount_destinations_with_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    container = _container(deployment, gpu=None)
    container.volumes = [
        VolumeMount("/data", "/mnt/data", "ro"),
        VolumeMount("/proc", "/host/proc", "ro"),
        VolumeMount("/sys", "/host/sys", "ro"),
        VolumeMount("/", "/rootfs", "ro"),
    ]
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: "/cache/driver-image.sqsh",
    )

    command = deployment._to_enroot_run(container, RunMode.ONESHOT)

    config_path = tmp_path / "enroot-configs" / "driver-0.sh"
    assert f"--conf={config_path}" in command
    assert " --rw " in command
    assert "/data:/mnt/data:none:x-create=auto,bind,ro" in command
    assert "/proc:/host/proc" not in command
    assert "/sys:/host/sys" not in command
    assert "/:/rootfs" not in command
    assert config_path.read_text() == (
        "hooks() {\n"
        '    mkdir -p -- "${ENROOT_ROOTFS}"/host/proc\n'
        "    printf '%s\\n' '/proc /host/proc none rbind,ro' "
        '>> "${ENROOT_MOUNTS}"\n'
        '    mkdir -p -- "${ENROOT_ROOTFS}"/host/sys\n'
        "    printf '%s\\n' '/sys /host/sys none rbind,ro' "
        '>> "${ENROOT_MOUNTS}"\n'
        '    mkdir -p -- "${ENROOT_ROOTFS}"/rootfs\n'
        "    printf '%s\\n' '/ /rootfs none rbind,ro' "
        '>> "${ENROOT_MOUNTS}"\n'
        "}\n"
    )
