# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from alpasim_wizard.context import TelemetryPorts, WizardContext
from alpasim_wizard.deployment.slurm import SlurmDeployment
from alpasim_wizard.schema import DebugFlags, RunMode


def _context(
    tmp_path: Path, *, dry_run: bool = False, enable_mps: bool = False
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
            debug_flags=DebugFlags(use_localhost=False),
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
    tmp_path: Path, *, dry_run: bool = False, enable_mps: bool = False
) -> SlurmDeployment:
    deployment = SlurmDeployment.__new__(SlurmDeployment)
    deployment.context = _context(tmp_path, dry_run=dry_run, enable_mps=enable_mps)
    return deployment


def _container(uuid: str) -> SimpleNamespace:
    return SimpleNamespace(uuid=uuid)


def _slurm_container(deployment: SlurmDeployment, gpu: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        uuid="driver-0",
        context=deployment.context,
        service_config=SimpleNamespace(image="driver-image", remap_root=False),
        gpu=gpu,
        environments=[],
        volumes=[],
        workdir=None,
        command="echo ok",
    )


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        (0, "export CUDA_VISIBLE_DEVICES=0;"),
        (2, "export CUDA_VISIBLE_DEVICES=2;"),
        (None, None),
    ],
)
def test_slurm_run_exports_explicit_gpu_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gpu: int | None,
    expected: str | None,
) -> None:
    deployment = _deployment(tmp_path)
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: f"{image}.sqsh",
    )

    command = deployment._to_slurm_run(
        _slurm_container(deployment, gpu), RunMode.ONESHOT
    )

    if expected is None:
        assert "CUDA_VISIBLE_DEVICES" not in command
    else:
        assert expected in command


def test_slurm_run_isolates_submit_environment_except_job_id_unless_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: f"{image}.sqsh",
    )

    default_command = deployment._to_slurm_run(
        _slurm_container(deployment, None), RunMode.ONESHOT
    )
    container = _slurm_container(deployment, None)
    container.environments = ["HF_TOKEN", "HOME=/tmp", "XDG_CACHE_HOME=/tmp/.cache"]
    explicit_command = deployment._to_slurm_run(container, RunMode.ONESHOT)

    assert "--export=SLURM_JOB_ID " in default_command
    assert "--export=SLURM_JOB_ID,HF_TOKEN " in explicit_command
    assert "export HOME=/tmp;" in explicit_command
    assert "export XDG_CACHE_HOME=/tmp/.cache;" in explicit_command


def test_slurm_cleanup_runs_after_blocking_runtime_srun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    driver = _container("driver-0")
    runtime = _container("runtime-0")
    events = []

    monkeypatch.setattr(
        deployment,
        "get_missing_containers",
        lambda containers: containers,
    )
    monkeypatch.setattr(deployment, "wait_for_containers", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        deployment,
        "_get_slurm_dispatch_command",
        lambda container, mode: object(),
    )

    def fake_dispatch(command, *, log_dir, dry_run, blocking):
        del command, log_dir, dry_run
        events.append(("dispatch", blocking))
        return ""

    def fake_cleanup(containers):
        events.append(("cleanup", [container.uuid for container in containers]))

    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.dispatch_command",
        fake_dispatch,
    )
    monkeypatch.setattr(deployment, "_cleanup_launched_service_steps", fake_cleanup)

    deployment.deploy([driver], containers_to_start_last=[runtime])

    assert events == [
        ("dispatch", False),
        ("dispatch", True),
        ("cleanup", ["driver-0"]),
    ]


def test_slurm_cleanup_targets_only_launched_non_runtime_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    driver = _container("driver-0")
    renderer = _container("renderer-0")
    runtime = _container("runtime-0")
    cleaned_up = []

    monkeypatch.setattr(
        deployment,
        "get_missing_containers",
        lambda _containers: [driver],
    )
    monkeypatch.setattr(deployment, "wait_for_containers", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        deployment,
        "_get_slurm_dispatch_command",
        lambda container, mode: object(),
    )

    def fake_dispatch(command, *, log_dir, dry_run, blocking):
        del log_dir, dry_run, blocking
        del command
        return ""

    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.dispatch_command",
        fake_dispatch,
    )
    monkeypatch.setattr(
        deployment,
        "_cleanup_launched_service_steps",
        lambda containers: cleaned_up.extend(
            container.uuid for container in containers
        ),
    )

    deployment.deploy([driver, renderer], containers_to_start_last=[runtime])

    assert cleaned_up == ["driver-0"]


def test_slurm_cleanup_failure_does_not_mask_runtime_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    deployment = _deployment(tmp_path)
    driver = _container("driver-0")
    runtime = _container("runtime-0")

    monkeypatch.setattr(
        deployment,
        "get_missing_containers",
        lambda containers: containers,
    )
    monkeypatch.setattr(deployment, "wait_for_containers", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        deployment,
        "_get_slurm_dispatch_command",
        lambda container, mode: (
            "runtime" if container.uuid == "runtime-0" else "service"
        ),
    )
    monkeypatch.setattr(
        deployment,
        "_get_slurm_cleanup_command",
        lambda container: "cleanup",
    )

    def fake_dispatch(command, *, log_dir, dry_run, blocking):
        del log_dir, dry_run, blocking
        if command == "runtime":
            raise RuntimeError("runtime failed")
        if command == "cleanup":
            raise RuntimeError("cleanup failed")
        return ""

    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.dispatch_command",
        fake_dispatch,
    )

    with caplog.at_level(logging.WARNING, logger="alpasim_wizard.deployment.slurm"):
        with pytest.raises(RuntimeError, match="runtime failed"):
            deployment.deploy([driver], containers_to_start_last=[runtime])

    assert "Failed to clean up SLURM step for driver-0" in caplog.text
    assert "cleanup failed" in caplog.text


def test_slurm_dry_run_does_not_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path, dry_run=True)
    driver = _container("driver-0")
    runtime = _container("runtime-0")
    commands = []

    monkeypatch.setattr(
        deployment,
        "get_missing_containers",
        lambda containers: containers,
    )
    monkeypatch.setattr(
        deployment,
        "_get_slurm_dispatch_command",
        lambda container, mode: object(),
    )

    def fake_dispatch(command, *, log_dir, dry_run, blocking):
        del command, log_dir, dry_run
        commands.append(blocking)
        return ""

    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.dispatch_command",
        fake_dispatch,
    )

    deployment.deploy([driver], containers_to_start_last=[runtime])

    assert commands == [False, True]


def test_slurm_run_mps_env_and_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path, enable_mps=True)
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: f"{image}.sqsh",
    )

    gpu_command = deployment._to_slurm_run(
        _slurm_container(deployment, gpu=1), RunMode.ONESHOT
    )
    assert "export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-123;" in gpu_command
    assert "--container-mounts=/tmp/nvidia-mps-123:/tmp/nvidia-mps-123" in gpu_command

    cpu_command = deployment._to_slurm_run(
        _slurm_container(deployment, gpu=None), RunMode.ONESHOT
    )
    assert "nvidia-mps" not in cpu_command


def test_slurm_run_without_mps_has_no_mps_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path)
    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.ensure_sqsh_path",
        lambda image, caches: f"{image}.sqsh",
    )

    command = deployment._to_slurm_run(
        _slurm_container(deployment, gpu=1), RunMode.ONESHOT
    )
    assert "nvidia-mps" not in command


def test_mps_daemon_start_and_stop_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path, enable_mps=True)
    commands = []

    def fake_dispatch(command, *, log_dir, dry_run, blocking):
        del log_dir, dry_run, blocking
        commands.append(command)
        return ""

    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.dispatch_command",
        fake_dispatch,
    )

    deployment._start_mps_daemon()
    deployment._stop_mps_daemon()

    start, stop, cleanup = commands
    assert "nvidia-cuda-mps-control -d" in start
    assert "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-123" in start
    assert f"CUDA_MPS_LOG_DIRECTORY={tmp_path}/mps" in start
    assert stop.startswith("echo quit |")
    assert cleanup == "rm -rf /tmp/nvidia-mps-123"


def test_mps_pipe_dir_cleanup_runs_when_daemon_shutdown_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment = _deployment(tmp_path, enable_mps=True)
    commands = []

    def fake_dispatch(command, *, log_dir, dry_run, blocking):
        del log_dir, dry_run, blocking
        commands.append(command)
        if command.startswith("echo quit |"):
            raise RuntimeError("MPS shutdown failed")
        return ""

    monkeypatch.setattr(
        "alpasim_wizard.deployment.slurm.dispatch_command",
        fake_dispatch,
    )

    deployment._stop_mps_daemon()

    assert len(commands) == 2
    assert commands[1] == "rm -rf /tmp/nvidia-mps-123"
