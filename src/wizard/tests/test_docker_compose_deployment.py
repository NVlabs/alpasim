# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from alpasim_wizard.deployment.docker_compose import DockerComposeDeployment


def _deployment(tmp_path: Path, *, dry_run: bool) -> DockerComposeDeployment:
    deployment = DockerComposeDeployment.__new__(DockerComposeDeployment)
    deployment.context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                dry_run=dry_run,
                log_dir=str(tmp_path),
            )
        )
    )
    deployment.container_set = SimpleNamespace(runtime=[object()])
    deployment.docker_compose_filepath = "docker-compose.yaml"
    return deployment


def test_docker_compose_dry_run_does_not_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    deployment = _deployment(tmp_path, dry_run=True)

    def fail_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("docker compose should not run in dry-run mode")

    monkeypatch.setattr(
        "alpasim_wizard.deployment.docker_compose.subprocess.run",
        fail_run,
    )

    with caplog.at_level(
        logging.INFO,
        logger="alpasim_wizard.deployment.docker_compose",
    ):
        deployment.deploy_all_services()

    assert "[DRY-RUN] Would execute: docker compose" in caplog.text
