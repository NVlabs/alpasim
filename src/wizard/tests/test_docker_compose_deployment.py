# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from alpasim_wizard.deployment.docker_compose import DockerComposeDeployment
from alpasim_wizard.schema import RunMode
from alpasim_wizard.services import Address, ContainerSet


def _local_addresses(*ports: int) -> list[Address]:
    return [Address(host="127.0.0.1", port=port) for port in ports]


def _deployment(tmp_path: Path, *, dry_run: bool) -> DockerComposeDeployment:
    deployment = DockerComposeDeployment.__new__(DockerComposeDeployment)
    deployment.context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                dry_run=dry_run,
                log_dir=str(tmp_path),
                debug_flags=SimpleNamespace(use_localhost=False),
                run_mode=RunMode.ONESHOT,
            )
        ),
        num_gpus=0,
    )
    deployment.container_set = SimpleNamespace(runtime=object())
    deployment.docker_compose_filepath = "docker-compose.yaml"
    return deployment


def _prometheus_container(
    *,
    name: str,
    port_name: str,
    port: int,
    command: str = "echo ok",
) -> SimpleNamespace:
    return SimpleNamespace(
        uuid=f"{name}-0",
        name=name,
        service_config=SimpleNamespace(
            image=f"{name}-image",
            external_image=True,
            pull_policy="missing",
        ),
        volumes=[],
        command=command,
        workdir=None,
        environments=[],
        gpu=None,
        published_ports={port_name: port},
        get_all_addresses=lambda: _local_addresses(port),
    )


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


def test_docker_compose_service_uses_configured_pull_policy(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path, dry_run=True)
    container = SimpleNamespace(
        name="renderer",
        service_config=SimpleNamespace(
            image="flashdreams-alpasim:local",
            external_image=True,
            pull_policy="never",
        ),
        volumes=[],
        command="",
        workdir=None,
        environments=[],
        gpu=None,
        published_ports={},
        get_all_addresses=lambda: _local_addresses(7100),
    )

    service = deployment._to_docker_compose_service(container)

    assert service["pull_policy"] == "never"


def test_docker_compose_adds_prometheus_before_runtime(
    tmp_path: Path,
) -> None:
    deployment = DockerComposeDeployment.__new__(DockerComposeDeployment)
    deployment.context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                log_dir=str(tmp_path),
                debug_flags=SimpleNamespace(use_localhost=False),
                run_mode=RunMode.ONESHOT,
            )
        ),
        num_gpus=0,
    )
    runtime_container = SimpleNamespace(
        uuid="runtime-0",
        name="runtime",
        service_config=SimpleNamespace(
            image="runtime-image",
            external_image=True,
            pull_policy="missing",
        ),
        volumes=[],
        command="uv run python -m alpasim_runtime.simulate",
        workdir=None,
        environments=[],
        gpu=None,
        published_ports={},
        get_all_addresses=lambda: _local_addresses(6200),
    )
    prometheus = _prometheus_container(
        name="prometheus", port_name="prometheus", port=6100
    )

    deployment.generate_docker_compose_yaml(
        ContainerSet(sim=[], prometheus=prometheus, runtime=runtime_container)
    )

    compose = yaml.safe_load((tmp_path / "docker-compose.yaml").read_text())
    services = compose["services"]
    assert list(services) == ["prometheus-0", "runtime-0"]
    assert services["runtime-0"].get("pid") is None
    assert services["runtime-0"].get("deploy") is None
    assert services["prometheus-0"].get("pid") is None
    assert services["prometheus-0"].get("cap_add") is None
    assert services["prometheus-0"].get("deploy") is None
    assert services["prometheus-0"]["ports"] == ["6100:6100"]
