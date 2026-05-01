# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import yaml
from alpasim_wizard.configuration import ConfigurationManager
from alpasim_wizard.context import WizardContext
from alpasim_wizard.deployment.docker_compose import DockerComposeDeployment
from alpasim_wizard.schema import (
    DebugFlags,
    RunMode,
    RuntimeServiceConfig,
    ServiceConfig,
)
from alpasim_wizard.services import build_container_set
from omegaconf import OmegaConf


def _port_assigner(start: int) -> Iterator[int]:
    port = start
    while True:
        yield port
        port += 1


def _service(command: list[str]) -> ServiceConfig:
    return ServiceConfig(
        volumes=[],
        image="test-image",
        command=command,
        replicas_per_container=1,
        gpus=None,
    )


def _runtime_service() -> RuntimeServiceConfig:
    return RuntimeServiceConfig(
        volumes=[],
        image="runtime-image",
        command=[
            "uv run python -m alpasim_runtime.simulate",
            "--user-config=/mnt/log_dir/generated-user-config-0.yaml",
            "--network-config=/mnt/log_dir/generated-network-config.yaml",
            "--eval-config=/mnt/log_dir/eval-config.yaml",
            "--log-dir=/mnt/log_dir",
        ],
        replicas_per_container=1,
        gpus=None,
        depends_on=[],
    )


def _cfg(tmp_path: Path, *, run_sim_services: list[str] | None = None):
    if run_sim_services is None:
        run_sim_services = [
            "driver",
            "sensorsim",
            "physics",
            "trafficsim",
            "controller",
            "runtime",
        ]

    return SimpleNamespace(
        wizard=SimpleNamespace(
            run_mode=RunMode.SERVER,
            run_method=SimpleNamespace(name="DOCKER_COMPOSE"),
            run_sim_services=run_sim_services,
            runtime_server_port=None,
            debug_flags=DebugFlags(use_localhost=False),
            validate_mount_points=False,
            log_dir=str(tmp_path),
            external_services=None,
            slurm_job_id=0,
            submitter=None,
            description=None,
        ),
        scenes=SimpleNamespace(
            nre_version_string="26.02",
            test_suite_id=None,
        ),
        services=SimpleNamespace(
            driver=_service(["driver", "--port={port}"]),
            sensorsim=_service(["sensorsim", "--port={port}"]),
            physics=_service(["physics", "--port={port}"]),
            trafficsim=_service(["trafficsim", "--port={port}"]),
            controller=_service(["controller", "--port={port}"]),
            runtime=_runtime_service(),
        ),
        runtime=OmegaConf.create(
            {
                "endpoints": {"do_shutdown": True},
                "simulation_config": {},
            }
        ),
    )


def _context(cfg, *, baseport: int = 6100) -> WizardContext:
    return WizardContext(
        cfg=cfg,
        port_assigner=_port_assigner(baseport),
        artifact_list=[],
        num_gpus=0,
        sceneset_path="sceneset-a",
    )


def test_server_mode_generates_and_publishes_runtime_endpoint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    context = _context(cfg, baseport=6100)
    container_set = build_container_set(context, "uuid")
    deployment = DockerComposeDeployment.__new__(DockerComposeDeployment)
    deployment.context = context

    runtime = container_set.runtime[0]
    assert "--serve" in runtime.command
    assert "--listen-address=0.0.0.0:6105" in runtime.command
    assert deployment._to_docker_compose_service(runtime)["ports"] == ["6105:6105"]

    manager = ConfigurationManager(str(tmp_path))
    manager._generate_runtime_server_config(container_set, cfg)

    endpoint = yaml.safe_load((tmp_path / "generated-runtime-server.yaml").read_text())
    assert endpoint == {
        "host": "localhost",
        "port": 6105,
    }


def test_external_services_are_marked_unmanaged_in_network_config(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, run_sim_services=["runtime"])
    cfg.wizard.external_services = {"driver": ["localhost:6789"]}
    manager = ConfigurationManager(str(tmp_path))

    manager._generate_network_config([], cfg)

    network = yaml.safe_load((tmp_path / "generated-network-config.yaml").read_text())
    assert network["driver"]["endpoints"] == [
        {"address": "localhost:6789", "managed": False}
    ]
