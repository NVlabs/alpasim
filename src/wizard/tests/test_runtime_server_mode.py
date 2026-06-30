# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest
import yaml
from alpasim_wizard.configuration import ConfigurationManager
from alpasim_wizard.context import TelemetryPorts, WizardContext
from alpasim_wizard.deployment.docker_compose import DockerComposeDeployment
from alpasim_wizard.schema import (
    ContainerConfig,
    DebugFlags,
    RunMethod,
    RunMode,
    RuntimeServiceConfig,
    ServiceConfig,
)
from alpasim_wizard.services import build_container_set
from alpasim_wizard.setup_omegaconf import validate_config
from alpasim_wizard.telemetry import prometheus
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


def _cfg(
    tmp_path: Path,
    *,
    run_sim_services: list[str] | None = None,
):
    if run_sim_services is None:
        run_sim_services = [
            "driver",
            "renderer",
            "physics",
            "trafficsim",
            "controller",
            "runtime",
        ]

    return SimpleNamespace(
        wizard=SimpleNamespace(
            run_mode=RunMode.SERVER,
            run_method=RunMethod.DOCKER_COMPOSE,
            run_sim_services=run_sim_services,
            runtime_server_port=None,
            debug_flags=DebugFlags(use_localhost=False),
            validate_mount_points=False,
            log_dir=str(tmp_path),
            external_services=None,
            prometheus=SimpleNamespace(
                scrape_interval="5s",
                file_sd_dir=None,
            ),
            slurm_job_id=0,
            run_name="test-run",
            submitter=None,
            description=None,
        ),
        scenes=SimpleNamespace(
            nre_version_string="26.02",
            test_suite_id=None,
            sceneset_path="sceneset-a",
        ),
        services=SimpleNamespace(
            driver=_service(["driver", "--port={port}"]),
            renderer=_service(["renderer", "--port={port}"]),
            physics=_service(["physics", "--port={port}"]),
            trafficsim=_service(["trafficsim", "--port={port}"]),
            controller=_service(["controller", "--port={port}"]),
            runtime=_runtime_service(),
            prometheus=_prometheus_service(),
        ),
        runtime=OmegaConf.create(
            {
                "nr_workers": 2,
                "endpoints": {"do_shutdown": True},
                "simulation_config": {},
            }
        ),
    )


def _prometheus_service() -> ContainerConfig:
    return ContainerConfig(
        volumes=[],
        image="test-image",
    )


def _context(cfg, *, baseport: int = 6100) -> WizardContext:
    port_assigner = _port_assigner(baseport)
    nr_workers = int(cfg.runtime.nr_workers)
    telemetry_ports = TelemetryPorts(
        workers=tuple(next(port_assigner) for _ in range(nr_workers)),
        prometheus=next(port_assigner),
        node_exporter=next(port_assigner),
        process_exporter=next(port_assigner),
        dcgm_exporter=next(port_assigner),
    )
    return WizardContext(
        cfg=cfg,
        port_assigner=port_assigner,
        telemetry_ports=telemetry_ports,
        artifact_list=[],
        num_gpus=0,
    )


def test_server_mode_generates_and_publishes_runtime_endpoint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    context = _context(cfg, baseport=6100)
    container_set = build_container_set(context, "uuid")
    deployment = DockerComposeDeployment.__new__(DockerComposeDeployment)
    deployment.context = context

    runtime = container_set.runtime
    assert runtime is not None
    assert "--serve" in runtime.command
    assert "--listen-address=0.0.0.0:6111" in runtime.command
    assert deployment._to_docker_compose_service(runtime)["ports"] == [
        "6100:6100",
        "6101:6101",
        "6111:6111",
    ]

    manager = ConfigurationManager(str(tmp_path))
    manager._generate_runtime_server_config(container_set, cfg)

    endpoint = yaml.safe_load((tmp_path / "generated-runtime-server.yaml").read_text())
    assert endpoint == {
        "host": "localhost",
        "port": 6111,
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


def test_managed_renderer_is_written_as_renderer_endpoint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    context = _context(cfg, baseport=6100)
    container_set = build_container_set(context, "uuid")
    manager = ConfigurationManager(str(tmp_path))

    manager._generate_network_config(container_set.sim, cfg)

    network = yaml.safe_load((tmp_path / "generated-network-config.yaml").read_text())
    assert network["renderer"]["endpoints"] == [
        {"address": "renderer-0:6107", "managed": True}
    ]
    assert "sensorsim" not in network


def test_prometheus_configs_publish_runtime_targets_and_recording_rules(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    context = _context(cfg, baseport=6100)
    manager = ConfigurationManager(str(tmp_path))

    run_metadata = manager._load_or_create_run_metadata(cfg)
    manager._generate_runtime_config(cfg, [], context)
    prometheus.generate_prometheus_configs(
        tmp_path,
        run_metadata,
        context,
    )
    manager._write_config("run_metadata.yaml", run_metadata)

    runtime_config = yaml.safe_load(
        (tmp_path / "generated-user-config-0.yaml").read_text()
    )
    assert runtime_config["prometheus"]["worker_ports"] == [6100, 6101]
    assert runtime_config["prometheus"]["url"] == "http://prometheus-0:6102"
    assert "scrape_interval" not in runtime_config["prometheus"]
    assert "file_sd_dir" not in runtime_config["prometheus"]
    assert "grafana" not in runtime_config["prometheus"]

    run_metadata = yaml.safe_load((tmp_path / "run_metadata.yaml").read_text())

    targets = json.loads((tmp_path / "prometheus/targets/alpasim.json").read_text())
    worker_target = targets[0]
    assert worker_target["targets"] == ["runtime-0:6100", "runtime-0:6101"]
    assert worker_target["labels"]["job"] == "alpasim-runtime-worker"
    assert worker_target["labels"]["run_uuid"] == run_metadata["run_uuid"]
    assert worker_target["labels"]["run_name"] == "test-run"
    assert "worker_id" not in worker_target["labels"]

    prometheus_config = yaml.safe_load(
        (tmp_path / "prometheus/prometheus.yml").read_text()
    )
    assert prometheus_config["rule_files"] == ["/mnt/log_dir/prometheus/rules/*.yml"]

    rules = yaml.safe_load(
        (tmp_path / "prometheus/rules/alpasim-recording-rules.yml").read_text()
    )
    rule_names = {rule["record"] for rule in rules["groups"][0]["rules"]}
    assert "alpasim:rpc_queue_depth_at_start_latest:max" in rule_names
    assert "alpasim:rpc_queue_depth_at_start_latest:min" in rule_names
    assert "alpasim:simulation_rollouts_completed:sum" in rule_names
    assert "alpasim:process_cpu_utilization_percent:max_by_group:rate30s" in rule_names
    assert "alpasim:gpu_memory_pressure_percent:avg" in rule_names


def test_run_metadata_is_loaded_for_resume(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    existing_metadata = {
        "run_time": "2026-06-23 12:00:00",
        "run_name": "original-run",
        "run_uuid": "original-run-uuid",
        "slurm_job_id": 123,
        "run_user": "original-user",
        "run_dir": "/original",
        "run_args": "original-args",
        "submitter": None,
        "description": None,
        "test_suite_id": None,
    }
    (tmp_path / "run_metadata.yaml").write_text(yaml.dump(existing_metadata))

    manager = ConfigurationManager(str(tmp_path))
    run_metadata = manager._load_or_create_run_metadata(cfg)

    assert run_metadata == existing_metadata


def test_file_sd_cleanup_removes_old_unreachable_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_sd_dir = tmp_path / "file-sd"
    file_sd_dir.mkdir(parents=True)
    old_file = file_sd_dir / "old.json"
    new_file = file_sd_dir / "new.json"
    payload = [
        {"targets": ["host-a:6100"], "labels": {"job": "alpasim-runtime-worker"}}
    ]
    old_file.write_text(json.dumps(payload))
    new_file.write_text(json.dumps(payload))
    old_mtime = time.time() - 6 * 60 * 60
    os.utime(old_file, (old_mtime, old_mtime))

    monkeypatch.setattr(prometheus, "_target_reachable", lambda target: False)

    prometheus._cleanup_stale_file_sd(file_sd_dir)

    assert not old_file.exists()
    assert new_file.exists()


def test_central_file_sd_publication_is_removed_on_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    file_sd_dir = tmp_path / "central-file-sd"
    cfg.wizard.prometheus.file_sd_dir = str(file_sd_dir)
    context = _context(cfg, baseport=6100)
    manager = ConfigurationManager(str(tmp_path))
    monkeypatch.setattr(prometheus.socket, "gethostbyname", lambda host: "192.0.2.10")

    run_metadata = manager._load_or_create_run_metadata(cfg)
    manager._central_file_sd_path = prometheus.generate_prometheus_configs(
        tmp_path,
        run_metadata,
        context,
    )
    central_path = file_sd_dir / f"{run_metadata['run_uuid']}.json"
    assert central_path.exists()
    central_targets = json.loads(central_path.read_text())
    assert central_targets[0]["targets"] == ["192.0.2.10:6100", "192.0.2.10:6101"]
    assert central_targets[0]["labels"]["node"] != "192.0.2.10"

    manager.cleanup_central_file_sd()

    assert not central_path.exists()


def test_external_video_model_is_first_class_network_endpoint(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, run_sim_services=["runtime"])
    cfg.wizard.external_services = {"renderer": ["localhost:50056"]}
    manager = ConfigurationManager(str(tmp_path))

    manager._generate_network_config([], cfg)

    network = yaml.safe_load((tmp_path / "generated-network-config.yaml").read_text())
    assert network["renderer"]["endpoints"] == [
        {"address": "localhost:50056", "managed": False}
    ]
    assert "sensorsim" not in network
    assert "video_model" not in network
    assert "extra_services" not in network


def test_run_sim_services_rejects_unset_renderer_service() -> None:
    cfg = OmegaConf.create(
        {
            "wizard": {"run_sim_services": ["renderer"]},
            "services": {"renderer": None},
        }
    )

    with pytest.raises(RuntimeError, match=r"Services \['renderer'\].*set to null"):
        validate_config(cfg)


def test_combined_renderer_physics_maps_physics_to_renderer_container(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, run_sim_services=["renderer", "physics", "runtime"])
    cfg.services.physics.image = "*renderer*"
    cfg.services.physics.command = ["noop"]
    context = _context(cfg, baseport=6100)
    container_set = build_container_set(context, "uuid")
    manager = ConfigurationManager(str(tmp_path))

    manager._generate_network_config(container_set.sim, cfg)

    network = yaml.safe_load((tmp_path / "generated-network-config.yaml").read_text())
    assert network["renderer"]["endpoints"] == [
        {"address": "renderer-0:6106", "managed": True}
    ]
    assert network["physics"]["endpoints"] == [
        {"address": "renderer-0:6106", "managed": True}
    ]
