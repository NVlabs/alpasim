# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import alpasim_wizard.configuration as configuration
import pytest
import yaml
from alpasim_utils.scene_timing import SceneTimingRejectionReason
from alpasim_utils.types import ClipDataSource, Scene
from alpasim_wizard.configuration import ConfigurationManager
from alpasim_wizard.context import TelemetryPorts, WizardContext
from alpasim_wizard.deployment.docker_compose import DockerComposeDeployment
from alpasim_wizard.scenes.materialization import SceneMaterialization, SceneRejection
from alpasim_wizard.schema import (
    ContainerConfig,
    DebugFlags,
    RunMethod,
    RunMode,
    RuntimeServiceConfig,
    SceneSuiteConfig,
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
            gpu_slots=None,
            run_sim_services=run_sim_services,
            runtime_server_port=None,
            debug_flags=DebugFlags(use_localhost=False),
            validate_mount_points=False,
            log_dir=str(tmp_path),
            external_services=None,
            prometheus=SimpleNamespace(
                scrape_interval="5s",
                file_sd_dir=None,
                start_prometheus=True,
                run_uuid=None,
            ),
            slurm_job_id=0,
            run_name="test-run",
            submitter=None,
            description=None,
        ),
        scenes=SimpleNamespace(
            nre_version_string="26.02",
            suite_configs=None,
            suite_groups={},
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
        gpu_allocation=(),
        scene_list=[],
    )


def test_service_gpu_subset_keeps_full_allocation_for_uuid_discovery(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.wizard.gpu_slots = [2]
    cfg.services.renderer.gpus = [0, 0]
    context = _context(cfg)
    context.gpu_allocation = (4, 5, 6)

    containers = build_container_set(context, "uuid")
    renderers = [
        container for container in containers.sim if container.name == "renderer"
    ]

    assert [container.gpu for container in renderers] == [2, 2]
    assert [context.resolve_gpu_slot(container.gpu) for container in renderers] == [
        6,
        6,
    ]
    assert context.gpu_allocation == (4, 5, 6)
    assert cfg.services.renderer.gpus == [0, 0]


@pytest.mark.parametrize(
    "pool, service, error",
    [
        ([3], [0], RuntimeError),
        ([2], [1], IndexError),
        ([-1], [0], RuntimeError),
        ([2], [-1], ValueError),
    ],
)
def test_service_gpu_subset_rejects_slots_outside_the_pool(
    tmp_path: Path,
    pool: list[int],
    service: list[int],
    error: type[Exception],
) -> None:
    cfg = _cfg(tmp_path)
    cfg.wizard.gpu_slots = pool
    cfg.services.renderer.gpus = service
    context = _context(cfg)
    context.gpu_allocation = (4, 5, 6)

    with pytest.raises(error):
        build_container_set(context, "uuid")


def test_service_gpu_subset_rejects_duplicate_pool_slots(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.wizard.gpu_slots = [2, 2]
    context = _context(cfg)
    context.gpu_allocation = (4, 5, 6)

    with pytest.raises(ValueError, match="must not contain duplicate slots"):
        build_container_set(context, "uuid")


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


@pytest.mark.parametrize("run_method", [RunMethod.SLURM, RunMethod.SLURM_ENROOT])
def test_slurm_disables_container_process_exporter(
    tmp_path: Path,
    run_method: RunMethod,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.wizard.run_method = run_method

    container_set = build_container_set(_context(cfg), "uuid")

    assert "ALPASIM_PROCESS_EXPORTER_MODE=host" in container_set.prometheus.environments


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


def test_runtime_config_serializes_complete_scenes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    context = _context(cfg)
    context.scene_list = [
        Scene(ClipDataSource("clipgt-a", "uuid-a"), 1_600_000),
        Scene(ClipDataSource("clipgt-b", "uuid-b"), "AUTO"),
    ]

    ConfigurationManager(str(tmp_path))._generate_runtime_config(cfg, context)

    runtime_config = yaml.safe_load(
        (tmp_path / "generated-user-config-0.yaml").read_text()
    )
    assert runtime_config["scenes"] == [
        {
            "source": {"clip_id": "clipgt-a", "usdz_uuid": "uuid-a"},
            "policy_start_us": 1_600_000,
        },
        {
            "source": {"clip_id": "clipgt-b", "usdz_uuid": "uuid-b"},
            "policy_start_us": "AUTO",
        },
    ]


def test_runtime_config_checks_recorded_navigation_without_rewriting_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    route_root = tmp_path / "route"
    route_dir = route_root / "indices"
    route_dir.mkdir(parents=True)
    (route_dir / "a.pkl").touch()
    cfg = _cfg(tmp_path)
    cfg.services.runtime.volumes = [
        f"{route_root}:/mnt/route:ro",
    ]
    cfg.runtime.simulation_config.nav_instruction = {
        "source": "recorded",
        "route_index_dir": "/mnt/route/indices",
    }
    context = _context(cfg)
    context.scene_list = [
        Scene(ClipDataSource("clipgt-a", "uuid-a"), "AUTO"),
        Scene(ClipDataSource("clipgt-b", "uuid-b"), "AUTO"),
    ]

    ConfigurationManager(str(tmp_path))._generate_runtime_config(cfg, context)

    assert "clipgt-b" in caplog.text
    runtime_config = yaml.safe_load(
        (tmp_path / "generated-user-config-0.yaml").read_text()
    )
    assert runtime_config["simulation_config"]["nav_instruction"] == {
        "source": "recorded",
        "route_index_dir": "/mnt/route/indices",
    }


@pytest.mark.parametrize("existing_manifest", [False, True])
def test_scene_manifest_records_minimal_suite_and_scene_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_manifest: bool
) -> None:
    suite_path = tmp_path / "suite-a.csv"
    suite_path.write_text("clip_id,uuid,event_t0_us\nclipgt-a,uuid-a,10\n")
    cfg = _cfg(tmp_path)
    cfg.scenes.suite_configs = [
        SceneSuiteConfig(csv_path=str(suite_path), t0_offset_us=-2)
    ]
    context = _context(cfg)
    context.scene_list = [Scene(ClipDataSource("clipgt-a", "uuid-a"), 8)]
    manifest_path = tmp_path / "scene-manifest.yaml"
    previous_manifest = "previous: true\n"
    if existing_manifest:
        manifest_path.write_text(previous_manifest)
    write_yaml = configuration.write_yaml

    def inspect_during_write(data: dict, file_path: str) -> None:
        Path(file_path).write_text("scenes:\n  - source:")
        if existing_manifest:
            assert manifest_path.read_text() == previous_manifest
        else:
            assert not manifest_path.exists()
        write_yaml(data, file_path)

    monkeypatch.setattr(configuration, "write_yaml", inspect_during_write)
    ConfigurationManager(str(tmp_path))._write_scene_manifest(context)

    manifest = yaml.safe_load((tmp_path / "scene-manifest.yaml").read_text())
    assert manifest == {
        "scene_suites": [
            {
                "scene_suite_id": "suite-a",
                "csv_path": str(suite_path.resolve()),
                "source_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                "t0_offset_us": -2,
            }
        ],
        "scenes": [
            {
                "source": {"clip_id": "clipgt-a", "usdz_uuid": "uuid-a"},
                "policy_start_us": 8,
            }
        ],
    }


def test_scene_manifest_records_accepted_suite_groups(tmp_path: Path) -> None:
    training_path = tmp_path / "training.csv"
    validation_path = tmp_path / "validation.csv"
    training_path.write_text("clip_id,uuid,event_t0_us\nclipgt-a,uuid-a,10\n")
    validation_path.write_text("clip_id,uuid,event_t0_us\nclipgt-b,uuid-b,20\n")
    cfg = _cfg(tmp_path)
    cfg.scenes.suite_groups = {
        "training": [SceneSuiteConfig(csv_path=str(training_path))],
        "validation": [SceneSuiteConfig(csv_path=str(validation_path))],
    }
    training = Scene(ClipDataSource("clipgt-a", "uuid-a"), 10)
    validation = Scene(ClipDataSource("clipgt-b", "uuid-b"), 20)
    context = _context(cfg)
    context.scene_list = [training, validation]
    context.scene_groups = {
        "training": (training,),
        "validation": (validation,),
    }

    ConfigurationManager(str(tmp_path))._write_scene_manifest(context)

    manifest = yaml.safe_load((tmp_path / "scene-manifest.yaml").read_text())
    assert [suite["group"] for suite in manifest["scene_suites"]] == [
        "training",
        "validation",
    ]
    assert manifest["scene_groups"] == {
        "training": ["clipgt-a@t0=10"],
        "validation": ["clipgt-b@t0=20"],
    }


def test_scene_manifest_records_materialization_audit(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    context = _context(cfg)
    accepted = Scene(ClipDataSource("clipgt-a", "uuid-a"), 13_000_000)
    rejected = Scene(ClipDataSource("clipgt-b", "uuid-b"), 14_000_000)
    context.scene_list = [accepted]
    context.scene_materialization = SceneMaterialization(
        requested_count=2,
        accepted_scenes=(accepted,),
        rejections=(
            SceneRejection(
                scene=rejected,
                reason=SceneTimingRejectionReason.INSUFFICIENT_HORIZON,
                clip_duration_us=19_733_638,
                requested_end_us=20_000_000,
            ),
        ),
        force_gt_duration_us=1_600_000,
        n_sim_steps=38,
        control_timestep_us=200_000,
        accepted_sha256="accepted-digest",
    )

    ConfigurationManager(str(tmp_path))._write_scene_manifest(context)

    manifest = yaml.safe_load((tmp_path / "scene-manifest.yaml").read_text())
    assert manifest["scenes"] == [
        {
            "source": {"clip_id": "clipgt-a", "usdz_uuid": "uuid-a"},
            "policy_start_us": 13_000_000,
        }
    ]
    assert manifest["scene_materialization"] == {
        "simulation_timing": {
            "force_gt_duration_us": 1_600_000,
            "n_sim_steps": 38,
            "control_timestep_us": 200_000,
        },
        "requested_count": 2,
        "accepted_count": 1,
        "rejected_count": 1,
        "invalid_percent": 50.0,
        "rejection_counts": {
            "insufficient_history": 0,
            "insufficient_horizon": 1,
        },
        "accepted_sha256": "accepted-digest",
        "rejections": [
            {
                "source": {"clip_id": "clipgt-b", "usdz_uuid": "uuid-b"},
                "policy_start_us": 14_000_000,
                "reason": "insufficient_horizon",
                "clip_duration_us": 19_733_638,
                "requested_end_us": 20_000_000,
            }
        ],
    }


def test_prometheus_configs_publish_runtime_targets_and_recording_rules(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    context = _context(cfg, baseport=6100)
    manager = ConfigurationManager(str(tmp_path))

    run_metadata = manager._load_or_create_run_metadata(cfg)
    manager._generate_runtime_config(cfg, context)
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
    assert "instance" not in worker_target["labels"]

    prometheus_config = yaml.safe_load(
        (tmp_path / "prometheus/prometheus.yml").read_text()
    )
    assert prometheus_config["rule_files"] == ["/mnt/log_dir/prometheus/rules/*.yml"]

    rules = yaml.safe_load(
        (tmp_path / "prometheus/rules/alpasim-recording-rules.yml").read_text()
    )
    rules_by_name = {
        rule["record"]: rule["expr"] for rule in rules["groups"][0]["rules"]
    }
    rule_names = set(rules_by_name)
    assert "alpasim:rpc_queue_depth_at_start_latest:max" in rule_names
    assert "alpasim:rpc_queue_depth_at_start_latest:min" in rule_names
    assert "alpasim:simulation_rollouts_completed:sum" in rule_names
    throughput_rule = rules_by_name["alpasim:simulation_seconds_per_rollout:avg"]
    assert "alpasim_simulation_elapsed_seconds" in throughput_rule
    assert "alpasim_rollouts_total" in throughput_rule
    assert "alpasim_rollout_duration_seconds" not in throughput_rule
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


def test_parent_run_uuid_and_disabled_prometheus_keep_exporter_discovery(
    tmp_path: Path,
) -> None:
    """An orchestrator can own Prometheus while AlpaSim runs its exporters."""
    cfg = _cfg(tmp_path)
    cfg.wizard.prometheus.start_prometheus = False
    cfg.wizard.prometheus.run_uuid = "4ed94022-fc8d-409b-a0d8-745e6bdb44ab"
    cfg.wizard.prometheus.file_sd_dir = str(tmp_path / "file-sd")
    context = _context(cfg, baseport=6100)
    container_set = build_container_set(context, "uuid")
    manager = ConfigurationManager(str(tmp_path))

    assert "START_PROMETHEUS=false" in container_set.prometheus.command
    assert container_set.prometheus.service_instances[0].port == 6103
    run_metadata = manager._load_or_create_run_metadata(cfg)
    manager._generate_runtime_config(cfg, context)
    prometheus.generate_prometheus_configs(tmp_path, run_metadata, context)
    runtime_config = yaml.safe_load(
        (tmp_path / "generated-user-config-0.yaml").read_text()
    )
    targets = json.loads(
        (tmp_path / "file-sd" / f"{run_metadata['run_uuid']}.json").read_text()
    )

    assert run_metadata["run_uuid"] == cfg.wizard.prometheus.run_uuid
    assert runtime_config["prometheus"]["url"] is None
    assert not (tmp_path / "prometheus/prometheus.yml").exists()
    assert (tmp_path / "prometheus/process-exporter.yml").exists()
    assert len(targets) == 4
    assert {target["labels"]["job"] for target in targets} == {
        "alpasim-runtime-worker",
        "alpasim-node",
        "alpasim-process",
        "alpasim-dcgm",
    }


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


def test_file_sd_cleanup_tolerates_a_concurrent_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_sd_dir = tmp_path / "file-sd"
    file_sd_dir.mkdir(parents=True)
    old_file = file_sd_dir / "old.json"
    old_file.write_text(
        json.dumps(
            [
                {
                    "targets": ["host-a:6100"],
                    "labels": {"job": "alpasim-runtime-worker"},
                }
            ]
        )
    )
    old_mtime = time.time() - 6 * 60 * 60
    os.utime(old_file, (old_mtime, old_mtime))

    def remove_while_checking(_target: str) -> bool:
        old_file.unlink()
        return False

    monkeypatch.setattr(prometheus, "_target_reachable", remove_while_checking)

    prometheus._cleanup_stale_file_sd(file_sd_dir)

    assert not old_file.exists()


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
    assert central_path.stat().st_mode & 0o7777 == 0o666
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


def _generate_runtime_simulation_config(tmp_path: Path, cfg) -> dict:
    ConfigurationManager(str(tmp_path))._generate_runtime_config(cfg, _context(cfg))
    user_config = yaml.safe_load(
        (tmp_path / "generated-user-config-0.yaml").read_text()
    )
    return user_config["simulation_config"]


def test_runtime_config_renderer_cache_fingerprint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.runtime = OmegaConf.create(
        {
            "nr_workers": 1,
            "endpoints": {"do_shutdown": True},
            "simulation_config": {"force_gt_frame_cache": {"enabled": True}},
        }
    )
    cfg.services.renderer = OmegaConf.create(
        {"image": "nre:26.04", "command": ["serve", "--enable-fixer"]}
    )
    key = _generate_runtime_simulation_config(tmp_path, cfg)["force_gt_frame_cache"][
        "extra_key"
    ]
    assert key.startswith("renderer-")

    # A renderer flag change must produce a different fingerprint.
    cfg.services.renderer = OmegaConf.create(
        {"image": "nre:26.04", "command": ["serve"]}
    )
    changed = _generate_runtime_simulation_config(tmp_path, cfg)[
        "force_gt_frame_cache"
    ]["extra_key"]
    assert changed != key

    # An explicit key is preserved (no auto fingerprint).
    cfg.runtime.simulation_config.force_gt_frame_cache.extra_key = "manual-key"
    assert (
        _generate_runtime_simulation_config(tmp_path, cfg)["force_gt_frame_cache"][
            "extra_key"
        ]
        == "manual-key"
    )


def test_no_fingerprint_when_caching_disabled(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.runtime = OmegaConf.create(
        {"nr_workers": 1, "endpoints": {"do_shutdown": True}, "simulation_config": {}}
    )
    cfg.services.renderer = OmegaConf.create(
        {"image": "nre:26.04", "command": ["serve"]}
    )
    sim_config = _generate_runtime_simulation_config(tmp_path, cfg)
    assert "extra_key" not in sim_config.get("force_gt_frame_cache", {})


def test_caching_requires_local_renderer_for_fingerprint(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.runtime = OmegaConf.create(
        {
            "nr_workers": 1,
            "endpoints": {"do_shutdown": True},
            "simulation_config": {"force_gt_frame_cache": {"enabled": True}},
        }
    )
    cfg.services.renderer = None
    with pytest.raises(ValueError, match="external_services"):
        _generate_runtime_simulation_config(tmp_path, cfg)


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
