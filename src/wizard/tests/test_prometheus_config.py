# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import alpasim_wizard.setup_omegaconf  # noqa: F401
from alpasim_wizard.services import resolve_prometheus_command
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_node_exporter_does_not_require_host_root() -> None:
    context = SimpleNamespace(
        cfg=SimpleNamespace(
            wizard=SimpleNamespace(
                prometheus=SimpleNamespace(start_prometheus=True),
            )
        ),
        telemetry_ports=SimpleNamespace(
            prometheus_service_ports=lambda: {
                "prometheus": 6100,
                "node_exporter": 6101,
                "process_exporter": 6102,
                "dcgm_exporter": 6103,
            }
        ),
    )

    command = resolve_prometheus_command(context)

    assert "--no-collector.os" in command
    assert "--no-collector.filesystem" in command
    assert "--path.rootfs" not in command


def test_prometheus_container_does_not_mount_host_root() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="base_config.yaml",
            overrides=[
                "deploy=local",
                "topology=1gpu",
                "driver=vavam",
                "wizard.log_dir=/tmp/alpasim-test",
            ],
        )

    assert "/proc:/host/proc:ro" in cfg.services.prometheus.volumes
    assert "/sys:/host/sys:ro" in cfg.services.prometheus.volumes
    assert "/:/rootfs:ro" not in cfg.services.prometheus.volumes
