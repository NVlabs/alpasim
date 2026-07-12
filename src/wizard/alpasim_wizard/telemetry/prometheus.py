# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Prometheus and file-SD config generation."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

from alpasim_wizard.context import WizardContext
from alpasim_wizard.utils import write_json, write_yaml

logger = logging.getLogger(__name__)

TELEMETRY_LOG_DIR = "/mnt/log_dir"
PROMETHEUS_CONFIG = f"{TELEMETRY_LOG_DIR}/prometheus/prometheus.yml"
PROMETHEUS_TARGETS = f"{TELEMETRY_LOG_DIR}/prometheus/targets"
PROMETHEUS_RULES = f"{TELEMETRY_LOG_DIR}/prometheus/rules"
PROCESS_EXPORTER_CONFIG = f"{TELEMETRY_LOG_DIR}/prometheus/process-exporter.yml"
DCGM_COUNTERS_CONFIG = f"{TELEMETRY_LOG_DIR}/prometheus/dcgm-counters.csv"
PROMETHEUS_DATA = f"{TELEMETRY_LOG_DIR}/prometheus/data"

FILE_SD_CLEANUP_MIN_AGE_S = 5 * 60 * 60
FILE_SD_CLEANUP_TIMEOUT_S = 1.0
FILE_SD_CLEANUP_MAX_WORKERS = 32


def _base_file_sd_labels(run_metadata: dict[str, Any], cfg: Any) -> dict[str, str]:
    """Build labels shared by every file-SD target group for a run."""
    return {
        "run_uuid": str(run_metadata["run_uuid"]),
        "run_name": str(run_metadata["run_name"]),
        "user": str(os.environ.get("USER", "unknownUser")),
        "node": socket.gethostname(),
        "slurm_job_id": str(cfg.wizard.slurm_job_id or ""),
    }


def _host_log_path(log_dir: Path, container_path: str) -> Path:
    """Map a telemetry container path under /mnt/log_dir to its host log path."""
    return log_dir / Path(container_path).relative_to(TELEMETRY_LOG_DIR)


def generate_prometheus_configs(
    log_dir: Path,
    run_metadata: dict[str, Any],
    context: WizardContext,
) -> Path | None:
    """Write Prometheus config, rules, and file-SD targets for this run.

    The generated files live under the run log directory, which is mounted into
    the telemetry sidecar at the container paths defined in this module. The
    optional central file-SD publication lets an external Prometheus discover
    this run while it is active.

    Args:
        log_dir: Host-side run log directory.
        run_metadata: Stable run identity labels written into scrape targets.
        context: Wizard context containing resolved config and telemetry ports.

    Returns:
        The central file-SD path to remove during cleanup, or None when central
        file-SD publication is disabled.
    """
    cfg = context.cfg
    write_json(
        {
            "process_names": [
                {"name": "runtime", "cmdline": ["alpasim_runtime.simulate"]},
                {"name": "driver", "cmdline": ["alpasim_driver"]},
                {"name": "renderer", "cmdline": ["pycena|nre|sensorsim"]},
                {"name": "physics", "cmdline": ["physics_server"]},
                {"name": "trafficsim", "cmdline": ["trafficsim"]},
                {"name": "controller", "cmdline": ["alpasim_controller.server"]},
            ]
        },
        _host_log_path(log_dir, PROCESS_EXPORTER_CONFIG),
    )
    dcgm_counters_path = _host_log_path(log_dir, DCGM_COUNTERS_CONFIG)
    dcgm_counters_path.parent.mkdir(parents=True, exist_ok=True)
    dcgm_counters_path.write_text(
        resource_files("alpasim_wizard")
        .joinpath("telemetry/resources/dcgm-counters.csv")
        .read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    central_file_sd_path = None
    if cfg.wizard.prometheus.file_sd_dir:
        file_sd_root = Path(cfg.wizard.prometheus.file_sd_dir)
        file_sd_root.mkdir(parents=True, exist_ok=True)
        if file_sd_root.stat().st_mode & 0o7777 != 0o2777:
            file_sd_root.chmod(0o2777)
        _cleanup_stale_file_sd(file_sd_root)
        central_file_sd_path = file_sd_root / f"{run_metadata['run_uuid']}.json"
        write_json(
            _build_file_sd_targets(run_metadata, context, local=False),
            central_file_sd_path,
            mode=0o666,
        )

    if cfg.wizard.prometheus.start_prometheus:
        _host_log_path(log_dir, PROMETHEUS_DATA).mkdir(parents=True, exist_ok=True)
        write_json(
            _build_file_sd_targets(run_metadata, context, local=True),
            _host_log_path(log_dir, PROMETHEUS_TARGETS) / "alpasim.json",
        )
        prometheus_config = {
            "global": {
                "scrape_interval": str(cfg.wizard.prometheus.scrape_interval),
                "evaluation_interval": str(cfg.wizard.prometheus.scrape_interval),
            },
            "rule_files": [f"{PROMETHEUS_RULES}/*.yml"],
            "scrape_configs": [
                {
                    "job_name": "alpasim",
                    "file_sd_configs": [
                        {
                            "files": [f"{PROMETHEUS_TARGETS}/*.json"],
                            "refresh_interval": "5s",
                        }
                    ],
                }
            ],
        }
        recording_rules = resource_files("alpasim_utils.telemetry").joinpath(
            "metrics_plot_recording_rules.yml"
        )
        rules_dir = _host_log_path(log_dir, PROMETHEUS_RULES)
        rules_dir.mkdir(parents=True, exist_ok=True)
        (rules_dir / "alpasim-recording-rules.yml").write_text(
            recording_rules.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        write_yaml(prometheus_config, str(_host_log_path(log_dir, PROMETHEUS_CONFIG)))
    return central_file_sd_path


def _build_file_sd_targets(
    run_metadata: dict[str, Any],
    context: WizardContext,
    *,
    local: bool,
) -> list[dict[str, Any]]:
    """Build Prometheus file-SD target groups for local or central scraping."""
    cfg = context.cfg
    labels = _base_file_sd_labels(run_metadata, cfg)
    if local:
        runtime_host = (
            "localhost"
            if cfg.wizard.run_method.name == "SLURM"
            or cfg.wizard.debug_flags.use_localhost
            else "runtime-0"
        )
        exporter_host = "localhost"
    else:
        runtime_host = _central_scrape_host()
        exporter_host = runtime_host

    telemetry_ports = context.telemetry_ports
    prometheus_ports = telemetry_ports.prometheus_service_ports()
    return [
        {
            "targets": [f"{runtime_host}:{port}" for port in telemetry_ports.workers],
            "labels": {
                **labels,
                "job": "alpasim-runtime-worker",
                "component": "runtime",
            },
        },
        {
            "targets": [f"{exporter_host}:{prometheus_ports['node_exporter']}"],
            "labels": {**labels, "job": "alpasim-node"},
        },
        {
            "targets": [f"{exporter_host}:{prometheus_ports['process_exporter']}"],
            "labels": {**labels, "job": "alpasim-process"},
        },
        {
            "targets": [f"{exporter_host}:{prometheus_ports['dcgm_exporter']}"],
            "labels": {**labels, "job": "alpasim-dcgm"},
        },
    ]


def _central_scrape_host() -> str:
    """Return a scrape address reachable by external Prometheus servers."""
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return hostname


def _cleanup_stale_file_sd(file_sd_dir: Path) -> None:
    """Delete stale central file-SD files whose targets are unreachable."""
    now = time.time()
    for path in file_sd_dir.glob("*.json"):
        try:
            if now - path.stat().st_mtime < FILE_SD_CLEANUP_MIN_AGE_S:
                continue

            with open(path, encoding="utf-8") as stream:
                groups = json.load(stream)
            targets = [target for group in groups for target in group["targets"]]

            with ThreadPoolExecutor(
                max_workers=FILE_SD_CLEANUP_MAX_WORKERS
            ) as executor:
                if any(executor.map(_target_reachable, targets)):
                    continue
        except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping invalid file-SD file %s: %s", path, exc)
            continue

        path.unlink()
        with suppress(OSError):
            path.parent.rmdir()


def _target_reachable(target: str) -> bool:
    """Return whether a host:port target accepts a TCP connection."""
    host, port_str = target.rsplit(":", 1)
    port = int(port_str)
    try:
        with socket.create_connection(
            (host, port),
            timeout=FILE_SD_CLEANUP_TIMEOUT_S,
        ):
            return True
    except OSError:
        return False
