# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Lightweight Slurm-scoped process metrics exporter."""

from __future__ import annotations

import argparse
import http.server
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, cast

GROUP_PATTERNS = (
    ("runtime", re.compile(r"alpasim_runtime\.simulate")),
    ("driver", re.compile(r"alpasim_driver")),
    ("physics", re.compile(r"physics_server")),
    ("renderer", re.compile(r"pycena|sensorsim|(?:^|\W)nre(?:\W|$)")),
    ("trafficsim", re.compile(r"trafficsim")),
    ("controller", re.compile(r"alpasim_controller\.server")),
)

CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
logger = logging.getLogger(__name__)

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass
class ProcessSample:
    cpu_seconds: float = 0.0
    resident_bytes: int = 0


@dataclass(frozen=True)
class ProcessMetric:
    pid: str
    group: str
    port: str
    cpu_seconds: float
    resident_bytes: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_cmdline(pid_dir: Path) -> str:
    raw = pid_dir.joinpath("cmdline").read_bytes()
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _read_environ(pid_dir: Path) -> list[str]:
    raw = pid_dir.joinpath("environ").read_bytes()
    return [
        value.decode("utf-8", errors="replace") for value in raw.split(b"\0") if value
    ]


def _read_cgroup(pid_dir: Path) -> str:
    return _read_text(pid_dir / "cgroup")


def _read_cpu_seconds(pid_dir: Path) -> float:
    stat = _read_text(pid_dir / "stat")
    fields = stat[stat.rfind(")") + 2 :].split()
    utime = int(fields[11])
    stime = int(fields[12])
    return (utime + stime) / CLOCK_TICKS


def _read_resident_bytes(pid_dir: Path) -> int:
    statm_fields = _read_text(pid_dir / "statm").split()
    return int(statm_fields[1]) * PAGE_SIZE


def _group_for_cmdline(cmdline: str) -> str | None:
    for name, pattern in GROUP_PATTERNS:
        if pattern.search(cmdline):
            return name
    return None


def _port_for_cmdline(cmdline: str) -> str:
    match = re.search(r"(?:^|\s)(?:--?)?port(?:=|\s+)(\d+)(?:\s|$)", cmdline)
    return match.group(1) if match else ""


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def parse_slurm_pids(output: str) -> set[str]:
    pids = set()
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        pid = fields[0]
        if pid == "-1" or not pid.isdigit():
            continue
        pids.add(pid)
    return pids


def discover_slurm_pids(
    job_id: str,
    *,
    run_command: CommandRunner = _run_command,
) -> set[str]:
    try:
        result = run_command(["scontrol", "listpids", job_id])
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr or exc.stdout or str(exc)
        raise RuntimeError(detail) from exc
    return parse_slurm_pids(result.stdout)


def discover_cgroup_pids(cgroupfs: Path, job_id: str) -> set[str]:
    """Read process IDs from the host Slurm job cgroup."""
    job_dirs = {
        *cgroupfs.glob(f"*/slurm/uid_*/job_{job_id}"),
        *cgroupfs.glob(f"slurm/uid_*/job_{job_id}"),
    }
    pids: set[str] = set()
    for job_dir in job_dirs:
        for path in job_dir.rglob("cgroup.procs"):
            try:
                pids.update(
                    line for line in _read_text(path).splitlines() if line.isdigit()
                )
            except (FileNotFoundError, PermissionError):
                continue
    return pids


def discover_procfs_pids(procfs: Path, job_id: str) -> set[str]:
    pids = set()
    job_env_names = ("SLURM_JOB_ID", "SLURM_JOBID")
    job_cgroup = f"/job_{job_id}/"
    for pid_dir in procfs.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            environ = _read_environ(pid_dir)
            cgroup = _read_cgroup(pid_dir)
            if not any(f"{name}={job_id}" in environ for name in job_env_names) and (
                job_cgroup not in cgroup
            ):
                continue
            if _group_for_cmdline(_read_cmdline(pid_dir)) is None:
                continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        pids.add(pid_dir.name)
    return pids


def discover_pids(
    job_id: str,
    procfs: Path,
    cgroupfs: Path = Path("/host/sys/fs/cgroup"),
) -> set[str]:
    pids = discover_cgroup_pids(cgroupfs, job_id)
    if pids:
        return pids
    try:
        pids = discover_slurm_pids(job_id)
    except RuntimeError as exc:
        logger.warning(
            "Failed to discover Slurm PIDs for job %s: %s; falling back to %s",
            job_id,
            exc,
            procfs,
        )
        pids = set()
    return pids | discover_procfs_pids(procfs, job_id)


def collect(procfs: Path, pids: Iterable[str]) -> dict[str, ProcessSample]:
    samples: dict[str, ProcessSample] = {}
    for process in collect_processes(procfs, pids):
        sample = samples.setdefault(process.group, ProcessSample())
        sample.cpu_seconds += process.cpu_seconds
        sample.resident_bytes += process.resident_bytes
    return samples


def collect_processes(procfs: Path, pids: Iterable[str]) -> list[ProcessMetric]:
    samples = []
    for pid in pids:
        pid_dir = procfs / pid
        try:
            cmdline = _read_cmdline(pid_dir)
            group = _group_for_cmdline(cmdline)
            if group is None:
                continue
            samples.append(
                ProcessMetric(
                    pid=pid,
                    group=group,
                    port=_port_for_cmdline(cmdline),
                    cpu_seconds=_read_cpu_seconds(pid_dir),
                    resident_bytes=_read_resident_bytes(pid_dir),
                )
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return samples


def _label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_metrics(
    samples: dict[str, ProcessSample],
    duration: float,
    process_samples: Sequence[ProcessMetric] = (),
) -> bytes:
    lines = [
        "# HELP namedprocess_namegroup_cpu_seconds_total Cpu usage in seconds",
        "# TYPE namedprocess_namegroup_cpu_seconds_total counter",
    ]
    for group, sample in samples.items():
        label = _label_value(group)
        lines.append(
            f'namedprocess_namegroup_cpu_seconds_total{{groupname="{label}"}} '
            f"{sample.cpu_seconds}"
        )
    lines.extend(
        [
            "# HELP namedprocess_namegroup_memory_bytes Memory usage in bytes",
            "# TYPE namedprocess_namegroup_memory_bytes gauge",
        ]
    )
    for group, sample in samples.items():
        label = _label_value(group)
        lines.append(
            f'namedprocess_namegroup_memory_bytes{{groupname="{label}",'
            f'memtype="resident"}} {sample.resident_bytes}'
        )
    lines.extend(
        [
            "# HELP alpasim_process_cpu_seconds_total "
            "Cpu usage in seconds by process",
            "# TYPE alpasim_process_cpu_seconds_total counter",
        ]
    )
    for process in process_samples:
        group = _label_value(process.group)
        pid = _label_value(process.pid)
        port = _label_value(process.port)
        lines.append(
            f'alpasim_process_cpu_seconds_total{{groupname="{group}",'
            f'pid="{pid}",port="{port}"}} {process.cpu_seconds}'
        )
    lines.extend(
        [
            "# HELP alpasim_slurm_process_exporter_scrape_duration_seconds "
            "Time spent collecting Slurm process metrics",
            "# TYPE alpasim_slurm_process_exporter_scrape_duration_seconds gauge",
            f"alpasim_slurm_process_exporter_scrape_duration_seconds {duration}",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    cache_until = 0.0
    cache_body = b""

    def do_GET(self) -> None:
        if self.path not in ("/metrics", "/"):
            self.send_error(404)
            return
        now = time.monotonic()
        server = cast(MetricsServer, self.server)
        if now >= MetricsHandler.cache_until:
            started = time.monotonic()
            pids = discover_pids(server.job_id, server.procfs, server.cgroupfs)
            process_samples = collect_processes(server.procfs, pids)
            samples: dict[str, ProcessSample] = {}
            for process in process_samples:
                sample = samples.setdefault(process.group, ProcessSample())
                sample.cpu_seconds += process.cpu_seconds
                sample.resident_bytes += process.resident_bytes
            duration = time.monotonic() - started
            MetricsHandler.cache_body = render_metrics(
                samples,
                duration,
                process_samples,
            )
            MetricsHandler.cache_until = now + server.cache_seconds
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(MetricsHandler.cache_body)))
        self.end_headers()
        self.wfile.write(MetricsHandler.cache_body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


class MetricsServer(http.server.HTTPServer):
    job_id: str
    procfs: Path
    cgroupfs: Path
    cache_seconds: float


def _job_id_from_env() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--procfs", default="/host/proc", type=Path)
    parser.add_argument("--cgroupfs", default="/host/sys/fs/cgroup", type=Path)
    parser.add_argument("--cache-seconds", default=5.0, type=float)
    args = parser.parse_args()

    server = MetricsServer(("0.0.0.0", args.port), MetricsHandler)
    server.job_id = args.job_id or _job_id_from_env()
    server.procfs = args.procfs
    server.cgroupfs = args.cgroupfs
    server.cache_seconds = args.cache_seconds
    server.serve_forever()


if __name__ == "__main__":
    main()
