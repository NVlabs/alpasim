# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from pathlib import Path

import pytest
from alpasim_wizard.telemetry.slurm_process_exporter import (
    collect,
    collect_processes,
    discover_pids,
    render_metrics,
)


def _write_proc(
    procfs: Path,
    pid: str,
    *,
    cmdline: str,
    utime: int,
    stime: int,
    resident_pages: int,
    environ: list[str] | None = None,
    cgroup: str = "",
) -> None:
    pid_dir = procfs / pid
    pid_dir.mkdir()
    (pid_dir / "cmdline").write_bytes(cmdline.encode("utf-8") + b"\0")
    (pid_dir / "environ").write_bytes(
        b"\0".join(value.encode("utf-8") for value in environ or []) + b"\0"
    )
    fields_after_comm = ["S", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
    fields_after_comm.extend([str(utime), str(stime), "0", "0"])
    (pid_dir / "stat").write_text(
        f"{pid} (python) {' '.join(fields_after_comm)}\n",
        encoding="utf-8",
    )
    (pid_dir / "statm").write_text(
        f"0 {resident_pages} 0 0 0 0 0\n",
        encoding="utf-8",
    )
    (pid_dir / "cgroup").write_text(cgroup, encoding="utf-8")


def test_discover_pids_falls_back_to_job_scoped_procfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_proc(
        tmp_path,
        "101",
        cmdline="uv run python -m alpasim_runtime.simulate",
        utime=100,
        stime=50,
        resident_pages=10,
        environ=["SLURM_JOBID=123"],
    )
    _write_proc(
        tmp_path,
        "102",
        cmdline="uv run python -m alpasim_driver.main",
        utime=100,
        stime=50,
        resident_pages=10,
        environ=["SLURM_JOB_ID=999"],
    )
    _write_proc(
        tmp_path,
        "103",
        cmdline="python unrelated.py",
        utime=100,
        stime=50,
        resident_pages=10,
        environ=["SLURM_JOB_ID=123"],
    )
    _write_proc(
        tmp_path,
        "104",
        cmdline="uv run python -m alpasim_controller.server --port=6132",
        utime=100,
        stime=50,
        resident_pages=10,
        cgroup="7:cpu,cpuacct:/slurm/uid_101499/job_123/step_1/task_0\n",
    )

    def raise_missing_scontrol(job_id: str) -> set[str]:
        raise RuntimeError("[Errno 2] No such file or directory: 'scontrol'")

    monkeypatch.setattr(
        "alpasim_wizard.telemetry.slurm_process_exporter.discover_slurm_pids",
        raise_missing_scontrol,
    )

    assert discover_pids("123", tmp_path) == {"101", "104"}


def test_discover_pids_prefers_host_cgroup_without_scontrol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    procfs = tmp_path / "proc"
    procfs.mkdir()
    _write_proc(
        procfs,
        "101",
        cmdline="uv run physics_server --port=6116",
        utime=100,
        stime=50,
        resident_pages=10,
    )
    cgroupfs = tmp_path / "cgroup"
    job_dir = cgroupfs / "freezer/slurm/uid_101499/job_123/step_1"
    job_dir.mkdir(parents=True)
    (job_dir / "cgroup.procs").write_text("101\n", encoding="utf-8")

    def fail_scontrol(job_id: str) -> set[str]:
        raise AssertionError("scontrol must not be needed when the job cgroup exists")

    monkeypatch.setattr(
        "alpasim_wizard.telemetry.slurm_process_exporter.discover_slurm_pids",
        fail_scontrol,
    )

    pids = discover_pids("123", procfs, cgroupfs)

    assert pids == {"101"}


def test_render_metrics_omits_unobserved_groups(tmp_path: Path) -> None:
    _write_proc(
        tmp_path,
        "101",
        cmdline=(
            "uv run physics_server --port=6116 "
            "--artifact-glob=/mnt/nre-data/**/*.usdz"
        ),
        utime=100,
        stime=50,
        resident_pages=10,
    )

    process_samples = collect_processes(tmp_path, {"101"})
    payload = render_metrics(
        collect(tmp_path, {"101"}),
        duration=0.125,
        process_samples=process_samples,
    ).decode("utf-8")

    assert 'namedprocess_namegroup_cpu_seconds_total{groupname="physics"}' in payload
    assert (
        'alpasim_process_cpu_seconds_total{groupname="physics",pid="101",port="6116"}'
        in payload
    )
    assert 'namedprocess_namegroup_cpu_seconds_total{groupname="driver"}' not in payload
    assert (
        'namedprocess_namegroup_memory_bytes{groupname="physics",memtype="resident"}'
        in payload
    )
    assert (
        'namedprocess_namegroup_memory_bytes{groupname="driver",memtype="resident"}'
        not in payload
    )
    assert "alpasim_slurm_process_exporter_scrape_duration_seconds 0.125" in payload
