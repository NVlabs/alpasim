# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Tests for the alpasim-reeval CLI tool."""

import pathlib
import subprocess
import sys
from unittest.mock import patch

import pytest
import yaml

from eval.reeval import detect_layout


def _make_single_job(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a single-job directory structure."""
    (tmp_path / "eval-config.yaml").write_text("num_processes: 16\n")
    (tmp_path / "wizard-config.yaml").write_text(
        yaml.dump(
            {
                "services": {"runtime": {"image": "nvcr.io/test/image:1.0"}},
                "scenes": {
                    "scene_cache": "/lustre/nre-artifacts",
                    "sceneset_path": "test_sceneset",
                },
                "wizard": {"sqshcaches": ["/cache"]},
            }
        )
    )
    rollout = tmp_path / "rollouts" / "scene_001" / "abc123"
    rollout.mkdir(parents=True)
    (rollout / "log.asl").write_bytes(b"fake")
    return tmp_path


def _make_array_job(tmp_path: pathlib.Path, n_tasks: int = 3) -> pathlib.Path:
    """Create an array-job directory structure."""
    for i in range(n_tasks):
        task_dir = tmp_path / f"12345_{i}_2025_01_01__00_00_00"
        task_dir.mkdir()
        (task_dir / "eval-config.yaml").write_text("num_processes: 16\n")
        (task_dir / "wizard-config.yaml").write_text(
            yaml.dump(
                {
                    "services": {"runtime": {"image": "nvcr.io/test/image:1.0"}},
                    "scenes": {
                        "scene_cache": "/lustre/nre-artifacts",
                        "sceneset_path": "test_sceneset",
                    },
                    "wizard": {"sqshcaches": ["/cache"]},
                }
            )
        )
        rollout = task_dir / "rollouts" / "scene_001" / f"uuid_{i}"
        rollout.mkdir(parents=True)
        (rollout / "log.asl").write_bytes(b"fake")
    return tmp_path


class TestDetectLayout:
    def test_single_job(self, tmp_path: pathlib.Path) -> None:
        _make_single_job(tmp_path)
        layout = detect_layout(tmp_path)
        assert layout.is_array_job is False
        assert layout.job_dirs == [tmp_path]
        assert layout.image == "nvcr.io/test/image:1.0"
        assert layout.scene_cache == "/lustre/nre-artifacts"

    def test_array_job(self, tmp_path: pathlib.Path) -> None:
        _make_array_job(tmp_path, n_tasks=3)
        layout = detect_layout(tmp_path)
        assert layout.is_array_job is True
        assert len(layout.job_dirs) == 3
        assert layout.image == "nvcr.io/test/image:1.0"

    def test_empty_dir_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError):
            detect_layout(tmp_path)

    def test_array_job_dirs_sorted(self, tmp_path: pathlib.Path) -> None:
        _make_array_job(tmp_path, n_tasks=3)
        layout = detect_layout(tmp_path)
        assert layout.job_dirs == sorted(layout.job_dirs)


class TestRunLocal:
    """Tests for run_local (mocks subprocess calls)."""

    @patch("eval.reeval.subprocess.run")
    def test_single_job_runs_eval_then_aggregation(
        self, mock_run, tmp_path: pathlib.Path
    ) -> None:
        _make_single_job(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        from eval.reeval import run_local

        rc = run_local(detect_layout(tmp_path))
        assert rc == 0
        assert mock_run.call_count == 2  # eval + aggregation

    @patch("eval.reeval.subprocess.run")
    def test_array_job_runs_eval_per_subdir_then_aggregation(
        self, mock_run, tmp_path: pathlib.Path
    ) -> None:
        _make_array_job(tmp_path, n_tasks=3)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        from eval.reeval import run_local

        rc = run_local(detect_layout(tmp_path))
        assert rc == 0
        assert mock_run.call_count == 4  # 3 eval + 1 aggregation

    @patch("eval.reeval.subprocess.run")
    def test_eval_failure_stops_early(self, mock_run, tmp_path: pathlib.Path) -> None:
        _make_single_job(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1)

        from eval.reeval import run_local

        rc = run_local(detect_layout(tmp_path))
        assert rc == 1
        assert mock_run.call_count == 1  # stopped after eval failure


class TestRunSlurm:
    """Tests for run_slurm (mocks subprocess calls)."""

    @patch("eval.reeval.subprocess.run")
    def test_single_job_submits_eval_and_aggregation(
        self, mock_run, tmp_path: pathlib.Path
    ) -> None:
        _make_single_job(tmp_path)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Submitted batch job 12345\n", stderr=""
        )

        from eval.reeval import run_slurm

        rc = run_slurm(
            detect_layout(tmp_path),
            partition="cpu_short",
            account="my_account",
        )
        assert rc == 0
        assert mock_run.call_count == 2  # eval sbatch + agg sbatch

    @patch("eval.reeval.subprocess.run")
    def test_array_job_submits_array_eval_and_aggregation(
        self, mock_run, tmp_path: pathlib.Path
    ) -> None:
        _make_array_job(tmp_path, n_tasks=3)
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Submitted batch job 12345\n", stderr=""
        )

        from eval.reeval import run_slurm

        rc = run_slurm(
            detect_layout(tmp_path),
            partition="cpu_short",
            account="my_account",
        )
        assert rc == 0
        # Check index file was created
        index_file = tmp_path / "reeval_scripts" / "job_dirs.txt"
        assert index_file.exists()
        lines = index_file.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_slurm_missing_account_raises(self, tmp_path: pathlib.Path) -> None:
        _make_single_job(tmp_path)

        from eval.reeval import run_slurm

        with pytest.raises(ValueError, match="--account is required"):
            run_slurm(
                detect_layout(tmp_path),
                partition="cpu_short",
                account=None,
            )


class TestCli:
    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "eval.reeval", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Re-evaluate" in result.stdout
        assert "single-job" in result.stdout
        assert "array-job" in result.stdout

    def test_missing_log_dir_exits_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "eval.reeval"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_invalid_dir_exits_nonzero(self, tmp_path: pathlib.Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "eval.reeval", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
