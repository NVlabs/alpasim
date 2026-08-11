# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_submit_removes_inherited_slurm_step_and_pmix_environment(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    script_dir = repo_root / "src" / "tools" / "run-on-slurm"
    script_dir.mkdir(parents=True)
    source_script = Path(__file__).parents[1] / "run-on-slurm" / "submit.sh"
    submit_script = script_dir / "submit.sh"
    submit_script.write_bytes(source_script.read_bytes())
    submit_script.chmod(0o755)

    (script_dir / "runs" / "slurm_output").mkdir(parents=True)
    (script_dir / "runs" / "slurm_output" / "123.log").touch()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "scontrol",
        f"#!/bin/bash\nprintf 'Command={submit_script}\\nAccount=test-account\\n'\n",
    )
    _write_executable(
        bin_dir / "sacct",
        "#!/bin/bash\nprintf 'SubmitLine|\\nsbatch submit.sh deploy=iad topology=test driver=test|\\n'\n",
    )
    captured_environment = tmp_path / "wizard-environment"
    _write_executable(
        bin_dir / "uv",
        f"#!/bin/bash\nenv | sort > {captured_environment}\n",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "DESCRIPTION": "test",
            "SLURM_JOB_ACCOUNT": "test-account",
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NAME": "test-job",
            "SLURM_STEP_ID": "0",
            "SLURM_STEP_NODELIST": "stale-node",
            "SLURM_SUBMIT_DIR": str(script_dir),
            "SUBMITTER": "test-user",
            "PMIX_NAMESPACE": "stale-namespace",
            "PMIX_RANK": "0",
        }
    )
    for variable in list(environment):
        if variable.startswith("SLURM_ARRAY_"):
            environment.pop(variable)

    result = subprocess.run(
        [
            str(submit_script),
            "deploy=iad",
            "topology=test",
            "driver=test",
        ],
        cwd=script_dir,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    wizard_environment = captured_environment.read_text().splitlines()
    assert "SLURM_JOB_ID=123" in wizard_environment
    assert not any(
        entry.startswith(("SLURM_STEP_", "PMIX_")) for entry in wizard_environment
    )
