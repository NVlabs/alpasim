# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Re-evaluation CLI tool.

Runs evaluation and aggregation on ASL logs from a previous simulation,
without requiring the wizard. Supports both local execution and SLURM
array job submission.

Usage (local):
    alpasim-reeval /path/to/sim/output

Usage (SLURM):
    alpasim-reeval /path/to/sim/output --slurm \\
        --partition cpu_short --account my_account
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from textwrap import dedent

from alpasim_utils.paths import find_repo_root, image_to_sqsh_basename
from alpasim_utils.yaml_utils import load_yaml_dict

logger = logging.getLogger("alpasim_reeval")

EVAL_CONFIG_FILE = "eval-config.yaml"
WIZARD_CONFIG_FILE = "wizard-config.yaml"


@dataclass
class JobLayout:
    """Detected job directory layout."""

    is_array_job: bool
    job_dirs: list[pathlib.Path]
    array_job_dir: pathlib.Path
    image: str
    scene_cache: str
    sceneset_path: str
    num_processes: int
    sqshcaches: list[str] = field(default_factory=list)


def _layout_from_wizard_cfg(
    wizard_cfg: dict, eval_cfg: dict, **kwargs: object
) -> JobLayout:
    """Extract layout fields from a resolved wizard config."""
    return JobLayout(
        **kwargs,
        image=wizard_cfg["services"]["runtime"]["image"],
        scene_cache=wizard_cfg["scenes"]["scene_cache"],
        sceneset_path=wizard_cfg["scenes"]["sceneset_path"],
        num_processes=eval_cfg["num_processes"],
        sqshcaches=wizard_cfg.get("wizard", {}).get("sqshcaches", []),
    )


def detect_layout(log_dir: pathlib.Path) -> JobLayout:
    """Auto-detect whether log_dir is a single job or array job parent.

    Single job: log_dir contains eval-config.yaml directly.
    Array job: log_dir contains subdirectories with eval-config.yaml.
    """
    log_dir = pathlib.Path(log_dir)

    # Single job: eval-config.yaml at root
    if (log_dir / EVAL_CONFIG_FILE).exists():
        wizard_cfg = load_yaml_dict(log_dir / WIZARD_CONFIG_FILE)
        eval_cfg = load_yaml_dict(log_dir / EVAL_CONFIG_FILE)
        return _layout_from_wizard_cfg(
            wizard_cfg,
            eval_cfg,
            is_array_job=False,
            job_dirs=[log_dir],
            array_job_dir=log_dir,
        )

    # Array job: subdirs with eval-config.yaml
    subdirs = sorted(
        d for d in log_dir.iterdir() if d.is_dir() and (d / EVAL_CONFIG_FILE).exists()
    )
    if subdirs:
        first = subdirs[0]
        wizard_cfg = load_yaml_dict(first / WIZARD_CONFIG_FILE)
        eval_cfg = load_yaml_dict(first / EVAL_CONFIG_FILE)
        return _layout_from_wizard_cfg(
            wizard_cfg,
            eval_cfg,
            is_array_job=True,
            job_dirs=subdirs,
            array_job_dir=log_dir,
        )

    raise FileNotFoundError(
        f"Could not detect job layout in {log_dir}. "
        f"Expected either {EVAL_CONFIG_FILE} at root (single job) "
        f"or subdirectories containing {EVAL_CONFIG_FILE} (array job)."
    )


def _run_eval_on_dir(
    job_dir: pathlib.Path,
    eval_config_path: pathlib.Path,
    usdz_glob: str,
) -> int:
    """Run eval.main on a single job directory's ASL files."""
    asl_glob = str(job_dir / "rollouts" / "**" / "*.asl")

    cmd = [
        sys.executable,
        "-m",
        "eval.main",
        "--asl_search_glob",
        asl_glob,
        "--config_path",
        str(eval_config_path),
        "--usdz_glob",
        usdz_glob,
    ]
    logger.info("Running evaluation on %s: %s", job_dir, " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def _run_aggregation(
    array_job_dir: pathlib.Path, eval_config_path: pathlib.Path
) -> int:
    """Run aggregation on the array job directory."""
    cmd = [
        sys.executable,
        "-m",
        "eval.aggregation.main",
        "--array_job_dir",
        str(array_job_dir),
        "--config_path",
        str(eval_config_path),
    ]
    logger.info("Running aggregation: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def run_local(layout: JobLayout) -> int:
    """Run re-evaluation locally."""
    for job_dir in layout.job_dirs:
        logger.info("Evaluating %s", job_dir)
        eval_config = job_dir / EVAL_CONFIG_FILE
        usdz_glob = str(
            pathlib.Path(layout.scene_cache) / layout.sceneset_path / "**" / "*.usdz"
        )
        rc = _run_eval_on_dir(job_dir, eval_config, usdz_glob)
        if rc != 0:
            logger.error("Evaluation failed for %s (exit code %d)", job_dir, rc)
            return rc

    logger.info("Running aggregation on %s", layout.array_job_dir)
    agg_eval_config = layout.job_dirs[0] / EVAL_CONFIG_FILE
    rc = _run_aggregation(layout.array_job_dir, agg_eval_config)
    if rc != 0:
        logger.error("Aggregation failed (exit code %d)", rc)
    return rc


def _resolve_image(layout: JobLayout) -> str:
    """Resolve container image to sqsh file if possible, otherwise return URL."""
    image = layout.image
    sqsh_fname = image_to_sqsh_basename(image)
    for cache_dir in layout.sqshcaches:
        sqsh_path = os.path.join(cache_dir, sqsh_fname)
        if os.path.isfile(sqsh_path):
            logger.info("Resolved image %s to sqsh: %s", image, sqsh_path)
            return sqsh_path
    return image


def _build_sbatch_script(
    srun_command: str,
    partition: str,
    account: str,
    gpus: int = 0,
    time_limit: str = "03:59:00",
    cpus: int = 8,
    mem: str = "32gb",
    array: str | None = None,
) -> str:
    """Build an sbatch submission script."""
    array_line = f"#SBATCH --array={array}\n" if array else ""
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --account={account}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --time={time_limit}\n"
        f"#SBATCH --gpus-per-node={gpus}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}\n"
        f"{array_line}"
        f"\n"
        f"{srun_command}\n"
    )


def _submit_sbatch(
    script_path: pathlib.Path,
    job_name: str,
    log_path: pathlib.Path,
    dependency: int | None = None,
) -> int:
    """Submit an sbatch script and return the job ID."""
    cmd = ["sbatch", f"--output={log_path}", f"--job-name={job_name}"]
    if dependency is not None:
        cmd.append(f"--dependency=afterany:{dependency}")
    cmd.append(str(script_path))

    logger.info("Submitting: %s", cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("sbatch failed: %s", result.stderr)
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from: {result.stdout}")
    return int(match.group(1))


def run_slurm(layout: JobLayout, partition: str, account: str | None) -> int:
    """Submit re-evaluation as SLURM jobs."""
    if not account:
        raise ValueError("--account is required for SLURM mode.")

    image = _resolve_image(layout)
    scripts_dir = layout.array_job_dir / "reeval_scripts"
    scripts_dir.mkdir(exist_ok=True)
    logs_dir = layout.array_job_dir / "txt-logs"
    logs_dir.mkdir(exist_ok=True)

    repo_src = find_repo_root(__file__) / "src"
    src_mount = f",{repo_src}:/repo/src"
    # Mount the full scene cache so symlinks in the sceneset dir can resolve.
    scene_mount = f",{layout.scene_cache}:/mnt/nre-data"
    usdz_arg = f"--usdz_glob='/mnt/nre-data/{layout.sceneset_path}/**/*.usdz'"

    srun_base = (
        f"srun --container-image={image} "
        f"--no-container-remap-root "
        f"--container-workdir=/repo/src "
    )

    if layout.is_array_job:
        # Write index file mapping SLURM_ARRAY_TASK_ID -> subdir path
        index_file = scripts_dir / "job_dirs.txt"
        index_file.write_text("\n".join(str(d) for d in layout.job_dirs) + "\n")

        eval_srun = (
            f'JOB_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {index_file})\n'
            f"mkdir -p $JOB_DIR/txt-logs\n"
            f"exec > $JOB_DIR/txt-logs/reeval_eval.log 2>&1\n"
            f"{srun_base}"
            f"--container-mounts=$JOB_DIR:/mnt/log_dir{src_mount}{scene_mount} "
            f'bash -c "uv run alpasim-eval '
            f"--asl_search_glob='/mnt/log_dir/rollouts/**/*.asl' "
            f"--config_path=/mnt/log_dir/eval-config.yaml "
            f'{usdz_arg}"'
        )
        eval_script = _build_sbatch_script(
            eval_srun,
            partition=partition,
            account=account,
            cpus=layout.num_processes,
            array=f"0-{len(layout.job_dirs) - 1}",
        )
    else:
        log_dir = layout.job_dirs[0]
        eval_srun = (
            f"{srun_base}"
            f"--container-mounts={log_dir}:/mnt/log_dir{src_mount}{scene_mount} "
            f'bash -c "uv run alpasim-eval '
            f"--asl_search_glob='/mnt/log_dir/rollouts/**/*.asl' "
            f"--config_path=/mnt/log_dir/eval-config.yaml "
            f'{usdz_arg}"'
        )
        eval_script = _build_sbatch_script(
            eval_srun,
            partition=partition,
            account=account,
            cpus=layout.num_processes,
        )

    eval_script_path = scripts_dir / "submit_eval.sh"
    eval_script_path.write_text(eval_script)

    eval_job_id = _submit_sbatch(
        eval_script_path,
        job_name="reeval-eval",
        log_path=logs_dir / "reeval_eval.log",
    )
    logger.info("Eval job submitted: %d", eval_job_id)

    # Submit aggregation with dependency
    array_job_dir = layout.array_job_dir
    # For array jobs, mount both the parent dir and pass eval config from first subdir
    # For single jobs, everything is in one dir
    if layout.is_array_job:
        first_subdir = layout.job_dirs[0]
        first_subdir_rel = first_subdir.relative_to(array_job_dir)
        config_mount_path = f"/mnt/array_job_dir/{first_subdir_rel}/{EVAL_CONFIG_FILE}"
    else:
        config_mount_path = f"/mnt/array_job_dir/{EVAL_CONFIG_FILE}"

    agg_srun = (
        f"{srun_base}"
        f"--container-mounts={array_job_dir}:/mnt/array_job_dir{src_mount} "
        f'bash -c "uv run alpasim-aggregation '
        f"--array_job_dir=/mnt/array_job_dir "
        f'--config_path={config_mount_path}"'
    )
    agg_script = _build_sbatch_script(agg_srun, partition=partition, account=account)
    agg_script_path = scripts_dir / "submit_aggregation.sh"
    agg_script_path.write_text(agg_script)

    agg_job_id = _submit_sbatch(
        agg_script_path,
        job_name="reeval-agg",
        log_path=logs_dir / "reeval_aggregation.log",
        dependency=eval_job_id,
    )
    logger.info(
        "Aggregation job submitted: %d (depends on %d)", agg_job_id, eval_job_id
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate simulation results from ASL logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """\
            Examples:
              # Local re-evaluation
              alpasim-reeval /path/to/sim/output

              # Local re-evaluation
              uv run --project src/eval alpasim-reeval /path/to/sim/output

              # SLURM re-evaluation (point to single-job or array-job)
              # Submits one eval job and one dependent aggregation job.
              alpasim-reeval /path/to/sim/output --slurm \\
                  --partition cpu_short --account my_account
        """
        ),
    )

    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to job directory (single-job) or array-job parent directory.",
    )

    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument(
        "--slurm",
        action="store_true",
        help="Submit as SLURM jobs instead of running locally.",
    )
    slurm_group.add_argument(
        "--partition",
        type=str,
        default="cpu_short",
        help="SLURM partition. Default: cpu_short",
    )
    slurm_group.add_argument(
        "--account",
        type=str,
        default=None,
        help="SLURM account (required with --slurm).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.is_dir():
        logger.error("Directory not found: %s", log_dir)
        return 1

    try:
        layout = detect_layout(log_dir)
    except FileNotFoundError as e:
        logger.exception("Failed to detect layout: %s", e)
        return 1

    logger.info(
        "Detected %s layout with %d job(s)",
        "array" if layout.is_array_job else "single",
        len(layout.job_dirs),
    )

    if args.slurm:
        return run_slurm(layout, partition=args.partition, account=args.account)
    else:
        return run_local(layout)


if __name__ == "__main__":
    sys.exit(main())
