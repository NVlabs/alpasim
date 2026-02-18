# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Post-eval aggregation: Aggregating results across all array jobs."""

import argparse
import logging
import os
import pathlib
import subprocess
import sys

import polars as pl
from omegaconf import OmegaConf

from eval.aggregation import processing, utils
from eval.aggregation.modifiers import (
    MetricAggregationModifiers,
    RemoveTimestepsAfterEvent,
)
from eval.aggregation.processing import ProcessedMetricDFs
from eval.metadata import get_metadata
from eval.schema import EvalConfig
from eval.video import VIDEO_FILE_NAME_FORMAT

WIZARD_FILE = "wizard-config.yaml"

# Configure the root logger first to affect all modules
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(logging.StreamHandler())

# Set up the specific logger for this module
logger = logging.getLogger("alpasim_eval.aggregation")
logger.setLevel(logging.INFO)
# No need to add handler to this logger as it will inherit from root

CONCAT_VIDEO_NAME = "00_all_clips"


def _collect_metrics_from_job_dir(job_dir: pathlib.Path) -> pl.DataFrame | None:
    """
    Collect metrics from a single job directory.

    Both post-eval and runtime-eval write metrics to the same unified path:
    <job_dir>/rollouts/<scene_id>/<rollout_uuid>/metrics.parquet

    Returns:
        DataFrame with metrics from this job directory, or None if no metrics found.
    """
    rollouts_dir = job_dir / "rollouts"
    if not rollouts_dir.exists():
        return None

    metrics_files = list(rollouts_dir.glob("**/metrics.parquet"))
    if not metrics_files:
        return None

    # Use Polars glob pattern to read all parquet files at once
    glob_pattern = str(rollouts_dir / "**" / "metrics.parquet")
    try:
        df = pl.read_parquet(glob_pattern)
        logger.info("Loaded metrics from %d files in %s", len(metrics_files), job_dir)
        return df
    except (OSError, pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
        logger.warning("Failed to read metrics from %s: %s", glob_pattern, e)
        return None


def _aggregate_metrics(
    job_dirs: list[pathlib.Path],
    aggregate_dir: str | pathlib.Path,
    modifiers: list[MetricAggregationModifiers],
) -> ProcessedMetricDFs:
    """
    Aggregate metrics from job directories.

    Both post-eval and runtime-eval use the same unified metrics path:
    <job_dir>/rollouts/<scene_id>/<rollout_uuid>/metrics.parquet
    """
    all_dfs: list[pl.DataFrame] = []
    for job_dir in job_dirs:
        df = _collect_metrics_from_job_dir(job_dir)
        if df is None:
            logger.warning("No metrics found in %s", job_dir)
        else:
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(
            "No metrics files found in any job directory. Ensure either post-eval "
            "or in-runtime evaluation has completed successfully."
        )

    logger.info(
        "Aggregating metrics from %d job directories with data (of %d total)",
        len(all_dfs),
        len(job_dirs),
    )
    df = pl.concat(all_dfs)

    return processing.aggregate_and_write_metrics_results_txt(
        df,
        force_same_run=True,
        output_path=str(aggregate_dir),
        additional_modifiers=modifiers,
    )


def _run_aggregation_core(
    job_dirs: list[pathlib.Path],
    aggregate_dir: pathlib.Path,
    cfg: EvalConfig,
) -> ProcessedMetricDFs:
    """
    Core aggregation logic shared between runtime and CLI entry points.

    Args:
        job_dirs: List of job directories to aggregate.
        aggregate_dir: Directory to write aggregated results.
        cfg: Evaluation configuration.

    Returns:
        ProcessedMetricDFs containing aggregated metrics data.

    Raises:
        ValueError: If no metrics files are found.
    """
    os.makedirs(aggregate_dir, exist_ok=True)

    modifiers = [
        RemoveTimestepsAfterEvent(
            pl.col("dist_to_gt_trajectory")
            >= cfg.aggregation_modifiers.max_dist_to_gt_trajectory
        ),
    ]

    processed_dfs = _aggregate_metrics(job_dirs, aggregate_dir, modifiers)
    processed_dfs.save_to(aggregate_dir)

    logger.info("Aggregation complete. Results saved to %s", aggregate_dir)

    # Handle video aggregation
    if cfg.video.render_video:
        conditions = {
            "collision_at_fault": pl.col("collision_at_fault") > 0.0,
            "collision_rear": pl.col("collision_rear") > 0.0,
            "offroad": pl.col("offroad") > 0.0,
            "dist_to_gt_trajectory": pl.col("dist_to_gt_trajectory")
            >= cfg.aggregation_modifiers.max_dist_to_gt_trajectory,
        }
        _aggregate_eval_videos(
            job_dirs, aggregate_dir / "videos", cfg, processed_dfs, conditions
        )
    else:
        logger.info(
            "Skipping video aggregation as render_video is disabled in the config."
        )

    return processed_dfs


def run_aggregation_from_runtime(
    log_dir: str | pathlib.Path,
    eval_config: EvalConfig,
    array_job_dir: str | pathlib.Path | None = None,
) -> bool:
    """
    Run metric aggregation from the runtime after all rollouts complete.

    This function is designed to be called at the end of the runtime simulation loop.
    It handles synchronization across SLURM array jobs using a file-based counter,
    ensuring aggregation only runs once when the last job finishes.

    Args:
        log_dir: The log directory for this job (contains asl/, metrics/, etc.)
        eval_config: The evaluation configuration
        array_job_dir: Parent directory containing all array job directories.
                       If None, defaults to log_dir (single job mode).

    Returns:
        True if aggregation was run or skipped (not last job).

    Raises:
        ValueError: If no job directories are found for aggregation.
    """
    log_dir = pathlib.Path(log_dir)
    array_job_dir = pathlib.Path(array_job_dir) if array_job_dir else log_dir

    # Check if we're the last job in the array
    if not utils.incr_counter_and_check_aggregation_start(array_job_dir):
        logger.info(
            "Not the last job in array, skipping aggregation. "
            "Aggregation will run when last job finishes."
        )
        return True

    job_dirs = _discover_job_dirs(array_job_dir, log_dir)
    if not job_dirs:
        raise ValueError(
            f"No job directories found for aggregation in {array_job_dir}. "
            "This indicates a configuration error or missing wizard-config.yaml files."
        )

    logger.info(
        "Running aggregation. Found %d job directories: %s",
        len(job_dirs),
        ", ".join([str(d) for d in job_dirs]),
    )

    aggregate_dir = array_job_dir / "aggregate"

    _run_aggregation_core(job_dirs, aggregate_dir, eval_config)
    return True


def _discover_job_dirs(
    array_job_dir: pathlib.Path,
    log_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """
    Discover job directories within an array job directory.

    In single job mode (log_dir provided and equals array_job_dir), returns
    just array_job_dir without searching subdirectories.
    In array job mode (log_dir not provided or different from array_job_dir),
    finds all subdirectories containing wizard-config.yaml.

    Args:
        array_job_dir: Parent directory containing all array job directories.
        log_dir: The log directory for this specific job. If None or different
            from array_job_dir, array mode is used.

    Returns:
        List of job directories to aggregate.
    """
    # If log_dir is provided and equals array_job_dir, it's single job mode
    if log_dir is not None and array_job_dir == log_dir:
        return [array_job_dir]

    # Array job mode - find subdirectories with wizard-config.yaml
    job_dirs: list[pathlib.Path] = []
    for d in array_job_dir.iterdir():
        if d.is_dir() and (d / WIZARD_FILE).exists():
            job_dirs.append(d)
        else:
            logger.info(
                "Skipping directory %s - not recognized as job dir (wizard config missing)",
                d,
            )

    # If no subdirectories found, fall back to using array_job_dir itself
    # (handles CLI running locally without SLURM array)
    if not job_dirs:
        return [array_job_dir]
    return sorted(job_dirs)


def _speed_up_video(video_file: pathlib.Path, speed_factor: float) -> None:
    # Create output filename by appending _fast before extension
    output_file = video_file.parent / (video_file.stem + "_fast" + video_file.suffix)

    try:
        # Run ffmpeg command to speed up video
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_file),
                "-vf",
                f"fps=30,setpts={speed_factor}*PTS",  # Speed up by 1 / speed_factor
                "-vsync",
                "vfr",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-an",  # Remove audio
                str(output_file),
            ],
            check=True,
        )
        os.remove(video_file)

    except subprocess.CalledProcessError as e:
        logger.error("Failed to speed up video %s: %s", video_file, e)


def _concatenate_videos(video_dir: pathlib.Path) -> None:
    videos = list(video_dir.glob("*.mp4"))
    if videos:
        # Create file listing all videos to concatenate
        with open(video_dir / "concat_list.txt", "w") as f:
            for video in videos:
                f.write(f"file '{video.name}'\n")

        try:
            # Use ffmpeg to concatenate videos
            concat_output = video_dir / f"{CONCAT_VIDEO_NAME}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(video_dir / "concat_list.txt"),
                    "-c",
                    "copy",
                    str(concat_output),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to concatenate videos: %s", e)

        os.remove(video_dir / "concat_list.txt")


def _aggregate_eval_videos(
    job_dirs: list[pathlib.Path],
    target_video_dir: pathlib.Path,
    cfg: EvalConfig,
    processed_dfs: ProcessedMetricDFs,
    conditions: dict[str, pl.Expr],
) -> None:
    """Aggregate eval videos across all array jobs.

    Videos are saved next to ASL files in the unified path structure:
    <job_dir>/rollouts/<scene_id>/<rollout_uuid>/<video_name>.mp4

    This function creates symlinks to all videos in the aggregate/videos/all/ directory.
    """
    logger.info("Aggregating eval videos across all array jobs.")
    all_videos_dir = target_video_dir / "all"
    os.makedirs(all_videos_dir, exist_ok=True)

    total_videos = 0
    for job_dir in job_dirs:
        rollouts_dir = job_dir / "rollouts"
        if not rollouts_dir.exists():
            logger.warning("Rollouts directory %s does not exist", rollouts_dir)
            continue

        # Find all video files in rollouts/**/*.mp4
        video_files = list(rollouts_dir.glob("**/*.mp4"))
        if not video_files:
            logger.warning("No videos found in %s", rollouts_dir)
            continue

        video_files.sort()
        for video_file in video_files:
            # Use the basename as the symlink name. By design, video_file.name is
            # "{clipgt_id}_{batch_id}_{rollout_id}.mp4" (VIDEO_FILE_NAME_FORMAT),
            # which ensures global uniqueness across jobs; collisions are not expected.
            symlink_path = all_videos_dir / video_file.name
            symlink_path.unlink(missing_ok=True)
            # Create relative symlink for portability across different mount points
            relative_target = pathlib.Path(os.path.relpath(video_file, all_videos_dir))
            symlink_path.symlink_to(relative_target)
            total_videos += 1

    logger.info(
        "Found and linked %d videos from %d job directories",
        total_videos,
        len(job_dirs),
    )
    if cfg.video.generate_combined_video:
        _concatenate_videos(all_videos_dir)
        _speed_up_video(
            all_videos_dir / f"{CONCAT_VIDEO_NAME}.mp4",
            cfg.video.combined_video_speed_factor,
        )
        for video_file in all_videos_dir.glob("*.mp4"):
            if not video_file.name.startswith(CONCAT_VIDEO_NAME):
                logger.info("Removing video file %s", video_file)
                os.remove(video_file)
        return

    # Note generating combined video. Instead create subfolders for conditions
    # with links to the video in "all"
    for condition_name, condition in conditions.items():
        filtered_df = processed_dfs.df_wide_avg_t.filter(condition)
        condition_folder = target_video_dir / "violations" / condition_name
        os.makedirs(condition_folder, exist_ok=True)
        layouts_to_link = (
            cfg.video.video_layouts if len(cfg.video.video_layouts) > 0 else ["default"]
        )
        for row in filtered_df.iter_rows(named=True):
            for layout_id in layouts_to_link:
                video_file_name = VIDEO_FILE_NAME_FORMAT.format(
                    clipgt_id=row["clipgt_id"],
                    batch_id=row["batch_id"],
                    rollout_id=row["rollout_id"],
                    camera_id=cfg.video.camera_id_to_render,
                    layout_id=layout_id,
                )
                (condition_folder / video_file_name).unlink(missing_ok=True)
                # Create a relative symlink to the video in all_videos_dir to ensure the link will be
                # valid even when running with different mount points.
                relative_path = pathlib.Path(
                    os.path.relpath(all_videos_dir, condition_folder)
                )
                (condition_folder / video_file_name).symlink_to(
                    relative_path / video_file_name
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--array_job_dir",
        type=str,
        required=True,
        help="Directory containing array job results",
    )
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    array_job_dir = pathlib.Path(args.array_job_dir)

    config_untyped = OmegaConf.load(args.config_path)
    cfg: EvalConfig = OmegaConf.merge(EvalConfig, config_untyped)

    # Check if we're the last job in the array. If not, we skip the aggregation.
    if not utils.incr_counter_and_check_aggregation_start(array_job_dir):
        logger.info(
            "Not array job or not the last job, skipping post-eval aggregation."
        )
        return 0

    # CLI mode: search for subdirectories with wizard-config.yaml
    # Falls back to using array_job_dir itself if no subdirectories found
    job_dirs = _discover_job_dirs(array_job_dir)

    logger.info(
        "Running aggregation. Found %d job directories in %s: %s",
        len(job_dirs),
        array_job_dir,
        ", ".join([str(d) for d in job_dirs]),
    )

    aggregate_dir = array_job_dir / "aggregate"

    _run_aggregation_core(job_dirs, aggregate_dir, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
