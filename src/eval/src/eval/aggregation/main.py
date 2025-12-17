# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Post-eval aggregation: Aggregating results across all array jobs."""

import argparse
import logging
import os
import pathlib
import shutil
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
from eval.schema import EvalConfig
from eval.video import VIDEO_FILE_NAME_FORMAT

# Configure the root logger first to affect all modules
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(logging.StreamHandler())

# Set up the specific logger for this module
logger = logging.getLogger("alpasim_eval.aggregation")
logger.setLevel(logging.INFO)
# No need to add handler to this logger as it will inherit from root

CONCAT_VIDEO_NAME = "00_all_clips"


def _aggregate_metrics(
    job_dirs: list[pathlib.Path],
    aggregate_dir: str | pathlib.Path,
    modifiers: list[MetricAggregationModifiers],
) -> ProcessedMetricDFs:
    # Get all parquet files in the job directories
    dfs = []
    for job_dir in job_dirs:
        file = job_dir / "eval" / "metrics_unprocessed.parquet"
        dfs.append(pl.read_parquet(file))
    df = pl.concat(dfs)

    # Overwrite the run_uuid to be the same for all rows, coming from different
    # array jobs.
    return processing.aggregate_and_write_metrics_results_txt(
        df,
        force_same_run=True,
        output_path=str(aggregate_dir),
        additional_modifiers=modifiers,
    )


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
    """Aggregate eval videos across all array jobs."""
    logger.info("Aggregating eval videos across all array jobs.")
    all_videos_dir = target_video_dir / "all"
    os.makedirs(all_videos_dir, exist_ok=True)
    for job_dir in job_dirs:
        video_dir = job_dir / "eval" / "videos"
        if not os.path.exists(video_dir):
            logger.warning("Video directory %s does not exist", video_dir)
            continue
        video_files = [f for f in video_dir.glob("*.mp4")]
        if len(video_files) == 0:
            logger.warning("No videos found in %s", video_dir)
            continue
        video_files.sort()
        for video_file in video_files:
            shutil.copy(video_file, all_videos_dir / video_file.name)
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
        for row in filtered_df.iter_rows(named=True):
            layout_id = (
                cfg.video.video_layouts[0].index
                if len(cfg.video.video_layouts) > 0
                else 0
            )
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

    # Check if we're running in an array job or single job.
    if int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0)) > 0:
        # Identify job directories by checking for the presence of
        # `wizard-config.yaml`, which is written at the start of every job
        # regardless of whether it eventually succeeds.  This approach avoids
        # maintaining a hard-coded exclusion list and ensures that helper
        # folders (e.g. `outputs`, `runs`, `aggregate`, etc.) are ignored automatically.
        job_dirs: list[pathlib.Path] = []
        WIZARD_FILE = "wizard-config.yaml"
        for d in array_job_dir.iterdir():
            if d.is_dir() and (d / WIZARD_FILE).exists():
                job_dirs.append(d)
            else:
                logger.info(
                    "Skipping directory %s – not recognized as job dir "
                    "(wizard config missing)",
                    d,
                )
        job_dirs.sort()
    else:
        job_dirs = sorted([array_job_dir])

    logger.info(
        "Running post-eval aggregation. Found %d job directories in %s: %s",
        len(job_dirs),
        array_job_dir,
        ", ".join([str(d) for d in job_dirs]),
    )

    aggregate_dir = array_job_dir / "aggregate"
    os.makedirs(aggregate_dir, exist_ok=True)

    modifiers = [
        RemoveTimestepsAfterEvent(
            pl.col("dist_to_gt_trajectory")
            >= cfg.aggregation_modifiers.max_dist_to_gt_trajectory
        ),
    ]

    processed_dfs = _aggregate_metrics(job_dirs, aggregate_dir, modifiers)
    processed_dfs.save_to(aggregate_dir)
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

    return_code = 0
    return return_code


if __name__ == "__main__":
    sys.exit(main())
