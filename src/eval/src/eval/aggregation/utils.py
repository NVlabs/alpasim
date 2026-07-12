# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import json
import logging
import os
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger("alpasim_eval.post_eval_aggregation")


def incr_counter_and_check_aggregation_start(log_dir: str) -> bool:
    """
    Increments counter and checks if post_eval_aggregation should be started in
    this job.

    Returns True if we're the last job in the array or if there is no array job.
    """
    lock = FileLock(Path(log_dir) / "post_eval_aggregation.lock")
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 0))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if task_count == 0:
        logger.info("No array job, don't need to check counter.")
        return True

    with lock:
        # Set `prev_finished_jobs` by loading it from the file if it exists.
        # Otherwise set it to 0.
        if not (Path(log_dir) / "post_eval_aggregation.json").is_file():
            logger.info("No post_eval_aggregation.json file, starting one.")
            prev_finished_jobs = 0
            prev_finished_job_ids = []
        else:
            with open(Path(log_dir) / "post_eval_aggregation.json", "r") as f:
                data = json.load(f)
                prev_finished_job_ids = data["finished_job_ids"]
                prev_finished_jobs = len(set(prev_finished_job_ids))
                logger.info(
                    "Loaded post_eval_aggregation.json file, prev_finished_jobs: %d, prev_finished_job_ids: %s",
                    prev_finished_jobs,
                    prev_finished_job_ids,
                )

        # Record each logical array task once. A requeued task may enter this
        # function more than once if aggregation itself was interrupted.
        finished_job_ids = sorted(set(prev_finished_job_ids) | {task_id})
        with open(Path(log_dir) / "post_eval_aggregation.json", "w") as f:
            json.dump(
                {
                    "finished_jobs": len(finished_job_ids),
                    "finished_job_ids": finished_job_ids,
                },
                f,
            )
        logger.info(
            "Wrote post_eval_aggregation.json file, finished_jobs: %d, finished_job_ids: %s",
            len(finished_job_ids),
            finished_job_ids,
        )

        if len(finished_job_ids) < task_count:
            logger.info(
                "Not the last job, skipping post_eval_aggregation. "
                "finished_jobs: %d, task_count: %d, finished_job_ids: %s",
                len(finished_job_ids),
                task_count,
                finished_job_ids,
            )
            return False
        if len(finished_job_ids) > task_count:
            logger.warning(
                "More task IDs finished than expected: expected %d, found %d",
                task_count,
                len(finished_job_ids),
            )
        logger.info("All logical array tasks finished, starting post_eval_aggregation")
        return True
