# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import json
import logging
from typing import Any, Optional

import polars as pl

from eval.aggregation.processing import ProcessedMetricDFs

logger = logging.getLogger("alpasim_eval.aggregation.leaderboard")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def upload_for_leaderboard(
    processed_dfs: ProcessedMetricDFs,
    run_metadata: dict[str, Any],
    leaderboard_table_name: Optional[str],
) -> int:
    """
    Prepares metrics for the leaderboard table.

    Note: Database upload functionality has been removed. The leaderboard_metrics
    DataFrame is constructed but not uploaded. To re-enable uploads, integrate
    with a new database backend.

    Args:
      processed_dfs: Processed dataframes containing metric results
      run_metadata: Run metadata dictionary
      leaderboard_table_name: Name of the leaderboard table to upload to

    Returns:
      0 (always succeeds as no upload is performed)
    """
    if run_metadata.get("test_suite_id", None) is None:
        logger.warning(
            "No test suite ID found in wizard config--skip leaderboard processing."
        )
        return 0

    weightings = {
        "avg_dist_between_incidents": 1.0,
        "progress": 0.2,
        "collision_front": 0.0,
        "collision_lateral": 0.0,
        "offroad": 0.0,
        "collision_rear": 0.0,
    }
    df = processed_dfs.df_wide_avg_t_clip_rollout
    weighted_score = sum(df[col] * weight for col, weight in weightings.items()) / sum(
        [abs(weight) for weight in weightings.values()]
    )

    relevant_metrics = {}
    for column in weightings.keys():
        relevant_metrics[column] = df[column][0]

    leaderboard_metrics = pl.DataFrame(
        {
            "run_name": run_metadata["run_name"],
            "run_uuid": run_metadata["run_uuid"],
            "submitter": run_metadata["submitter"],
            "description": run_metadata["description"],
            "slurm_job_id": run_metadata["slurm_job_id"],
            "suite_name": run_metadata["test_suite_id"],
            "date": run_metadata["run_time"],
            "aggregated_metric": weighted_score,
            "metric_details": json.dumps(relevant_metrics, indent=2),
        }
    )

    # TODO: Database upload functionality has been removed.
    # The leaderboard_metrics DataFrame is ready but not uploaded.
    # To re-enable uploads, integrate with a new database backend here.
    logger.info(
        "Leaderboard metrics prepared but not uploaded (database integration removed)"
    )
    logger.debug("Leaderboard metrics: %s", leaderboard_metrics)

    return 0
