# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Post-eval main entry point for evaluation.
"""

import argparse
import asyncio
import glob
import logging
import multiprocessing as mp
import os
import pathlib
import pprint
import sys
from functools import partial
from typing import Any

import polars as pl
from alpasim_utils.artifact import Artifact
from alpasim_utils.paths import extract_ids_from_path
from omegaconf import OmegaConf
from tqdm import tqdm

from eval.asl_loader import load_scenario_eval_input_from_asl
from eval.metadata import get_metadata
from eval.scenario_evaluator import ScenarioEvaluator
from eval.schema import EvalConfig
from eval.video import render_video_from_eval_result

logger = logging.getLogger("alpasim_eval")

TRACKER_FILE_NAME = "_complete"


def process_asl_file(
    asl_file_path: str,
    cfg: EvalConfig,
    artifacts: dict[str, Artifact],
    run_metadata: dict[str, Any],
) -> pl.DataFrame:
    """
    Process a single ASL file using the unified evaluation path.

    This function:
    1. Loads ASL into ScenarioEvalInput via load_scenario_eval_input_from_asl
    2. Evaluates using ScenarioEvaluator.evaluate
    3. Writes per-rollout metrics parquet next to the ASL file
    4. Optionally renders video next to the ASL file

    Args:
        asl_file_path: Path to the ASL file.
        cfg: Evaluation configuration.
        artifacts: Dictionary of scene artifacts (maps, etc.).
        run_metadata: Run metadata containing run_uuid and run_name.

    Returns:
        DataFrame with unprocessed metrics.
    """
    # Load ASL into ScenarioEvalInput (unified loading path)
    scenario_input = asyncio.run(
        load_scenario_eval_input_from_asl(asl_file_path, cfg, artifacts, run_metadata)
    )

    # Use unified evaluator
    evaluator = ScenarioEvaluator(cfg)
    result = evaluator.evaluate(scenario_input)

    if result.metrics_df is None or len(result.metrics_df) == 0:
        raise ValueError(f"No metrics were computed for ASL file: {asl_file_path}")

    # Write per-rollout metrics parquet next to the ASL file
    # This creates a unified metrics path structure (rollouts/**/metrics.parquet)
    rollout_dir = os.path.dirname(asl_file_path)
    metrics_path = os.path.join(rollout_dir, "metrics.parquet")
    if result.metrics_df is not None and len(result.metrics_df) > 0:
        result.metrics_df.write_parquet(metrics_path)
        logger.debug("Wrote per-rollout metrics to %s", metrics_path)

    # Video rendering (if enabled) - save next to ASL file (unified path structure)
    if cfg.video.render_video:
        clipgt_id, batch_id, rollout_id = extract_ids_from_path(asl_file_path)
        render_video_from_eval_result(
            scenario_input=scenario_input,
            metrics_df=result.metrics_df,
            cfg=cfg,
            output_dir=rollout_dir,
            clipgt_id=clipgt_id,
            batch_id=batch_id,
            rollout_id=rollout_id,
        )
    else:
        logger.info("Skipping video rendering as it is disabled in the config.")

    return result.metrics_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perform KPIs evaluation of ASL files."
    )
    parser.add_argument("--asl_search_glob", type=str)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--trajdata_cache_dir", type=str)
    parser.add_argument("--usdz_glob", type=str)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    assert args.config_path.endswith(".yaml")

    # TODO(pkarkus): consider skipping eval if all metrics files are already present.
    # It's less trivial with per-rollout parquet files compared to legacy metrics_unprocessed.parquet.

    # Needed for matplotlib to work in multiprocessing
    mp.set_start_method("forkserver")

    config_untyped = OmegaConf.load(args.config_path)
    cfg: EvalConfig = OmegaConf.merge(EvalConfig, config_untyped)

    logger.info("Config details:")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg, resolve=True)))

    artifacts = Artifact.discover_from_glob(args.usdz_glob)
    files = glob.glob(args.asl_search_glob, recursive=True)
    run_metadata = get_metadata(pathlib.Path(args.config_path).parent)

    filtered_files = [
        fname
        for fname in files
        if (pathlib.Path(fname).parent / TRACKER_FILE_NAME).exists()
    ]
    if not filtered_files:
        raise ValueError(
            f"No files found in {args.asl_search_glob}. "
            "This case should be handled by the wizard by not dispatching KPIs at all."
        )
    if len(filtered_files) != len(files):
        raise ValueError(
            f"Only {len(filtered_files)} out of {len(files)} files have been found to be complete."
        )

    num_workers = min(
        mp.cpu_count(),
        cfg.num_processes,
        len(filtered_files),
    )

    logger.info(
        "Using %d workers: %d CPUs available, config asks for %d",
        num_workers,
        mp.cpu_count(),
        cfg.num_processes,
    )

    logger.info(
        "Processing %d ASL files...",
        len(filtered_files),
    )

    # Use partial to fix the arguments that are the same for all calls
    process_func = partial(
        process_asl_file,
        cfg=cfg,
        artifacts=artifacts,
        run_metadata=run_metadata,
    )

    if num_workers > 1:
        logger.info("Processing ASL files in multi-process mode.")
        with mp.Pool(processes=num_workers) as pool:
            df_list = list(
                tqdm(
                    pool.imap_unordered(process_func, filtered_files),
                    total=len(filtered_files),
                    desc="Processing ASL files",
                )
            )
    else:  # num_workers == 1
        logger.info("Processing ASL files in single-process mode.")
        df_list = [
            process_func(asl_file)
            for asl_file in tqdm(
                filtered_files,
                desc="Processing ASL files",
            )
        ]

    # Ensure no empty dataframes returned (fail-fast)
    if any(len(df) == 0 for df in df_list):
        raise ValueError(
            "Empty metrics DataFrame returned. This should not happen as process_asl_file validates non-empty results."
        )

    return 0


if __name__ == "__main__":  #
    sys.exit(main())
