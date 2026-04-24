# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
RuntimeEvaluator - handles in-runtime evaluation for the simulation loop.

This module provides the RuntimeEvaluator class that encapsulates all evaluation
logic needed during runtime simulation, including data accumulation, building
evaluation inputs, running metrics computation, and video rendering.

Evaluation runs when `eval_config.enabled` is True; otherwise, calls are no-ops.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import yaml
from alpasim_grpc.v0.logging_pb2 import LogEntry
from trajdata.maps import VectorMap

from eval.accumulator import EvalDataAccumulator
from eval.data import ScenarioEvalInput
from eval.scenario_evaluator import ScenarioEvalResult, ScenarioEvaluator
from eval.schema import EvalConfig
from eval.video import render_video_from_eval_result

logger = logging.getLogger(__name__)


def _evaluate_in_subprocess(
    eval_config: EvalConfig,
    scenario_input: ScenarioEvalInput,
    metrics_path: str,
    render_video: bool,
    video_output_dir: str,
    scene_id: str,
    rollout_uuid: str,
) -> ScenarioEvalResult:
    """Run evaluation in a subprocess. Module-level for pickling.

    Reconstructs a ScenarioEvaluator from config, runs scorer computation,
    writes metrics to parquet, and optionally renders video.
    """
    evaluator = ScenarioEvaluator(eval_config)
    eval_result = evaluator.evaluate(scenario_input)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    if eval_result.metrics_df is not None and len(eval_result.metrics_df) > 0:
        eval_result.metrics_df.write_parquet(metrics_path)

    if render_video:
        render_video_from_eval_result(
            scenario_input=scenario_input,
            metrics_df=eval_result.metrics_df,
            cfg=eval_config,
            output_dir=video_output_dir,
            clipgt_id=scene_id,
            rollout_id=rollout_uuid,
        )

    return eval_result


class RuntimeEvaluator:
    """
    Handles in-runtime evaluation for a simulation rollout.

    This class encapsulates all evaluation-related logic including data
    accumulation during simulation and evaluation at rollout end.

    The implementation uses EvalDataAccumulator internally to process messages,
    ensuring identical behavior between runtime evaluation and post-hoc ASL
    file evaluation. All trajectory and metadata information is extracted from
    the messages broadcast during simulation (rollout_metadata, actor_poses, etc.).

    Usage:
        # Create at rollout start (in BoundRollout.__post_init__)
        evaluator = RuntimeEvaluator(
            eval_config=eval_config,
            rollout_uuid=rollout_uuid,
            scene_id=scene_id,
            save_path_root=save_path_root,
            vector_map=vector_map,
        )

        # During simulation, messages are accumulated via on_message (called by broadcaster):
        # - rollout_metadata: extracts session metadata, AABB dims, gt trajectory
        # - actor_poses: builds trajectories
        # - driver_camera_image, route_request, driver_request/return: accumulated

        # At rollout end, run evaluation in a ProcessPoolExecutor:
        await evaluator.run_evaluation(executor)
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        rollout_uuid: str,
        scene_id: str,
        save_path_root: str,
        vector_map: VectorMap | None,
    ) -> None:
        """
        Initialize the RuntimeEvaluator.

        Args:
            eval_config: Evaluation configuration.
            rollout_uuid: Unique identifier for this rollout.
            scene_id: Scene identifier.
            save_path_root: Root path for this scenario's rollouts.
            vector_map: Vector map for offroad metrics (can be None).
        """
        # Configuration (immutable after init)
        self.eval_config = eval_config
        self.rollout_uuid = rollout_uuid
        self.scene_id = scene_id
        self.save_path_root = save_path_root
        self.vector_map = vector_map

        # Initialize accumulator (accumulates info for eval) if evaluation is enabled
        self._accumulator: EvalDataAccumulator | None = (
            EvalDataAccumulator(cfg=eval_config) if self._enabled else None
        )

    @property
    def _enabled(self) -> bool:
        """Check if in-runtime evaluation is enabled."""
        return self.eval_config.enabled

    @property
    def _should_render_video(self) -> bool:
        """Check if video rendering is enabled."""
        return self._enabled and self.eval_config.video.render_video

    async def on_message(self, message: LogEntry) -> None:
        """
        Handle a LogEntry message by delegating to the accumulator.

        This method implements the MessageHandler protocol, allowing RuntimeEvaluator
        to be used as a handler in MessageBroadcaster. It processes all eval-relevant
        message types via the EvalDataAccumulator:
        - rollout_metadata: Session metadata, AABB dims, transforms, gt trajectory
        - actor_poses: Accumulate poses to build trajectories
        - driver_camera_image: Camera images for image metrics
        - route_request: Route data for video rendering
        - driver_request: Stores timestamps for pairing with driver_return
        - driver_return: Driver responses for trajectory metrics
        - available_cameras_return: Camera calibrations

        The accumulator ensures consistent behavior between runtime evaluation
        and post-hoc ASL file evaluation.

        Args:
            message: The LogEntry protobuf message to process.
        """
        if self._enabled and self._accumulator is not None:
            self._accumulator.handle_message(message)

    def _get_run_metadata(self) -> tuple[str, str]:
        """
        Get run_uuid and run_name from run_metadata.yaml.

        Returns:
            Tuple of (run_uuid, run_name).

        Raises:
            FileNotFoundError: If run_metadata.yaml is not found.
            ValueError: If the metadata file is empty or invalid.
        """
        # Try to find run_metadata.yaml in parent directories of save_path_root
        # save_path_root is typically: <log_dir>/rollouts/<scene_id>
        # run_metadata.yaml is in: <log_dir>/
        log_dir = os.path.dirname(os.path.dirname(self.save_path_root))
        metadata_path = os.path.join(log_dir, "run_metadata.yaml")

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
            if metadata is None:
                raise ValueError(f"Empty metadata file at {metadata_path}")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Invalid metadata file at {metadata_path}: expected dict, got {type(metadata)}"
                )
            run_uuid = metadata.get("run_uuid", self.rollout_uuid)
            run_name = metadata.get("run_name", "runtime_eval")
            return run_uuid, run_name

    def _rollout_dir(self) -> str:
        """Directory for this rollout's output files."""
        return os.path.join(self.save_path_root, self.rollout_uuid)

    def _metrics_path(self) -> str:
        """Path to per-rollout metrics parquet file."""
        return os.path.join(self._rollout_dir(), "metrics.parquet")

    def build_eval_input(self) -> ScenarioEvalInput | None:
        """Build ScenarioEvalInput from accumulated message data.

        This runs in the main process and extracts all accumulated data
        into a picklable input object suitable for subprocess evaluation.

        Returns:
            ScenarioEvalInput if accumulation succeeded, None otherwise.
        """
        if not self._enabled or self._accumulator is None:
            return None

        run_uuid, run_name = self._get_run_metadata()
        return self._accumulator.build_scenario_eval_input(
            run_uuid=run_uuid,
            run_name=run_name,
            vec_map=self.vector_map,
        )

    async def run_evaluation(
        self,
        executor: ProcessPoolExecutor,
    ) -> ScenarioEvalResult | None:
        """Run evaluation in a ProcessPoolExecutor.

        Builds the eval input in the main process, then submits the
        CPU-bound scorer computation, parquet writing, and optional video
        rendering to the provided executor.

        Eval failures are caught and logged; they do not fail the rollout.

        Args:
            executor: ProcessPoolExecutor to run evaluation in.

        Returns:
            ScenarioEvalResult if evaluation succeeded, None otherwise.
        """
        if not self._enabled:
            return None

        scenario_input = self.build_eval_input()
        if scenario_input is None:
            logger.warning(
                "RuntimeEvaluator enabled but build_eval_input returned None"
            )
            return None

        logger.info(
            "Submitting evaluation to process pool for session %s",
            self.rollout_uuid,
        )

        metrics_path = self._metrics_path()
        loop = asyncio.get_running_loop()

        try:
            eval_result = await loop.run_in_executor(
                executor,
                functools.partial(
                    _evaluate_in_subprocess,
                    eval_config=self.eval_config,
                    scenario_input=scenario_input,
                    metrics_path=metrics_path,
                    render_video=self._should_render_video,
                    video_output_dir=os.path.dirname(metrics_path),
                    scene_id=self.scene_id,
                    rollout_uuid=self.rollout_uuid,
                ),
            )
        except Exception:
            logger.exception(
                "Evaluation failed in subprocess for session %s",
                self.rollout_uuid,
            )
            return None

        # Log results in the main process
        if eval_result.metrics_df is not None and len(eval_result.metrics_df) > 0:
            logger.info(
                "Saved %d metric rows to %s",
                len(eval_result.metrics_df),
                metrics_path,
            )
        else:
            logger.warning(
                "No metrics computed for session %s",
                self.rollout_uuid,
            )

        if eval_result.aggregated_metrics:
            logger.info(
                "Aggregated metrics for %s: %s",
                self.rollout_uuid,
                {k: f"{v:.4f}" for k, v in eval_result.aggregated_metrics.items()},
            )

        return eval_result
