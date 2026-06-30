# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
The main entrypoint to start simulations with alpasim.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from alpasim_grpc.v0 import runtime_pb2
from alpasim_runtime.autoresume import (
    find_num_complete_rollouts,
    remove_incomplete_rollouts,
)
from alpasim_runtime.config import UserSimulatorConfig
from alpasim_runtime.daemon.app import RuntimeDaemonApp
from alpasim_runtime.daemon.engine import DaemonEngine
from alpasim_runtime.runtime_context import parse_simulator_config
from alpasim_runtime.telemetry.plot_metrics import generate_metrics_plot
from alpasim_runtime.validation import validate_array_job_config
from alpasim_utils.yaml_utils import typed_parse_config

from eval.aggregation.failed_rollouts import FailedRollout
from eval.aggregation.main import run_aggregation_from_runtime
from eval.schema import EvalConfig

logger = logging.getLogger(__name__)


def get_run_name(log_dir: str) -> str:
    run_metadata_path = os.path.join(log_dir, "run_metadata.yaml")
    with open(run_metadata_path, "r") as f:
        run_metadata = yaml.safe_load(f)
    return run_metadata["run_name"]


def _failed_rollouts_from_returns(
    rollout_returns: list[runtime_pb2.SimulationReturn.RolloutReturn],
    *,
    run_name: str | None,
) -> list[FailedRollout]:
    failed_rollouts = []
    for idx, rollout_return in enumerate(rollout_returns):
        if rollout_return.success:
            continue
        failed_rollouts.append(
            FailedRollout(
                run_name=run_name,
                run_uuid=None,
                clipgt_id=rollout_return.rollout_spec.scenario_id,
                rollout_id=rollout_return.rollout_uuid or f"failed-{idx}",
                error=rollout_return.error,
            )
        )
    return failed_rollouts


def _write_metrics_artifact_error(prometheus_dir: Path, exc: BaseException) -> None:
    prometheus_dir.mkdir(parents=True, exist_ok=True)
    error_path = prometheus_dir / "metrics_plot_error.txt"
    error_path.write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
    logger.warning("Telemetry plot generation failed: %s", exc)


def _generate_metrics_artifacts(
    *,
    prometheus_url: str,
    log_dir: Path,
) -> None:
    """Generate best-effort runtime metrics artifacts from local Prometheus."""
    prometheus_dir = log_dir / "prometheus"
    try:
        generate_metrics_plot(
            prometheus_url=prometheus_url,
            output_path=log_dir / "metrics_plot.png",
        )
    except Exception as exc:
        _write_metrics_artifact_error(prometheus_dir, exc)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # We split user and network config files because the latter is commonly generated.
    parser.add_argument("--user-config", type=str, required=True)
    parser.add_argument("--network-config", type=str, required=True)

    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Root directory for all simulation outputs (rollouts/, prometheus/, txt-logs/)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        required=False,
        default="INFO",
        help="Python logging level (e.g. DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--array-job-dir",
        type=str,
        required=False,
        default=None,
        help="Parent directory for SLURM array jobs. Used for aggregation across jobs. "
        "Defaults to --log-dir for single job runs.",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        required=True,
        help="Path to evaluation config file (mandatory). Controls in-runtime evaluation settings.",
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--listen-address", type=str, default="[::]:50051")

    return parser


async def _serve(args: argparse.Namespace) -> None:
    """Start the runtime in daemon (gRPC server) mode."""
    engine = DaemonEngine(
        user_config=args.user_config,
        network_config=args.network_config,
        eval_config=args.eval_config,
        log_dir=args.log_dir,
        validate_config_scenes=False,
    )
    app = RuntimeDaemonApp(
        engine=engine,
        listen_address=args.listen_address,
    )
    await app.run()


def build_simulation_request(
    user_config: UserSimulatorConfig,
    rollouts_dir: str,
) -> runtime_pb2.SimulationRequest:
    """Build a SimulationRequest from user config, respecting autoresume settings.

    Iterates over configured scenes, subtracts already-completed rollouts when
    autoresume is enabled, and constructs a RolloutSpec per scene with the
    remaining rollout count.
    """
    rollout_specs = []
    total_rollouts = 0

    for scene in user_config.scenes:
        n_rollouts_to_dispatch = (
            scene.n_rollouts
            if scene.n_rollouts is not None
            else user_config.simulation_config.n_rollouts
        )

        # Handle autoresume: skip already-completed rollouts
        if user_config.enable_autoresume:
            remove_incomplete_rollouts(rollouts_dir, scene.scene_id)
            num_finished_rollouts = find_num_complete_rollouts(
                rollouts_dir, scene.scene_id
            )
            if num_finished_rollouts > 0:
                logger.info(
                    f"Found {num_finished_rollouts} already completed rollouts for "
                    f"scene_id={scene.scene_id}"
                )
                n_rollouts_to_dispatch -= num_finished_rollouts

        if n_rollouts_to_dispatch <= 0:
            continue

        rollout_specs.append(
            runtime_pb2.RolloutSpec(
                scenario_id=scene.scene_id,
                nr_rollouts=n_rollouts_to_dispatch,
            )
        )
        total_rollouts += n_rollouts_to_dispatch

    logger.info("Built %d jobs to execute", total_rollouts)
    return runtime_pb2.SimulationRequest(rollout_specs=rollout_specs)


async def run_simulation(args: argparse.Namespace) -> bool:
    """Main simulation orchestration."""
    config = parse_simulator_config(args.user_config, args.network_config)
    eval_config = typed_parse_config(args.eval_config, EvalConfig)

    # Derive output directories from log_dir
    rollouts_dir = os.path.join(args.log_dir, "rollouts")

    # Validate nr_workers
    if config.user.nr_workers < 1:
        raise ValueError(f"nr_workers must be >= 1, got {config.user.nr_workers}")

    request = build_simulation_request(config.user, rollouts_dir)

    failed_rollouts: list[FailedRollout] = []
    all_rollouts_successful = True

    if not request.rollout_specs:
        logger.info("No jobs to run (all rollouts already complete or no scenarios).")
    else:
        simulation_return = await _run_one_shot_request(args, request)

        # Validate result count
        expected_results = sum(spec.nr_rollouts for spec in request.rollout_specs)
        actual_results = len(simulation_return.rollout_returns)
        if actual_results != expected_results:
            raise RuntimeError(
                "Daemon returned unexpected number of job results: "
                f"expected {expected_results} results, got {actual_results}"
            )

        # Check for failures
        all_rollouts_successful = all(
            rr.success for rr in simulation_return.rollout_returns
        )
        if not all_rollouts_successful:
            failed = [rr for rr in simulation_return.rollout_returns if not rr.success]
            failed_rollouts = _failed_rollouts_from_returns(
                list(simulation_return.rollout_returns),
                run_name=get_run_name(args.log_dir),
            )
            logger.error("%d jobs failed:", len(failed))
            for rr in failed[:3]:
                logger.error("  Scene %s: %s", rr.rollout_spec.scenario_id, rr.error)
            if len(failed) > 3:
                logger.error("  ... and %d more", len(failed) - 3)

        _generate_metrics_artifacts(
            prometheus_url=config.user.prometheus.url,
            log_dir=Path(args.log_dir),
        )

    success = all_rollouts_successful

    allow_aggregation_with_failed_rollouts = getattr(
        eval_config,
        "allow_aggregation_with_failed_rollouts",
        False,
    )
    if eval_config.enabled:
        if failed_rollouts:
            if not allow_aggregation_with_failed_rollouts:
                logger.warning(
                    "Rollouts failed; skipping aggregation because "
                    "eval.allow_aggregation_with_failed_rollouts is false"
                )
                return False
            logger.warning(
                "Rollouts failed; running aggregation with %d failed rollout row(s)",
                len(failed_rollouts),
            )
        logger.info("Running post-rollout aggregation...")
        # Determine array job directory: CLI arg > log_dir
        array_job_dir = args.array_job_dir or args.log_dir
        aggregation_success = run_aggregation_from_runtime(
            log_dir=args.log_dir,
            eval_config=eval_config,
            array_job_dir=array_job_dir,
            failed_rollouts=failed_rollouts,
        )
        if not aggregation_success:
            logger.warning("Aggregation completed with errors")
        success = aggregation_success
    else:
        logger.info("Evaluation disabled, skipping aggregation")

    return success


async def _run_one_shot_request(
    args: argparse.Namespace,
    request: runtime_pb2.SimulationRequest,
) -> runtime_pb2.SimulationReturn:
    """Run a single simulation request using a temporary DaemonEngine.

    Creates the engine, starts it, runs the request, then shuts down the
    engine. Deployment-managed services are cleaned up by the deployment layer.
    Used for CLI (non-daemon) mode.
    """
    engine = DaemonEngine(
        user_config=args.user_config,
        network_config=args.network_config,
        eval_config=args.eval_config,
        log_dir=args.log_dir,
    )

    try:
        await engine.startup()
        return await engine.simulate(request)
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    if args.serve:
        asyncio.run(_serve(args))
    else:
        validate_array_job_config(args.array_job_dir)

        success = asyncio.run(run_simulation(args))
        logging.info("Alpasim finished.")

        sys.exit(0 if success else 1)
