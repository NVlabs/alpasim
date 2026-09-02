# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Reproduce the challenge Drive-IRT aggregation from local run summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).with_name("data")
VALID_TRACKS = ("pai", "nuplan")


@dataclass(frozen=True)
class RunInput:
    subject_id: str
    summary_path: Path
    is_reference: bool = False


@dataclass(frozen=True)
class RunScores:
    subject_id: str
    summary_path: Path
    scene_scores: dict[str, float]
    avg_dist_between_incidents_at_fault: float | None
    is_reference: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=VALID_TRACKS)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="MODEL_ID=PATH",
        help="Completed run directory, aggregate directory, or results-summary.json. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="Override data/<track>/reference_manifest.json.",
    )
    parser.add_argument(
        "--without-references",
        action="store_true",
        help="Do not add published reference runs (experimental only).",
    )
    parser.add_argument("--algorithm", choices=("zoib", "average"), default="zoib")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-decay", type=float, default=0.975)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-particles", type=int, default=16)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--average-iterate-tail-percentage", type=float, default=0.025)
    parser.add_argument("--rank-interval-confidence-level", type=float, default=0.95)
    parser.add_argument("--rank-interval-monte-carlo-samples", type=int, default=100000)
    parser.add_argument("--rank-interval-seed", type=int, default=0)
    return parser.parse_args()


def resolve_summary_path(path: Path) -> Path:
    if path.name == "results-summary.json":
        summary_path = path
    elif path.name == "aggregate":
        summary_path = path / "results-summary.json"
    else:
        summary_path = path / "aggregate" / "results-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Could not find aggregate/results-summary.json under {path}"
        )
    return summary_path.resolve()


def parse_run_spec(value: str, *, is_reference: bool = False) -> RunInput:
    subject_id, separator, raw_path = value.partition("=")
    if not separator or not subject_id or not raw_path:
        raise ValueError("--run must have the form MODEL_ID=PATH")
    return RunInput(subject_id, resolve_summary_path(Path(raw_path)), is_reference)


def _safe_reference_path(manifest_path: Path, raw_path: str) -> Path:
    root = manifest_path.parent.resolve()
    candidate = (root / raw_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Reference path escapes data directory: {raw_path}")
    return candidate


def load_reference_inputs(
    track: str, manifest_path: Path | None
) -> tuple[list[RunInput], dict[str, Any] | None]:
    manifest_path = manifest_path or DATA_DIR / track / "reference_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Reference data for {track!r} is not installed at {manifest_path}. "
            "Use --without-references for an experimental run, or install the published data bundle."
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("track") != track:
        raise ValueError(f"{manifest_path} is not a {track!r} reference manifest")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"{manifest_path} must contain a non-empty runs list")
    inputs: list[RunInput] = []
    for raw_run in raw_runs:
        if not isinstance(raw_run, dict):
            raise TypeError(f"{manifest_path} contains a non-object reference run")
        subject_id, summary_path = (
            raw_run.get("subject_id"),
            raw_run.get("summary_path"),
        )
        if (
            not isinstance(subject_id, str)
            or not subject_id
            or not isinstance(summary_path, str)
            or not summary_path
        ):
            raise ValueError(
                f"{manifest_path} reference runs require subject_id and summary_path"
            )
        inputs.append(
            RunInput(
                subject_id,
                resolve_summary_path(_safe_reference_path(manifest_path, summary_path)),
                True,
            )
        )
    return inputs, manifest


def _score_value(rollout: dict[str, Any], summary_path: Path) -> float | None:
    value = rollout.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{summary_path}: rollout score is outside [0, 1]")
    return score


def load_run_scores(run_input: RunInput) -> RunScores:
    with run_input.summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("scene_score_enabled") is False:
        raise ValueError(f"{run_input.summary_path} has scene scoring disabled")
    scores_by_scene: dict[str, list[float]] = defaultdict(list)
    for rollout in summary.get("rollouts") or []:
        if not isinstance(rollout, dict):
            continue
        scene_id = rollout.get("clipgt_id") or rollout.get("scene_id")
        if not scene_id:
            continue
        score = _score_value(rollout, run_input.summary_path)
        if score is not None:
            # A zero is valid (for example, an at-fault collision).
            scores_by_scene[str(scene_id)].append(score)
    scene_scores = {
        scene_id: float(sum(scores) / len(scores))
        for scene_id, scores in scores_by_scene.items()
        if scores
    }
    if not scene_scores:
        raise ValueError(f"{run_input.summary_path} contains no valid scene scores")
    metrics_results = summary.get("metrics_results") or []
    metrics = (
        metrics_results[0]
        if metrics_results and isinstance(metrics_results[0], dict)
        else {}
    )
    try:
        distance = float(metrics["avg_dist_between_incidents_at_fault"])
        distance = distance if np.isfinite(distance) else None
    except (KeyError, TypeError, ValueError):
        distance = None
    return RunScores(
        run_input.subject_id,
        run_input.summary_path,
        scene_scores,
        distance,
        run_input.is_reference,
    )


def build_matrix(
    runs: list[RunScores],
) -> tuple[np.ndarray, list[str], list[str], list[dict[str, Any]]]:
    sets = [tuple(sorted(run.scene_scores)) for run in runs]
    set_counts = Counter(sets)
    canonical = max(set_counts, key=lambda value: (len(value), set_counts[value]))
    subject_ids: list[str] = []
    matrix_rows: list[list[float]] = []
    exclusions: list[dict[str, Any]] = []
    for run in runs:
        item_ids = tuple(sorted(run.scene_scores))
        if item_ids != canonical:
            exclusions.append(
                {
                    "subject_id": run.subject_id,
                    "summary_path": str(run.summary_path),
                    "reason": "scenario_set_mismatch",
                    "scenario_count": len(item_ids),
                    "expected_scenario_count": len(canonical),
                }
            )
            continue
        subject_ids.append(run.subject_id)
        matrix_rows.append([run.scene_scores[item_id] for item_id in canonical])
    if not matrix_rows:
        raise ValueError("No runs share a common scored scenario set")
    matrix = np.asarray(matrix_rows, dtype=np.float32)
    if not np.isfinite(matrix).all() or ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError("Scene score matrix contains values outside [0, 1]")
    return matrix, subject_ids, list(canonical), exclusions


def effective_algorithm(
    requested: str, matrix: np.ndarray, subjects: list[str], items: list[str]
) -> tuple[str, dict[str, Any] | None]:
    if requested != "zoib":
        return requested, None
    observations, minimum = int(matrix.size), len(subjects) + 5 * len(items)
    if observations >= minimum:
        return requested, None
    return "average", {
        "reason": "not_enough_observations_for_requested_algorithm",
        "requested_algorithm": requested,
        "fallback_algorithm": "average",
        "included_submission_count": len(subjects),
        "scenario_count": len(items),
        "observation_count": observations,
        "minimum_observation_count": minimum,
    }


def fit_algorithm(
    args: argparse.Namespace,
    algorithm: str,
    matrix: np.ndarray,
    subjects: list[str],
    items: list[str],
) -> Any:
    try:
        from drive_irt.irt import Average, BetaIrt
    except ImportError as error:
        raise RuntimeError(
            "drive-irt is not installed; run with `uv run --extra local-evaluation ...`"
        ) from error
    if algorithm == "average":
        return Average(seed=args.seed).fit(matrix, subjects, items)
    return BetaIrt(
        epochs=args.epochs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        num_particles=args.num_particles,
        clip_norm=args.clip_norm,
        average_iterate_tail_percentage=args.average_iterate_tail_percentage,
    ).fit(matrix, subjects, items)


def rank_interval_settings(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.rank_interval_confidence_level < 1.0:
        raise ValueError("--rank-interval-confidence-level must be in (0, 1)")
    if args.rank_interval_monte_carlo_samples < 1:
        raise ValueError("--rank-interval-monte-carlo-samples must be positive")
    tail = (1.0 - args.rank_interval_confidence_level) / 2.0
    return {
        "applied": False,
        "kind": "posterior_rank_interval",
        "nominal_confidence_level": args.rank_interval_confidence_level,
        "rank_lo_quantile": tail,
        "rank_hi_quantile": 1.0 - tail,
        "monte_carlo_samples": args.rank_interval_monte_carlo_samples,
        "monte_carlo_seed": args.rank_interval_seed,
        "uncertainty_source": "drive_irt_posterior_score_std",
    }


def rank_intervals(
    result: Any, settings: dict[str, Any]
) -> dict[str, tuple[float, float]]:
    score_std = getattr(result, "score_uncertainty", None)
    if score_std is None:
        settings["reason"] = "algorithm_does_not_provide_posterior_score_uncertainty"
        return {}
    scores, stds = (
        np.asarray(result.scores, dtype=float),
        np.asarray(score_std, dtype=float),
    )
    if (
        len(scores) != len(stds)
        or len(scores) != len(result.subject_ids)
        or not np.isfinite(scores).all()
        or not np.isfinite(stds).all()
    ):
        raise ValueError("drive-irt result contains invalid score uncertainty")
    from drive_irt.ensemble.help import rank_stats_monte_carlo

    rank_lo, rank_hi = rank_stats_monte_carlo(
        mean_value=scores,
        std=stds,
        n_samples=int(settings["monte_carlo_samples"]),
        rank_lo_quantile=float(settings["rank_lo_quantile"]),
        rank_hi_quantile=float(settings["rank_hi_quantile"]),
        seed=int(settings["monte_carlo_seed"]),
    )
    settings["applied"] = True
    return {
        str(subject_id): (float(lo), float(hi))
        for subject_id, lo, hi in zip(result.subject_ids, rank_lo, rank_hi)
    }


def score_scale(
    result: Any, reference_manifest: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the same positive affine score scale used by leaderboard anchors."""
    base: dict[str, Any] = {"type": "anchor_affine", "applied": False}
    config = (reference_manifest or {}).get("score_scale")
    if not isinstance(config, dict):
        return base | {"reason": "anchor_scale_not_configured"}
    low_id, high_id = config.get("low_subject_id"), config.get("high_subject_id")
    low_target, high_target = (
        config.get("low_target_score"),
        config.get("high_target_score"),
    )
    if not isinstance(low_id, str) or not isinstance(high_id, str):
        return base | {"reason": "anchor_subject_not_configured"}
    try:
        low_target, high_target = float(low_target), float(high_target)
    except (TypeError, ValueError):
        return base | {"reason": "anchor_target_not_configured"}
    scores = {
        str(subject): float(score)
        for subject, score in zip(result.subject_ids, result.scores)
    }
    low_raw, high_raw = scores.get(low_id), scores.get(high_id)
    if low_raw is None or high_raw is None:
        return base | {
            "reason": "anchor_subject_not_included",
            "missing_anchors": [
                subject
                for subject, score in ((low_id, low_raw), (high_id, high_raw))
                if score is None
            ],
        }
    raw_delta, target_delta = high_raw - low_raw, high_target - low_target
    if raw_delta <= 0.0 or target_delta <= 0.0:
        return base | {"reason": "anchor_scores_not_monotone"}
    scale = target_delta / raw_delta
    return base | {
        "applied": True,
        "low_subject_id": low_id,
        "high_subject_id": high_id,
        "low_target_score": low_target,
        "high_target_score": high_target,
        "low_raw_score": low_raw,
        "high_raw_score": high_raw,
        "scale": scale,
        "offset": low_target - scale * low_raw,
    }


def _apply_score_scale(value: float, scale: dict[str, Any]) -> float:
    if not scale.get("applied"):
        return value
    return float(scale["offset"]) + float(scale["scale"]) * value


def _apply_score_std_scale(value: float, scale: dict[str, Any]) -> float:
    return abs(float(scale["scale"])) * value if scale.get("applied") else value


def ranking_rows(
    result: Any,
    runs: list[RunScores],
    rank_interval: dict[str, Any],
    scale: dict[str, Any],
) -> list[dict[str, Any]]:
    by_subject = {run.subject_id: run for run in runs}
    score_std, intervals = (
        getattr(result, "score_uncertainty", None),
        rank_intervals(result, rank_interval),
    )
    rows: list[dict[str, Any]] = []
    for point_rank, (subject_id, score) in enumerate(
        zip(result.subject_ids, result.scores), start=1
    ):
        subject_id = str(subject_id)
        run = by_subject[subject_id]
        raw_score = float(score)
        row: dict[str, Any] = {
            "point_estimate_rank": point_rank,
            "subject_id": subject_id,
            "is_reference": run.is_reference,
            "policy_capability_score": _apply_score_scale(raw_score, scale),
            "raw_policy_capability_score": raw_score,
            "average_scene_score": float(np.mean(list(run.scene_scores.values()))),
            "avg_dist_between_incidents_at_fault": run.avg_dist_between_incidents_at_fault,
        }
        if score_std is not None:
            raw_std = float(score_std[point_rank - 1])
            row["policy_capability_score_std"] = _apply_score_std_scale(raw_std, scale)
            row["raw_policy_capability_score_std"] = raw_std
        if subject_id in intervals:
            row["rank_lo"], row["rank_hi"] = intervals[subject_id]
            row["rank_spread"] = float(row["rank_hi"] - row["rank_lo"])
        rows.append(row)
    if intervals:
        rows.sort(
            key=lambda row: (
                float(row["rank_hi"]),
                -_incident_distance_sort_value(
                    row["avg_dist_between_incidents_at_fault"]
                ),
                int(row["point_estimate_rank"]),
                str(row["subject_id"]),
            )
        )
    else:
        rows.sort(
            key=lambda row: (int(row["point_estimate_rank"]), str(row["subject_id"]))
        )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _incident_distance_sort_value(value: float | None) -> float:
    return float("-inf") if value is None else value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    try:
        contestant_inputs = [parse_run_spec(value) for value in args.run]
        reference_inputs: list[RunInput] = []
        reference_manifest: dict[str, Any] | None = None
        if not args.without_references:
            reference_inputs, reference_manifest = load_reference_inputs(
                args.track, args.reference_manifest
            )
        inputs = [*reference_inputs, *contestant_inputs]
        if len({run.subject_id for run in inputs}) != len(inputs):
            raise ValueError("Every reference and --run subject ID must be unique")
        runs = [load_run_scores(run) for run in inputs]
        matrix, subjects, items, exclusions = build_matrix(runs)
        algorithm, fallback_warning = effective_algorithm(
            args.algorithm, matrix, subjects, items
        )
        result = fit_algorithm(args, algorithm, matrix, subjects, items)
        rank_interval = rank_interval_settings(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "scene_score_matrix.csv",
        [
            dict(subject_id=subject, **dict(zip(items, map(float, row))))
            for subject, row in zip(subjects, matrix)
        ],
        ["subject_id", *items],
    )
    scale = score_scale(result, reference_manifest)
    rows = ranking_rows(result, runs, rank_interval, scale)
    write_csv(
        args.output_dir / "capability_ranking.csv",
        rows,
        [
            "rank",
            "point_estimate_rank",
            "rank_lo",
            "rank_hi",
            "rank_spread",
            "subject_id",
            "is_reference",
            "policy_capability_score",
            "raw_policy_capability_score",
            "policy_capability_score_std",
            "raw_policy_capability_score_std",
            "average_scene_score",
            "avg_dist_between_incidents_at_fault",
        ],
    )
    result.save(args.output_dir / "fit")
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "track": args.track,
        "requested_algorithm": args.algorithm,
        "effective_algorithm": algorithm,
        "score_scale": scale,
        "rank_interval": rank_interval,
        "reference_data_metadata": reference_manifest,
        "included_subject_ids": subjects,
        "reference_subject_ids": [
            run.subject_id
            for run in runs
            if run.is_reference and run.subject_id in subjects
        ],
        "scenario_count": len(items),
        "exclusions": exclusions,
        "warnings": [fallback_warning] if fallback_warning else [],
        "ranking_policy": {
            "primary": "rank_interval_upper_bound_ascending",
            "tiebreakers": [
                "avg_dist_between_incidents_at_fault_descending",
                "point_estimate_rank_ascending",
                "subject_id_ascending",
            ],
            "fallback_without_rank_interval": "point_estimate_rank_ascending",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote local {args.track} capability analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
