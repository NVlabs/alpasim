# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json

import numpy as np
import polars as pl
import pytest
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_utils.geometry import Pose, Trajectory
from conftest import SimpleScenarioEvaluator
from shapely.geometry import LineString, Point

from eval.aggregation.modifiers import RemoveTimestepsAfterEvent
from eval.aggregation.processing import aggregate_and_write_metrics_results_txt
from eval.data import AggregationType, MetricReturn, ScenarioEvalInput
from eval.schema import EvalConfig
from eval.scorers.ground_truth import _end_direction, _lateral_distance_to_gt


def _lat(x: float, y: float) -> float:
    gt = LineString([(0, 0), (50, 0)])
    return _lateral_distance_to_gt(gt, Point(x, y), _end_direction(gt))


def test_forward_overshoot_is_not_lateral() -> None:
    # `LineString.distance` clamps to the endpoint, so these read as 6 m and
    # 20 m away; laterally the ego never left the corridor.
    assert _lat(56.0, 0.0) == 0.0
    assert _lat(70.0, 0.0) == 0.0


def test_sideways_excursion_is_lateral() -> None:
    assert _lat(25.0, 4.5) == 4.5


def test_overshoot_and_sideways_reports_only_the_sideways_part() -> None:
    assert _lat(56.0, 5.0) == 5.0


def test_degenerate_gt_falls_back_to_plain_distance() -> None:
    degenerate = LineString([(0, 0), (0, 0)])
    assert (
        _lateral_distance_to_gt(degenerate, Point(3.0, 4.0), _end_direction(degenerate))
        == 5.0
    )


@pytest.mark.parametrize(
    "coords, point, expected",
    [
        # A stopped endpoint jitters backwards, putting the nearest projection
        # before the endpoint and making the final segment point backwards.
        ([(0, 0), (50, 0), (49.999, 0)], (55, 0.5), 0.5),
        ([(0, 0), (50, 0), (50, 0)], (55, 0.5), 0.5),
        # A curved path must use its final motion rather than its full chord.
        ([(0, 0), (50, 0), (50, 10)], (50.5, 15), 0.5),
        ([(0, 0), (1, 0), (1, 0.5), (1, 1), (1, 1.5)], (1.5, 6.5), 0.5),
        # Reverse travel follows motion, irrespective of vehicle orientation.
        ([(50, 0), (0, 0), (0.001, 0)], (-5, 0.5), 0.5),
        # Loopbacks keep their final direction and their recorded segments.
        ([(0, 0), (50, 0), (50, 10), (0, 10)], (-5, 10.5), 0.5),
        ([(0, 0), (50, 0), (50, 10), (0, 10)], (25, 0.5), 0.5),
        # Behind the endpoint, the forward ray must not become an infinite line.
        ([(0, 0), (50, 0), (50, 10)], (50.5, -5), np.hypot(0.5, 5.0)),
        ([(0, 0), (50, 0), (49.999, 0)], (55, 5), 5.0),
        # Stationary position noise provides no supported travel direction.
        ([(0, 0), (0.001, 0), (0, 0)], (3, 4), np.hypot(3.0 - 0.001, 4.0)),
        ([(0, 0), (0.5, 0)], (3.5, 4), 5.0),
    ],
)
def test_endpoint_corridor_geometry(coords, point, expected) -> None:
    gt = LineString(coords)
    assert _lateral_distance_to_gt(
        gt, Point(point), _end_direction(gt)
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "motion_sign, sideways_m, event_name, event_idx, failure_reason",
    [
        (1, 0.5, "offroad", 5, None),
        (-1, 0.5, "offroad", 5, None),
        (1, 0.5, "collision_front", 5, None),
        (1, 0.5, "offroad", 4, "offroad"),
        (1, 0.5, "collision_front", 4, "collision_at_fault"),
        (1, 5.0, "offroad", 5, "left_corridor_laterally"),
    ],
)
def test_stopped_endpoint_scoring_keeps_inclusive_support_cutoff(
    motion_sign,
    sideways_m,
    event_name,
    event_idx,
    failure_reason,
    default_eval_config: EvalConfig,
    tmp_path,
    monkeypatch,
) -> None:
    cfg = default_eval_config
    cfg.aggregation_modifiers.max_dist_to_gt_trajectory = 4.0
    timestamps = np.arange(6, dtype=np.uint64) * 100_000

    def trajectory(coords):
        return Trajectory.from_poses(
            timestamps=timestamps,
            poses=[
                Pose(
                    position=np.array([motion_sign * x, y, 0.0], dtype=np.float32),
                    quaternion=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                )
                for x, y in coords
            ],
        )

    ego = trajectory([(0, 0), (25, 0), (49, 0.5), (53, 0.5), (55, sideways_m), (70, 5)])
    gt = trajectory([(0, 0), (25, 0), (50, 0), (49.999, 0), (49.998, 0), (49.997, 0)])
    scene_input = ScenarioEvalInput(
        session_metadata=RolloutMetadata.SessionMetadata(
            session_uuid="stopped-endpoint",
            scene={"source": {"clip": {"clip_id": "test", "usdz_uuid": "test"}}},
        ),
        run_uuid="test-run",
        run_name="test-run",
        ego_coords_rig_to_aabb_center=Pose.identity(),
        ego_aabb_x_m=4.5,
        ego_aabb_y_m=2.0,
        ego_aabb_z_m=1.5,
        actor_trajectories={"EGO": (ego, (4.5, 2.0, 1.5))},
        ego_recorded_ground_truth_trajectory=gt,
        vec_map=None,
        recorded_metrics=[
            MetricReturn(
                name=name,
                timestamps_us=timestamps.tolist(),
                values=[float(name == event_name and i == event_idx) for i in range(6)],
                valid=[True] * 6,
                time_aggregation=AggregationType.MAX,
            )
            for name in ["offroad", "img_is_black"]
        ],
    )
    result = SimpleScenarioEvaluator(cfg).evaluate(scene_input)
    metrics = result.metrics_df
    if event_name == "collision_front":
        metrics = metrics.with_columns(
            pl.when(
                pl.col("name").is_in(["collision_front", "collision_any"])
                & (pl.col("timestamps_us") == timestamps[event_idx])
            )
            .then(1.0)
            .otherwise(pl.col("values"))
            .alias("values")
        )
    monkeypatch.setattr(
        "eval.aggregation.processing.plot_metrics_results", lambda *a: None
    )
    processed = aggregate_and_write_metrics_results_txt(
        metrics,
        output_path=str(tmp_path),
        additional_modifiers=[
            RemoveTimestepsAfterEvent(pl.col("dist_to_gt_trajectory") >= 4.0)
        ],
    )
    # The first out-of-support sample survives, and the next sample is removed.
    assert (
        processed.df_wide_modified["timestamps_us"].to_list() == timestamps[:5].tolist()
    )
    rollout = json.loads((tmp_path / "results-summary.json").read_text())["rollouts"][0]
    assert rollout["failure_reason"] == failure_reason
    assert rollout["score"] == (1.0 if failure_reason is None else 0.0)
    assert rollout["score_metrics"]["progress_clipped_rel"] > 0.99
    assert rollout["metrics"]["gt_dist_traveled_m"] == pytest.approx(
        LineString(gt.positions[:, :2]).length
    )
    assert rollout["metrics"]["dist_traveled_m"] == pytest.approx(
        LineString(ego.positions[:5, :2]).length
    )
