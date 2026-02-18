# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Shared test fixtures and helpers for eval tests."""

import pytest
from omegaconf import OmegaConf

from eval.schema import EvalConfig
from eval.scorers.base import ScorerGroup


def create_test_eval_config(
    vehicle_shrink_factor: float = 0.0,
    vehicle_corner_roundness: float = 0.0,
) -> EvalConfig:
    """Create an EvalConfig with test defaults."""
    config_dict = {
        "num_processes": 1,
        "vehicle": {
            "vehicle_shrink_factor": vehicle_shrink_factor,
            "vehicle_corner_roundness": vehicle_corner_roundness,
        },
        "video": {
            "render_video": False,
            "video_layouts": ["DEFAULT"],
            "camera_id_to_render": "",
            "reasoning_text_refresh_interval_s": 1.0,
            "map_video": {
                "map_radius_m": 50.0,
                "ego_loc": "CENTER",
                "rotate_map_to_ego": False,
            },
            "render_every_nth_frame": 1,
            "generate_combined_video": False,
            "combined_video_speed_factor": 1.0,
        },
        "scorers": {
            "min_ade": {
                "time_deltas": [1.0, 2.0, 3.0],
                "incl_z": False,
                "target": "SELF",
            },
            "plan_deviation": {
                "incl_z": False,
                "avg_decay_rate": 0.5,
                "min_timesteps": 5,
            },
            "image": {
                "camera_logical_id": "camera_front_wide_120fov",
            },
        },
        "database": {
            "upload_metadata": False,
            "upload_leaderboard": False,
            "upload_full_metrics": False,
        },
        "aggregation_modifiers": {
            "max_dist_to_gt_trajectory": float("inf"),
        },
        "vec_map": {},
    }
    return OmegaConf.merge(OmegaConf.structured(EvalConfig), config_dict)


def create_test_scorer_group(cfg: EvalConfig) -> ScorerGroup:
    """Create a scorer group for testing without scorers requiring complex fixtures.

    Excluded scorers and reasons:
    - OffRoadScorer: Requires VectorMap, which is complex to construct for unit tests
      (needs proper road geometry with RoadLane objects, center lines, edges, and
      spatial indices from the trajdata library).
    - ImageScorer: Requires camera data.
    - MinADEScorer: Requires driver responses with sampled trajectories.
    - PlanDeviationScorer: Requires driver responses with sampled trajectories.
    - SafetyScorer: Requires driver responses.

    For tests that need these metrics, use real scenario data with actual fixtures.
    """
    from eval.scorers.collision import CollisionScorer
    from eval.scorers.ground_truth import GroundTruthScorer

    scorers = [
        CollisionScorer(cfg),
        GroundTruthScorer(cfg),
    ]
    return ScorerGroup(scorers)


@pytest.fixture
def default_eval_config() -> EvalConfig:
    """Create a default EvalConfig for testing."""
    return create_test_eval_config()


class SimpleScenarioEvaluator:
    """ScenarioEvaluator for tests that excludes scorers requiring complex fixtures.

    This is a thin wrapper around ScenarioEvaluator that uses a filtered set of
    scorers appropriate for unit tests without VectorMap, camera, or driver response
    fixtures.
    """

    def __init__(self, cfg: EvalConfig) -> None:
        from eval.scenario_evaluator import ScenarioEvaluator

        self._evaluator = ScenarioEvaluator(cfg)
        # Replace the scorer group with one that excludes complex-fixture scorers
        self._evaluator._scorer_group = create_test_scorer_group(cfg)

    @property
    def cfg(self) -> EvalConfig:
        return self._evaluator.cfg

    def evaluate(self, scenario_input):  # noqa: ANN001
        return self._evaluator.evaluate(scenario_input)
