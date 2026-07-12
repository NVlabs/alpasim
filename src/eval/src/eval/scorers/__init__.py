# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from eval.schema import EvalConfig
from eval.scorers.base import ScorerGroup
from eval.scorers.collision import CollisionScorer
from eval.scorers.ground_truth import GroundTruthScorer
from eval.scorers.image import ImageScorer
from eval.scorers.min_distance_to_obstacle import MinDistanceToObstacleScorer
from eval.scorers.minADE import MinADEScorer
from eval.scorers.offroad import OffRoadScorer
from eval.scorers.open_loop_collision import OpenLoopCollisionScorer
from eval.scorers.plan_deviation import PlanDeviationScorer
from eval.scorers.safety import SafetyScorer

SCORERS = [
    CollisionScorer,
    OffRoadScorer,
    MinDistanceToObstacleScorer,
    OpenLoopCollisionScorer,
    GroundTruthScorer,
    MinADEScorer,
    PlanDeviationScorer,
    ImageScorer,
    SafetyScorer,
]


def create_scorer_group(cfg: EvalConfig) -> ScorerGroup:
    """Initialize all scorers."""
    scorers = []
    for scorer in SCORERS:
        scorers.append(scorer(cfg))
    return ScorerGroup(scorers)
