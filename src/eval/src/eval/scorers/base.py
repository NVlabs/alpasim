# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging
from abc import ABC, abstractmethod

from eval.data import MetricReturn, SimulationResult
from eval.schema import EvalConfig

logger = logging.getLogger("alpasim_eval")


class Scorer(ABC):
    """Abstract base class for scoring metrics."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        """Calculate metrics for entire trajectory."""
        pass


class ScorerGroup(Scorer):
    """Group of scorers."""

    def __init__(self, scorers: list[Scorer]) -> None:
        self.scorers = scorers

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        """Calculate metrics for entire trajectory."""
        results = []
        for scorer in self.scorers:
            try:
                results.extend(scorer.calculate(simulation_result))
            except Exception as e:
                # We're not raising an error here because we want to continue
                # scoring other metrics even if one metric fails.
                logger.error(
                    "Error calculating metrics for %s: %s", scorer.__class__.__name__, e
                )
        return results
