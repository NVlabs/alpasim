# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricFilter:
    """Single metric filter configuration. Defines a _good_ trajectory."""

    column: str
    operator: str  # one of: >, <, >=, <=, ==, !=
    value: float


@dataclass
class TrajectoryFilterConfig:
    """Configuration for filtering metrics."""

    remove_trajectories_with_event: List[MetricFilter] = field(default_factory=list)
    remove_timesteps_after_event: List[MetricFilter] = field(default_factory=list)
