# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from pathlib import Path
from typing import Sequence

import polars as pl
from omegaconf import OmegaConf

from eval.aggregation.modifiers import (
    MetricAggregationModifiers,
    RemoveTimestepsAfterEvent,
    RemoveTrajectoryWithEvent,
)
from eval.filtering.schema import TrajectoryFilterConfig

OPERATOR_MAP = {
    ">": lambda x, y: x > y,
    "<": lambda x, y: x < y,
    ">=": lambda x, y: x >= y,
    "<=": lambda x, y: x <= y,
    "==": lambda x, y: x == y,
    "!=": lambda x, y: x != y,
}


def read_filter_config(filter_path: Path | str) -> TrajectoryFilterConfig:

    filter_config = OmegaConf.merge(
        OmegaConf.structured(TrajectoryFilterConfig), OmegaConf.load(filter_path)
    )
    return filter_config


def get_modifiers_from_filter_config(
    filter_config: TrajectoryFilterConfig,
) -> Sequence[MetricAggregationModifiers]:
    return [
        RemoveTrajectoryWithEvent(
            OPERATOR_MAP[filter.operator](pl.col(filter.column), filter.value)
        )
        for filter in filter_config.remove_trajectories_with_event
    ] + [
        RemoveTimestepsAfterEvent(
            OPERATOR_MAP[filter.operator](pl.col(filter.column), filter.value)
        )
        for filter in filter_config.remove_timesteps_after_event
    ]
