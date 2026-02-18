# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np

from eval.data import AggregationType, MetricReturn, SimulationResult
from eval.schema import EvalConfig, ImageScorerConfig
from eval.scorers.base import Scorer

BLACK_THRESHOLD = 10


class ImageScorer(Scorer):
    """Scorer for image quality metrics.

    Adds the following metrics:
    * img_is_black: Whether the image is black.
    * More to come...

    Requires camera data to be available in the simulation result.
    """

    def __init__(self, cfg: EvalConfig):
        super().__init__(cfg)
        scorer_config: ImageScorerConfig = cfg.scorers.image
        self.camera_logical_id = scorer_config.camera_logical_id

    def calculate(self, simulation_result: SimulationResult) -> list[MetricReturn]:
        # Fail fast if camera data is missing
        if simulation_result.cameras is None:
            raise ValueError("ImageScorer requires camera data but cameras is None")

        if not simulation_result.cameras.camera_by_logical_id:
            raise ValueError(
                "ImageScorer requires camera data but no cameras available"
            )

        if self.camera_logical_id not in simulation_result.cameras.camera_by_logical_id:
            raise ValueError(
                f"ImageScorer requires camera '{self.camera_logical_id}' but available "
                f"cameras are: {list(simulation_result.cameras.camera_by_logical_id.keys())}"
            )

        camera = simulation_result.cameras.camera_by_logical_id[self.camera_logical_id]

        all_black_values = []
        timestamps_us = []

        for time in simulation_result.timestamps_us:
            img = camera.image_at_time(time)
            if img is None:
                continue
            all_black_values.append(float(np.all(np.array(img) <= BLACK_THRESHOLD)))
            timestamps_us.append(time)

        return [
            MetricReturn(
                name="img_is_black",
                values=all_black_values,
                valid=[True] * len(all_black_values),
                timestamps_us=timestamps_us,
                time_aggregation=AggregationType.MAX,
            ),
        ]
