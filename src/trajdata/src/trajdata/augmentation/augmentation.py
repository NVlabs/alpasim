# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pandas as pd

from trajdata.data_structures.batch import AgentBatch, SceneBatch


class Augmentation:
    def __init__(self) -> None:
        raise NotImplementedError()


class DatasetAugmentation(Augmentation):
    def apply(self, scene_data_df: pd.DataFrame) -> None:
        raise NotImplementedError()


class BatchAugmentation(Augmentation):
    def apply_agent(self, agent_batch: AgentBatch) -> None:
        raise NotImplementedError()

    def apply_scene(self, scene_batch: SceneBatch) -> None:
        raise NotImplementedError()
