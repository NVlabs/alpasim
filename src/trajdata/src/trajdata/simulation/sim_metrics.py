# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Dict

import numpy as np
import pandas as pd


class SimMetric:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        raise NotImplementedError()


class ADE(SimMetric):
    def __init__(self) -> None:
        super().__init__("ade")

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        err_df = pd.DataFrame(index=gt_df.index, columns=["error"])
        err_df["error"] = np.linalg.norm(gt_df[["x", "y"]] - sim_df[["x", "y"]], axis=1)
        return err_df.groupby("agent_id")["error"].mean().to_dict()


class FDE(SimMetric):
    def __init__(self) -> None:
        super().__init__("fde")

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        err_df = pd.DataFrame(index=gt_df.index, columns=["error"])
        err_df["error"] = np.linalg.norm(gt_df[["x", "y"]] - sim_df[["x", "y"]], axis=1)
        return err_df.groupby("agent_id")["error"].last().to_dict()
