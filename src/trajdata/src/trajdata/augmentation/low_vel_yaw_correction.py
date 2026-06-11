# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import numpy as np
import pandas as pd

from trajdata.augmentation.augmentation import DatasetAugmentation


class LowSpeedYawCorrection(DatasetAugmentation):
    def __init__(self, speed_threshold: float = 0.0) -> None:
        self.speed_threshold = speed_threshold

    def apply(self, scene_data_df: pd.DataFrame) -> None:
        speed = np.linalg.norm(scene_data_df[["vx", "vy"]], axis=1)

        scene_data_df["yaw_diffs"] = scene_data_df["heading"].diff()
        # Doing this because the first row is always nan
        scene_data_df["yaw_diffs"].iat[0] = 0.0

        agent_ids: pd.Series = scene_data_df.index.get_level_values(0).astype(
            "category"
        )

        # The point of the border mask is to catch data like this:
        # index    agent_id     vx    vy
        #     0           1    7.3   9.1
        #     1           2    0.0   0.0
        #                  ...
        # As implemented, we would currently only
        # return index 0 (since we chop off the top with the 1: in the slice below), but
        # we want to return 1 so that's why the + 1 at the end.
        border_mask: np.ndarray = np.concatenate(
            [[0], np.nonzero(agent_ids[1:] != agent_ids[:-1])[0] + 1]
        )

        scene_data_df["yaw_diffs"].iloc[border_mask] = 0.0
        scene_data_df["yaw_diffs"].iloc[speed < self.speed_threshold] = 0.0

        mask_arr = np.ones((len(scene_data_df),), dtype=np.bool)
        mask_arr[border_mask] = False
        scene_data_df["heading"].iloc[mask_arr] = 0.0

        scene_data_df.loc[:, "yaw_diffs"] += scene_data_df["heading"]
        scene_data_df.loc[:, "heading"] = scene_data_df.groupby("agent_id")[
            "yaw_diffs"
        ].cumsum()

        scene_data_df.drop(columns="yaw_diffs", inplace=True)
