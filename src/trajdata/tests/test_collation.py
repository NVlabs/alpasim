# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import unittest
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from trajdata.data_structures.agent import AgentType
from trajdata.dataset import UnifiedDataset
from trajdata.utils import arr_utils


class TestCollation(unittest.TestCase):
    def test_convert_with_dir(self):
        input = np.array([1, 2, 3])
        converted = arr_utils.convert_with_dir(
            input, torch.long, -1, arr_utils.PadDirection.AFTER
        )

        self.assertTrue(torch.allclose(torch.tensor(input), converted))

        input = np.array([1, 2, 3])
        converted = arr_utils.convert_with_dir(
            input, torch.long, -1, arr_utils.PadDirection.BEFORE
        )

        self.assertTrue(torch.allclose(torch.tensor(np.array([3, 2, 1])), converted))

    def test_pad_with_dir(self):
        inputs = [
            torch.tensor([[6], [5], [4], [3], [2], [1], [0]], dtype=torch.float),
            torch.tensor([[3], [2], [1], [0]], dtype=torch.float),
            torch.tensor([[9], [10], [11]], dtype=torch.float),
            torch.tensor([[9], [10], [11], [12], [13], [14]], dtype=torch.float),
        ]
        converted = arr_utils.pad_with_dir(
            inputs,
            -2,
            arr_utils.PadDirection.AFTER,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[6], [5], [4], [3], [2], [1], [0]],
                        [[3], [2], [1], [0], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [np.nan], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [12], [13], [14], [np.nan]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )

        converted = arr_utils.pad_with_dir(
            inputs,
            -2,
            arr_utils.PadDirection.BEFORE,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[0], [1], [2], [3], [4], [5], [6]],
                        [[np.nan], [np.nan], [np.nan], [0], [1], [2], [3]],
                        [[np.nan], [np.nan], [np.nan], [np.nan], [11], [10], [9]],
                        [[np.nan], [14], [13], [12], [11], [10], [9]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )

    def test_pad_sequences(self):
        inputs = [
            np.array([[6], [5], [4], [3], [2], [1], [0]], dtype=float),
            np.array([[3], [2], [1], [0]], dtype=float),
            np.array([[9], [10], [11]], dtype=float),
            np.array([[9], [10], [11], [12], [13], [14]], dtype=float),
        ]
        converted = arr_utils.pad_sequences(
            inputs,
            torch.float,
            -2,
            arr_utils.PadDirection.AFTER,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[6], [5], [4], [3], [2], [1], [0]],
                        [[3], [2], [1], [0], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [np.nan], [np.nan], [np.nan], [np.nan]],
                        [[9], [10], [11], [12], [13], [14], [np.nan]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )

        converted = arr_utils.pad_sequences(
            inputs,
            torch.float,
            -2,
            arr_utils.PadDirection.BEFORE,
            batch_first=True,
            padding_value=np.nan,
        )

        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        [[6], [5], [4], [3], [2], [1], [0]],
                        [[np.nan], [np.nan], [np.nan], [3], [2], [1], [0]],
                        [[np.nan], [np.nan], [np.nan], [np.nan], [9], [10], [11]],
                        [[np.nan], [9], [10], [11], [12], [13], [14]],
                    ],
                ),
                converted,
                equal_nan=True,
            )
        )

    def test_zero_neighbor_dict_collation(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_val"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 0.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=0,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=dataset.get_collate_fn(return_dict=True),
            num_workers=0,
        )

        i = 0
        for batch in dataloader:
            i += 1

            self.assertIsInstance(batch["curr_agent_state"], dataset.torch_state_type)
            self.assertIsInstance(batch["agent_hist"], dataset.torch_obs_type)
            self.assertIsInstance(batch["agent_fut"], dataset.torch_obs_type)
            self.assertIsInstance(batch["robot_fut"], dataset.torch_obs_type)

            if i == 5:
                break

        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_val"],
            centric="scene",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 0.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=0,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=dataset.get_collate_fn(return_dict=True),
            num_workers=0,
        )

        i = 0
        for batch in dataloader:
            i += 1

            self.assertIsInstance(
                batch["centered_agent_state"], dataset.torch_state_type
            )
            self.assertIsInstance(batch["agent_hist"], dataset.torch_obs_type)
            self.assertIsInstance(batch["agent_fut"], dataset.torch_obs_type)
            self.assertIsInstance(batch["robot_fut"], dataset.torch_obs_type)

            if i == 5:
                break


if __name__ == "__main__":
    unittest.main()
