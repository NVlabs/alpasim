# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import unittest
from collections import defaultdict

from torch.utils.data import DataLoader

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.state import NP_STATE_TYPES, TORCH_STATE_TYPES
from trajdata.dataset import UnifiedDataset


class TestDataset(unittest.TestCase):
    def test_dataloading(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_val"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 30.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=4,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=dataset.get_collate_fn(),
            num_workers=0,
        )

        i = 0
        batch: AgentBatch
        for batch in dataloader:
            i += 1

            batch.to("cuda")

            self.assertIsInstance(batch.curr_agent_state, dataset.torch_state_type)
            self.assertIsInstance(batch.agent_hist, dataset.torch_obs_type)
            self.assertIsInstance(batch.agent_fut, dataset.torch_obs_type)
            self.assertIsInstance(batch.robot_fut, dataset.torch_obs_type)

            if i == 5:
                break

    def test_dict_dataloading(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_val"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 30.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=4,
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
            agent_interaction_distances=defaultdict(lambda: 30.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=4,
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

    def test_default_datatypes_agent(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_val"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 30.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=4,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        elem: AgentBatchElement = dataset[0]
        self.assertIsInstance(elem.curr_agent_state_np, dataset.np_state_type)
        self.assertIsInstance(elem.agent_history_np, dataset.np_obs_type)
        self.assertIsInstance(elem.agent_future_np, dataset.np_obs_type)
        self.assertIsInstance(elem.robot_future_np, dataset.np_obs_type)

    def test_default_datatypes_scene(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_val"],
            centric="scene",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 30.0),
            incl_robot_future=True,
            incl_raster_map=True,
            standardize_data=False,
            raster_map_params={
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            num_workers=4,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        elem: SceneBatchElement = dataset[0]
        self.assertIsInstance(elem.centered_agent_state_np, dataset.np_state_type)
        self.assertIsInstance(elem.agent_histories[0], dataset.np_obs_type)
        self.assertIsInstance(elem.agent_futures[0], dataset.np_obs_type)
        self.assertIsInstance(elem.robot_future_np, dataset.np_obs_type)


if __name__ == "__main__":
    unittest.main()
