# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import random
import unittest

from trajdata import AgentType, UnifiedDataset


class TestDatasetSizes(unittest.TestCase):
    def test_two_datasets(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
        )

        self.assertEqual(len(dataset), 27_054_719)

    def test_splits(self):
        dataset = UnifiedDataset(desired_data=["nusc_mini-mini_train"], centric="agent")

        self.assertEqual(len(dataset), 10_598)

        dataset = UnifiedDataset(desired_data=["nusc_mini-mini_val"], centric="agent")

        self.assertEqual(len(dataset), 4_478)

    def test_geography(self):
        dataset = UnifiedDataset(desired_data=["singapore"], centric="agent")

        self.assertEqual(len(dataset), 8_965)

        dataset = UnifiedDataset(desired_data=["boston"], centric="agent")

        self.assertEqual(len(dataset), 6_111)

        dataset = UnifiedDataset(
            desired_data=["pittsburgh"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
        )

        self.assertEqual(len(dataset), 846_605)

        dataset = UnifiedDataset(
            desired_data=["boston", "pittsburgh"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
        )

        self.assertEqual(len(dataset), 2_381_216)

    def test_exclusion(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            no_types=[AgentType.VEHICLE],
        )

        self.assertEqual(len(dataset), 13_099_040)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            no_types=[AgentType.VEHICLE, AgentType.BICYCLE],
        )

        self.assertEqual(len(dataset), 12_989_300)

    def test_inclusion(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            only_types=[AgentType.PEDESTRIAN],
        )

        self.assertEqual(len(dataset), 12_988_830)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            only_types=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        )

        self.assertEqual(len(dataset), 26_944_509)

    def test_prediction_inclusion(self):
        unfiltered_dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
        )

        filtered_dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            only_predict=[AgentType.VEHICLE],
        )

        self.assertGreaterEqual(len(unfiltered_dataset), len(filtered_dataset))

        for _ in range(20):
            sample_idx = random.randint(0, len(filtered_dataset) - 1)
            self.assertEqual(filtered_dataset[sample_idx].agent_type, AgentType.VEHICLE)

        filtered_dataset2 = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
        )

        for _ in range(20):
            sample_idx = random.randint(0, len(filtered_dataset2) - 1)
            self.assertIn(
                filtered_dataset2[sample_idx].agent_type, filtered_dataset2.only_predict
            )

        self.assertGreaterEqual(len(filtered_dataset2), len(filtered_dataset))

    def test_history_future(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            history_sec=(0.1, 2.0),
            future_sec=(0.1, 2.0),
        )

        self.assertEqual(len(dataset), 26_283_199)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            history_sec=(0.5, 2.0),
            future_sec=(0.5, 3.0),
        )

        self.assertEqual(len(dataset), 23_489_292)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            centric="agent",
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
            history_sec=(0.5, 1.0),
            future_sec=(0.5, 0.7),
        )

        self.assertEqual(len(dataset), 23_489_292)

    def test_interpolation(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_train"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_types=[AgentType.VEHICLE],
            incl_robot_future=False,
            incl_raster_map=False,
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

        self.assertEqual(len(dataset), 11_046)

    def test_simple_scene(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_train"],
            centric="scene",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_types=[AgentType.VEHICLE],
            max_agent_num=20,
            num_workers=0,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self.assertEqual(len(dataset), 943)

    def test_hist_fut_len(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_train"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(2.3, 2.3),
            future_sec=(2.4, 2.4),
            only_types=[AgentType.VEHICLE],
            max_agent_num=20,
            num_workers=0,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self.assertEqual(dataset[0].agent_history_np.shape[0], 24)
        self.assertEqual(dataset[0].agent_future_np.shape[0], 24)
        self.assertEqual(dataset[0].scene_ts, 23)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_train"],
            centric="agent",
            desired_dt=0.1,
            history_sec=(3.2, 3.2),
            future_sec=(4.8, 4.8),
            only_types=[AgentType.VEHICLE],
            max_agent_num=20,
            num_workers=0,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self.assertEqual(dataset[0].agent_history_np.shape[0], 33)
        self.assertEqual(dataset[0].agent_future_np.shape[0], 48)
        self.assertEqual(dataset[0].scene_ts, 32)

        dataset = UnifiedDataset(
            desired_data=["nusc_mini-mini_train"],
            centric="scene",
            desired_dt=0.1,
            history_sec=(2.3, 2.3),
            future_sec=(2.4, 2.4),
            only_types=[AgentType.VEHICLE],
            max_agent_num=20,
            num_workers=0,
            verbose=True,
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self.assertEqual(dataset[0].agent_histories[0].shape[0], 24)
        self.assertEqual(dataset[0].agent_futures[0].shape[0], 24)
        self.assertEqual(dataset[0].scene_ts, 23)


if __name__ == "__main__":
    unittest.main()
