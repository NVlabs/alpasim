# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import unittest

from trajdata import UnifiedDataset


class TestDescriptionMatching(unittest.TestCase):
    def test_night(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini"], scene_description_contains=["night"]
        )

        for scene_info in dataset.scenes():
            self.assertIn("night", scene_info.description)

    def test_intersection(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini"], scene_description_contains=["intersection"]
        )

        for scene_info in dataset.scenes():
            self.assertIn("intersection", scene_info.description)

    def test_intersection_more_initial(self):
        dataset = UnifiedDataset(
            desired_data=["nusc_mini", "nuplan_mini"],
            scene_description_contains=["intersection"],
            data_dirs={  # Remember to change this to match your filesystem!
                "nusc_mini": "~/datasets/nuScenes",
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
        )

        for scene_info in dataset.scenes():
            self.assertIn("intersection", scene_info.description)


if __name__ == "__main__":
    unittest.main()
