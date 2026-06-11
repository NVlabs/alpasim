# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import unittest
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

from trajdata import AgentType, MapAPI, SceneBatch, UnifiedDataset, VectorMap
from trajdata.dataset import DataLoader
from trajdata.utils.batch_utils import get_raster_maps_for_scene_batch


class TestRasterMap(unittest.TestCase):
    def __init__(self, methodName: str = "batchConversion") -> None:
        super().__init__(methodName)

        data_source = "nusc_mini"
        history_sec = 2.0
        prediction_sec = 6.0

        attention_radius = defaultdict(
            lambda: 20.0
        )  # Default range is 20m unless otherwise specified.
        attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
        attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
        attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
        attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

        self._map_params = {
            "px_per_m": 2,
            "map_size_px": 100,
            "offset_frac_xy": (-0.75, 0.0),
        }

        self._scene_dataset = UnifiedDataset(
            centric="scene",
            desired_data=[data_source],
            history_sec=(history_sec, history_sec),
            future_sec=(prediction_sec, prediction_sec),
            agent_interaction_distances=attention_radius,
            incl_robot_future=False,
            incl_raster_map=True,
            raster_map_params=self._map_params,
            only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
            no_types=[AgentType.UNKNOWN],
            num_workers=0,
            standardize_data=True,
            data_dirs={
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self._scene_dataloader = DataLoader(
            self._scene_dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=self._scene_dataset.get_collate_fn(),
            num_workers=0,
        )

    def _assert_allclose_with_nans(self, tensor1, tensor2, atol=1e-8):
        """
        asserts that the two tensors have nans in the same locations, and the non-nan
        elements all are close.
        """
        # Check nans are in the same place
        self.assertFalse(
            torch.any(  # True if there's any mismatch
                torch.logical_xor(  # True where either tensor1 or tensor 2 has nans, but not both (mismatch)
                    torch.isnan(tensor1),  # True where tensor1 has nans
                    torch.isnan(tensor2),  # True where tensor2 has nans
                )
            ),
            msg="Nans occur in different places.",
        )
        valid_mask = torch.logical_not(torch.isnan(tensor1))
        self.assertTrue(
            torch.allclose(tensor1[valid_mask], tensor2[valid_mask], atol=atol),
            msg="Non-nan values don't match.",
        )

    def test_map_transform_scenebatch(self):
        scene_batch: SceneBatch
        for i, scene_batch in enumerate(self._scene_dataloader):

            # Make the tf double for more accurate transform.
            scene_batch.centered_world_from_agent_tf = (
                scene_batch.centered_world_from_agent_tf.double()
            )

            maps, maps_resolution, raster_from_world_tf = (
                get_raster_maps_for_scene_batch(
                    scene_batch,
                    self._scene_dataset.cache_path,
                    "nusc_mini",
                    self._map_params,
                )
            )

            self._assert_allclose_with_nans(
                scene_batch.rasters_from_world_tf, raster_from_world_tf, atol=1e-2
            )
            self._assert_allclose_with_nans(
                scene_batch.maps_resolution, maps_resolution
            )
            self._assert_allclose_with_nans(scene_batch.maps, maps, atol=1e-4)

            if i > 50:
                break


if __name__ == "__main__":
    unittest.main(catchbreak=False)
