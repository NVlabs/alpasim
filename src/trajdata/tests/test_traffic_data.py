# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import unittest

from trajdata import UnifiedDataset
from trajdata.caching.df_cache import DataFrameCache


class TestTrafficLightData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        kwargs = {
            "desired_data": ["nuplan_mini-mini_val"],
            "centric": "scene",
            "history_sec": (3.2, 3.2),
            "future_sec": (4.8, 4.8),
            "incl_robot_future": False,
            "incl_raster_map": True,
            "cache_location": "~/.unified_data_cache",
            "raster_map_params": {
                "px_per_m": 2,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
            },
            "num_workers": 64,
            "verbose": True,
            "data_dirs": {  # Remember to change this to match your filesystem!
                "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
            },
        }

        cls.dataset = UnifiedDataset(
            **kwargs,
            desired_dt=0.05,
        )

        cls.downsampled_dataset = UnifiedDataset(
            **kwargs,
            desired_dt=0.1,
        )

        cls.upsampled_dataset = UnifiedDataset(
            **kwargs,
            desired_dt=0.025,
        )

        cls.scene_num: int = 100

    def test_traffic_light_loading(self):
        # get random scene
        scene = self.dataset.get_scene(self.scene_num)
        scene_cache = DataFrameCache(self.dataset.cache_path, scene)
        traffic_light_status = scene_cache.get_traffic_light_status_dict()

        # just check if the loading works without errors
        self.assertTrue(traffic_light_status is not None)

    def test_downsampling(self):
        # get random scene from both datasets
        scene = self.dataset.get_scene(self.scene_num)
        downsampled_scene = self.downsampled_dataset.get_scene(self.scene_num)

        self.assertEqual(scene.name, downsampled_scene.name)

        scene_cache = DataFrameCache(self.dataset.cache_path, scene)
        downsampled_scene_cache = DataFrameCache(
            self.downsampled_dataset.cache_path, downsampled_scene
        )
        traffic_light_status = scene_cache.get_traffic_light_status_dict()
        downsampled_traffic_light_status = (
            downsampled_scene_cache.get_traffic_light_status_dict()
        )

        orig_lane_ids = set(key[0] for key in traffic_light_status.keys())
        downsampled_lane_ids = set(
            key[0] for key in downsampled_traffic_light_status.keys()
        )
        self.assertSetEqual(orig_lane_ids, downsampled_lane_ids)

        # check that matching indices match
        for (
            lane_id,
            scene_ts,
        ), downsampled_status in downsampled_traffic_light_status.items():
            if scene_ts % 2 == 0:
                try:
                    prev_status = traffic_light_status[lane_id, scene_ts * 2]
                except KeyError:
                    prev_status = None

                try:
                    next_status = traffic_light_status[lane_id, scene_ts * 2 + 1]
                except KeyError:
                    next_status = None

                self.assertTrue(
                    prev_status is not None or next_status is not None,
                    f"Lane {lane_id} at t={scene_ts} has status {downsampled_status} "
                    f"in the downsampled dataset, but neither t={2*scene_ts} nor "
                    f"t={2*scene_ts + 1} were found in the original dataset.",
                )
                self.assertTrue(
                    downsampled_status == prev_status
                    or downsampled_status == next_status,
                    f"Lane {lane_id} at t={scene_ts*2, scene_ts*2 + 1} in the original dataset "
                    f"had status {prev_status, next_status}, but in the downsampled dataset, "
                    f"{lane_id} at t={scene_ts} had status {downsampled_status}",
                )

    def test_upsampling(self):
        # get random scene from both datasets
        scene = self.dataset.get_scene(self.scene_num)
        upsampled_scene = self.upsampled_dataset.get_scene(self.scene_num)
        scene_cache = DataFrameCache(self.dataset.cache_path, scene)
        upsampled_scene_cache = DataFrameCache(
            self.upsampled_dataset.cache_path, upsampled_scene
        )
        traffic_light_status = scene_cache.get_traffic_light_status_dict()
        upsampled_traffic_light_status = (
            upsampled_scene_cache.get_traffic_light_status_dict()
        )

        # check that matching indices match
        for (lane_id, scene_ts), status in upsampled_traffic_light_status.items():
            if scene_ts % 2 == 0:
                orig_status = traffic_light_status[lane_id, scene_ts // 2]
                self.assertEqual(
                    status,
                    orig_status,
                    f"Lane {lane_id} at t={scene_ts // 2} in the original dataset "
                    f"had status {orig_status}, but in the upsampled dataset, "
                    f"{lane_id} at t={scene_ts} had status {status}",
                )
            else:
                try:
                    prev_status = traffic_light_status[lane_id, scene_ts // 2]
                except KeyError:
                    prev_status = None
                try:
                    next_status = traffic_light_status[lane_id, scene_ts // 2 + 1]
                except KeyError as k:
                    next_status = None

                self.assertTrue(
                    prev_status is not None or next_status is not None,
                    f"Lane {lane_id} at t={scene_ts} has status {status} "
                    f"in the upsampled dataset, but neither t={scene_ts // 2} nor "
                    f"t={scene_ts // 2 + 1} were found in the original dataset.",
                )

                self.assertTrue(
                    status == prev_status or status == next_status,
                    f"Lane {lane_id} at t={scene_ts // 2, scene_ts // 2 + 1} in the original dataset "
                    f"had status {prev_status, next_status}, but in the upsampled dataset, "
                    f"{lane_id} at t={scene_ts} had status {status}",
                )


if __name__ == "__main__":
    unittest.main()
