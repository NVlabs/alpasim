# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import importlib
import importlib.util
import unittest
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.data_structures import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.state import NP_STATE_TYPES, TORCH_STATE_TYPES

_has_nuscenes = importlib.util.find_spec("nuscenes") is not None

AgentStateArray = NP_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,h"]
AgentObsArray = NP_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,s,c"]
AgentStateTensor = TORCH_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,h"]
AgentObsTensor = TORCH_STATE_TYPES["x,y,z,xd,yd,xdd,ydd,s,c"]


class TestStateTensor(unittest.TestCase):
    def test_construction(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = torch.rand(8).as_subclass(AgentStateTensor)
        c = AgentObsTensor(torch.rand(5, 9))

    def test_class_propagation(self):
        a = AgentStateTensor(torch.rand(2, 8))
        self.assertTrue(isinstance(a.to("cpu"), AgentStateTensor))

        a = AgentStateTensor(torch.rand(2, 8))
        self.assertTrue(isinstance(a.cpu(), AgentStateTensor))

        b = AgentStateTensor(torch.rand(2, 8))
        self.assertTrue(isinstance(a + b, AgentStateTensor))

        b = torch.rand(2, 8)
        self.assertTrue(isinstance(a + b, AgentStateTensor))

        a += 1
        self.assertTrue(isinstance(a, AgentStateTensor))

    def test_property_access(self):
        a = AgentStateTensor(torch.rand(2, 8))
        position = a[..., :3]
        velocity = a[..., 3:5]
        acc = a[..., 5:7]
        h = a[..., 7:]

        self.assertTrue(torch.allclose(a.position3d, position))
        self.assertTrue(torch.allclose(a.velocity, velocity))
        self.assertTrue(torch.allclose(a.acceleration, acc))
        self.assertTrue(torch.allclose(a.heading, h))

    def test_heading_conversion(self):
        a = AgentStateTensor(torch.rand(2, 8))
        h = a[..., 7:]
        hv = a.heading_vector
        self.assertTrue(torch.allclose(torch.atan2(hv[..., 1], hv[..., 0])[:, None], h))

    def test_long_lat_velocity(self):
        a = AgentStateTensor(torch.rand(8))
        velocity = a[3:5]
        h = a[7]
        lonlat_v = a.as_format("v_lon,v_lat")
        lonlat_v_correct = (
            torch.tensor([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])[None, ...]
            @ velocity[..., None]
        )[..., 0]

        self.assertTrue(torch.allclose(lonlat_v, lonlat_v_correct))

        b = a.as_format("x,y,xd,yd,s,c")
        s = b[-2]
        c = b[-1]
        lonlat_v = b.as_format("v_lon,v_lat")
        lonlat_v_correct = (
            torch.tensor([[c, s], [-s, c]])[None, ...] @ velocity[..., None]
        )[..., 0]

        self.assertTrue(torch.allclose(lonlat_v, lonlat_v_correct))

    def test_long_lat_conversion(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = a.as_format("xd,yd,h")
        c = b.as_format("v_lon,v_lat,h")
        d = c.as_format("xd,yd,h")
        self.assertTrue(torch.allclose(b, d))

    def test_as_format(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = a.as_format("x,y,z,xd,yd,xdd,ydd,s,c")
        self.assertTrue(isinstance(b, AgentObsTensor))
        self.assertTrue(torch.allclose(a, b.as_format(a._format)))

    def test_as_tensor(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = a.as_tensor()
        self.assertTrue(isinstance(b, torch.Tensor))
        self.assertFalse(isinstance(b, AgentStateTensor))

    def test_tensor_ops(self):
        a = AgentStateTensor(torch.rand(2, 8))
        b = a[0] + a[1]
        c = torch.mean(b)
        self.assertFalse(isinstance(c, AgentStateTensor))
        self.assertTrue(isinstance(c, torch.Tensor))

    def test_reshape_keeps_class_if_possible(self):
        a = AgentStateTensor(torch.rand(2, 3, 8))
        self.assertTrue(isinstance(a.reshape(6, 8), AgentStateTensor))
        self.assertTrue(isinstance(a.reshape((6, 8)), AgentStateTensor))
        self.assertTrue(isinstance(torch.reshape(a, (6, 8)), AgentStateTensor))
        self.assertFalse(isinstance(a.reshape(12, 4), AgentStateTensor))
        self.assertFalse(isinstance(a.reshape((12, 4)), AgentStateTensor))
        self.assertFalse(isinstance(torch.reshape(a, (12, 4)), AgentStateTensor))


class TestStateArray(unittest.TestCase):
    def test_construction(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        c = np.random.rand(5, 9).view(AgentObsArray)

    def test_property_access(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        position = a[..., :3]
        velocity = a[..., 3:5]
        acc = a[..., 5:7]
        h = a[..., 7:]

        self.assertTrue(np.allclose(a.position3d, position))
        self.assertTrue(np.allclose(a.velocity, velocity))
        self.assertTrue(np.allclose(a.acceleration, acc))
        self.assertTrue(np.allclose(a.heading, h))

    def test_property_setting(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        a.heading = 0.0
        self.assertTrue(np.allclose(a[..., -1], np.zeros([2, 1])))

    def test_heading_conversion(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        h = a[..., 7:]
        hv = a.heading_vector
        self.assertTrue(np.allclose(np.arctan2(hv[..., 1], hv[..., 0])[:, None], h))

    def test_long_lat_velocity(self):
        a = np.random.rand(8).view(AgentStateArray)
        velocity = a[3:5]
        h = a[7]
        lonlat_v = a.as_format("v_lon,v_lat")
        lonlat_v_correct = (
            np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])[None, ...]
            @ velocity[..., None]
        )[..., 0]

        self.assertTrue(np.allclose(lonlat_v, lonlat_v_correct))

        b = a.as_format("x,y,xd,yd,s,c")
        s = b[-2]
        c = b[-1]
        lonlat_v = b.as_format("v_lon,v_lat")
        lonlat_v_correct = (
            np.array([[c, s], [-s, c]])[None, ...] @ velocity[..., None]
        )[..., 0]

        self.assertTrue(np.allclose(lonlat_v, lonlat_v_correct))

    def test_long_lat_conversion(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        b = a.as_format("xd,yd,h")
        c = b.as_format("v_lon,v_lat,h")
        d = c.as_format("xd,yd,h")
        self.assertTrue(np.allclose(b, d))

    def test_as_format(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        b = a.as_format("x,y,z,xd,yd,xdd,ydd,s,c")
        self.assertTrue(isinstance(b, AgentObsArray))
        self.assertTrue(np.allclose(a, b.as_format(a._format)))

    def test_as_ndarray(self):
        a: AgentStateArray = np.random.rand(2, 8).view(AgentStateArray)
        b = a.as_ndarray()
        self.assertTrue(isinstance(b, np.ndarray))
        self.assertFalse(isinstance(b, AgentStateArray))

    def test_tensor_ops(self):
        a = np.random.rand(2, 8).view(AgentStateArray)
        b = a[0] + a[1]
        c = np.mean(b)
        self.assertFalse(isinstance(c, AgentStateArray))
        self.assertTrue(isinstance(c, float))


@unittest.skipUnless(_has_nuscenes, "nuscenes package not installed")
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
