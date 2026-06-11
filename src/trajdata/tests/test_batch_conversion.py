# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import unittest
from collections import defaultdict

import torch

from trajdata import AgentType, UnifiedDataset
from trajdata.caching.env_cache import EnvCache
from trajdata.data_structures import AgentBatch
from trajdata.utils.batch_utils import SceneTimeBatcher, convert_to_agent_batch


class TestSceneToAgentBatchConversion(unittest.TestCase):
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

        map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}

        self._scene_dataset = UnifiedDataset(
            centric="scene",
            desired_data=[data_source],
            history_sec=(history_sec, history_sec),
            future_sec=(prediction_sec, prediction_sec),
            agent_interaction_distances=attention_radius,
            incl_robot_future=False,
            incl_raster_map=True,
            raster_map_params=map_params,
            only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
            no_types=[AgentType.UNKNOWN],
            num_workers=0,
            standardize_data=True,
            data_dirs={
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self._agent_dataset = UnifiedDataset(
            centric="agent",
            desired_data=[data_source],
            history_sec=(history_sec, history_sec),
            future_sec=(prediction_sec, prediction_sec),
            agent_interaction_distances=attention_radius,
            incl_robot_future=False,
            incl_raster_map=True,
            raster_map_params=map_params,
            only_predict=[AgentType.VEHICLE, AgentType.PEDESTRIAN],
            no_types=[AgentType.UNKNOWN],
            num_workers=0,
            standardize_data=True,
            data_dirs={
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

    def _assert_allclose_with_nans(self, tensor1, tensor2):
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
            torch.allclose(tensor1[valid_mask], tensor2[valid_mask]),
            msg="Non-nan values don't match.",
        )

    def _test_agent_idx(self, agent_dataset_idx: int, verbose=False):
        for offset in range(50):
            agent_batch_element = self._agent_dataset[agent_dataset_idx]
            agent_scene_path, _, _ = self._agent_dataset._data_index[agent_dataset_idx]
            agent_batch = self._agent_dataset.get_collate_fn(pad_format="right")(
                [agent_batch_element]
            )
            scene_ts = agent_batch_element.scene_ts
            scene_id = agent_batch_element.scene_id
            agent_name = agent_batch_element.agent_name
            if verbose:
                print(
                    f"From the agent-centric dataset at index {agent_dataset_idx}, we're looking at:\nAgent {agent_name} in {scene_id} at timestep {scene_ts}"
                )

            # find same scene and ts in scene-centric dataset
            scene_dataset_idx = 0
            for scene_dataset_idx in range(len(self._scene_dataset)):
                scene_path, ts = self._scene_dataset._data_index[scene_dataset_idx]
                if ts == scene_ts and scene_path == agent_scene_path:
                    # load scene to check scene name
                    scene = EnvCache.load(scene_path)
                    if scene.name == scene_id:
                        break

            if verbose:
                print(
                    f"We found a matching scene in the scene-centric dataset at index {scene_dataset_idx}"
                )

            scene_batch_element = self._scene_dataset[scene_dataset_idx]
            converted_agent_batch = convert_to_agent_batch(
                scene_batch_element,
                self._scene_dataset.only_types,
                self._scene_dataset.no_types,
                self._scene_dataset.agent_interaction_distances,
                self._scene_dataset.incl_raster_map,
                self._scene_dataset.raster_map_params,
                self._scene_dataset.max_neighbor_num,
                self._scene_dataset.state_format,
                self._scene_dataset.standardize_data,
                self._scene_dataset.standardize_derivatives,
                pad_format="right",
            )

            agent_idx = -1
            for j, name in enumerate(converted_agent_batch.agent_name):
                if name == agent_name:
                    agent_idx = j

            if agent_idx < 0:
                if verbose:
                    print("no matching scene containing agent, checking next index")
                agent_dataset_idx += 1
            else:
                break

        self.assertTrue(
            agent_idx >= 0, "Matching scene not found in scene-centric dataset!"
        )

        if verbose:
            print(
                f"Agent {converted_agent_batch.agent_name[agent_idx]} appears in {scene_batch_element.scene_id} at timestep {scene_batch_element.scene_ts}, as agent number {agent_idx}"
            )

        attrs_to_ignore = ["data_idx", "extras", "history_pad_dir"]

        variable_length_keys = {
            "neigh_types": "num_neigh",
            "neigh_hist": "num_neigh",
            "neigh_hist_extents": "num_neigh",
            "neigh_hist_len": "num_neigh",
            "neigh_fut": "num_neigh",
            "neigh_fut_extents": "num_neigh",
            "neigh_fut_len": "num_neigh",
        }

        for attr, val in converted_agent_batch.__dict__.items():
            if attr in attrs_to_ignore:
                continue
            if verbose:
                print(f"Checking {attr}")

            if val is None:
                self.assertTrue(agent_batch.__dict__[attr] is None)
            elif isinstance(val[agent_idx], torch.Tensor):
                if attr in variable_length_keys:
                    attr_len = converted_agent_batch.__dict__[
                        variable_length_keys[attr]
                    ][agent_idx]
                    convertedTensor = val[agent_idx, :attr_len, ...]
                    targetTensor = agent_batch.__dict__[attr][0, :attr_len, ...]
                else:
                    convertedTensor = val[agent_idx]
                    targetTensor = agent_batch.__dict__[attr][0]
                try:
                    self._assert_allclose_with_nans(convertedTensor, targetTensor)
                except RuntimeError as e:
                    print(f"Error at {attr=}")
                    raise e
            else:
                self.assertTrue(
                    val[agent_idx] == agent_batch.__dict__[attr][0],
                    f"Failed at {attr=}",
                )

    def test_index_1(self):
        self._test_agent_idx(0, verbose=False)

    def test_index_2(self):
        self._test_agent_idx(116, verbose=False)

    def test_index_3(self):
        self._test_agent_idx(222, verbose=False)


class TestSceneSampler(unittest.TestCase):
    def setUp(self) -> None:
        self._scene_dataset = UnifiedDataset(
            centric="scene",
            desired_data=["nusc_mini-mini_val"],
            data_dirs={
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self._agent_dataset = UnifiedDataset(
            centric="agent",
            desired_data=["nusc_mini-mini_val"],
            data_dirs={
                "nusc_mini": "~/datasets/nuScenes",
            },
        )

        self._scene_sampler = SceneTimeBatcher(self._agent_dataset)

    def _test_len(self, agent_idx=0):
        """
        Len of dataset should be equal to number of timesteps
        the agent appears in the dataset
        """
        sampler = SceneTimeBatcher(self._agent_dataset, agent_idx)

        total_len = sum(
            lengths[agent_idx + 1] - lengths[agent_idx]
            for lengths in self._agent_dataset._data_index._cumulative_scene_lengths
        )

        dl = torch.utils.data.DataLoader(self._agent_dataset, batch_sampler=sampler)

        self.assertEqual(len(dl), total_len)

    def test_len_ego(self):
        self.assertEqual(len(self._scene_sampler), len(self._scene_dataset))

    def test_len_nonego_1(self):
        return self._test_len(15)

    def test_len_nonego_2(self):
        return self._test_len(30)

    def test_consistency(self):
        dl = torch.utils.data.DataLoader(
            self._agent_dataset,
            batch_sampler=self._scene_sampler,
            collate_fn=self._agent_dataset.get_collate_fn(pad_format="right"),
        )
        scene_idx = 0
        agent_batch: AgentBatch
        for agent_batch in dl:
            scene_batch_elem = self._scene_dataset[scene_idx]
            for scene_id in agent_batch.scene_ids:
                self.assertEqual(scene_batch_elem.scene_id, scene_id)

            # ensure all elements have the same scene id
            self.assertEqual(
                torch.abs(agent_batch.scene_ts - scene_batch_elem.scene_ts)
                .float()
                .mean()
                .item(),
                0,
            )

            for agent_name in agent_batch.agent_name:
                self.assertIn(agent_name, scene_batch_elem.agent_names)

            scene_idx += 1

            if scene_idx == 10:
                break


if __name__ == "__main__":
    unittest.main()
