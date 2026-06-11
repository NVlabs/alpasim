# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
This is an example of how to extend a batch to include custom data
"""

from collections import defaultdict
from functools import partial
from typing import Tuple, Union

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement


def custom_random_data(
    batch_elem: Union[AgentBatchElement, SceneBatchElement]
) -> np.ndarray:
    # create new data to add to each batch element
    return np.random.random((10, 10))


def custom_goal_location(
    batch_elem: Union[AgentBatchElement, SceneBatchElement]
) -> np.ndarray:
    # simply access existing element attributes
    return batch_elem.agent_future_np.position


def custom_min_distance_from_others(
    batch_elem: Union[AgentBatchElement, SceneBatchElement]
) -> np.ndarray:
    # ... or more complicated calculations
    current_ego_loc = batch_elem.agent_history_np[-1, :2]
    all_distances = [
        np.linalg.norm(current_ego_loc - veh[-1, :2])
        for veh in batch_elem.neighbor_histories
    ]

    if not len(all_distances):
        return np.inf
    else:
        return np.min(all_distances)


def custom_distances_squared(
    batch_elem: Union[AgentBatchElement, SceneBatchElement]
) -> np.ndarray:
    # we can chain extras together if needed
    return batch_elem.extras["min_distance"] ** 2


def custom_raster(
    batch_elem: Union[AgentBatchElement, SceneBatchElement],
    raster_size: Tuple[int, ...],
) -> np.ndarray:
    # draw a custom raster
    img = np.zeros(raster_size)

    # ...
    return img


def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini-mini_train"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
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
        extras={  # a dictionary that contains functions that generate our custom data. Can be any function and has access to the batch element.
            "random_data": custom_random_data,
            "goal_location": custom_goal_location,
            "min_distance": custom_min_distance_from_others,
            "min_distance_sq": custom_distances_squared,  # in Python >= 3.7 dictionaries are guaranteed to maintain order => you can use previously computed keys
            "raster": partial(custom_raster, raster_size=(100, 100)),
        },
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        assert "random_data" in batch.extras
        assert "goal_location" in batch.extras
        assert "min_distance" in batch.extras
        assert "raster" in batch.extras


if __name__ == "__main__":
    main()
