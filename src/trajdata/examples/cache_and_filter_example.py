# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.data_structures.batch_element import AgentBatchElement
from trajdata.visualization.vis import plot_agent_batch


def main():
    noise_hists = NoiseHistories()

    create_dataset = lambda: UnifiedDataset(
        desired_data=["nusc_mini-mini_val"],
        centric="agent",
        desired_dt=0.5,
        history_sec=(2.0, 2.0),
        future_sec=(4.0, 4.0),
        only_predict=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=False,
        # map_params={"px_per_m": 2, "map_size_px": 224, "offset_frac_xy": (-0.5, 0.0)},
        augmentations=[noise_hists],
        num_workers=0,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_mini": "~/datasets/nuScenes",
        },
    )

    dataset = create_dataset()

    print(f"# Data Samples: {len(dataset):,}")

    print(
        "To demonstrate how to use caching we will first save the "
        "entire dataset (all BatchElements) to a cache file and then load from "
        "the cache file. Note that for large datasets and/or high time resolution "
        "this will create a large file and will use a lot of RAM."
    )
    cache_path = "./temp_cache_file.dill"

    print(
        "We also use a custom filter function that only keeps elements with more "
        "than 5 neighbors"
    )

    def my_filter(el: AgentBatchElement) -> bool:
        return el.num_neighbors > 5

    print(
        f"In the first run we will iterate through the entire dataset and save all "
        f"BatchElements to the cache file {cache_path}"
    )
    print("This may take several minutes.")
    dataset.load_or_create_cache(
        cache_path=cache_path, num_workers=0, filter_fn=my_filter
    )
    assert os.path.isfile(cache_path)

    print(
        "To demonstrate a consecuitve run we create a new dataset and load elements "
        "from the cache file."
    )
    del dataset
    dataset = create_dataset()

    dataset.load_or_create_cache(
        cache_path=cache_path, num_workers=0, filter_fn=my_filter
    )

    # Remove the temp cache file, we dont need it anymore.
    os.remove(cache_path)

    print(
        "We can iterate through the dataset the same way as normally, but this "
        "time it will be much faster because all BatchElements are in memory."
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        plot_agent_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()
