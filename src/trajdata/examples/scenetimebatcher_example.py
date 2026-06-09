# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.utils.batch_utils import SceneTimeBatcher
from trajdata.visualization.vis import plot_agent_batch_all


def main():
    """
    Here, we use SceneTimeBatcher to loop through an
    Agent-centric dataset with batches grouped by scene and timestep
    """
    dataset = UnifiedDataset(
        desired_data=["nusc_mini-mini_train"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(3.2, 3.2),
        future_sec=(4.8, 4.8),
        only_predict=[AgentType.VEHICLE],
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
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_sampler=SceneTimeBatcher(dataset),
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )

    batch: AgentBatch
    for batch in tqdm(dataloader):
        plot_agent_batch_all(batch)


if __name__ == "__main__":
    main()
