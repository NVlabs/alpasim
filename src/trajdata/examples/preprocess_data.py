# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os

from trajdata import UnifiedDataset


def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini", "lyft_sample", "nuplan_mini"],
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_mini": "~/datasets/nuScenes",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
            "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
        },
    )
    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()
