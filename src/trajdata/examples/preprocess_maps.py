# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from trajdata import UnifiedDataset


# @profile
def main():
    dataset = UnifiedDataset(
        # TODO(bivanovic@nvidia.com) Remove lyft from default examples
        desired_data=["nusc_mini", "lyft_sample", "nuplan_mini"],
        rebuild_maps=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_mini": "~/datasets/nuScenes",
            "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
            "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
        },
        verbose=True,
    )
    print(f"Finished Caching Maps!")


if __name__ == "__main__":
    main()
