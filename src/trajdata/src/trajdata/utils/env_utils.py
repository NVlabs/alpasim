# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Any, Dict, List

from trajdata.dataset_specific import RawDataset

_NUPLAN_VALID_KWARGS = frozenset(
    {"central_tokens_config", "num_timesteps_before", "num_timesteps_after"}
)


def get_raw_dataset(dataset_name: str, data_dir: str, **dataset_kwargs) -> RawDataset:
    """
    Get a RawDataset instance for the specified dataset.

    Args:
        dataset_name: Name of the dataset (e.g., "nuplan_mini")
        data_dir: Path to the dataset directory
        **dataset_kwargs: Dataset-specific keyword arguments.
                         For Nuplan: central_tokens_config, num_timesteps_before,
                                    num_timesteps_after.
                         Unknown keys (e.g. config_dir, asset_base_path) are silently
                         ignored — callers are responsible for converting config_dir to
                         central_tokens_config before passing dataset_kwargs.

    Returns:
        RawDataset instance for the specified dataset

    Raises:
        ValueError: If the dataset name is not supported
    """
    if "nusc" in dataset_name:
        from trajdata.dataset_specific.nusc import NuscDataset

        return NuscDataset(dataset_name, data_dir, parallelizable=False, has_maps=True)

    if "lyft" in dataset_name:
        from trajdata.dataset_specific.lyft import LyftDataset

        return LyftDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "eupeds" in dataset_name:
        from trajdata.dataset_specific.eth_ucy_peds import EUPedsDataset

        return EUPedsDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=False
        )

    if "sdd" in dataset_name:
        from trajdata.dataset_specific.sdd_peds import SDDPedsDataset

        return SDDPedsDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=False
        )

    if "nuplan" in dataset_name:
        from trajdata.dataset_specific.nuplan import NuplanDataset

        nuplan_kwargs = {
            k: v for k, v in dataset_kwargs.items() if k in _NUPLAN_VALID_KWARGS
        }
        return NuplanDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=True, **nuplan_kwargs
        )

    if "waymo" in dataset_name:
        from trajdata.dataset_specific.waymo import WaymoDataset

        return WaymoDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    if "interaction" in dataset_name:
        from trajdata.dataset_specific.interaction import InteractionDataset

        return InteractionDataset(
            dataset_name, data_dir, parallelizable=True, has_maps=True
        )
    if "mads" in dataset_name.lower():
        from trajdata.dataset_specific.mads import MADSDataset

        return MADSDataset(dataset_name, data_dir, parallelizable=True, has_maps=True)

    raise ValueError(f"Dataset with name '{dataset_name}' is not supported")


def get_raw_datasets(data_dirs: Dict[str, str], **dataset_kwargs) -> List[RawDataset]:
    """
    Get RawDataset instances for multiple datasets.

    Args:
        data_dirs: Dictionary mapping dataset names to their data directories
        **dataset_kwargs: Dataset-specific keyword arguments in nested dict format:
                         {'nuplan_mini': {...}}
                         Each dataset gets its own specific parameters.

    Returns:
        List of RawDataset instances
    """
    raw_datasets: List[RawDataset] = list()

    for dataset_name, data_dir in data_dirs.items():
        # Extract dataset-specific kwargs (empty dict if not specified)
        specific_kwargs = dataset_kwargs.get(dataset_name, {})
        raw_datasets.append(get_raw_dataset(dataset_name, data_dir, **specific_kwargs))

    return raw_datasets
