# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging

from .scenes import LOCAL_SUITE_ID, USDZManager
from .schema import AlpasimConfig
from .setup_omegaconf import main_wrapper

logger = logging.getLogger("alpasim_wizard")
logger.setLevel(logging.INFO)


def check_config(cfg: AlpasimConfig) -> None:
    """
    Sanity-checks the config file. Can be used on the login node.
    """
    manager = USDZManager.from_cfg(cfg.scenes)

    # TODO The logic here duplicates what is in context.py:fetch_artifacts. Unify.

    # Determine which selection method to use
    test_suite_id = cfg.scenes.test_suite_id
    scene_ids = cfg.scenes.scene_ids

    # If local_usdz_dir is set and neither scene_ids nor test_suite_id is provided,
    # default to using the "local" test suite (all scenes in the directory)
    if cfg.scenes.local_usdz_dir is not None:
        if test_suite_id is None and scene_ids is None:
            test_suite_id = LOCAL_SUITE_ID

    if test_suite_id is not None:
        artifacts = manager.query_by_suite_id(test_suite_id)
    elif scene_ids is not None:
        artifacts = manager.query_by_scene_ids(scene_ids)
    else:
        print("No scene_ids or test_suite_id specified.")
        return

    # Sort to ensure deterministic ordering. This is important for resume runs when
    # limit_to_first_n but also makes our life a bit easier.
    artifacts = sorted(artifacts, key=lambda x: x.scene_id)

    # Apply limit_to_first_n if specified (positive value)
    limit_n = cfg.scenes.limit_to_first_n
    if limit_n > 0 and len(artifacts) > limit_n:
        print(f"Limiting scenes from {len(artifacts)} to first {limit_n}")
        artifacts = artifacts[:limit_n]

    print(f"Found {len(artifacts)} scenes.")


def main() -> None:
    main_wrapper(check_config)


if __name__ == "__main__":
    main()
