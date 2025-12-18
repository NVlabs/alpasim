# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging

from .compatibility import CompatibilityMatrix
from .scenes import LOCAL_SUITE_ID, USDZManager
from .schema import AlpasimConfig
from .setup_omegaconf import main_wrapper
from .utils import nre_image_to_nre_version

logger = logging.getLogger("alpasim_wizard")
logger.setLevel(logging.INFO)


def check_config(cfg: AlpasimConfig) -> None:
    """
    Sanity-checks the config file. Can be used on the login node.
    """
    if cfg.services.sensorsim is None:
        # TODO: could we run in these conditions?
        raise ValueError("Missing 'sensorsim' config in 'services' section.")

    nre_version_string = nre_image_to_nre_version(cfg.services.sensorsim.image)
    compatibility_matrix = CompatibilityMatrix.from_config(
        cfg.scenes.artifact_compatibility_matrix
    )
    compatible_versions = list(compatibility_matrix.lookup(nre_version_string))

    manager = USDZManager.from_cfg(cfg.scenes)

    # Determine which selection method to use
    test_suite_id = cfg.scenes.test_suite_id
    scene_ids = cfg.scenes.scene_ids

    # If local_usdz_dir is set and neither scene_ids nor test_suite_id is provided,
    # default to using the "local" test suite (all scenes in the directory)
    if cfg.scenes.local_usdz_dir is not None:
        if test_suite_id is None and scene_ids is None:
            test_suite_id = LOCAL_SUITE_ID

    if test_suite_id is not None:
        artifacts = manager.query_by_suite_id(test_suite_id, compatible_versions)
    elif scene_ids is not None:
        artifacts = manager.query_by_scene_ids(scene_ids, compatible_versions)
    else:
        print("No scene_ids or test_suite_id specified.")
        return

    print(f"Found {len(artifacts)} scenes compatible with {nre_version_string=}.")


def main() -> None:
    main_wrapper(check_config)


if __name__ == "__main__":
    main()
