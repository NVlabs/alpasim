# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging
from typing import cast

from alpasim_utils.types import ClipDataSource

from .scenes import USDZManager, resolve_scene_groups
from .schema import AlpasimConfig
from .setup_omegaconf import main_wrapper, update_scene_config

logger = logging.getLogger("alpasim_wizard")
logger.setLevel(logging.INFO)


def check_config(cfg: AlpasimConfig) -> None:
    """Sanity-check the selected clips without downloading them."""
    update_scene_config(cfg)
    manager = USDZManager.from_cfg(cfg.scenes)
    scenes, _ = resolve_scene_groups(cfg.scenes, manager)
    print(
        f"Found {len(scenes)} scenes backed by "
        f"{len({cast(ClipDataSource, scene.source).usdz_uuid for scene in scenes})} "
        "USDZs."
    )


def main() -> None:
    main_wrapper(check_config)


if __name__ == "__main__":
    main()
