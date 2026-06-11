# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import os

from .schema import AlpasimConfig
from .setup_omegaconf import main_wrapper, update_scene_config, validate_config
from .wizard import AlpasimWizard


def run_wizard(cfg: AlpasimConfig) -> None:
    cfg.wizard.log_dir = os.path.abspath(cfg.wizard.log_dir)
    # First validate the configuration
    validate_config(cfg)
    # Then update scene config if needed
    update_scene_config(cfg)
    # Finally create and run the wizard
    AlpasimWizard.create(cfg).cast()


def main() -> None:
    main_wrapper(run_wizard)


if __name__ == "__main__":
    main()
