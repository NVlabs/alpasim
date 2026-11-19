# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import os
from contextlib import ExitStack

from .deployment.slurm import mps_daemon
from .schema import AlpasimConfig
from .setup_omegaconf import main_wrapper, update_scene_config, validate_config
from .wizard import AlpasimWizard


def run_wizard(cfg: AlpasimConfig) -> None:
    cfg.wizard.log_dir = os.path.abspath(cfg.wizard.log_dir)
    if cfg.wizard.slurm_job_id is None:
        cfg.wizard.slurm_job_id = int(os.environ.get("SLURM_JOB_ID", "0"))
    if cfg.wizard.enable_mps and cfg.wizard.mps_pipe_dir is None:
        cfg.wizard.mps_pipe_dir = f"/tmp/nvidia-mps-{cfg.wizard.slurm_job_id}"
    # First validate the configuration
    validate_config(cfg)
    with ExitStack() as stack:
        # Start MPS before scene preparation, which takes minutes, so processes
        # outside the Wizard that join the daemon need not wait for it.
        if cfg.wizard.enable_mps:
            stack.enter_context(mps_daemon(cfg.wizard))
        # Then update scene config if needed
        update_scene_config(cfg)
        # Finally create and run the wizard
        AlpasimWizard.create(cfg).cast()


def main() -> None:
    main_wrapper(run_wizard)


if __name__ == "__main__":
    main()
