# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Contains the code necessary for any entrypoint which uses Hydra to parse a config file
(currently the main wizard entry point and check_config).

Sets up the logger and config schema for the wizard, and provides a main_wrapper function.
"""

import logging
import os
from pathlib import Path
from typing import Callable

import git
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .schema import AlpasimConfig, RunMode

logger = logging.getLogger("alpasim_wizard")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config_schema", node=AlpasimConfig)


def get_repo_root() -> Path:
    path = Path(__file__)
    try:

        repo = git.Repo(path, search_parent_directories=True)
        return Path(repo.git.rev_parse("--show-toplevel"))
    except ModuleNotFoundError:
        logger.info(
            "GitPython module not installed, please install it into the "
            "alpasim environment."
        )
    # Assumes .../repo_root/wizard/src/alpasim_wizard/__main__.py
    return path.resolve().parent.parent.parent.parent


def validate_config(cfg: AlpasimConfig) -> None:
    """Validate the configuration for consistency and completeness.

    This function performs all validation checks that should happen
    after the configuration is loaded but before it's used.
    """
    # Validate NRE version configuration
    if cfg.scenes.nre_version_string is None and not cfg.services.sensorsim:
        raise RuntimeError(
            "Either `scenes.nre_version_string` or `services.sensorsim.image` must be set "
            "to determine the NRE version."
        )

    # Validate service lists
    services_dict = OmegaConf.to_container(cfg.services) or {}
    service_lists = {
        "run_sim_services": cfg.wizard.run_sim_services,
        "run_eval_services": cfg.wizard.run_eval_services,
        "run_aggregation_services": cfg.wizard.run_aggregation_services,
    }

    for list_name, service_list in service_lists.items():
        if service_list:
            undefined_services = [s for s in service_list if s not in services_dict]
            if undefined_services:
                raise RuntimeError(
                    f"Services {undefined_services} in `wizard.{list_name}` "
                    f"are not defined in the `services` section."
                )

    if cfg.wizard.run_mode != RunMode.BATCH:
        # Count total services to run
        total_services = 0
        if cfg.wizard.run_sim_services:
            total_services += len(cfg.wizard.run_sim_services)
        if cfg.wizard.run_eval_services:
            total_services += len(cfg.wizard.run_eval_services)
        if cfg.wizard.run_aggregation_services:
            total_services += len(cfg.wizard.run_aggregation_services)

        if total_services != 1:
            raise AssertionError(
                "When specifying a run mode other than BATCH, "
                "only one service may be run. Ensure only one service is specified "
                "across run_sim_services, run_eval_services, and run_aggregation_services."
            )


def update_scene_config(cfg: AlpasimConfig) -> None:
    """Remove scene_ids from the config if multiple scene sources are specified or
    add all available artifacts if source is set to local and scene_ids is None.

    Only one of scene_ids or test_suite_id should be specified in the config.
    However, we specify a default scene_ids in the stable_manifest/oss.yaml,
    requiring users to explicitly set scene_ids to None if they want to use a
    test_suite_id.

    This function removes this requirement by removing scene_ids from the config
    if:
    - exactly one of scene_ids or test_suite_id is specified in the command line
        arguments
    - scene_ids has exactly one element (which we assume to be the default one)
    """
    scene_config = cfg.scenes
    # Database scene handling
    scene_config_keys = ("scene_ids", "test_suite_id")
    if sum(getattr(scene_config, key) is not None for key in scene_config_keys) == 1:
        # Exactly one specified in config, all good, no need to do anything.
        return

    hydra_cfg = HydraConfig.get()
    cmd_line_overrides = hydra_cfg.overrides.task
    cmd_line_overrides_str = " ".join(cmd_line_overrides)

    # We specify a default scene_id in the config so simulations can run by default.
    # However, when users specify test_suite_id on the command line, we need to
    # clear the default scene_ids to avoid conflicts.
    # Here, we check that exactly one scene source was specified via command line
    # and scene_ids has only one element (which we assume to be the default one).
    if (
        sum(key in cmd_line_overrides_str for key in scene_config_keys) == 1
        and scene_config.scene_ids is not None
        and len(scene_config.scene_ids) == 1
    ):
        scene_config.scene_ids = None


def cmd_line_args(cfg: DictConfig) -> str:
    """Returns a (possibly nested) config as a string of cmd line args."""

    def _convert_to_cmd_line_args_list(cfg: DictConfig, prefix: str = "") -> list[str]:
        args = []
        for key, value in cfg.items():
            if isinstance(value, DictConfig):
                args.extend(
                    _convert_to_cmd_line_args_list(value, f"{prefix}{str(key)}.")
                )
            else:
                args.append(f"{prefix}{str(key)}='{str(value)}'")
        return args

    # Need to escape $ so that omegaconf doesn't interpolate it
    # (e.g. if we want to pass ${num_historical_waypoints} as a command line argument)
    return " ".join(_convert_to_cmd_line_args_list(cfg)).replace("$", r"\$")


OmegaConf.register_new_resolver(
    "repo-relative", lambda path: str(get_repo_root() / path)
)

OmegaConf.register_new_resolver("cmd-line-args", lambda cfg: cmd_line_args(cfg))
OmegaConf.register_new_resolver("or", lambda a, b: a or b)


def main_wrapper(main: Callable) -> None:
    """Wraps a main function with Hydra config parsing."""
    repo_root = get_repo_root()
    config_path = repo_root / "src" / "wizard" / "configs"
    if not os.path.isdir(config_path):
        raise OSError(
            f"Wizard config dir not found at {config_path=}. "
            "Make sure you're invoking the wizard from the alpasim-docker repo root "
            f"or a directory below it (detected {os.getcwd()=})."
        )

    main_with_bound_config = hydra.main(
        config_name="base_config.yaml", version_base="1.3", config_path=str(config_path)
    )(main)

    main_with_bound_config()
