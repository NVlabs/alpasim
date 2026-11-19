# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
Contains the code necessary for any entrypoint which uses Hydra to parse a config file
(currently the main wizard entry point and check_config).

Sets up the logger and config schema for the wizard, and provides a main_wrapper function.
"""

import logging
import os
import re
import subprocess
from typing import Callable

import hydra
from alpasim_utils.paths import find_repo_root
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf

from .scenes.scene_suite import find_scene_suite_csv
from .schema import AlpasimConfig, SceneSuiteConfig

logger = logging.getLogger("alpasim_wizard")

cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config_schema", node=AlpasimConfig)

REPO_ROOT = find_repo_root(__file__)

_DEFAULT_CLIP_IDS = ("clipgt-01d503d4-449b-46fc-8d78-9085e70d3554",)


def _list_config_group_options(group: str) -> list[str]:
    """List available options for a Hydra config group by scanning the search path.

    Returns sorted list of config names (without extension) found under
    ``group/`` across all Hydra search paths.
    """

    gh = GlobalHydra.instance()
    if not gh.is_initialized():
        return []

    config_loader = gh.config_loader()
    try:
        options = config_loader.get_group_options(group)
    except (
        hydra.errors.HydraException,
        KeyError,
    ):  # Hydra may raise various internal errors for missing/invalid groups
        options = []
    # Filter out companion configs (e.g., vavam_configs, dino_configs used by driver defaults)
    options = [o for o in options if not o.endswith("_configs")]
    return sorted(set(options))


def _is_missing_or_empty(cfg: AlpasimConfig, path: str) -> bool:
    """Check if a config value is MISSING or an empty DictConfig.

    Some config groups (e.g., driver) default to an empty DictConfig in the
    schema, so MISSING never survives the merge.  An empty dict means the
    user didn't select a config for that group.
    """
    try:
        val = OmegaConf.select(cfg, path, throw_on_missing=True)  # type: ignore[arg-type]
    except MissingMandatoryValue:
        return True
    if isinstance(val, DictConfig) and len(val) == 0:
        return True
    return False


def _check_required_config_groups(cfg: AlpasimConfig) -> None:
    """Check that all required Hydra config groups were specified.

    Detects MISSING sentinel values left behind when a required config
    group (deploy, topology, driver) was not provided on the command line,
    and raises a single helpful error listing everything that's missing.
    Available options are discovered dynamically from the Hydra search path.
    """
    # Each entry: (dotted config path to probe, Hydra group name, CLI syntax)
    required_checks = [
        ("defines.filesystem", "deploy", "deploy=<target>"),
        ("services.renderer.replicas_per_container", "topology", "topology=<layout>"),
    ]
    # Driver config is only required when the driver service is being launched.
    # In daemon mode (run_sim_services excludes driver), no driver model is needed.
    run_sim_services = list(cfg.wizard.run_sim_services or [])
    if "driver" in run_sim_services:
        required_checks.append(("driver", "driver", "driver=<model>"))

    missing = []
    for path, group, arg in required_checks:
        if _is_missing_or_empty(cfg, path):
            options = _list_config_group_options(group)
            hint = f"  {arg:30s}"
            if options:
                hint += f"  (available: {', '.join(options)})"
            missing.append(hint)

    if missing:
        lines = ["", "Missing required configuration:"]
        lines.extend(missing)
        lines.append("")
        lines.append("Example:")
        lines.append(
            "  uv run alpasim_wizard deploy=local topology=1gpu"
            " driver=vavam wizard.log_dir=./out"
        )
        lines.append("")
        raise SystemExit("\n".join(lines))


def _check_run_sim_services_defined(cfg: AlpasimConfig) -> None:
    """Check that launched services exist and are configured."""
    services_dict = OmegaConf.to_container(cfg.services) or {}
    if not isinstance(services_dict, dict):
        raise TypeError("Expected `services` to be a mapping.")
    if not cfg.wizard.run_sim_services:
        return

    undefined_services = [
        s for s in cfg.wizard.run_sim_services if s not in services_dict
    ]
    if undefined_services:
        raise RuntimeError(
            f"Services {undefined_services} in `wizard.run_sim_services` "
            f"are not defined in the `services` section."
        )

    unset_services = [
        s
        for s in cfg.wizard.run_sim_services
        if s in services_dict and services_dict.get(s) is None
    ]
    if unset_services:
        raise RuntimeError(
            f"Services {unset_services} in `wizard.run_sim_services` "
            f"are set to null in the `services` section."
        )


def validate_config(cfg: AlpasimConfig) -> None:
    """Validate the configuration for consistency and completeness.

    This function performs all validation checks that should happen
    after the configuration is loaded but before it's used.
    """
    _check_run_sim_services_defined(cfg)
    _check_required_config_groups(cfg)
    if cfg.wizard.enable_mps and not cfg.wizard.run_method.is_slurm:
        raise ValueError("wizard.enable_mps is only supported on SLURM")
    # The pipe directory goes unquoted into shell commands (including `rm -rf`)
    # and comma-separated container mount lists.
    if cfg.wizard.mps_pipe_dir is not None and not re.fullmatch(
        r"/[A-Za-z0-9._/-]+", cfg.wizard.mps_pipe_dir
    ):
        raise ValueError(
            "wizard.mps_pipe_dir must be an absolute path of letters, digits, "
            f"'.', '_', '-' and '/'; got {cfg.wizard.mps_pipe_dir!r}"
        )

    do_shutdown = OmegaConf.select(cfg.runtime, "endpoints.do_shutdown", default=True)
    if do_shutdown is False:
        logger.warning(
            "runtime.endpoints.do_shutdown=false is ignored: shutdown behavior "
            "changed and managed services now shut down after the runtime exits."
        )

    # Validate NRE version configuration
    if cfg.scenes.nre_version_string is None and not cfg.services.renderer:
        raise RuntimeError(
            "Either `scenes.nre_version_string` or `services.renderer.image` must be set "
            "to determine the NRE version."
        )

    driver_in_run_list = "driver" in (cfg.wizard.run_sim_services or [])
    external_services = cfg.wizard.external_services or {}
    external_driver_addresses = external_services.get("driver") or []

    if driver_in_run_list and external_driver_addresses:
        raise RuntimeError(
            "Driver service is listed in `wizard.run_sim_services`, but "
            "`wizard.external_services.driver` is also configured. Remove the "
            "driver service or the external driver addresses."
        )


def update_scene_config(cfg: AlpasimConfig) -> None:
    """Validate scene selectors and normalize suite IDs to suite configs.

    A local USDZ directory selects all discovered clips unless
    ``scenes.clip_ids`` provides a filter.
    """
    scene_config = cfg.scenes

    hydra_cfg = HydraConfig.get()
    cmd_line_overrides = hydra_cfg.overrides.task
    # Compare normalized Hydra keys exactly so exclude_clip_ids does not count as
    # selecting clip_ids. Hydra keys may use +/~ prefixes.
    override_keys = {
        override.partition("=")[0].strip().lstrip("+~")
        for override in cmd_line_overrides
    }
    clip_ids_selected_on_cli = "scenes.clip_ids" in override_keys
    local_directory_selected_on_cli = "scenes.local_usdz_dir" in override_keys
    suite_ids_selected_on_cli = "scenes.suite_ids" in override_keys
    suite_configs_selected_on_cli = "scenes.suite_configs" in override_keys
    suite_groups_selected_on_cli = "scenes.suite_groups" in override_keys

    inherited_default_selected = tuple(scene_config.clip_ids or ()) == _DEFAULT_CLIP_IDS
    if (
        (
            suite_ids_selected_on_cli
            or suite_configs_selected_on_cli
            or suite_groups_selected_on_cli
        )
        and not clip_ids_selected_on_cli
        and inherited_default_selected
    ):
        scene_config.clip_ids = None

    if (
        scene_config.local_usdz_dir is not None
        and local_directory_selected_on_cli
        and not clip_ids_selected_on_cli
        and inherited_default_selected
    ):
        scene_config.clip_ids = None

    suite_ids = scene_config.suite_ids
    suite_ids_selected = suite_ids is not None
    suite_configs_selected = scene_config.suite_configs is not None
    suite_groups = getattr(scene_config, "suite_groups", {})
    if suite_groups_selected_on_cli and not suite_groups:
        raise ValueError("scenes.suite_groups must not be empty.")
    if suite_groups:
        if any(not configs for configs in suite_groups.values()):
            raise ValueError("Every scenes.suite_groups entry must be non-empty.")
        if scene_config.exclude_clip_ids or scene_config.limit_to_first_n > 0:
            raise ValueError(
                "scenes.suite_groups cannot be combined with exclude_clip_ids or "
                "limit_to_first_n."
            )
        if suite_ids_selected or suite_configs_selected:
            raise ValueError(
                "Set either scenes.suite_groups or a single suite selector, not both."
            )
        if scene_config.local_usdz_dir is not None or scene_config.clip_ids is not None:
            raise ValueError(
                "Set either scenes.suite_groups or another scene selector, not both."
            )
        return
    if suite_ids_selected and not suite_ids:
        raise ValueError("scenes.suite_ids must not be empty.")
    if suite_ids_selected and suite_configs_selected:
        raise ValueError(
            "Set either scenes.suite_ids or scenes.suite_configs, not both."
        )

    suite_selected = suite_ids_selected or suite_configs_selected
    selected_suite_key = (
        "scenes.suite_ids" if suite_ids_selected else "scenes.suite_configs"
    )
    if suite_selected and scene_config.local_usdz_dir is not None:
        raise ValueError(
            f"Set either {selected_suite_key} or scenes.local_usdz_dir, not both."
        )
    if suite_selected and scene_config.clip_ids is not None:
        raise ValueError(
            f"Set either {selected_suite_key} or scenes.clip_ids, not both."
        )
    if (
        not suite_selected
        and scene_config.clip_ids is None
        and scene_config.local_usdz_dir is None
    ):
        raise ValueError(
            "Set one scene selector: scenes.clip_ids, scenes.suite_ids, "
            "scenes.suite_configs, or scenes.local_usdz_dir."
        )

    if suite_ids is not None:
        scene_config.suite_configs = [
            SceneSuiteConfig(
                csv_path=str(
                    find_scene_suite_csv(suite_id, scene_config.scene_suite_dirs)
                )
            )
            for suite_id in suite_ids
        ]
        scene_config.suite_ids = None


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


def _read_repo_version() -> str:
    """Read the project version from the root pyproject.toml."""
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _read_base_image_tag() -> str:
    """Return the stable or CI-built base image tag for this checkout."""
    version = _read_repo_version()
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    if branch is None or branch == "main":
        return version

    merge_base = _git_output("merge-base", "HEAD", "origin/main")
    if merge_base is None:
        return version

    changed_files = _git_output("diff", "--name-only", f"{merge_base}...HEAD")
    if changed_files is None or "pyproject.toml" not in changed_files.splitlines():
        return version

    commit_sha = _git_output("rev-parse", "HEAD")
    return commit_sha[:8] if commit_sha else version


OmegaConf.register_new_resolver("repo-relative", lambda path: str(REPO_ROOT / path))
OmegaConf.register_new_resolver("cmd-line-args", lambda cfg: cmd_line_args(cfg))
OmegaConf.register_new_resolver("or", lambda a, b: a or b)
OmegaConf.register_new_resolver("repo-version", _read_repo_version)
OmegaConf.register_new_resolver("repo-base-image-tag", _read_base_image_tag)


def main_wrapper(main: Callable) -> None:
    """Wraps a main function with Hydra config parsing."""
    config_path = REPO_ROOT / "src" / "wizard" / "configs"
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
