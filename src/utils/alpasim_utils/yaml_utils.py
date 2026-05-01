# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from pathlib import Path
from typing import Any, Type, TypeVar, cast

import yaml
from omegaconf import OmegaConf

C = TypeVar("C")


def load_yaml_dict(
    file_path: str | Path, *, missing_ok: bool = False
) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        file_path: Path to YAML file.
        missing_ok: If True, return an empty dict for missing files.

    Returns:
        Parsed YAML mapping as a dictionary. Empty files return {}.

    Raises:
        FileNotFoundError: If file does not exist and missing_ok is False.
        TypeError: If YAML top-level value is not a mapping.
    """
    path = Path(file_path)
    if not path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected YAML mapping at {path}, got {type(loaded).__name__}")
    return loaded


def typed_parse_config(path: str | Path, config_type: Type[C]) -> C:
    """Reads a yaml file at `path` and parses it into a provided type using omegaconf."""
    yaml_config = OmegaConf.create(load_yaml_dict(path))

    schema = OmegaConf.structured(config_type)
    config: C = cast(C, OmegaConf.merge(schema, yaml_config))
    return config
