# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Hydra SearchPathPlugin that auto-discovers config directories from installed alpasim plugins.

Any installed package can register its config directory by adding an entry point
in the ``alpasim.configs`` group.  For example, in ``pyproject.toml``::

    [project.entry-points."alpasim.configs"]
    my_plugin = "my_plugin.configs"

When Hydra initialises, this plugin will discover all such entry points and
add ``pkg://<entry_point_value>`` to Hydra's config search path.  This makes
plugin configs available for composition without manual
``hydra.searchpath`` overrides on the command line.

After all paths are registered the plugin walks every config directory and
raises ``ValueError`` if two providers supply a YAML file at the same
relative path (e.g. both ``wizard`` and an internal plugin ship
``deploy/foo.yaml``).
"""

from __future__ import annotations

import importlib.util
import logging
from importlib.metadata import entry_points
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

logger = logging.getLogger(__name__)

_FILE_SCHEME = "file://"
_PKG_SCHEME = "pkg://"


def find_duplicate_configs(
    config_dirs: dict[str, Path],
) -> dict[str, list[tuple[str, Path]]]:
    """Find YAML config files that exist in more than one provider directory.

    Args:
        config_dirs: Mapping from provider name to its config root directory.

    Returns:
        Mapping from relative YAML path to a list of ``(provider, absolute_path)``
        tuples, **only** for paths that appear in two or more providers.
    """
    seen: dict[str, list[tuple[str, Path]]] = {}
    for provider, root in config_dirs.items():
        if not root.is_dir():
            continue
        for yaml_file in sorted(root.rglob("*.yaml")):
            rel = str(yaml_file.relative_to(root))
            seen.setdefault(rel, []).append((provider, yaml_file))
    return {rel: providers for rel, providers in seen.items() if len(providers) > 1}


def _resolve_search_path_element(path: str) -> Path | None:
    """Resolve a Hydra search-path string (``file://`` or ``pkg://``) to a directory."""
    if path.startswith(_FILE_SCHEME):
        return Path(path[len(_FILE_SCHEME) :])
    if path.startswith(_PKG_SCHEME):
        module_path = path[len(_PKG_SCHEME) :]
        spec = importlib.util.find_spec(module_path)
        if spec and spec.submodule_search_locations:
            return Path(spec.submodule_search_locations[0])
    return None


class AlpasimConfigDiscoveryPlugin(SearchPathPlugin):
    """Discover and register config search paths from ``alpasim.configs`` entry points."""

    provider = "alpasim-config-discovery"

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        eps = entry_points(group="alpasim.configs")

        for ep in eps:
            path = f"pkg://{ep.value}"
            logger.debug(
                "Auto-registering config search path: %s (from %s)", path, ep.name
            )
            search_path.append(
                provider=f"alpasim-plugin-{ep.name}",
                path=path,
            )

        # Collect every config root directory for duplicate detection.
        # The wizard's own config dir (registered by hydra.main) and any
        # plugin dirs are all on the search path by this point.
        config_dirs: dict[str, Path] = {}
        for element in search_path.get_path():
            resolved = _resolve_search_path_element(element.path)
            if resolved is None or not resolved.is_dir():
                continue
            if element.provider in config_dirs:
                existing = config_dirs[element.provider]
                if resolved != existing:
                    raise ValueError(
                        f"Provider name collision: {element.provider!r} maps to "
                        f"both {existing} and {resolved}. Each alpasim.configs "
                        f"entry point must use a unique name."
                    )
            config_dirs[element.provider] = resolved

        # Fail fast when two providers ship the same relative config path.
        duplicates = find_duplicate_configs(config_dirs)
        if duplicates:
            lines = []
            for rel, providers in sorted(duplicates.items()):
                providers_str = ", ".join(
                    f"{name} ({fpath})" for name, fpath in providers
                )
                lines.append(f"  {rel}: {providers_str}")
            raise ValueError(
                "Duplicate Hydra config files found across config providers. "
                "Each config file path must be unique across all providers:\n"
                + "\n".join(lines)
            )
