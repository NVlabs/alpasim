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
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

logger = logging.getLogger(__name__)


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
