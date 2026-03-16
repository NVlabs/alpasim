# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Plugin system for alpasim extensibility.

This module provides the infrastructure for discovering and loading
plugins (project components) from installed packages.

Plugins register themselves via Python entry points in their pyproject.toml.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PluginNotFoundError(ValueError):
    """Raised when a requested plugin is not found in the registry."""


class PluginRegistry:
    """Generic registry for discovering plugins via entry points.

    Usage:
        model_registry = PluginRegistry("alpasim.models")
        models = model_registry.get_available()  # {"ar1": <class>, "manual": <class>}
        model = model_registry.create("ar1", checkpoint="...", device="cuda")
    """

    def __init__(self, group: str) -> None:
        """Initialize registry for an entry point group.

        Args:
            group: Entry point group name (e.g., "alpasim.models")
        """
        self._group = group
        self._cache: dict[str, Any] | None = None

    def get_available(self) -> dict[str, Any]:
        """Get all available plugins in this group.

        Returns:
            Dict mapping plugin name to loaded class/factory.
        """
        if self._cache is None:
            self._cache = {}
            eps = entry_points(group=self._group)
            for ep in eps:
                if ep.name in self._cache:
                    raise ValueError(
                        f"Duplicate entry point '{ep.name}' in group "
                        f"'{self._group}': provided by both "
                        f"{self._cache[ep.name].__module__} and {ep.value}"
                    )
                try:
                    self._cache[ep.name] = ep.load()
                    logger.debug("Loaded plugin %s:%s", self._group, ep.name)
                except Exception as e:
                    logger.warning("Failed to load %s:%s: %s", self._group, ep.name, e)
        return self._cache

    def get_names(self) -> list[str]:
        """Get names of available plugins."""
        return sorted(self.get_available().keys())

    def is_available(self, name: str) -> bool:
        """Check if a plugin is available."""
        return name in self.get_available()

    def get(self, name: str) -> Any:
        """Get a plugin class/factory by name.

        Raises:
            PluginNotFoundError: If plugin not found.
        """
        available = self.get_available()
        if name not in available:
            raise PluginNotFoundError(
                f"Plugin '{name}' not found in {self._group}. "
                f"Available: {self.get_names()}. "
                "Install a package that provides this plugin."
            )
        return available[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a plugin instance.

        Args:
            name: Plugin name.
            **kwargs: Arguments to pass to the plugin factory/class.

        Returns:
            Plugin instance.
        """
        factory = self.get(name)
        return factory(**kwargs)


# Pre-defined registries for common plugin types
models = PluginRegistry("alpasim.models")
mpc_controllers = PluginRegistry("alpasim.mpc")
scorers = PluginRegistry("alpasim.scorers")
tools = PluginRegistry("alpasim.tools")


def get_plugin_info() -> dict[str, list[str]]:
    """Get summary of all installed plugins.

    Returns:
        Dict mapping plugin group to list of available names.
    """
    groups = [
        "alpasim.models",
        "alpasim.mpc",
        "alpasim.scorers",
        "alpasim.tools",
        "alpasim.configs",
    ]
    return {group: PluginRegistry(group).get_names() for group in groups}
