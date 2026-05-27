# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import sys
import types
from pathlib import Path

import pytest

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(parent_dir))


def patch_plugin_registry(monkeypatch: pytest.MonkeyPatch, registry_cls: type) -> None:
    """Patch ``alpasim_plugins.PluginRegistry`` for tests.

    The wizard's plugin-config validator does a deferred import of
    ``alpasim_plugins.PluginRegistry`` at submit time, but the wizard
    package's CI test env does not have ``alpasim-plugins`` installed.
    Mirrors the helper in ``src/runtime/tests/conftest.py``.
    """
    if "alpasim_plugins" in sys.modules:
        monkeypatch.setattr("alpasim_plugins.PluginRegistry", registry_cls)
        return
    fake = types.ModuleType("alpasim_plugins")
    fake.PluginRegistry = registry_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "alpasim_plugins", fake)
