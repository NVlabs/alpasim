# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the duplicate config detection in AlpasimConfigDiscoveryPlugin."""

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra_plugins.alpasim_config_discovery import (
    AlpasimConfigDiscoveryPlugin,
    find_duplicate_configs,
)


def _write_yaml(path: Path) -> None:
    """Create a minimal YAML file (and parent dirs) at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# placeholder\n")


class TestFindDuplicateConfigs:
    """Unit tests for :func:`find_duplicate_configs`."""

    def test_no_duplicates(self, tmp_path: Path) -> None:
        """Disjoint config trees should produce no duplicates."""
        wizard = tmp_path / "wizard"
        plugin = tmp_path / "plugin"

        _write_yaml(wizard / "deploy" / "local.yaml")
        _write_yaml(plugin / "deploy" / "remote.yaml")

        duplicates = find_duplicate_configs({"wizard": wizard, "plugin": plugin})
        assert duplicates == {}

    def test_detects_duplicate(self, tmp_path: Path) -> None:
        """Two providers shipping the same relative path must be detected."""
        wizard = tmp_path / "wizard"
        plugin = tmp_path / "plugin"

        _write_yaml(wizard / "deploy" / "local.yaml")
        _write_yaml(plugin / "deploy" / "local.yaml")

        duplicates = find_duplicate_configs({"wizard": wizard, "plugin": plugin})
        assert "deploy/local.yaml" in duplicates
        providers = {name for name, _ in duplicates["deploy/local.yaml"]}
        assert providers == {"wizard", "plugin"}

    def test_detects_nested_duplicate(self, tmp_path: Path) -> None:
        """Duplicates in nested subdirectories are caught."""
        a = tmp_path / "a"
        b = tmp_path / "b"

        _write_yaml(a / "exp" / "presets" / "foo.yaml")
        _write_yaml(b / "exp" / "presets" / "foo.yaml")

        duplicates = find_duplicate_configs({"a": a, "b": b})
        assert "exp/presets/foo.yaml" in duplicates

    def test_missing_directory_is_skipped(self, tmp_path: Path) -> None:
        """A provider whose directory does not exist should not cause errors."""
        wizard = tmp_path / "wizard"
        _write_yaml(wizard / "deploy" / "local.yaml")

        nonexistent = tmp_path / "does_not_exist"
        duplicates = find_duplicate_configs({"wizard": wizard, "missing": nonexistent})
        assert duplicates == {}

    def test_multiple_duplicates(self, tmp_path: Path) -> None:
        """Multiple duplicate files are all reported."""
        a = tmp_path / "a"
        b = tmp_path / "b"

        _write_yaml(a / "deploy" / "local.yaml")
        _write_yaml(b / "deploy" / "local.yaml")
        _write_yaml(a / "driver" / "vavam.yaml")
        _write_yaml(b / "driver" / "vavam.yaml")

        duplicates = find_duplicate_configs({"a": a, "b": b})
        assert len(duplicates) == 2
        assert "deploy/local.yaml" in duplicates
        assert "driver/vavam.yaml" in duplicates


class TestDuplicateConfigIntegration:
    """Integration-style test: simulate what happens when a plugin shadows a wizard config."""

    def test_shadow_copy_raises(self, tmp_path: Path) -> None:
        """Copying an existing wizard YAML into a plugin tree should trigger the check.

        Steps:
        a) Create a plugin config dir with a shadow copy of an existing wizard config.
        b) Verify that find_duplicate_configs catches the overlap.
        c) tmp_path fixture handles cleanup automatically.
        """
        wizard_configs = Path(__file__).resolve().parent.parent / "configs"
        # Pick a real wizard config to shadow.
        real_config = wizard_configs / "deploy" / "local.yaml"
        assert real_config.exists(), f"Expected wizard config at {real_config}"

        # (a) Create a fake plugin tree with the same relative path.
        fake_plugin = tmp_path / "fake_internal" / "configs"
        shadow = fake_plugin / "deploy" / "local.yaml"
        _write_yaml(shadow)

        # (b) The duplicate checker must flag it.
        duplicates = find_duplicate_configs(
            {"wizard": wizard_configs, "fake_internal": fake_plugin}
        )
        assert (
            "deploy/local.yaml" in duplicates
        ), "Shadow copy of deploy/local.yaml was not detected as duplicate"
        providers = {name for name, _ in duplicates["deploy/local.yaml"]}
        assert providers == {"wizard", "fake_internal"}


class TestProviderCollision:
    """Test that duplicate provider names on the search path are caught."""

    def test_same_provider_different_paths_raises(self, tmp_path: Path) -> None:
        """Two search-path entries with the same provider but different dirs must error."""
        from hydra._internal.config_search_path_impl import ConfigSearchPathImpl

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        search_path = ConfigSearchPathImpl()
        search_path.append(provider="my-provider", path=f"file://{dir_a}")
        search_path.append(provider="my-provider", path=f"file://{dir_b}")

        plugin = AlpasimConfigDiscoveryPlugin()
        with patch(
            "hydra_plugins.alpasim_config_discovery.entry_points", return_value=[]
        ):
            with pytest.raises(ValueError, match="Provider name collision"):
                plugin.manipulate_search_path(search_path)
