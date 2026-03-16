# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for the alpasim-info CLI (Phase 1 extensible framework)."""

import pytest
from alpasim_plugins.info import main as info_main
from alpasim_plugins.plugins import get_plugin_info


def test_get_plugin_info_structure() -> None:
    """get_plugin_info() returns dict of group -> list of names."""
    info = get_plugin_info()
    assert "alpasim.models" in info
    assert "alpasim.mpc" in info
    assert all(isinstance(v, list) for v in info.values())
    assert all(all(isinstance(x, str) for x in v) for v in info.values())


def test_info_main_prints_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """alpasim_plugins.info.main() prints lines to stdout with expected labels."""
    info_main()
    captured = capsys.readouterr()
    out = captured.out
    assert "Models:" in out
    assert "MPC:" in out
    assert "Scorers:" in out
    assert "Tools:" in out
    assert "Configs:" in out


def test_info_main_includes_models_when_driver_installed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When driver is installed, 'Models:' line lists ar1, manual, vam."""
    info_main()
    out = capsys.readouterr().out
    for line in out.splitlines():
        if line.startswith("Models:"):
            models_line = line
            break
    else:
        pytest.fail("No 'Models:' line in output")
    for name in ("ar1", "manual", "vam"):
        assert name in models_line, f"Expected {name} in Models line: {models_line}"


def test_info_main_includes_mpc_when_controller_installed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When controller is installed, 'MPC:' line lists linear and nonlinear."""
    info_main()
    out = capsys.readouterr().out
    for line in out.splitlines():
        if line.startswith("MPC:"):
            assert "linear" in line
            assert "nonlinear" in line
            break
    else:
        pytest.fail("No 'MPC:' line in output")
