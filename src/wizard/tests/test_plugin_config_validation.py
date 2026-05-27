# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Tests for wizard-side validation of plugin-owned config sections.

Stage 5 of the renderer plugin refactor: the wizard validates
``runtime.renderer_config`` against the active plugin's typed schema (exposed
via ``alpasim.services``) at config-generation time, so bad configs fail the
wizard rather than surfacing at SLURM-worker startup minutes later.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest
from alpasim_wizard.configuration import ConfigurationManager
from omegaconf import OmegaConf


@dataclass
class _FakeRendererConfig:
    fps: int = 30
    label: str = "demo"


class _FakeServiceWithSchema:
    @classmethod
    def get_config_schema(cls) -> type:
        return _FakeRendererConfig


class _FakeRegistry:
    mapping: ClassVar[dict[str, object]] = {}

    def __init__(self, group: str):
        assert group == "alpasim.services"

    def get(self, name: str):
        if name not in self.mapping:
            raise LookupError(name)
        return self.mapping[name]


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[str, object]
) -> None:
    from tests.conftest import patch_plugin_registry

    _FakeRegistry.mapping = mapping
    patch_plugin_registry(monkeypatch, _FakeRegistry)


def _make_cfg(renderer_type: str, renderer_config: dict | None) -> SimpleNamespace:
    runtime = OmegaConf.create(
        {"renderer_config": renderer_config} if renderer_config is not None else {}
    )
    wizard = SimpleNamespace(renderer_type=renderer_type)
    return SimpleNamespace(runtime=runtime, wizard=wizard)


def test_wizard_validates_plugin_renderer_config_accepts_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_registry(monkeypatch, {"video_model": _FakeServiceWithSchema})
    mgr = ConfigurationManager(log_dir=str(tmp_path))
    cfg = _make_cfg("video_model", {"fps": 24})
    # Should not raise.
    mgr._validate_plugin_configs(cfg)


def test_wizard_rejects_invalid_plugin_renderer_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_registry(monkeypatch, {"video_model": _FakeServiceWithSchema})
    mgr = ConfigurationManager(log_dir=str(tmp_path))
    cfg = _make_cfg("video_model", {"definitely_not_a_field": "oops"})
    with pytest.raises(ValueError, match="renderer_type='video_model'"):
        mgr._validate_plugin_configs(cfg)


def test_wizard_sensorsim_renderer_config_passes_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    # No registry lookup for the default sensorsim path.
    _patch_registry(monkeypatch, {})
    mgr = ConfigurationManager(log_dir=str(tmp_path))
    cfg = _make_cfg("sensorsim", {"arbitrary": 1})
    mgr._validate_plugin_configs(cfg)  # no raise


def test_wizard_missing_renderer_config_ok(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _patch_registry(monkeypatch, {"video_model": _FakeServiceWithSchema})
    mgr = ConfigurationManager(log_dir=str(tmp_path))
    cfg = _make_cfg("video_model", None)
    mgr._validate_plugin_configs(cfg)  # no raise
