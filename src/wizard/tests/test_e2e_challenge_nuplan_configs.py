# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Contract tests for the public nuPlan/MTGS challenge presets."""

from __future__ import annotations

from pathlib import Path

import alpasim_wizard.setup_omegaconf  # noqa: F401
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def _compose_config(*overrides: str):
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="base_config.yaml", overrides=list(overrides))


def test_full_preset_is_the_standard_navtest_evaluation(monkeypatch) -> None:
    monkeypatch.setenv("ALPASIM_NUPLAN_ROOT", "/tmp/alpasim-nuplan-track")

    cfg = _compose_config("+e2e_challenge_nuplan=full")

    assert cfg.runtime.scene_provider.kind == "trajdata"
    assert cfg.runtime.scene_provider.trajdata.dataset.name == "nuplan_test"
    assert cfg.runtime.endpoints.renderer.skip is False
    assert cfg.scenes.test_suite_id is None
    assert cfg.scenes.limit_to_first_n == 0
    assert len(cfg.scenes.scene_ids) == 1485
    assert len(set(cfg.scenes.scene_ids)) == len(cfg.scenes.scene_ids)
    assert cfg.eval.scene_score.enabled is True


def test_dev_scenes_are_available_in_full_navtest(monkeypatch) -> None:
    monkeypatch.setenv("ALPASIM_NUPLAN_ROOT", "/tmp/alpasim-nuplan-track")

    dev_cfg = _compose_config("+e2e_challenge_nuplan=dev")
    full_cfg = _compose_config("+e2e_challenge_nuplan=full")

    assert dev_cfg.scenes.limit_to_first_n == 1
    assert set(dev_cfg.scenes.scene_ids) <= set(full_cfg.scenes.scene_ids)


def test_documented_local_challenge_commands_compose(monkeypatch) -> None:
    """Keep the three commands published in the challenge README valid."""
    monkeypatch.setenv("ALPASIM_NUPLAN_ROOT", "/tmp/alpasim-nuplan-track")

    pai_smoke = _compose_config(
        "+e2e_challenge=dev",
        "wizard.log_dir=./runs/e2e_challenge_pai_smoke",
    )
    nuplan_smoke = _compose_config(
        "+e2e_challenge_nuplan=dev",
        "wizard.log_dir=./runs/e2e_challenge_nuplan_smoke",
    )
    nuplan_full = _compose_config(
        "+e2e_challenge_nuplan=full",
        "wizard.log_dir=./runs/e2e_challenge_nuplan_navtest",
    )

    assert pai_smoke.runtime.scene_provider.kind == "usdz"
    assert nuplan_smoke.runtime.scene_provider.kind == "trajdata"
    assert nuplan_smoke.scenes.limit_to_first_n == 1
    assert nuplan_full.runtime.scene_provider.kind == "trajdata"
    assert nuplan_full.scenes.limit_to_first_n == 0
