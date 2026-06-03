# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

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


def test_standard_deploy_keeps_controller_enabled_after_topology_overrides() -> None:
    cfg = _compose_config(
        "deploy=local",
        "topology=1gpu",
        "driver=vavam",
        "wizard.log_dir=/tmp/alpasim-test",
    )

    assert cfg.runtime.endpoints.controller.skip is False
    assert cfg.runtime.endpoints.controller.n_concurrent_rollouts == 4


def test_managed_flashdreams_uses_local_image_without_pulling() -> None:
    cfg = _compose_config(
        "deploy=managed_flashdreams",
        "topology=1gpu",
        "driver=alpamayo1_5_1cam",
        "+chunking=8frame",
        "wizard.log_dir=/tmp/alpasim-test",
    )

    assert cfg.services.renderer.external_image is True
    assert cfg.services.renderer.pull_policy == "never"
