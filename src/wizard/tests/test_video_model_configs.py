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


def test_vavam_requests_direct_pinhole_rendering() -> None:
    cfg = _compose_config(
        "deploy=local",
        "topology=1gpu",
        "driver=vavam",
        "wizard.log_dir=/tmp/alpasim-test",
    )

    assert "rectification" not in cfg.driver
    assert len(cfg.runtime.extra_cameras) == 1
    camera = cfg.runtime.extra_cameras[0]
    assert camera.logical_id == "camera_front_wide_120fov"
    assert camera.intrinsics.model == "opencv_pinhole"


def test_alpamayo2_driver_config_excludes_service_settings() -> None:
    cfg = _compose_config(
        "deploy=local",
        "topology=1gpu",
        "driver=alpamayo2",
        "wizard.log_dir=/tmp/alpasim-test",
    )

    assert "services" not in cfg.driver


def test_vavam_video_model_rectifies_recorded_ftheta_images() -> None:
    cfg = _compose_config(
        "deploy=local",
        "topology=1gpu",
        "driver=vavam_video_model",
        "wizard.log_dir=/tmp/alpasim-test",
    )

    assert cfg.runtime.extra_cameras == []
    target = cfg.driver.rectification.camera_front_wide_120fov
    assert target.focal_length == [1545.0, 1545.0]
    assert target.principal_point == [960.0, 560.0]
    assert target.resolution_hw == [1080, 1920]


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
