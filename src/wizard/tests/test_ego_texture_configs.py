# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import csv
from pathlib import Path

import alpasim_wizard.setup_omegaconf  # noqa: F401
import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.plugins import Plugins
from hydra_plugins import alpasim_config_discovery
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/scenes/ego_textures"
HOODS = ROOT / "data/nre-artifacts/ego-hoods"
CAMERAS = {
    "camera_front_wide_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
}


@pytest.mark.parametrize(
    "suite",
    [path.stem for path in sorted((ROOT / "data/scenes/scene_suites").rglob("*.csv"))],
)
def test_public_carline_transformed_preset_keeps_vehicle_and_cameras(
    monkeypatch, suite
):
    # Public presets must compose without any optional config plugins installed.
    Plugins.instance()
    monkeypatch.setattr(alpasim_config_discovery, "entry_points", lambda **kwargs: [])
    overrides = [
        "deploy=local",
        "topology=1gpu",
        "driver=alpamayo1_5",
        "wizard.log_dir=/tmp/alpasim-carline-test",
        f"scenes.suite_ids=[{suite}]",
    ]
    with initialize_config_dir(
        config_dir=str(ROOT / "src/wizard/configs"), version_base="1.3"
    ):
        base = compose(config_name="base_config", overrides=overrides)
        cfg = compose(
            config_name="base_config",
            overrides=overrides + ["+ego_textures=public_carline_transformed"],
        )
    assert set(base.services.runtime.volumes) <= set(cfg.services.runtime.volumes)
    assert cfg.runtime.simulation_config.ego_mask_rig_config_id is None
    configured = OmegaConf.to_container(cfg.runtime.simulation_config)
    original = OmegaConf.to_container(base.runtime.simulation_config)
    for key in ["ego_mask_rig_config_id", "ego_mask_overrides_file"]:
        configured.pop(key)
        original.pop(key)
    assert configured == original


def test_public_carline_selection_covers_all_public_catalogs_and_suites():
    selections = yaml.safe_load((DATA / "public_carline.yaml").read_text())
    catalogs = set()
    for path in (ROOT / "data/scenes").glob("clip_catalog*.csv"):
        with path.open() as stream:
            catalogs.update(
                row["clip_id"]
                for row in csv.DictReader(stream)
                if row["clip_id"].startswith("clipgt-")
            )
    assert selections.keys() == catalogs
    for path in (ROOT / "data/scenes/scene_suites").rglob("*.csv"):
        with path.open() as stream:
            clips = {
                row["clip_id"]
                for row in csv.DictReader(stream)
                if row["clip_id"].startswith("clipgt-")
            }
        assert clips <= selections.keys(), path
    for selection in selections.values():
        assert set(selection["camera_logical_ids"]) == CAMERAS
        for camera in CAMERAS:
            assert (HOODS / selection["rig_config_id"] / f"{camera}.png").is_file()


def test_bundled_carline_banks_contain_only_requested_images():
    selections = yaml.safe_load((DATA / "public_carline.yaml").read_text())
    assert {selection["rig_config_id"] for selection in selections.values()} <= {
        p.parent.name for p in HOODS.glob("*/source.json")
    }
    for source in HOODS.glob("*/source.json"):
        assert {p.stem for p in source.parent.glob("*.png")} == CAMERAS
        for path in source.parent.glob("*.png"):
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                assert image.mode == "RGBA"
                assert image.getchannel("A").getextrema()[1] > 0
