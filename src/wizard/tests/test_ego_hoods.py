# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

import alpasim_wizard.setup_omegaconf  # noqa: F401
import numpy as np
import pytest
import yaml
from alpasim_utils.types import ClipDataSource, Scene
from alpasim_wizard import ego_hoods
from alpasim_wizard.context import slurm_array_task_scenes
from alpasim_wizard.ego_hoods import (
    FthetaCamera,
    generate_ego_hood_banks,
    load_image,
    warp,
)
from hydra import compose, initialize_config_dir
from hydra.core.plugins import Plugins
from hydra_plugins import alpasim_config_discovery
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "data/ego-hood-transforms/public_carline.yaml"
CARLINES = str(ROOT / "data/scenes/ego_textures/public_carline.yaml")
FRONT = "camera_front_wide_120fov"
TRANSFORMED = "carline_002"
UNLISTED = "carline_009"  # a carline the spec does not list


def _spec_camera(carline: str) -> FthetaCamera:
    return FthetaCamera.from_dict(
        yaml.safe_load(SPEC.read_text())["sources"][carline]["camera"]
    )


def _write_usdz(path: Path, camera: FthetaCamera, rotate_deg: float = 0.0) -> None:
    """A USDZ holding only the front-wide calibration, optionally rolled about the optical axis."""
    a = np.asarray(camera.A)
    t = np.eye(4)
    c, s = np.cos(np.radians(rotate_deg)), np.sin(np.radians(rotate_deg))
    t[:3, :3] = np.asarray(camera.R) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    calibration = {
        "camera_model": {
            "type": "ftheta",
            "parameters": {
                "resolution": [1920, 1080],
                "principal_point": [camera.cx, camera.cy],
                "pixeldist_to_angle_poly": list(camera.pixel_to_angle),
                "angle_to_pixeldist_poly": list(camera.angle_to_pixel),
                "linear_cde": [a[0, 0], a[0, 1], a[1, 0]],
            },
        },
        "T_sensor_rig": t.tolist(),
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "rig_trajectories.json",
            json.dumps({"camera_calibrations": {f"{FRONT}@sensor": calibration}}),
        )


def test_usdz_calibration_round_trips_and_same_camera_warp_is_identity(tmp_path):
    camera = _spec_camera(TRANSFORMED)
    _write_usdz(tmp_path / "scene.usdz", camera)
    loaded = FthetaCamera.from_usdz(str(tmp_path / "scene.usdz"), FRONT)
    np.testing.assert_allclose(np.asarray(loaded.R), np.asarray(camera.R))
    np.testing.assert_allclose(np.asarray(loaded.A), np.asarray(camera.A))
    np.testing.assert_allclose([loaded.cx, loaded.cy], [camera.cx, camera.cy])

    # The lens polynomials are not exact inverses, so a same-camera warp moves pixels by
    # a fraction of a pixel rather than none.
    v, u = np.mgrid[0:1080, 0:1920].astype(float)
    rig = loaded.rays(u, v) @ np.asarray(loaded.R).T
    x, y = camera.project(rig @ np.asarray(camera.R))
    assert np.hypot(x - u, y - v).max() < 0.2

    image = load_image(
        str(ROOT / "data/nre-artifacts/ego-hoods" / TRANSFORMED / f"{FRONT}.png")
    )
    assert np.abs(warp(image, camera, loaded).astype(int) - image).mean() < 0.1


def test_generates_transformed_banks_once_and_keeps_other_carlines(tmp_path, caplog):
    base = yaml.safe_load(
        (ROOT / "data/scenes/ego_textures/public_carline.yaml").read_text()
    )
    transformed_clip = next(
        k for k, v in base.items() if v["rig_config_id"] == TRANSFORMED
    )
    unlisted_clip = next(k for k, v in base.items() if v["rig_config_id"] == UNLISTED)
    _write_usdz(
        tmp_path / "transformed.usdz", _spec_camera(TRANSFORMED), rotate_deg=2.0
    )
    _write_usdz(tmp_path / "unlisted.usdz", _spec_camera(TRANSFORMED))
    banks = tmp_path / "out/banks"
    paths = {
        transformed_clip: str(tmp_path / "transformed.usdz"),
        unlisted_clip: str(tmp_path / "unlisted.usdz"),
        "clipgt-not-in-public-carline": str(tmp_path / "unlisted.usdz"),
    }
    overrides_file = tmp_path / "log/ego_mask_overrides.yaml"
    with caplog.at_level(logging.WARNING):
        generate_ego_hood_banks(
            [str(SPEC)], CARLINES, str(banks), paths, str(overrides_file)
        )
    assert "clipgt-not-in-public-carline" in caplog.text

    overrides = yaml.safe_load(overrides_file.read_text())
    assert set(overrides) == {transformed_clip, unlisted_clip}
    unlisted_bank = banks / overrides[unlisted_clip]["rig_config_id"]
    assert unlisted_bank.name.startswith(f"{UNLISTED}_")
    bank = banks / overrides[transformed_clip]["rig_config_id"]
    assert bank.name.startswith(f"transformed_{TRANSFORMED}_")
    stock = ROOT / "data/nre-artifacts/ego-hoods" / TRANSFORMED
    for camera in overrides[transformed_clip]["camera_logical_ids"]:
        assert (bank / f"{camera}.png").is_file()
        assert (unlisted_bank / f"{camera}.png").is_file()
        if camera != FRONT:
            assert (bank / f"{camera}.png").read_bytes() == (
                stock / f"{camera}.png"
            ).read_bytes()
    front = np.asarray(Image.open(bank / f"{FRONT}.png"))
    assert front.shape == (1080, 1920, 4)
    # A 2 degree roll moves the hood: the transformed image differs from the source.
    assert (front != load_image(str(stock / f"{FRONT}.png"))).any()

    mtime = (bank / f"{FRONT}.png").stat().st_mtime_ns
    banks_mtime = banks.stat().st_mtime_ns
    generate_ego_hood_banks(
        [str(SPEC)], CARLINES, str(banks), paths, str(overrides_file)
    )
    assert (bank / f"{FRONT}.png").stat().st_mtime_ns == mtime
    assert banks.stat().st_mtime_ns == banks_mtime
    assert not list(banks.glob(".tmp-*"))

    # The renderer's hood index (<banks>/.egocar_hood_index) is current only while the
    # banks directory's mtime is unchanged, so a new bank must change it.
    other_clip = next(
        k
        for k, v in base.items()
        if v["rig_config_id"] == TRANSFORMED and k != transformed_clip
    )
    _write_usdz(tmp_path / "other.usdz", _spec_camera(TRANSFORMED), rotate_deg=3.0)
    os.utime(banks, ns=(banks_mtime - 10**10, banks_mtime - 10**10))
    generate_ego_hood_banks(
        [str(SPEC)],
        CARLINES,
        str(banks),
        {other_clip: str(tmp_path / "other.usdz")},
        str(overrides_file),
    )
    assert banks.stat().st_mtime_ns != banks_mtime - 10**10
    for path in [bank, *bank.iterdir(), unlisted_bank]:
        assert path.stat().st_mode & 0o020, f"{path} is not group-writable"


@pytest.mark.parametrize("carline", [TRANSFORMED, UNLISTED])
def test_warns_when_front_wide_reprojection_is_unavailable(tmp_path, caplog, carline):
    base = yaml.safe_load(Path(CARLINES).read_text())
    clip = next(k for k, v in base.items() if v["rig_config_id"] == carline)
    usdz = tmp_path / "scene.usdz"
    _write_usdz(usdz, _spec_camera(TRANSFORMED))
    for _ in range(2):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            generate_ego_hood_banks(
                [str(SPEC)],
                CARLINES,
                str(tmp_path / "banks"),
                {clip: str(usdz)},
                str(tmp_path / "overrides.yaml"),
            )
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        if carline == UNLISTED:
            assert len(warnings) == 1
            assert "1 of 1 selected clips use unchanged textures" in warnings[0]
            assert carline in warnings[0]
            assert str(SPEC) in warnings[0]
            assert "may not align with the scene camera" in warnings[0]
        else:
            assert not warnings


def test_warns_when_the_banks_directory_holds_many_unused_banks(tmp_path, caplog):
    base = yaml.safe_load(Path(CARLINES).read_text())
    clip = next(k for k, v in base.items() if v["rig_config_id"] == TRANSFORMED)
    _write_usdz(tmp_path / "scene.usdz", _spec_camera(TRANSFORMED))
    banks = tmp_path / "banks"
    for i in range(2):
        (banks / f"stale_{i}").mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        generate_ego_hood_banks(
            [str(SPEC)],
            CARLINES,
            str(banks),
            {clip: str(tmp_path / "scene.usdz")},
            str(tmp_path / "overrides.yaml"),
        )
    assert "holds 3 banks, of which this run uses 1" in caplog.text


def test_changed_carline_bank_gets_new_banks(tmp_path):
    base = yaml.safe_load(
        (ROOT / "data/scenes/ego_textures/public_carline.yaml").read_text()
    )
    transformed_clip = next(
        k for k, v in base.items() if v["rig_config_id"] == TRANSFORMED
    )
    unlisted_clip = next(k for k, v in base.items() if v["rig_config_id"] == UNLISTED)
    _write_usdz(tmp_path / "transformed.usdz", _spec_camera(TRANSFORMED))
    base_banks = tmp_path / "base_banks"
    for carline in (TRANSFORMED, UNLISTED):
        shutil.copytree(
            ROOT / "data/nre-artifacts/ego-hoods" / carline, base_banks / carline
        )
    spec = yaml.safe_load(SPEC.read_text())
    spec["banks"] = str(base_banks)
    for entry in spec["sources"].values():
        entry["image"] = str((SPEC.parent / entry["image"]).resolve())
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    banks = tmp_path / "out/banks"
    paths = {
        transformed_clip: str(tmp_path / "transformed.usdz"),
        unlisted_clip: str(tmp_path / "transformed.usdz"),
    }
    overrides_file = tmp_path / "ego_mask_overrides.yaml"
    before = generate_ego_hood_banks(
        [str(spec_path)], CARLINES, str(banks), paths, str(overrides_file)
    )

    for carline in (TRANSFORMED, UNLISTED):
        (base_banks / carline / "camera_rear_left_70fov.png").write_bytes(b"changed")
    after = generate_ego_hood_banks(
        [str(spec_path)], CARLINES, str(banks), paths, str(overrides_file)
    )
    for clip in paths:
        new_bank = banks / after[clip]["rig_config_id"]
        assert after[clip]["rig_config_id"] != before[clip]["rig_config_id"]
        assert (new_bank / "camera_rear_left_70fov.png").read_bytes() == b"changed"


def test_render_version_change_gets_new_banks(tmp_path, monkeypatch):
    base = yaml.safe_load(
        (ROOT / "data/scenes/ego_textures/public_carline.yaml").read_text()
    )
    transformed_clip = next(
        k for k, v in base.items() if v["rig_config_id"] == TRANSFORMED
    )
    _write_usdz(tmp_path / "transformed.usdz", _spec_camera(TRANSFORMED))
    banks = tmp_path / "banks"
    paths = {transformed_clip: str(tmp_path / "transformed.usdz")}
    overrides_file = str(tmp_path / "ego_mask_overrides.yaml")
    before = generate_ego_hood_banks(
        [str(SPEC)], CARLINES, str(banks), paths, overrides_file
    )
    monkeypatch.setattr(ego_hoods, "RENDER_VERSION", ego_hoods.RENDER_VERSION + 1)
    after = generate_ego_hood_banks(
        [str(SPEC)], CARLINES, str(banks), paths, overrides_file
    )
    bank = after[transformed_clip]["rig_config_id"]
    assert bank != before[transformed_clip]["rig_config_id"]
    assert (banks / bank / f"{FRONT}.png").is_file()


def test_public_carline_transformed_preset_mounts_generated_banks(monkeypatch):
    Plugins.instance()
    monkeypatch.setattr(alpasim_config_discovery, "entry_points", lambda **kwargs: [])
    with initialize_config_dir(
        config_dir=str(ROOT / "src/wizard/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="base_config",
            overrides=[
                "deploy=local",
                "topology=1gpu",
                "driver=alpamayo1_5",
                "wizard.log_dir=/tmp/alpasim-transformed-test",
                "+ego_textures=public_carline_transformed",
            ],
        )
    banks = cfg.ego_hoods.banks
    assert banks == str(ROOT / "data/ego-textures/public_carline_transformed/banks")
    assert f"{banks}:/mnt/ego-hoods" in cfg.services.renderer.volumes
    assert cfg.ego_hoods.carlines == CARLINES
    assert (
        cfg.runtime.simulation_config.ego_mask_overrides_file
        == "/mnt/log_dir/ego_mask_overrides.yaml"
    )
    assert "/tmp/alpasim-transformed-test:/mnt/log_dir" in cfg.services.runtime.volumes
    assert list(cfg.ego_hoods.transforms) == [str(SPEC)]
    spec = yaml.safe_load(SPEC.read_text())
    for carline, entry in spec["sources"].items():
        assert (SPEC.parent / entry["image"]).is_file()
        assert (SPEC.parent / spec["banks"] / carline).is_dir()


def test_slurm_array_tasks_share_the_scenes(monkeypatch):
    """Each array task makes banks for, and runs, a disjoint share of the scenes."""
    scenes = [
        Scene(ClipDataSource(clip_id=f"clipgt-{i:08d}", usdz_uuid=str(i)), t0)
        for i in (3, 1, 2, 0, 4)
        for t0 in ("AUTO", 5_000_000)
    ]
    assert slurm_array_task_scenes(scenes) == scenes
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "3")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "1")
    shares = []
    for task in (1, 2, 3):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task))
        shares.append(slurm_array_task_scenes(scenes))
    assert sorted(map(len, shares)) == [3, 3, 4]
    assert sorted(s.scene_key for share in shares for s in share) == sorted(
        s.scene_key for s in scenes
    )


def test_specs_must_not_repeat_a_carline(tmp_path):
    with pytest.raises(ValueError, match="carline_002"):
        generate_ego_hood_banks(
            [str(SPEC), str(SPEC)],
            CARLINES,
            str(tmp_path),
            {},
            str(tmp_path / "o.yaml"),
        )
