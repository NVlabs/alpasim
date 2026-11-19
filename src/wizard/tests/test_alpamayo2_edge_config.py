# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
import subprocess
from pathlib import Path

import alpasim_wizard.setup_omegaconf  # noqa: F401
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.plugins import Plugins
from hydra_plugins import alpasim_config_discovery
from omegaconf import OmegaConf


def test_edge_preset_provides_six_cameras_history_and_shared_image(monkeypatch):
    Plugins.instance()
    monkeypatch.setattr(alpasim_config_discovery, "entry_points", lambda **kwargs: [])
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[1] / "configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="base_config",
            overrides=[
                "deploy=local",
                "topology=1gpu",
                "driver=alpamayo2_edge",
                "defines.base_image=alpasim-base:local-unified",
                "+ego_textures=public_carline_transformed",
                "scenes.clip_ids=[clipgt-01d503d4-449b-46fc-8d78-9085e70d3554]",
                "runtime.simulation_config.n_rollouts=1",
                "runtime.simulation_config.n_sim_steps=50",
                "wizard.log_dir=/tmp/edge-config-test",
            ],
        )
    cameras = cfg.runtime.simulation_config.cameras
    assert {camera.logical_id for camera in cameras} == set(
        cfg.driver.inference.use_cameras
    )
    assert len(cameras) == 6
    assert all(camera.frame_interval_us == 100_000 for camera in cameras)
    assert cfg.driver.inference.context_length == 21
    assert cfg.runtime.simulation_config.force_gt_duration_us >= 2_000_000
    assert cfg.runtime.simulation_config.skip_driver_during_force_gt
    assert cfg.runtime.simulation_config.nav_instruction.source == "geometry"
    assert cfg.driver.model.model_type == "alpamayo2_edge"
    assert cfg.driver.model.image_decode_device == "cpu"
    assert not cfg.driver.model.diffusion_expert_cuda_graph
    assert not cfg.services.driver.external_image
    assert cfg.services.driver.image == cfg.services.runtime.image
    assert cfg.services.driver.image == "alpasim-base:local-unified"
    assert cfg.runtime.simulation_config.n_sim_steps == 50
    assert cfg.runtime.simulation_config.n_rollouts == 1


@pytest.mark.parametrize("deployment", ["iad", "ord"])
def test_edge_preserves_deployment_offline_environment(deployment):
    pytest.importorskip("alpasim_internal.configs")
    with initialize_config_dir(
        config_dir=str(Path(__file__).parents[1] / "configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="base_config",
            overrides=[
                f"deploy={deployment}",
                "topology=1gpu",
                "driver=alpamayo2_edge",
                "wizard.log_dir=/tmp/edge-iad-config-test",
            ],
        )
    assert "HF_HUB_OFFLINE=1" in cfg.services.driver.environments
    assert "TRANSFORMERS_OFFLINE=1" in cfg.services.driver.environments
    assert "HF_HOME=/root/.cache/huggingface" in cfg.services.driver.environments


def test_edge_download_menu_uses_the_driver_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[3]
    data = tmp_path / "data"
    data.mkdir()
    script = data / "auto-init.sh"
    script.write_bytes((root / "data/auto-init.sh").read_bytes())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    arguments = tmp_path / "download-args.txt"
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "$ALPASIM_TEST_DOWNLOAD_ARGS"\n'
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        input="1\n5\n\n",  # Deselect the default VaVAM-B; select Edge.
        text=True,
        capture_output=True,
        check=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "ALPASIM_TEST_DOWNLOAD_ARGS": str(arguments),
        },
    )
    model = OmegaConf.load(root / "src/wizard/configs/driver/alpamayo2_edge.yaml").model
    assert "Downloading A2 Edge" in result.stdout
    assert arguments.read_text().splitlines() == [
        "run",
        "hf",
        "download",
        model.checkpoint_path,
    ]
