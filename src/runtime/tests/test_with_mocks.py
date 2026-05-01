# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import asyncio
from pathlib import Path

import pytest
import yaml
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_runtime.simulate.__main__ import create_arg_parser, run_simulation


@pytest.mark.asyncio
async def test_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path):
    async def fake_get_available_cameras(self, scene_id: str):
        del scene_id  # skip-specific scenes ignored in mock mode
        cameras = []
        for logical_id in (
            "camera_front_wide_120fov",
            "camera_front_tele_30fov",
        ):
            camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
                logical_id=logical_id,
                intrinsics=sensorsim_pb2.CameraSpec(
                    logical_id=logical_id,
                    shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
                ),
            )
            camera.rig_to_camera.quat.w = 1.0
            cameras.append(camera)
        return cameras

    monkeypatch.setattr(
        "alpasim_runtime.services.sensorsim_service.SensorsimService.get_available_cameras",
        fake_get_available_cameras,
    )

    mock_data_dir = Path(__file__).parent / "data" / "mock"

    # Create required run_metadata.yaml for get_run_name()
    run_metadata = tmp_path / "run_metadata.yaml"
    run_metadata.write_text("run_name: test_mocks\n")

    user_config_src = mock_data_dir / "user-config.yaml"
    user_config = yaml.safe_load(user_config_src.read_text())
    user_config["simulation_config"]["n_sim_steps"] = 1
    user_config["scene_provider"] = {
        "kind": "usdz",
        "usdz": {
            "data_dir": str(mock_data_dir),
        },
        "trajdata": {
            "cache_location": str(tmp_path / "trajdata-cache"),
            "desired_dt": 0.1,
            "load_vector_map": True,
            "rebuild_cache": False,
            "rebuild_maps": False,
            "num_workers": 1,
            "dataset": None,
        },
    }
    user_config_path = tmp_path / "user-config.yaml"
    user_config_path.write_text(yaml.safe_dump(user_config))

    eval_config = yaml.safe_load((mock_data_dir / "eval-config.yaml").read_text())
    eval_config["enabled"] = False
    eval_config_path = tmp_path / "eval-config.yaml"
    eval_config_path.write_text(yaml.safe_dump(eval_config))

    parser = create_arg_parser()
    parsed_args = parser.parse_args(
        [
            f"--user-config={user_config_path}",
            f"--network-config={mock_data_dir / 'network-config.yaml'}",
            f"--eval-config={eval_config_path}",
            f"--log-dir={tmp_path}",
        ]
    )
    success = await asyncio.wait_for(run_simulation(parsed_args), timeout=90)
    assert success
