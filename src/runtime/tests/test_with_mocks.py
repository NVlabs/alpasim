# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import asyncio

import pytest
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
            cameras.append(
                sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
                    logical_id=logical_id,
                    intrinsics=sensorsim_pb2.CameraSpec(
                        logical_id=logical_id,
                        shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
                    ),
                )
            )
        return cameras

    monkeypatch.setattr(
        "alpasim_runtime.services.sensorsim_service.SensorsimService.get_available_cameras",
        fake_get_available_cameras,
    )

    # Create required run_metadata.yaml for get_run_name()
    run_metadata = tmp_path / "run_metadata.yaml"
    run_metadata.write_text("run_name: test_mocks\n")

    parser = create_arg_parser()
    parsed_args = parser.parse_args(
        [
            "--user-config=tests/data/mock/user-config.yaml",
            "--network-config=tests/data/mock/network-config.yaml",
            "--eval-config=tests/data/mock/eval-config.yaml",
            "--usdz-glob=tests/data/**/*.usdz",
            f"--log-dir={tmp_path}",
        ]
    )
    success = await asyncio.wait_for(run_simulation(parsed_args), timeout=35)
    assert success
