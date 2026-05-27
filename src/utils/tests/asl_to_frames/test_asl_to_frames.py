# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import numpy as np
import pytest
from alpasim_grpc.v0.egodriver_pb2 import RolloutCameraImage
from alpasim_utils.asl_to_frames.__main__ import (
    determine_save_dir,
    save_frames_as_files,
)


def test_determine_save_dir():
    log_path = "mnt/rollouts/cliggt-hash/rollout.asl"

    # nominal path, log_save_dir unspecified
    save_dir = determine_save_dir(log_path, None)
    expected_save_dir = "mnt/rollouts/cliggt-hash/rollout_asl_frames"
    assert save_dir == expected_save_dir, f"{save_dir=} {expected_save_dir=}"

    # path required by kpi, log_save_dir specified
    save_dir = determine_save_dir(log_path, "mnt/outputs")
    expected_save_dir = "mnt/outputs/rollouts/cliggt-hash/rollout"
    assert save_dir == expected_save_dir, f"{save_dir=} {expected_save_dir=}"


@pytest.mark.asyncio
async def test_save_frames_as_files_uses_supplied_timestamps(monkeypatch, tmp_path):
    written_paths = []

    async def fake_write_image(content: bytes, path: str) -> None:
        del content
        written_paths.append(path)

    monkeypatch.setattr(
        "alpasim_utils.asl_to_frames.__main__._write_image", fake_write_image
    )

    images = [
        RolloutCameraImage.CameraImage(image_bytes=b"first"),
        RolloutCameraImage.CameraImage(image_bytes=b"second"),
    ]
    await save_frames_as_files(
        images,
        np.array([200, 100], dtype=np.uint64),
        str(tmp_path),
    )

    assert sorted(written_paths) == [
        f"{tmp_path}/100",
        f"{tmp_path}/200",
    ]
