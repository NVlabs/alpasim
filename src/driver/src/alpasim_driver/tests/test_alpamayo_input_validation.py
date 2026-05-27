# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import pytest
from alpasim_driver.models.alpamayo_base import _validate_ego_history_span
from alpasim_driver.models.base import ModelInputValidationError
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3


def _pose(timestamp_us: int) -> PoseAtTime:
    return PoseAtTime(
        timestamp_us=timestamp_us,
        pose=Pose(
            vec=Vec3(x=float(timestamp_us) / 1e6, y=0.0, z=0.0),
            quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
    )


def test_validate_ego_history_span_raises_clear_error_for_too_few_poses() -> None:
    with pytest.raises(
        ModelInputValidationError,
        match=(
            r"AlpamayoTest needs at least 2 ego poses spanning 1500\.0ms.*"
            r"latest_camera_frame_us=2"
        ),
    ):
        _validate_ego_history_span(
            [_pose(1_000_000)],
            latest_camera_frame_us=2_000_000,
            num_history_steps=16,
            history_time_step=0.1,
            model_name="AlpamayoTest",
        )


def test_validate_ego_history_span_raises_clear_error_for_short_span() -> None:
    with pytest.raises(
        ModelInputValidationError,
        match=(
            r"AlpamayoTest ego pose history is too short: "
            r"available_span=500\.0ms, required_span=1500\.0ms"
        ),
    ):
        _validate_ego_history_span(
            [_pose(1_500_000), _pose(2_000_000)],
            latest_camera_frame_us=2_000_000,
            num_history_steps=16,
            history_time_step=0.1,
            model_name="AlpamayoTest",
        )


def test_validate_ego_history_span_raises_clear_error_for_stale_history() -> None:
    with pytest.raises(
        ModelInputValidationError,
        match=(
            r"AlpamayoTest ego pose history is stale: "
            r"available_span=1500\.0ms, required_span=1500\.0ms, "
            r"latest_available_us=1500000, latest_camera_frame_us=2000000"
        ),
    ):
        _validate_ego_history_span(
            [_pose(0), _pose(1_500_000)],
            latest_camera_frame_us=2_000_000,
            num_history_steps=16,
            history_time_step=0.1,
            model_name="AlpamayoTest",
        )


def test_validate_ego_history_span_accepts_required_span() -> None:
    _validate_ego_history_span(
        [_pose(500_000), _pose(2_000_000)],
        latest_camera_frame_us=2_000_000,
        num_history_steps=16,
        history_time_step=0.1,
        model_name="AlpamayoTest",
    )
