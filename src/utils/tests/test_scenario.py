# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import json

import pytest
from alpasim_utils.scenario import Rig


def _identity_matrix() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rig_json(cameras_frame_timestamps_us: dict | None) -> str:
    trajectory = {
        "sequence_id": "clipgt-test",
        "T_rig_world_timestamps_us": [1_000, 2_000],
        "T_rig_worlds": [_identity_matrix(), _identity_matrix()],
        "rig_bbox": None,
    }
    if cameras_frame_timestamps_us is not None:
        trajectory["cameras_frame_timestamps_us"] = cameras_frame_timestamps_us

    return json.dumps(
        {
            "world_to_nre": {"matrix": _identity_matrix()},
            "camera_calibrations": {
                "unique-front": {"logical_sensor_name": "camera_front"},
                "unique-left": {"logical_sensor_name": "camera_left"},
            },
            "rig_trajectories": [trajectory],
        }
    )


def test_rig_loads_camera_frame_ranges() -> None:
    (rig,) = Rig.load_from_json(
        _rig_json(
            {
                "unique-front": [[-1, 1_200], [-1, 1_300]],
                "unique-left": [[-1, 1_100], [-1, 1_400]],
            }
        )
    )

    assert rig.camera_frame_timestamps_us == {
        "unique-front": [1_200, 1_300],
        "unique-left": [1_100, 1_400],
    }
    assert rig.camera_frame_ranges_us["unique-front"][0] == range(-1, 1_200)
    assert rig.camera_frame_ranges_us["unique-left"][0] == range(-1, 1_100)
    assert rig.first_camera_frame_ranges_us(["camera_front", "camera_left"]) == {
        "camera_front": range(-1, 1_200),
        "camera_left": range(-1, 1_100),
    }
    assert rig.first_camera_frame_end_us(["camera_front", "camera_left"]) == 1_100


def test_rig_clip_start_drops_early_frames_and_trims_trajectory() -> None:
    """clip_start trims the trajectory to the offset and drops earlier frames."""
    (rig,) = Rig.load_from_json(
        _rig_json(
            {
                "unique-front": [[1_050, 1_100], [1_250, 1_300]],
                "unique-left": [[1_000, 1_050], [1_300, 1_350]],
            }
        )
    )

    # Trajectory spans [1_000, 2_000]; offset 200 -> new start 1_200, so the
    # first frame of each camera closes before the cut and is dropped.
    clipped = rig.clip_start(200)

    assert clipped.trajectory.time_range_us.start == 1_200
    assert clipped.camera_frame_ranges_us["unique-front"] == [range(1_250, 1_300)]
    assert clipped.camera_frame_ranges_us["unique-left"] == [range(1_300, 1_350)]
    assert clipped.first_camera_frame_end_us(["camera_front", "camera_left"]) == 1_300

    # The original rig is left unchanged.
    assert rig.trajectory.time_range_us.start == 1_000
    assert len(rig.camera_frame_ranges_us["unique-front"]) == 2


def test_rig_clip_start_drops_frame_exposing_across_the_new_start() -> None:
    """A frame that opens before the new start is dropped even if it closes after."""
    (rig,) = Rig.load_from_json(
        _rig_json(
            {
                "unique-front": [[1_150, 1_250], [1_250, 1_350]],
                "unique-left": [[1_150, 1_250], [1_250, 1_350]],
            }
        )
    )

    # Trajectory spans [1_000, 2_000]; offset 200 -> new start 1_200. The first
    # frame closes after the new start but opens 50 us before it, so the trimmed
    # trajectory cannot supply its start pose.
    clipped = rig.clip_start(200)

    assert clipped.camera_frame_ranges_us["unique-front"] == [range(1_250, 1_350)]
    assert clipped.camera_frame_timestamps_us["unique-front"] == [1_350]
    assert clipped.first_camera_frame_end_us(["camera_front", "camera_left"]) == 1_350


def test_rig_load_rejects_missing_camera_frame_timestamps() -> None:
    with pytest.raises(ValueError, match="Missing cameras_frame_timestamps_us"):
        Rig.load_from_json(_rig_json(None))


def test_rig_load_rejects_malformed_camera_frame_timestamp() -> None:
    with pytest.raises(ValueError, match="malformed frame timestamp"):
        Rig.load_from_json(
            _rig_json(
                {
                    "unique-front": [["not-a-frame"]],
                    "unique-left": [[-1, 1_100]],
                }
            )
        )


def test_rig_load_rejects_single_int_camera_frame_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"Camera 'unique-front'.*sequence_id='clipgt-test'.*"
            r"frame timestamp at index 0"
        ),
    ):
        Rig.load_from_json(
            _rig_json(
                {
                    "unique-front": [1_200],
                    "unique-left": [[-1, 1_100]],
                }
            )
        )


def test_simulation_start_rejects_configured_camera_without_timestamps() -> None:
    (rig,) = Rig.load_from_json(
        _rig_json(
            {
                "unique-front": [[-1, 1_200]],
                "unique-left": [],
            }
        )
    )

    with pytest.raises(ValueError, match="has no frame timestamps"):
        rig.first_camera_frame_ranges_us(["camera_left"])
