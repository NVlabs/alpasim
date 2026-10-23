# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Parity of the copied input logic with rsim and drivr.

Fixtures come from the originals; see ``fixtures/make_*_fixtures.py``.
"""

from pathlib import Path

import numpy as np
import pytest
from alpasim_onnx_traj.rsim_port import (
    FLEET_CALIBRATION_PATH,
    DecoderGraphSpec,
    Defisher,
    FisheyeCalibration,
    KvCacheRing,
    frame_rope_cos_sin,
    model_xy_to_rig,
    preprocess_image,
    rig_xy_to_model,
    route_to_model_waypoints,
)
from fixtures.images import smooth_image

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def rsim() -> dict[str, np.ndarray]:
    return dict(np.load(FIXTURES / "rsim_parity.npz"))


def test_rope_matches_rsim(rsim: dict[str, np.ndarray]) -> None:
    for frame in rsim["rope_frames"]:
        cos, sin = frame_rope_cos_sin(int(frame), 64)
        np.testing.assert_array_equal(cos, rsim[f"rope_cos_{frame}"])
        np.testing.assert_array_equal(sin, rsim[f"rope_sin_{frame}"])


def test_preprocess_matches_rsim(rsim: dict[str, np.ndarray]) -> None:
    out = preprocess_image(smooth_image(324, 576), (224, 224))

    assert out.shape == (3, 224, 224)
    # torch versions round the uint8 antialiased resize differently by up to
    # one level (rsim's torch 2.11 vs this env's 2.8).
    np.testing.assert_allclose(
        np.rint(out * 255), rsim["preprocess_output_u8"], rtol=0, atol=1
    )


def test_defish_matches_drivr() -> None:
    expected = np.load(FIXTURES / "drivr_defish.npz")
    stride = int(expected["stride"])
    defisher = Defisher(FisheyeCalibration.from_json(FLEET_CALIBRATION_PATH))

    out = defisher.apply(smooth_image(1080, 1920))

    np.testing.assert_array_equal(out[::stride, ::stride], expected["output"])


def test_defish_rejects_non_native_frames() -> None:
    defisher = Defisher(FisheyeCalibration.from_json(FLEET_CALIBRATION_PATH))

    with pytest.raises(ValueError, match="1080x1920"):
        defisher.apply(np.zeros((540, 960, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", ["straight", "curved_left"])
def test_waypoints_match_rsim(rsim: dict[str, np.ndarray], name: str) -> None:
    waypoints = route_to_model_waypoints(rsim[f"route_rig_{name}"], 10)

    np.testing.assert_allclose(waypoints, rsim[f"waypoints_{name}"], atol=1e-5)


def test_left_curve_bends_to_negative_model_x(rsim: dict[str, np.ndarray]) -> None:
    waypoints = route_to_model_waypoints(rsim["route_rig_curved_left"], 10)

    assert np.all(waypoints[:, 1] > 0)  # forward
    assert waypoints[-1, 0] < waypoints[0, 0] < 0  # left is -x_right


def test_waypoints_drop_nan_padding_and_default_to_zeros() -> None:
    route = np.array([[0.0, 0.0], [50.0, 0.0], [np.nan, np.nan]])

    padded = route_to_model_waypoints(route, 10)
    empty = route_to_model_waypoints(route[2:], 10)

    np.testing.assert_allclose(
        padded[:, 1], np.minimum(1 + 10 * np.arange(10), 50) / 100
    )
    np.testing.assert_array_equal(empty, np.zeros((10, 2)))


def test_axis_mapping_round_trips() -> None:
    xy_rig = np.array([[3.0, 1.0], [5.0, -2.0]])

    np.testing.assert_array_equal(rig_xy_to_model(xy_rig), [[-1.0, 3.0], [2.0, 5.0]])
    np.testing.assert_array_equal(model_xy_to_rig(rig_xy_to_model(xy_rig)), xy_rig)


def test_kv_ring_writes_one_slot_per_frame_and_wraps() -> None:
    spec = DecoderGraphSpec(
        camera_inputs={"cam_front_left": "inputs_image_cam_front_left"},
        image_hw=(8, 8),
        num_waypoints=10,
        past_kv_shape=(2, 1, 2, 6, 4),
        tokens_per_frame=3,
        num_hypotheses=5,
        num_poses=5,
    )
    ring = KvCacheRing(spec)
    new = np.empty((2, 1, 2, 3, 4), dtype=np.float32)

    for value in (1.0, 2.0, 3.0):
        np.testing.assert_array_equal(
            ring.feed()["inputs_rope_cos"], frame_rope_cos_sin(ring.frame_index, 4)[0]
        )
        new.fill(value)
        ring.advance(new, new)

    # Frame 2 overwrote slot 0; frame 1 is still in slot 1.
    np.testing.assert_array_equal(ring.past_k[..., 0:3, :], 3.0)
    np.testing.assert_array_equal(ring.past_k[..., 3:6, :], 2.0)
    np.testing.assert_array_equal(ring.cache_bias, 0.0)
    assert ring.frame_index == 3
