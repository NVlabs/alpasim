# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Writes rsim_parity.npz from rsim's own functions.

Run with rsim's interpreter from this directory:
    ~/rsim/.venv/bin/python make_rsim_fixtures.py
"""

from pathlib import Path

import numpy as np
from images import smooth_image
from rsim.agents.onnx import _preprocess_image
from rsim.agents.onnx_decoder import frame_rope_cos_sin
from rsim.carla.waypoints import ego_local_m_to_world, world_route_to_ego_normalized

ROPE_FRAMES = np.array([0, 1, 7, 100])
HEAD_DIM = 64
IMAGE_HW = (224, 224)
EGO_XY = np.array([12.0, -7.0])
HEADING_DEG = 30.0


def routes_rig() -> dict[str, np.ndarray]:
    """Rig-frame (x forward, y left) routes, ~4 m spacing like Alpasim's."""
    s = np.arange(0.0, 84.0, 4.2)
    straight = np.stack([s, np.full_like(s, 0.8)], axis=-1)
    radius = 40.0
    angle = s / radius
    curved_left = np.stack(
        [radius * np.sin(angle), radius * (1 - np.cos(angle))], axis=-1
    )
    return {"straight": straight, "curved_left": curved_left}


def main() -> None:
    out: dict[str, np.ndarray] = {"rope_frames": ROPE_FRAMES}
    for f in ROPE_FRAMES:
        out[f"rope_cos_{f}"], out[f"rope_sin_{f}"] = frame_rope_cos_sin(
            int(f), HEAD_DIM
        )

    image = smooth_image(324, 576)
    # The pipeline resizes in uint8, so output * 255 is exact.
    out["preprocess_output_u8"] = np.rint(
        _preprocess_image(image, IMAGE_HW, imagenet_norm=False) * 255
    ).astype(np.uint8)

    for name, route in routes_rig().items():
        # rsim works in the model's ego frame (x right, y forward) and CARLA world.
        route_model_m = np.stack([-route[:, 1], route[:, 0]], axis=-1)
        route_world = ego_local_m_to_world(
            route_model_m, ego_xy=EGO_XY, heading_deg=HEADING_DEG
        )
        out[f"route_rig_{name}"] = route
        out[f"waypoints_{name}"] = np.clip(
            world_route_to_ego_normalized(
                route_world,
                ego_xy=EGO_XY,
                heading_deg=HEADING_DEG,
                speed_kmh=30.0,
                n=10,
                spacing="distance",
                first_offset_m=1.0,
            ),
            -1.0,
            1.0,
        )

    np.savez_compressed(Path(__file__).with_name("rsim_parity.npz"), **out)


if __name__ == "__main__":
    main()
