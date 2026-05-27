# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Video model renderer configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VideoModelConfig:
    """Configuration for video model renderer.

    Parsed from the untyped ``renderer_config`` dict in ``UserSimulatorConfig``
    by :meth:`VideoModelService.from_config`.
    """

    fps: int = 30
    # Chunk sizes must match the external server's --num_frames_per_block and
    # initial-frame contract. Config group chunking/<N>frame.yaml keeps these
    # in sync for the common 8/12/16-frame variants.
    first_chunk_frames: int = 5
    chunk_frames: int = 8
    text_prompt_positive: str = ""
    text_prompt_negative: str = ""

    # Request debug streams from the video model server. Forwarding them to the
    # driver is opt-in because most policies expect RGB camera inputs only.
    return_hdmap_frames: bool = False
    return_bev_map: bool = False
    forward_hdmap_to_driver: bool = False
    forward_bev_to_driver: bool = False

    bev_height_m: float = 40.0
    bev_fov_deg: float = 50.0

    # "all" forwards every generated frame to the driver. "subsample" forwards
    # the nearest frames matching subsample_count/subsample_interval_us within
    # each returned chunk. Prefer driver.inference.subsample_factor for policy
    # input-rate matching because it operates on the continuous frame cache.
    frame_forwarding_mode: str = "all"
    subsample_count: int = 4
    subsample_interval_us: int = 100_000

    def __post_init__(self) -> None:
        # Numeric guards: fps=0 trips ZeroDivisionError in
        # VideoModelService.__init__ (1_000_000 // config.fps); zero or
        # negative chunk sizes break trajectory building.
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.first_chunk_frames <= 0:
            raise ValueError(
                f"first_chunk_frames must be positive, got {self.first_chunk_frames}"
            )
        if self.chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {self.chunk_frames}")
        if self.forward_hdmap_to_driver and not self.return_hdmap_frames:
            raise ValueError(
                "forward_hdmap_to_driver=True requires return_hdmap_frames=True."
            )
        if self.frame_forwarding_mode not in {"all", "subsample"}:
            raise ValueError(
                f"frame_forwarding_mode must be 'all' or 'subsample', "
                f"got {self.frame_forwarding_mode!r}"
            )
        if self.forward_bev_to_driver and not self.return_bev_map:
            raise ValueError("forward_bev_to_driver=True requires return_bev_map=True.")
