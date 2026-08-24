# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from os import PathLike

import numpy as np
import torch
from torch import Tensor

from .preprocessing import CAMERA_IDS, build_status_feature, preprocess_images
from .simscale_diffusiondrive.config import DiffusionDriveConfig
from .simscale_diffusiondrive.model import DiffusionDriveModel

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "agent._transfuser_model."
ANCHOR_KEY = CHECKPOINT_PREFIX + "_trajectory_head.plan_anchor"
EXPECTED_STATE_DICT_COUNT = 764


@dataclass(frozen=True)
class InferenceInput:
    images: Mapping[str, np.ndarray]
    command_one_hot: np.ndarray
    velocity_xy: np.ndarray
    acceleration_xy: np.ndarray
    noise_seed: int | None = None
    noise_index: int | None = None

    def __post_init__(self) -> None:
        missing = [
            camera_id for camera_id in CAMERA_IDS if camera_id not in self.images
        ]
        if missing:
            raise ValueError(f"missing required cameras: {', '.join(missing)}")
        if (self.noise_seed is None) != (self.noise_index is None):
            raise ValueError("noise_seed and noise_index must be provided together")
        if self.noise_seed is not None:
            if isinstance(self.noise_seed, bool) or not isinstance(
                self.noise_seed, Integral
            ):
                raise ValueError("noise_seed must be an integer")
            if (
                isinstance(self.noise_index, bool)
                or not isinstance(self.noise_index, Integral)
                or self.noise_index < 0
            ):
                raise ValueError("noise_index must be a non-negative integer")


@dataclass(frozen=True)
class Prediction:
    trajectory: np.ndarray


def normalized_checkpoint(payload: object) -> tuple[dict[str, Tensor], np.ndarray]:
    """Validate and normalize the exact released checkpoint wrapper."""
    if not isinstance(payload, dict) or set(payload) != {"state_dict"}:
        raise ValueError(
            "checkpoint payload must be a dict containing only 'state_dict'"
        )
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint state_dict must be a dict")
    if len(state_dict) != EXPECTED_STATE_DICT_COUNT:
        raise ValueError(
            f"checkpoint must contain exactly {EXPECTED_STATE_DICT_COUNT} tensors; "
            f"got {len(state_dict)}"
        )

    normalized: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.startswith(CHECKPOINT_PREFIX):
            raise ValueError(f"unexpected checkpoint key: {key}")
        if not isinstance(value, Tensor):
            raise ValueError(f"checkpoint value for {key} must be a tensor")
        normalized[key.removeprefix(CHECKPOINT_PREFIX)] = value

    anchor = state_dict.get(ANCHOR_KEY)
    if not isinstance(anchor, Tensor):
        raise ValueError("checkpoint plan anchor tensor is missing")
    if tuple(anchor.shape) != (20, 8, 2):
        raise ValueError(
            "checkpoint plan anchor must have shape (20, 8, 2); "
            f"got {tuple(anchor.shape)}"
        )
    if not anchor.is_floating_point():
        raise ValueError("checkpoint plan anchor must use a floating-point dtype")
    if anchor.dtype != torch.float32:
        raise ValueError(f"checkpoint plan anchor must use float32; got {anchor.dtype}")
    if not torch.isfinite(anchor).all():
        raise ValueError("checkpoint plan anchor contains non-finite values")

    return normalized, anchor.detach().cpu().numpy().copy()


class DiffusionDrivePolicy:
    def __init__(
        self,
        checkpoint_path: str | PathLike[str],
        device: str | torch.device = "cuda",
        use_autocast: bool = True,
        warm_up: bool = True,
        noise_seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but CUDA is not available"
            )
        self.use_autocast = bool(use_autocast) and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_autocast else torch.float32

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict, plan_anchor = normalized_checkpoint(payload)
        self.model = DiffusionDriveModel(DiffusionDriveConfig(), plan_anchor)
        self.model.load_state_dict(state_dict, strict=True)
        self.state_dict_count = len(state_dict)
        LOGGER.info(
            "Loaded %d tensors from checkpoint %s",
            self.state_dict_count,
            checkpoint_path,
        )
        self.model.to(self.device).eval()

        self.generator = torch.Generator(device=self.device)
        if noise_seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(noise_seed)

        if warm_up:
            zero_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.predict_batch(
                [
                    InferenceInput(
                        images={camera_id: zero_image for camera_id in CAMERA_IDS},
                        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
                        velocity_xy=np.zeros(2, dtype=np.float32),
                        acceleration_xy=np.zeros(2, dtype=np.float32),
                    )
                ]
            )
            LOGGER.info("DiffusionDrive policy warm-up completed")

    def predict_batch(self, requests: list[InferenceInput]) -> list[Prediction]:
        if not requests:
            return []

        camera_feature = torch.stack(
            [
                preprocess_images(request.images, dtype=torch.float32)
                for request in requests
            ]
        ).to(device=self.device, dtype=self.dtype)
        status_feature = torch.stack(
            [
                build_status_feature(
                    command_one_hot=request.command_one_hot,
                    velocity_xy=request.velocity_xy,
                    acceleration_xy=request.acceleration_xy,
                )
                for request in requests
            ]
        ).to(device=self.device, dtype=self.dtype)
        noise_samples = []
        for request in requests:
            generator = self.generator
            if request.noise_seed is not None:
                generator = torch.Generator(device=self.device)
                derived_seed = (
                    int(request.noise_seed)
                    + 0x9E3779B97F4A7C15 * (int(request.noise_index) + 1)
                ) & 0xFFFFFFFFFFFFFFFF
                generator.manual_seed(derived_seed)
            noise_samples.append(
                torch.randn(
                    (20, 8, 2),
                    device=self.device,
                    dtype=torch.float32,
                    generator=generator,
                )
            )
        noise = torch.stack(noise_samples)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_autocast
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            output = self.model(
                {
                    "camera_feature": camera_feature,
                    "status_feature": status_feature,
                },
                noise=noise,
            )

        trajectory = output.detach().float().cpu().numpy()
        expected_shape = (len(requests), 8, 3)
        if trajectory.shape != expected_shape:
            raise ValueError(
                f"model trajectory must have shape {expected_shape}; got {trajectory.shape}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError("model trajectory contains non-finite values")
        return [Prediction(trajectory=sample.copy()) for sample in trajectory]
