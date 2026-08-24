# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from os import PathLike

import numpy as np
import torch
from torch import Tensor, nn

from .preprocessing import CAMERA_IDS, build_status_feature, preprocess_images
from .simscale_ltf import TransfuserConfig, TransfuserModel

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "agent._transfuser_model."


@dataclass(frozen=True)
class InferenceInput:
    images: Mapping[str, np.ndarray]
    command_one_hot: np.ndarray
    velocity_xy: np.ndarray
    acceleration_xy: np.ndarray

    def __post_init__(self) -> None:
        missing = [
            camera_id for camera_id in CAMERA_IDS if camera_id not in self.images
        ]
        if missing:
            raise ValueError(f"missing required cameras: {', '.join(missing)}")


@dataclass(frozen=True)
class Prediction:
    trajectory: np.ndarray


def normalized_state_dict(payload: object) -> dict[str, Tensor]:
    if not isinstance(payload, dict) or set(payload) != {"state_dict"}:
        raise ValueError(
            "checkpoint payload must be a dict containing only 'state_dict'"
        )

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint state_dict must be a dict")

    normalized: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.startswith(CHECKPOINT_PREFIX):
            raise ValueError(f"unexpected checkpoint key: {key}")
        normalized[key.removeprefix(CHECKPOINT_PREFIX)] = value
    return normalized


def load_checkpoint(model: nn.Module, path: str | PathLike[str]) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = normalized_state_dict(payload)
    model.load_state_dict(state_dict, strict=True)
    return len(state_dict)


class LtfPolicy:
    def __init__(
        self,
        checkpoint_path: str | PathLike[str],
        device: str | torch.device = "cuda",
        use_autocast: bool = True,
        warm_up: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but CUDA is not available"
            )

        self.use_autocast = bool(use_autocast) and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_autocast else torch.float32

        self.model = TransfuserModel(TransfuserConfig())
        self._validate_trajectory_head()
        self.state_dict_count = load_checkpoint(self.model, checkpoint_path)
        LOGGER.info(
            "Loaded %d tensors from checkpoint %s",
            self.state_dict_count,
            checkpoint_path,
        )
        self.model.to(self.device).eval()

        if warm_up:
            zero_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            request = InferenceInput(
                images={camera_id: zero_image for camera_id in CAMERA_IDS},
                command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
                velocity_xy=np.zeros(2, dtype=np.float32),
                acceleration_xy=np.zeros(2, dtype=np.float32),
            )
            self.predict_batch([request])
            LOGGER.info("LTF policy warm-up completed")

    def _validate_trajectory_head(self) -> None:
        head = getattr(self.model, "_trajectory_head", None)
        num_poses = getattr(head, "_num_poses", None)
        mlp = getattr(head, "_mlp", None)
        output_count = getattr(mlp[-1], "out_features", None) if mlp else None
        if num_poses != 8 or output_count != 24:
            raise RuntimeError(
                "Transfuser trajectory head must predict 8 poses / 24 outputs; "
                f"got {num_poses} poses / {output_count} outputs"
            )

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
                }
            )

        trajectory = output["trajectory"].detach().float().cpu().numpy()
        expected_shape = (len(requests), 8, 3)
        if trajectory.shape != expected_shape:
            raise ValueError(
                f"model trajectory must have shape {expected_shape}; "
                f"got {trajectory.shape}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError("model trajectory contains non-finite values")

        return [Prediction(trajectory=sample.copy()) for sample in trajectory]
