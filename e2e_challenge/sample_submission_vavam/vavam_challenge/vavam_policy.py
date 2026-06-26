# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Thin VAVAM inference wrapper used by the challenge driver."""

from __future__ import annotations

import logging
import platform
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import omegaconf.dictconfig
import omegaconf.listconfig
import torch
import torch.serialization
from PIL import Image

LOGGER = logging.getLogger(__name__)

torch.serialization.add_safe_globals(
    [
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
    ]
)


@dataclass(frozen=True)
class VavamPrediction:
    trajectory_xy: np.ndarray
    headings: np.ndarray


class VavamPolicy:
    """Loads VAVAM and predicts a 2 Hz ego-relative trajectory."""

    expected_height = 900
    expected_width = 1600
    output_frequency_hz = 2.0
    dtype = torch.float32 if platform.machine() == "aarch64" else torch.float16

    def __init__(
        self,
        *,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = "cuda",
    ) -> None:
        from vam.action_expert import VideoActionModelInference
        from vam.datalib.transforms import NeuroNCAPTransform

        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        if resolved_device.type != "cuda":
            LOGGER.warning("CUDA is unavailable; loading VAVAM on CPU")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt["hyper_parameters"]["vam_conf"].copy()
        config.pop("_target_", None)
        config.pop("_recursive_", None)
        config["gpt_checkpoint_path"] = None
        config["action_checkpoint_path"] = None
        config["gpt_mup_base_shapes"] = None
        config["action_mup_base_shapes"] = None

        LOGGER.info("Loading VAVAM checkpoint from %s", checkpoint_path)
        vam = VideoActionModelInference(**config)
        state_dict = OrderedDict()
        for key, value in ckpt["state_dict"].items():
            state_dict[key.replace("vam.", "")] = value
        vam.load_state_dict(state_dict, strict=True)
        self._vam = vam.eval().to(resolved_device)

        LOGGER.info("Loading VQ tokenizer from %s", tokenizer_path)
        self._tokenizer = torch.jit.load(tokenizer_path, map_location=resolved_device)
        self._tokenizer.to(resolved_device).eval()

        self._device = resolved_device
        self._preproc_pipeline = NeuroNCAPTransform()
        self._use_autocast = (
            resolved_device.type == "cuda" and platform.machine() != "aarch64"
        )

    def predict(self, image_hwc: np.ndarray, command: int) -> VavamPrediction:
        """Predict a trajectory from the latest RGB frame.

        Args:
            image_hwc: uint8 RGB image in HWC layout.
            command: VAVAM command id, where 0=right, 1=left, 2=straight.
        """

        image = self._resize_and_center_crop(
            image_hwc,
            self.expected_height,
            self.expected_width,
        )
        tensor = self._preproc_pipeline(image).unsqueeze(0).to(self._device)
        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self.dtype)
            if self._use_autocast
            else nullcontext()
        )

        with torch.no_grad():
            with autocast_ctx:
                tokens = self._tokenizer(tensor)
                batched_tokens = tokens.unsqueeze(1)
                batched_command = torch.tensor(
                    [[command]],
                    device=self._device,
                    dtype=torch.long,
                )
                trajectory = self._vam(batched_tokens, batched_command, self.dtype)

        trajectory_xy = _format_trajectory(trajectory)
        return VavamPrediction(
            trajectory_xy=trajectory_xy,
            headings=_compute_headings(trajectory_xy),
        )

    @staticmethod
    def _resize_and_center_crop(
        image: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        if h == target_height and w == target_width:
            return image

        pil_img = Image.fromarray(image)
        scale = target_height / h
        new_w = int(w * scale)
        pil_img = pil_img.resize((new_w, target_height), Image.Resampling.BILINEAR)

        if new_w > target_width:
            left = (new_w - target_width) // 2
            pil_img = pil_img.crop((left, 0, left + target_width, target_height))
        elif new_w < target_width:
            raise ValueError(
                f"Image width {new_w} too small after resize, need {target_width}"
            )

        return np.array(pil_img)


def _format_trajectory(trajectory: torch.Tensor) -> np.ndarray:
    array = trajectory.detach().float().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array.squeeze(0)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Unexpected VAVAM trajectory shape {array.shape}")
    return array


def _compute_headings(trajectory_xy: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(trajectory_xy)
    prev[1:, :] = trajectory_xy[:-1, :]
    deltas = trajectory_xy - prev
    return np.arctan2(deltas[:, 1], deltas[:, 0])
