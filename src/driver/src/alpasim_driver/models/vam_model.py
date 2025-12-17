# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""VAM (Video Action Model) wrapper implementing the common interface."""

from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

import numpy as np
import omegaconf.dictconfig
import omegaconf.listconfig
import torch
import torch.serialization
from vam.action_expert import VideoActionModelInference
from vam.datalib.transforms import NeuroNCAPTransform

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction

logger = logging.getLogger(__name__)


# Allow torch.load to recreate OmegaConf containers embedded in checkpoints
torch.serialization.add_safe_globals(
    [
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
    ]
)


def load_inference_VAM(
    checkpoint_path: str,
    device: torch.device | str = "cuda",
) -> VideoActionModelInference:
    """Load VAM model from checkpoint.

    Custom loader that handles PyTorch 2.6+ weights_only issue.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = ckpt["hyper_parameters"]["vam_conf"].copy()
    config.pop("_target_", None)
    config.pop("_recursive_", None)
    config["gpt_checkpoint_path"] = None
    config["action_checkpoint_path"] = None
    config["gpt_mup_base_shapes"] = None
    config["action_mup_base_shapes"] = None

    logger.info("Loading VAM checkpoint from %s", checkpoint_path)
    logger.debug("VAM config: %s", config)

    vam = VideoActionModelInference(**config)
    state_dict = OrderedDict()
    for key, value in ckpt["state_dict"].items():
        state_dict[key.replace("vam.", "")] = value
    vam.load_state_dict(state_dict, strict=True)
    vam = vam.eval().to(device)
    return vam


def _format_trajs(trajs: torch.Tensor) -> np.ndarray:
    """Normalize VAM trajectory tensor shape to (T, 2)."""
    array = trajs.detach().float().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array.squeeze(0)

    if array.ndim != 2:
        raise ValueError(f"Unexpected trajectory shape {array.shape}")

    return array


class VAMModel(BaseTrajectoryModel):
    """VAM wrapper implementing the common interface."""

    # VAM uses float16 for inference
    DTYPE = torch.float16
    # VAM only supports single camera
    NUM_CAMERAS = 1
    # NeuroNCAPTransform expects 900x1600 input
    EXPECTED_HEIGHT = 900
    EXPECTED_WIDTH = 1600

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = 8,
        cache_size: int = 32,
    ):
        """Initialize VAM model.

        Args:
            checkpoint_path: Path to VAM model checkpoint.
            tokenizer_path: Path to JIT-compiled VQ tokenizer.
            device: Torch device for inference.
            camera_ids: List of camera IDs (must be exactly 1).
            context_length: Number of temporal frames (default 8).
            cache_size: Maximum number of tokenized frames to cache.
        """
        if len(camera_ids) != self.NUM_CAMERAS:
            raise ValueError(
                f"VAM requires exactly {self.NUM_CAMERAS} camera, "
                f"got {len(camera_ids)}"
            )

        self._vam = load_inference_VAM(checkpoint_path, device)
        self._tokenizer = torch.jit.load(tokenizer_path, map_location=device)
        self._tokenizer.to(device).eval()
        self._device = device
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._preproc_pipeline = NeuroNCAPTransform()
        self._use_autocast = device.type == "cuda"

        # Token cache: timestamp_us -> tokenized tensor (LRU)
        self._token_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._cache_size = cache_size

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return 2  # VAM outputs trajectory at 2Hz

    def _encode_command(self, command: DriveCommand) -> int:
        """Convert DriveCommand to VAM format.

        DriveCommand:  LEFT=0, STRAIGHT=1, RIGHT=2, UNKNOWN=3
        VAM Command:   RIGHT=0, LEFT=1, STRAIGHT=2
        """
        COMMAND_MAP = {
            DriveCommand.LEFT: 1,  # VAM LEFT
            DriveCommand.STRAIGHT: 2,  # VAM STRAIGHT
            DriveCommand.RIGHT: 0,  # VAM RIGHT
            DriveCommand.UNKNOWN: 2,  # Default to STRAIGHT
        }
        return COMMAND_MAP[command]

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize and apply NeuroNCAP transform."""
        image = self._resize_and_center_crop(
            image, self.EXPECTED_HEIGHT, self.EXPECTED_WIDTH
        )
        return self._preproc_pipeline(image)

    def _get_or_tokenize(self, timestamp_us: int, image: np.ndarray) -> torch.Tensor:
        """Get cached tokens or tokenize and cache."""
        if timestamp_us in self._token_cache:
            self._token_cache.move_to_end(timestamp_us)
            return self._token_cache[timestamp_us]

        # Preprocess and tokenize
        tensor = self._preprocess(image)
        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self.DTYPE)
            if self._use_autocast
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                tokens = self._tokenizer(tensor.unsqueeze(0).to(self._device))
        tokens = tokens.squeeze(0).cpu()

        # Cache with LRU eviction
        self._token_cache[timestamp_us] = tokens
        if len(self._token_cache) > self._cache_size:
            self._token_cache.popitem(last=False)

        return tokens

    def predict(
        self,
        camera_images: dict[str, list[tuple[int, np.ndarray]]],
        command: DriveCommand,
        speed: float,
        acceleration: float,
        ego_pose_at_time_history_local: list[Any] | None = None,
    ) -> ModelPrediction:
        """Generate trajectory prediction.

        Args:
            camera_images: Dict mapping camera_id to list of
                (timestamp_us, image) tuples. List length must equal
                context_length.
            command: Canonical navigation command.
            speed: Current vehicle speed in m/s (unused by VAM).
            acceleration: Current longitudinal acceleration (unused by VAM).
            ego_pose_at_time_history_local: Optional list of PoseAtTime for building ego history.
                PoseAtTime contains pairs of (timestamp_us, Pose) where Pose is 3D position and
                orientation in local frame.
        Returns:
            ModelPrediction with trajectory in rig frame.
        """
        del ego_pose_at_time_history_local
        self._validate_cameras(camera_images)

        # VAM uses single camera
        cam_id = self._camera_ids[0]
        frames = camera_images[cam_id]

        if len(frames) != self._context_length:
            raise ValueError(
                f"VAM expects {self._context_length} frames, got {len(frames)}"
            )

        # VAM ignores speed/acceleration - uses only visual tokens + command
        tokens = [self._get_or_tokenize(ts, img) for ts, img in frames]
        token_tensor = torch.stack(tokens, dim=0).unsqueeze(0).to(self._device)

        # Encode command using VAM-specific encoding
        encoded_command = self._encode_command(command)
        command_tensor = torch.tensor(
            [[encoded_command]], device=self._device, dtype=torch.long
        )

        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self.DTYPE)
            if self._use_autocast
            else nullcontext()
        )

        with torch.no_grad():
            with autocast_ctx:
                trajectory = self._vam(token_tensor, command_tensor, self.DTYPE)

        trajectory_xy = _format_trajs(trajectory)
        headings = self._compute_headings_from_trajectory(trajectory_xy)
        return ModelPrediction(trajectory_xy=trajectory_xy, headings=headings)
