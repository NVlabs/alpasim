# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""VAM (Video Action Model) wrapper implementing the common interface."""

from __future__ import annotations

import logging
import platform
from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import omegaconf.dictconfig
import omegaconf.listconfig
import torch
import torch.serialization
from vam.action_expert import VideoActionModelInference
from vam.datalib.transforms import NeuroNCAPTransform

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput

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

    # VAM uses float16 for inference; float32 on ARM (torch.amp.autocast on aarch64
    # doesn't auto-cast Float32 weights in F.linear — fails across PyTorch versions)
    DTYPE = torch.float32 if platform.machine() == "aarch64" else torch.float16
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
    ):
        """Initialize VAM model.

        Args:
            checkpoint_path: Path to VAM model checkpoint.
            tokenizer_path: Path to JIT-compiled VQ tokenizer.
            device: Torch device for inference.
            camera_ids: List of camera IDs (must be exactly 1).
            context_length: Number of temporal frames (default 8).
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
        self._use_autocast = device.type == "cuda" and platform.machine() != "aarch64"

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

    def _preprocess_batch(self, images: list[np.ndarray]) -> torch.Tensor:
        """Preprocess and stack multiple images into a batch tensor.

        Args:
            images: List of HWC uint8 numpy arrays.

        Returns:
            Tensor of shape ``(B, C, H, W)`` ready for the tokenizer.
        """
        tensors = [self._preprocess(img) for img in images]
        return torch.stack(tensors, dim=0)

    def _tokenize_frames(self, images: list[np.ndarray]) -> torch.Tensor:
        """Preprocess and tokenize a batch of images in a single GPU call.

        Args:
            images: List of HWC uint8 numpy arrays.

        Returns:
            Token tensor of shape ``(B, h, w)``.
        """
        batch_tensor = self._preprocess_batch(images).to(self._device)
        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self.DTYPE)
            if self._use_autocast
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                tokens = self._tokenizer(batch_tensor)
        return tokens

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate trajectory prediction.

        Delegates to :meth:`predict_batch` so there is one inference path.
        """
        return self.predict_batch([prediction_input])[0]

    def predict_batch(
        self, prediction_inputs: list[PredictionInput]
    ) -> list[ModelPrediction]:
        """Generate trajectory predictions for a batch of inputs.

        All frames are batch-tokenized in a single GPU call, then a single
        VAM forward pass produces all trajectories.

        Args:
            prediction_inputs: List of prediction inputs, one per concurrent session.

        Returns:
            List of ModelPrediction, one per input, in the same order.
        """
        if not prediction_inputs:
            return []

        cam_id = self._camera_ids[0]
        batch_size = len(prediction_inputs)

        # --- Validate and collect all frames ---
        all_images: list[np.ndarray] = []
        all_commands: list[int] = []

        for inp in prediction_inputs:
            self._validate_cameras(inp.camera_images)
            frames = inp.camera_images[cam_id]
            if len(frames) != self._context_length:
                raise ValueError(
                    f"VAM expects {self._context_length} frames, got {len(frames)}"
                )
            for _ts, img in frames:
                all_images.append(img)
            all_commands.append(self._encode_command(inp.command))

        # --- Batch tokenize: (B*T, C, H, W) -> (B*T, h, w) ---
        all_tokens = self._tokenize_frames(all_images)

        # Reshape to (B, T, h, w)
        token_h, token_w = all_tokens.shape[1], all_tokens.shape[2]
        batched_tokens = all_tokens.reshape(
            batch_size, self._context_length, token_h, token_w
        )

        # --- Batch commands: (B, 1) ---
        batched_commands = torch.tensor(
            all_commands, device=self._device, dtype=torch.long
        ).unsqueeze(1)

        # --- Single VAM forward pass ---
        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self.DTYPE)
            if self._use_autocast
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                trajectories = self._vam(batched_tokens, batched_commands, self.DTYPE)

        # --- Unpack per-sample results ---
        results: list[ModelPrediction] = []
        for i in range(batch_size):
            trajectory_xy = _format_trajs(trajectories[i : i + 1])
            headings = self._compute_headings_from_trajectory(trajectory_xy)
            results.append(
                ModelPrediction(trajectory_xy=trajectory_xy, headings=headings)
            )

        return results
