# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 1 wrapper implementing the common interface."""

from __future__ import annotations

import logging

import torch
from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from ..schema import ModelConfig

# Re-export for backward compatibility (tests import these from here).
from .alpamayo_base import (  # noqa: F401
    CAMERA_NAME_TO_INDEX,
    AlpamayoBaseModel,
    build_ego_history,
)

logger = logging.getLogger(__name__)


class Alpamayo1Model(AlpamayoBaseModel):
    """Alpamayo 1 wrapper implementing the common interface."""

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "Alpamayo1Model":
        """Create Alpamayo1Model from driver configuration."""
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
        )

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = AlpamayoBaseModel.DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
    ):
        """Initialize Alpamayo 1 model.

        Args:
            checkpoint_path: Path or HuggingFace model ID for Alpamayo 1 checkpoint.
            device: Torch device for inference.
            camera_ids: List of camera IDs (supports multiple cameras).
            context_length: Number of temporal frames per camera (default 4).
            num_traj_samples: Number of trajectory samples to generate.
            top_p: Top-p sampling parameter for VLM generation.
            temperature: Temperature for VLM sampling.
        """
        logger.info("Loading Alpamayo 1 checkpoint from %s", checkpoint_path)

        model = AlpamayoR1.from_pretrained(checkpoint_path, dtype=self.DTYPE).to(device)
        processor = helper.get_processor(model.tokenizer)

        self._init_common(
            model=model,
            processor=processor,
            helper_module=helper,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length,
            num_traj_samples=num_traj_samples,
            top_p=top_p,
            temperature=temperature,
        )

    def _create_chat_message(self, image_frames: torch.Tensor) -> list:
        """Create chat message using Alpamayo 1's helper (no camera indices)."""
        return self._helper.create_message(image_frames.flatten(0, 1))
