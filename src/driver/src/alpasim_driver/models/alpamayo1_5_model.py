# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 1.5 wrapper implementing the common interface."""

from __future__ import annotations

import logging
from typing import Any

import torch
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

from ..schema import ModelConfig
from .alpamayo_base import CAMERA_NAME_TO_INDEX, AlpamayoBaseModel

logger = logging.getLogger(__name__)


class Alpamayo15Model(AlpamayoBaseModel):
    """Alpamayo 1.5 wrapper implementing the common interface.

    Compared to Alpamayo 1, Alpamayo 1.5 adds:
    - Camera index awareness via ``helper.create_message(frames, camera_indices)``
    - Optional classifier-free guidance navigation
      (``sample_trajectories_from_data_with_vlm_rollout_cfg_nav``)
    """

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "Alpamayo15Model":
        """Create Alpamayo15Model from driver configuration."""
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
            use_classifier_free_guidance_nav=model_cfg.use_classifier_free_guidance_nav,
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
        use_classifier_free_guidance_nav: bool = False,
    ):
        """Initialize Alpamayo 1.5 model.

        Args:
            checkpoint_path: Path or HuggingFace model ID for Alpamayo 1.5 checkpoint.
            device: Torch device for inference.
            camera_ids: List of camera IDs (supports multiple cameras).
            context_length: Number of temporal frames per camera (default 4).
            num_traj_samples: Number of trajectory samples to generate.
            top_p: Top-p sampling parameter for VLM generation.
            temperature: Temperature for VLM sampling.
            use_classifier_free_guidance_nav: If True, use classifier-free guidance navigation
                sampling.  Requires roughly 60 GB VRAM (vs ~40 GB standard).
        """
        logger.info("Loading Alpamayo 1.5 checkpoint from %s", checkpoint_path)

        model = Alpamayo1_5.from_pretrained(checkpoint_path, dtype=self.DTYPE).to(
            device
        )
        processor = helper.get_processor(model.tokenizer)

        self._use_classifier_free_guidance_nav = use_classifier_free_guidance_nav

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

        if use_classifier_free_guidance_nav:
            logger.info("CFG nav sampling enabled (requires ~60 GB VRAM)")

    def _create_chat_message(self, image_frames: torch.Tensor) -> list:
        """Create chat message with camera indices for Alpamayo 1.5."""
        # Sort camera IDs by index (same order used in _preprocess_images)
        sorted_camera_ids = sorted(
            self._camera_ids, key=lambda cam_id: CAMERA_NAME_TO_INDEX[cam_id]
        )
        camera_indices = torch.tensor(
            [CAMERA_NAME_TO_INDEX[cam_id] for cam_id in sorted_camera_ids]
        )

        return self._helper.create_message(image_frames.flatten(0, 1), camera_indices)

    def _run_inference(
        self, model_inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Run inference, optionally using CFG nav sampling."""
        if self._use_classifier_free_guidance_nav:
            return self._model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
                data=model_inputs,
                top_p=self._top_p,
                temperature=self._temperature,
                num_traj_samples=self._num_traj_samples,
                return_extra=True,
            )
        return super()._run_inference(model_inputs)
