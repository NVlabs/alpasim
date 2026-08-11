# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Native AlpaSim adapter for Alpamayo Recipes 1.5 SFT checkpoints.

Loads ``TrainableAlpamayoR1`` from Alpamayo Recipes:
https://github.com/NVlabs/alpamayo-recipes/blob/bfebb8451366b166611f4a84920bff1857d69d46/recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py#L33
"""

import logging

import torch
from alpamayo1_5 import helper

from .alpamayo1_5_model import Alpamayo15Model
from .alpamayo_base import configure_deterministic_runtime
from .trajectory_selection import TrajectorySelectionStrategy

logger = logging.getLogger(__name__)


def _load_recipes_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    try:
        from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The Alpamayo 1.5 SFT driver requires the recipes extra: "
            "uv sync --extra all --extra recipes"
        ) from error

    return (
        TrainableAlpamayoR1.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )


def _load_recipes_chat_template():
    from alpamayo.chat_template import get_template

    return get_template("r1_5")


class Alpamayo15RecipesSFTModel(Alpamayo15Model):
    """Run an Alpamayo Recipes 1.5 SFT checkpoint in the native driver."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = Alpamayo15Model.DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
        cfg_guidance_weight: float | None = None,
        force_determinism: bool = False,
        selection_strategy: TrajectorySelectionStrategy = (
            TrajectorySelectionStrategy.ALWAYS_FIRST
        ),
        max_num_distance_points: int = 64,
        skip_first_n_distance_points: int = 0,
    ) -> None:
        if cfg_guidance_weight is not None:
            raise ValueError("Alpamayo 1.5 SFT does not support CFG navigation")
        if force_determinism:
            configure_deterministic_runtime()

        logger.info(
            "Loading Alpamayo Recipes 1.5 SFT checkpoint from %s", checkpoint_path
        )
        model = _load_recipes_model(checkpoint_path, device)
        processor = helper.get_processor(model.tokenizer)

        self._chat_template = _load_recipes_chat_template()
        self._cfg_guidance_weight = None
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
            force_determinism=force_determinism,
            selection_strategy=selection_strategy,
            max_num_distance_points=max_num_distance_points,
            skip_first_n_distance_points=skip_first_n_distance_points,
        )

    def _create_chat_message(
        self, image_frames: torch.Tensor, nav_text: str | None
    ) -> list:
        """Build the future-only conversation used by Recipes SFT.

        The SFT conversation has no route component, so ``nav_text`` is unused.
        """
        return self._chat_template.build_conversation(
            data={"image_frames": image_frames},
            num_tokens_per_history_traj=self._model.config.tokens_per_history_traj,
            num_tokens_per_future_traj=self._model.config.tokens_per_future_traj,
            components_order=["image", "traj_history", "prompt", "traj_future"],
            components_prompt=["traj_future"],
            generation_mode=True,
            include_camera_ids=self._model.config.include_camera_ids,
            camera_ids=self._camera_indices(),
            include_frame_nums=self._model.config.include_frame_nums,
        )
