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
    configure_deterministic_runtime,
)
from .trajectory_selection import TrajectorySelectionStrategy

logger = logging.getLogger(__name__)


class Alpamayo1Model(AlpamayoBaseModel):
    """Alpamayo 1 wrapper implementing the common interface."""

    @classmethod
    def _variant_init_kwargs(cls, model_cfg: ModelConfig) -> dict[str, object]:
        return {
            "diffusion_expert_cuda_graph": model_cfg.diffusion_expert_cuda_graph,
            "diffusion_expert_cuda_graph_max_graphs": (
                model_cfg.diffusion_expert_cuda_graph_max_graphs
            ),
        }

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = AlpamayoBaseModel.DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        top_p: float = 0.98,
        temperature: float = 0.6,
        diffusion_expert_cuda_graph: bool = False,
        diffusion_expert_cuda_graph_max_graphs: int = 4,
        force_determinism: bool = False,
        selection_strategy: TrajectorySelectionStrategy = (
            TrajectorySelectionStrategy.ALWAYS_FIRST
        ),
        max_num_distance_points: int = 64,
        skip_first_n_distance_points: int = 0,
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
            diffusion_expert_cuda_graph: Replay diffusion expert forwards with
                exact-shape CUDA graphs.
            diffusion_expert_cuda_graph_max_graphs: Maximum prompt-shape graphs
                retained before new shapes use eager execution.
            force_determinism: Whether to make stochastic inference repeatable from
                each prediction's inference seed.
            selection_strategy: How to pick one of the sampled trajectories.
            max_num_distance_points: Waypoints entering the selection distance
                average.
            skip_first_n_distance_points: Leading waypoints excluded from the
                selection distance average.
        """
        if force_determinism:
            configure_deterministic_runtime()
        logger.info("Loading Alpamayo 1 checkpoint from %s", checkpoint_path)

        model = AlpamayoR1.from_pretrained(
            checkpoint_path, dtype=self.DTYPE, attn_implementation="sdpa"
        ).to(device)
        model.eval()
        if diffusion_expert_cuda_graph:
            model.enable_diffusion_expert_cuda_graph(
                max_batch_size=num_traj_samples,
                max_graphs=diffusion_expert_cuda_graph_max_graphs,
            )
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
            force_determinism=force_determinism,
            selection_strategy=selection_strategy,
            max_num_distance_points=max_num_distance_points,
            skip_first_n_distance_points=skip_first_n_distance_points,
        )

        if diffusion_expert_cuda_graph:
            logger.info(
                "Diffusion expert CUDA graphs enabled with at most %d signatures",
                diffusion_expert_cuda_graph_max_graphs,
            )

    def _create_chat_message(
        self, image_frames: torch.Tensor, nav_text: str | None
    ) -> list:
        """Create chat message using Alpamayo 1's helper.

        Alpamayo 1 has no camera indices and no navigation conditioning, so
        ``nav_text`` is unused.
        """
        return self._helper.create_message(image_frames.flatten(0, 1))
