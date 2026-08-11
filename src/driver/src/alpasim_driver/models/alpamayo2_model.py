# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 2 Super wrapper implementing the common trajectory-model interface.

Alpamayo 2 Super is a 32B expert reasoning VLA. Unlike Alpamayo 1/1.5, its
released inference path (``alpamayo2_super``) assembles model inputs through its
own ``helper.prepare_model_inputs`` and produces the future trajectory with a
diffusion action expert rather than a discrete trajectory tokenizer, so this
wrapper implements :class:`BaseTrajectoryModel` directly instead of sharing the
Alpamayo 1.x template in :mod:`alpamayo_base`. It still reuses the family's ego
history construction and rig-frame conventions.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
import torch
from alpamayo2_super import helper
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

from ..schema import ModelConfig
from .alpamayo_base import (
    CAMERA_NAME_TO_INDEX,
    _validate_camera_frame_at_or_before_t0,
    _validate_ego_history_span,
    build_ego_history,
    configure_deterministic_runtime,
)
from .base import (
    BaseTrajectoryModel,
    CameraImages,
    DriveCommand,
    ModelInputValidationError,
    ModelPrediction,
    PredictionInput,
)
from .trajectory_selection import (
    TrajectorySelectionStrategy,
    plan_in_local_frame,
    select_trajectory,
)

logger = logging.getLogger(__name__)


class Alpamayo2Model(BaseTrajectoryModel):
    """Alpamayo 2 Super wrapper implementing the common interface."""

    DTYPE: torch.dtype = torch.bfloat16
    NUM_HISTORY_STEPS: int = 16
    HISTORY_TIME_STEP: float = 0.1
    DEFAULT_CONTEXT_LENGTH: int = 4
    OUTPUT_FREQUENCY_HZ: int = 10
    IMAGE_INPUT_FREQUENCY_HZ: int = 10
    # Token budget for the chain of thought generated before the trajectory
    # tokens.  Passed explicitly rather than left to the checkpoint's fallback,
    # so the budget the expert conditions on is visible at the call site.
    MAX_GENERATION_LENGTH: int = 256

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> Alpamayo2Model:
        """Create an Alpamayo2Model from driver configuration."""
        if output_frequency_hz != cls.OUTPUT_FREQUENCY_HZ:
            raise ValueError(
                f"{cls.__name__} predicts waypoints at {cls.OUTPUT_FREQUENCY_HZ}Hz "
                f"and cannot resample, but output_frequency_hz is "
                f"{output_frequency_hz}."
            )
        if model_cfg.cfg_guidance_weight is not None:
            raise ValueError(
                "Alpamayo2Model navigation CFG is not available through the standard "
                "driver API. The upstream implementation requires explicit two-GPU "
                "VLM/expert placement and paired guided/unguided KV caches."
            )
        selection = model_cfg.trajectory_selection
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
            context_length=context_length or cls.DEFAULT_CONTEXT_LENGTH,
            force_determinism=model_cfg.force_determinism,
            num_traj_samples=model_cfg.num_trajectory_samples,
            trajectory_candidate_microbatch_size=(
                model_cfg.trajectory_candidate_microbatch_size
            ),
            selection_strategy=TrajectorySelectionStrategy(selection.strategy),
            max_num_distance_points=selection.max_num_distance_points,
            skip_first_n_distance_points=selection.skip_first_n_distance_points,
        )

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        num_traj_samples: int = 1,
        trajectory_candidate_microbatch_size: int | None = None,
        top_p: float = 0.98,
        temperature: float = 0.6,
        diffusion_steps: int = 10,
        force_determinism: bool = False,
        selection_strategy: TrajectorySelectionStrategy = (
            TrajectorySelectionStrategy.ALWAYS_FIRST
        ),
        max_num_distance_points: int = 64,
        skip_first_n_distance_points: int = 0,
    ):
        """Initialize Alpamayo 2 Super model.

        Args:
            checkpoint_path: Path or HuggingFace model ID for a release-native
                Alpamayo 2 Super checkpoint (``config.json`` + tokenizer/processor
                files + ``*.safetensors`` shards).
            device: Torch device for inference.
            camera_ids: Camera logical IDs; must be a subset of the Alpamayo rig.
            context_length: Number of temporal frames per camera.
            num_traj_samples: Trajectory samples to draw before selection.
            trajectory_candidate_microbatch_size: Maximum candidates sampled in
                each sequential model call. Unset samples all candidates at once.
            top_p: Top-p sampling parameter for CoT generation.
            temperature: Temperature for CoT sampling.
            diffusion_steps: Euler integration steps for the action-expert flow
                matching.
            force_determinism: Whether to make stochastic inference repeatable
                from each prediction's inference seed.
            selection_strategy: How to select one sampled trajectory.
            max_num_distance_points: Waypoints entering the selection distance
                average.
            skip_first_n_distance_points: Leading waypoints excluded from the
                selection distance average.
        """
        if context_length < 1:
            raise ValueError(
                f"context_length must be at least 1, got {context_length}."
            )
        if num_traj_samples < 1:
            raise ValueError(
                f"num_traj_samples must be at least 1, got {num_traj_samples}."
            )
        if (
            trajectory_candidate_microbatch_size is not None
            and trajectory_candidate_microbatch_size < 1
        ):
            raise ValueError(
                "trajectory_candidate_microbatch_size must be at least 1 when "
                "set, got "
                f"{trajectory_candidate_microbatch_size}."
            )
        if (
            selection_strategy is not TrajectorySelectionStrategy.ALWAYS_FIRST
            and num_traj_samples < 2
        ):
            raise ValueError(
                f"Trajectory selection strategy {selection_strategy} needs at least "
                f"2 samples to choose from, got num_traj_samples={num_traj_samples}."
            )
        if max_num_distance_points < 1:
            raise ValueError(
                "max_num_distance_points must be at least 1, got "
                f"{max_num_distance_points}."
            )
        if skip_first_n_distance_points < 0:
            raise ValueError(
                "skip_first_n_distance_points must not be negative, got "
                f"{skip_first_n_distance_points}."
            )
        if diffusion_steps < 1:
            raise ValueError(
                f"diffusion_steps must be at least 1, got {diffusion_steps}."
            )

        missing_cameras = [c for c in camera_ids if c not in CAMERA_NAME_TO_INDEX]
        if missing_cameras:
            raise ValueError(
                f"Cameras {missing_cameras} not found in Alpamayo camera mapping."
            )

        if force_determinism:
            configure_deterministic_runtime()

        logger.info("Loading Alpamayo 2 Super checkpoint from %s", checkpoint_path)
        # transformers 4.57.x runs local paths through huggingface_hub's repo-ID
        # validator, causing HFValidationError. local_files_only bypasses that path;
        # HF model IDs (no local dir) go through the normal hub flow.
        load_kwargs: dict = {}
        if os.path.isdir(os.path.expanduser(checkpoint_path)):
            load_kwargs["local_files_only"] = True
        self._model = Alpamayo2Super.from_pretrained(
            checkpoint_path,
            dtype=self.DTYPE,
            device_map=str(device),
            **load_kwargs,
        )
        self._model.eval()

        action_space = self._model.expert.action_space
        self._pred_num_waypoints, _ = action_space.get_action_space_dims()
        reported_time_step = 1.0 / self.OUTPUT_FREQUENCY_HZ
        if not math.isclose(
            action_space.dt, reported_time_step, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError(
                f"{self.__class__.__name__} checkpoint predicts a waypoint every "
                f"{action_space.dt}s, but the driver reports waypoints every "
                f"{reported_time_step}s (OUTPUT_FREQUENCY_HZ="
                f"{self.OUTPUT_FREQUENCY_HZ})."
            )

        self._device = device
        self._camera_ids = camera_ids
        self._context_length = context_length
        self._num_traj_samples = num_traj_samples
        self._trajectory_candidate_microbatch_size = (
            trajectory_candidate_microbatch_size
        )
        self._top_p = top_p
        self._temperature = temperature
        self._diffusion_steps = diffusion_steps
        self._force_determinism = force_determinism
        self._selection_strategy = selection_strategy
        self._max_num_distance_points = max_num_distance_points
        self._skip_first_n_distance_points = skip_first_n_distance_points

        logger.info(
            "Initialised Alpamayo2Model with %d cameras, context_length=%d, "
            "num_traj_samples=%d, trajectory_candidate_microbatch_size=%s, "
            "selection_strategy=%s",
            len(camera_ids),
            context_length,
            num_traj_samples,
            trajectory_candidate_microbatch_size,
            selection_strategy,
        )

    # ------------------------------------------------------------------
    # BaseTrajectoryModel interface
    # ------------------------------------------------------------------

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> Any:
        """Alpamayo 2 Super reasons about navigation from context.

        The released inference path takes no discrete command; route intent, when
        used, is supplied as free-form language. This wrapper runs the plain
        (no-route) path, so the canonical command is unused.
        """
        return None

    # ------------------------------------------------------------------
    # Input assembly
    # ------------------------------------------------------------------

    def _sorted_camera_ids(self) -> list[str]:
        """Camera IDs sorted ascending by their Alpamayo rig index."""
        return sorted(self._camera_ids, key=lambda cam_id: CAMERA_NAME_TO_INDEX[cam_id])

    def _preprocess_images(self, camera_images: CameraImages) -> torch.Tensor:
        """Stack multi-camera frames into ``(N_cameras, num_frames, 3, H, W)``.

        Frames stay uint8 so ``helper.prepare_model_inputs`` performs the
        ``/255`` rescaling. Cameras are ordered by their rig index to match
        ``camera_indices``.
        """
        frames_list = []
        for cam_id in self._sorted_camera_ids():
            camera_frames = [
                # as_tensor takes a host array or a frame already on the
                # inference device.
                torch.as_tensor(img).permute(2, 0, 1)  # HWC uint8 -> CHW uint8
                for _, img in camera_images[cam_id]
            ]
            frames_list.append(torch.stack(camera_frames, dim=0))
        return torch.stack(frames_list, dim=0)

    def _camera_indices(self) -> torch.Tensor:
        """Ascending rig indices for the configured cameras."""
        return torch.tensor(
            [CAMERA_NAME_TO_INDEX[cam_id] for cam_id in self._sorted_camera_ids()],
            dtype=torch.int64,
        )

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate a single trajectory prediction from cameras and ego history."""
        if self._force_determinism:
            torch.manual_seed(prediction_input.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prediction_input.inference_seed)

        self._validate_cameras(prediction_input.camera_images)

        for cam_id in self._camera_ids:
            n_frames = len(prediction_input.camera_images[cam_id])
            if n_frames != self._context_length:
                raise ModelInputValidationError(
                    f"Alpamayo2Model expects {self._context_length} frames per "
                    f"camera, got {n_frames} for {cam_id}."
                )

        if not prediction_input.ego_pose_history:
            raise ModelInputValidationError(
                "Alpamayo2Model needs ego pose history to select planning t0."
            )
        # Match the internal RL handler: Session keeps poses sorted, and t0 is
        # the latest submitted ego timestamp rather than the newest camera time.
        planning_t0_us = int(prediction_input.ego_pose_history[-1].timestamp_us)
        _validate_camera_frame_at_or_before_t0(
            prediction_input.camera_images,
            planning_t0_us=planning_t0_us,
            model_name=self.__class__.__name__,
        )
        _validate_ego_history_span(
            prediction_input.ego_pose_history,
            planning_t0_us=planning_t0_us,
            num_history_steps=self.NUM_HISTORY_STEPS,
            history_time_step=self.HISTORY_TIME_STEP,
            model_name=self.__class__.__name__,
        )
        ego_history_xyz, ego_history_rot, pose_local_to_rig_t0 = build_ego_history(
            prediction_input.ego_pose_history,
            planning_t0_us,
            self.NUM_HISTORY_STEPS,
            self.HISTORY_TIME_STEP,
        )

        data = {
            "image_frames": self._preprocess_images(prediction_input.camera_images),
            "camera_indices": self._camera_indices(),
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        model_inputs = helper.prepare_model_inputs(
            data, self._model.config, self._model.tokenizer
        )
        model_inputs = helper.to_device(model_inputs, self._device)

        pred_xyz, pred_rot, _logprob, extra = self._sample_trajectory_candidates(
            model_inputs
        )

        # Candidates are in the rig frame at t0, shaped
        # [batch=1, num_traj_sets=1, K, T, ...] -> (K, T, ...).
        candidate_positions = pred_xyz[0, 0].float().cpu().numpy()
        candidate_rotations = pred_rot[0, 0].float().cpu().numpy()
        waypoint_timestamps_us = self._waypoint_timestamps_us(
            planning_t0_us, num_waypoints=candidate_positions.shape[1]
        )
        select_ix = select_trajectory(
            candidate_positions=candidate_positions,
            candidate_timestamps_us=waypoint_timestamps_us,
            previous_plan_in_local=prediction_input.previous_plan,
            pose_local_to_rig_t0=pose_local_to_rig_t0,
            strategy=self._selection_strategy,
            max_num_distance_points=self._max_num_distance_points,
            skip_first_n_distance_points=self._skip_first_n_distance_points,
        )
        selected_positions = candidate_positions[select_ix]
        selected_rotations = candidate_rotations[select_ix]

        reasoning_text = None
        if "cot" in extra:
            reasoning_text = str(extra["cot"][0, 0, select_ix])
            logger.info("Alpamayo2Model Chain-of-Causation: %s", reasoning_text)

        return ModelPrediction(
            candidate_positions=candidate_positions,
            candidate_rotations=candidate_rotations,
            selected_index=select_ix,
            reasoning_text=reasoning_text,
            model_t0_us=planning_t0_us,
            pose_local_to_rig_t0=pose_local_to_rig_t0,
            waypoint_timestamps_us=waypoint_timestamps_us,
            selected_plan=plan_in_local_frame(
                selected_positions,
                selected_rotations,
                waypoint_timestamps_us,
                pose_local_to_rig_t0,
            ),
        )

    def _sample_trajectory_candidates(
        self, model_inputs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
        """Sample candidates sequentially when a microbatch limit is configured."""
        microbatch_size = self._trajectory_candidate_microbatch_size
        if microbatch_size is None or microbatch_size >= self._num_traj_samples:
            sample_sizes = [self._num_traj_samples]
        else:
            sample_sizes = [
                min(microbatch_size, self._num_traj_samples - start)
                for start in range(0, self._num_traj_samples, microbatch_size)
            ]

        samples = []
        with torch.no_grad(), torch.autocast(self._device.type, dtype=self.DTYPE):
            for sample_size in sample_sizes:
                samples.append(
                    self._model.sample_trajectories_from_data(
                        data=model_inputs,
                        top_p=self._top_p,
                        temperature=self._temperature,
                        num_traj_samples=sample_size,
                        max_generation_length=self.MAX_GENERATION_LENGTH,
                        diffusion_kwargs={"inference_step": self._diffusion_steps},
                        return_extra=True,
                    )
                )
                if len(samples) != len(sample_sizes) and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if len(samples) == 1:
            return samples[0]

        pred_xyz_samples, pred_rot_samples, logprob_samples, extra_samples = zip(
            *samples, strict=True
        )
        extra_keys = extra_samples[0].keys()
        if any(sample.keys() != extra_keys for sample in extra_samples[1:]):
            raise RuntimeError(
                "Alpamayo2 candidate microbatches returned inconsistent extra keys."
            )
        return (
            torch.cat(pred_xyz_samples, dim=2),
            torch.cat(pred_rot_samples, dim=2),
            torch.cat(logprob_samples, dim=2),
            {
                key: np.concatenate([sample[key] for sample in extra_samples], axis=2)
                for key in extra_keys
            },
        )

    def _waypoint_timestamps_us(self, t0_us: int, num_waypoints: int) -> np.ndarray:
        """Timestamps of predicted waypoints, starting one step after t0."""
        step_us = 1_000_000 // self.output_frequency_hz
        return t0_us + np.arange(1, num_waypoints + 1, dtype=np.uint64) * step_us
