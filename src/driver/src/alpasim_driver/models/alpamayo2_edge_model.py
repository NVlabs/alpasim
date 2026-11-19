# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Alpamayo 2 Edge adapter with checkpoint-native six-camera input sampling."""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Any

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, resize

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


class Alpamayo2EdgeModel(BaseTrajectoryModel):
    """Consume 10 Hz camera buffers and return 10 Hz poses in the planning rig."""

    DTYPE = torch.bfloat16
    CONTEXT_LENGTH = 21
    HISTORY_STEPS = 20
    OUTPUT_FREQUENCY_HZ = 10
    CAMERA_IDS = tuple(
        camera for camera, index in CAMERA_NAME_TO_INDEX.items() if index != 4
    )
    FRONT_OFFSETS_US = (-2_000_000, -1_000_000, -500_000, -100_000, 0)
    OTHER_OFFSETS_US = (-1_000_000, -100_000, 0)

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> Alpamayo2EdgeModel:
        if output_frequency_hz != cls.OUTPUT_FREQUENCY_HZ:
            raise ValueError("Alpamayo 2 Edge requires output_frequency_hz=10")
        if context_length not in (None, cls.CONTEXT_LENGTH):
            raise ValueError("Alpamayo 2 Edge requires context_length=21 at 10 Hz")
        if set(camera_ids) != set(cls.CAMERA_IDS) or len(camera_ids) != 6:
            raise ValueError(f"Alpamayo 2 Edge requires cameras {cls.CAMERA_IDS}")
        if model_cfg.cfg_guidance_weight is not None:
            raise ValueError("Alpamayo 2 Edge does not support navigation CFG")
        if model_cfg.diffusion_expert_cuda_graph:
            raise ValueError("This Edge adapter uses eager inference, not CUDA graphs")
        if model_cfg.trajectory_candidate_microbatch_size is not None:
            raise ValueError("Edge candidate microbatching is not implemented")
        if model_cfg.image_decode_device != "cpu":
            raise ValueError("Edge input preparation requires image_decode_device=cpu")
        return cls(model_cfg, device)

    def __init__(self, model_cfg: ModelConfig, device: torch.device):
        from alpamayo2_edge import helper
        from alpamayo2_edge.checkpoint import load_model
        from huggingface_hub import constants as hf_constants

        if model_cfg.num_trajectory_samples < 1:
            raise ValueError("num_trajectory_samples must be positive")
        self._selection = model_cfg.trajectory_selection
        self._strategy = TrajectorySelectionStrategy(self._selection.strategy)
        if (
            self._strategy != TrajectorySelectionStrategy.ALWAYS_FIRST
            and model_cfg.num_trajectory_samples < 2
        ):
            raise ValueError("Trajectory selection needs at least two candidates")
        if (
            self._selection.max_num_distance_points < 1
            or self._selection.skip_first_n_distance_points < 0
        ):
            raise ValueError("Invalid trajectory selection distance window")
        self._device = device
        self._num_samples = model_cfg.num_trajectory_samples
        self._deterministic = model_cfg.force_determinism
        if self._deterministic:
            configure_deterministic_runtime()
        self._helper = helper
        self._model, _ = load_model(
            model_cfg.checkpoint_path,
            local_files_only=hf_constants.HF_HUB_OFFLINE,
            device=str(device),
            dtype="bfloat16",
            attn_implementation="sdpa",
        )
        if not math.isclose(
            self._model.expert.action_space.dt, 0.1, rel_tol=1e-5, abs_tol=1e-6
        ):
            raise ValueError("Edge checkpoint must predict poses at 10 Hz")

    @property
    def camera_ids(self) -> list[str]:
        return list(self.CAMERA_IDS)

    @property
    def context_length(self) -> int:
        return self.CONTEXT_LENGTH

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> None:
        # Navigation is supplied as text, rather than a discrete turn command.
        return None

    def _prepare_sample(
        self, prediction: PredictionInput, t0_us: int
    ) -> dict[str, Any]:
        self._validate_cameras(prediction.camera_images)
        _validate_camera_frame_at_or_before_t0(
            prediction.camera_images,
            planning_t0_us=t0_us,
            model_name=type(self).__name__,
        )
        navigation = prediction.navigation
        if navigation is None or not navigation.instruction_text.strip():
            raise ModelInputValidationError("Edge requires nonempty navigation text")
        _validate_ego_history_span(
            prediction.ego_pose_history,
            planning_t0_us=t0_us,
            num_history_steps=self.HISTORY_STEPS,
            history_time_step=0.1,
            model_name=type(self).__name__,
        )
        xyz, rotations, origin = build_ego_history(
            prediction.ego_pose_history, t0_us, self.HISTORY_STEPS, 0.1
        )
        frames, indices, timestamps = [], [], []
        for camera in self.CAMERA_IDS:
            entries = prediction.camera_images[camera]
            times = [int(entry[0]) for entry in entries]
            if not times or any(a >= b for a, b in zip(times, times[1:])):
                raise ModelInputValidationError(f"{camera}: timestamps must increase")
            index = CAMERA_NAME_TO_INDEX[camera]
            offsets = (
                self.FRONT_OFFSETS_US if index in (1, 6) else self.OTHER_OFFSETS_US
            )
            for offset in offsets:
                target = t0_us + offset
                selected = bisect_right(times, target) - 1
                if selected < 0 or target - times[selected] > 100_000:
                    raise ModelInputValidationError(
                        f"{camera}: no frame within 100 ms at or before {target}"
                    )
                frame = torch.as_tensor(entries[selected][1])
                if (
                    frame.device.type != "cpu"
                    or frame.dtype != torch.uint8
                    or frame.ndim != 3
                    or frame.shape[-1] != 3
                ):
                    raise ModelInputValidationError(
                        f"{camera}: expected CPU uint8 HWC RGB"
                    )
                if frames and frame.shape != frames[0].shape:
                    # Rendered widths vary slightly between cameras in some rigs;
                    # the SDK's raw input requires one shared image shape.
                    frame = resize(
                        frame.permute(2, 0, 1),
                        list(frames[0].shape[:2]),
                        interpolation=InterpolationMode.BICUBIC,
                    ).permute(1, 2, 0)
                frames.append(frame)
                indices.append(index)
                timestamps.append(times[selected])
        return {
            "sample": {
                "image_frames": torch.stack(frames)
                .permute(0, 3, 1, 2)
                .unsqueeze(1)
                .contiguous(),
                "camera_indices": torch.tensor(indices, dtype=torch.int64),
                "absolute_timestamps": torch.tensor(timestamps, dtype=torch.int64),
                "ego_history_xyz": xyz,
                "ego_history_rot": rotations,
                "nav_text": [navigation.instruction_text],
            },
            "origin": origin,
        }

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        if not prediction_input.ego_pose_history:
            raise ModelInputValidationError("Edge requires ego pose history")
        t0_us = int(prediction_input.ego_pose_history[-1].timestamp_us)
        prepared = self._prepare_sample(prediction_input, t0_us)
        if self._deterministic:
            torch.manual_seed(prediction_input.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prediction_input.inference_seed)
        inputs = self._helper.prepare_model_inputs(
            prepared["sample"], self._model.config, self._model.tokenizer
        )
        inputs = self._helper.to_device(inputs, device=self._device)
        with (
            torch.inference_mode(),
            torch.autocast(self._device.type, dtype=self.DTYPE),
        ):
            xyz, rotations, _, extra = self._model.sample_trajectories_from_data(
                inputs,
                num_traj_sets=1,
                num_traj_samples=self._num_samples,
                max_generation_length=512,
                diffusion_kwargs={"inference_step": 10},
                return_extra=True,
            )
        positions = xyz[0, 0].float().cpu().numpy()
        rotations = rotations[0, 0].float().cpu().numpy()
        timestamps = (
            t0_us + np.arange(1, positions.shape[1] + 1, dtype=np.uint64) * 100_000
        )
        selected = select_trajectory(
            candidate_positions=positions,
            candidate_timestamps_us=timestamps,
            previous_plan_in_local=prediction_input.previous_plan,
            pose_local_to_rig_t0=prepared["origin"],
            strategy=self._strategy,
            max_num_distance_points=self._selection.max_num_distance_points,
            skip_first_n_distance_points=self._selection.skip_first_n_distance_points,
        )
        return ModelPrediction(
            candidate_positions=positions,
            candidate_rotations=rotations,
            selected_index=selected,
            reasoning_text=(
                str(extra["cot"][0, 0, selected]) if "cot" in extra else None
            ),
            model_t0_us=t0_us,
            pose_local_to_rig_t0=prepared["origin"],
            waypoint_timestamps_us=timestamps,
            selected_plan=plan_in_local_frame(
                positions[selected], rotations[selected], timestamps, prepared["origin"]
            ),
        )
