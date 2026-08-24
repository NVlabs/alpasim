# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Minimal DDIM scheduler used by the vendored inference graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DDIMSchedulerOutput:
    """Output of one deterministic DDIM reverse step."""

    prev_sample: torch.Tensor


class DDIMScheduler:
    """Diffusers-compatible subset for SimScale sample prediction."""

    def __init__(
        self,
        num_train_timesteps: int,
        beta_schedule: str,
        prediction_type: str,
    ) -> None:
        if not isinstance(num_train_timesteps, int) or num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be a positive integer")
        if beta_schedule != "scaled_linear":
            raise ValueError("only the scaled_linear beta schedule is supported")
        if prediction_type != "sample":
            raise ValueError("only sample prediction is supported")

        self.num_train_timesteps = num_train_timesteps
        self.betas = (
            torch.linspace(
                0.0001**0.5,
                0.02**0.5,
                num_train_timesteps,
                dtype=torch.float32,
            )
            ** 2
        )
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0, dtype=torch.float32)
        self.num_inference_steps: int | None = None
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1, dtype=torch.long)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None,
    ) -> None:
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps cannot exceed num_train_timesteps")

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            torch.arange(num_inference_steps, dtype=torch.long, device=device)
            .mul(step_ratio)
            .flip(0)
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if original_samples.shape != noise.shape:
            raise ValueError("original_samples and noise must have identical shapes")

        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        timesteps = timesteps.to(device=original_samples.device, dtype=torch.long)
        sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt().flatten()
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt().flatten()
        while sqrt_alpha_prod.ndim < original_samples.ndim:
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> DDIMSchedulerOutput:
        if self.num_inference_steps is None:
            raise ValueError("set_timesteps must be called before step")
        if model_output.shape != sample.shape:
            raise ValueError("model_output and sample must have identical shapes")

        timestep_value = (
            int(timestep.item()) if isinstance(timestep, torch.Tensor) else timestep
        )
        if timestep_value < 0 or timestep_value >= self.num_train_timesteps:
            raise ValueError("timestep is outside the training schedule")
        prev_timestep = (
            timestep_value - self.num_train_timesteps // self.num_inference_steps
        )

        alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alpha_prod_t = alphas_cumprod[timestep_value]
        alpha_prod_t_prev = (
            alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod.to(device=sample.device)
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = model_output.clamp(-1, 1)
        pred_epsilon = (
            sample - alpha_prod_t.sqrt() * model_output
        ) / beta_prod_t.sqrt()
        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * pred_epsilon
        prev_sample = (
            alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        )
        return DDIMSchedulerOutput(prev_sample=prev_sample.to(dtype=sample.dtype))
