# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Numerical parity tests for the local DDIM scheduler."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from navsim_diffusiondrive_challenge.simscale_diffusiondrive.config import (
    DiffusionDriveConfig,
    TrajectorySampling,
)
from navsim_diffusiondrive_challenge.simscale_diffusiondrive.scheduler import (
    DDIMScheduler,
    DDIMSchedulerOutput,
)


def _scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )


def test_scaled_linear_schedule_matches_diffusers_0_35_1() -> None:
    scheduler = _scheduler()

    expected_betas = torch.tensor(
        [
            9.99999974737875164e-05,
            1.02648366009816527e-04,
            1.28041196148842573e-04,
            2.00000014156103134e-02,
        ],
        dtype=torch.float32,
    )
    expected_alphas = torch.tensor(
        [
            9.99899983406066895e-01,
            9.99002158641815186e-01,
            9.98749315738677979e-01,
            7.33411987312138081e-04,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(scheduler.betas[[0, 1, 10, 999]], expected_betas)
    torch.testing.assert_close(
        scheduler.alphas_cumprod[[0, 8, 10, 999]], expected_alphas
    )


def test_add_noise_matches_diffusers_and_preserves_dtype() -> None:
    scheduler = _scheduler()
    sample = torch.tensor([[[[0.25, -0.5], [1.0, -1.5]]]], dtype=torch.float64)
    noise = torch.tensor([[[[0.1, -0.2], [0.3, -0.4]]]], dtype=torch.float64)
    expected = torch.tensor(
        [
            [
                [
                    [0.25338011702032553, -0.50676023404065107],
                    [1.00998396661612211, -1.51320769919159326],
                ]
            ]
        ],
        dtype=torch.float64,
    )

    actual = scheduler.add_noise(sample, noise, torch.tensor([10]))

    assert actual.dtype == sample.dtype
    assert actual.device == sample.device
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("timestep", "expected"),
    [
        (
            10,
            [
                [
                    [
                        [-0.46730638606064706, 0.21727466345147517],
                        [0.37245604535059601, -0.60351769981775771],
                    ]
                ]
            ],
        ),
        (0, [[[[-0.75, 0.5], [0.125, -0.25]]]]),
    ],
)
def test_step_matches_diffusers_0_35_1(
    timestep: int, expected: list[list[list[list[float]]]]
) -> None:
    scheduler = _scheduler()
    scheduler.set_timesteps(100, device=torch.device("cpu"))
    sample = torch.tensor([[[[0.25, -0.5], [1.0, -1.5]]]], dtype=torch.float64)
    model_output = torch.tensor([[[[-0.75, 0.5], [0.125, -0.25]]]], dtype=torch.float64)

    output = scheduler.step(model_output, timestep, sample)

    assert isinstance(output, DDIMSchedulerOutput)
    assert output.prev_sample.dtype == sample.dtype
    assert output.prev_sample.device == sample.device
    torch.testing.assert_close(
        output.prev_sample,
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(FrozenInstanceError):
        output.prev_sample = sample


def test_set_timesteps_matches_leading_spacing() -> None:
    scheduler = _scheduler()

    scheduler.set_timesteps(100, device=torch.device("cpu"))

    assert scheduler.num_inference_steps == 100
    assert scheduler.timesteps[:3].tolist() == [990, 980, 970]
    assert scheduler.timesteps[-3:].tolist() == [20, 10, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "num_train_timesteps": 0,
            "beta_schedule": "scaled_linear",
            "prediction_type": "sample",
        },
        {
            "num_train_timesteps": 1000,
            "beta_schedule": "linear",
            "prediction_type": "sample",
        },
        {
            "num_train_timesteps": 1000,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
    ],
)
def test_constructor_rejects_unsupported_options(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DDIMScheduler(**kwargs)


def test_config_uses_local_trajectory_sampling() -> None:
    config = DiffusionDriveConfig()

    assert config.trajectory_sampling == TrajectorySampling(
        num_poses=8,
        time_horizon=4.0,
        interval_length=0.5,
    )
