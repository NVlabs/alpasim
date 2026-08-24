# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import inspect

import torch
from navsim_gtrs_dense_challenge.simscale_gtrs_dense import (
    attention as attention_module,
)
from navsim_gtrs_dense_challenge.simscale_gtrs_dense.attention import (
    Attention,
    MemoryEffTransformer,
    attention,
    memory_efficient_attention,
)


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    return (
        torch.randn(2, 2, 7, 4),
        torch.randn(2, 2, 9, 4),
        torch.randn(2, 2, 9, 4),
    )


def test_chunked_attention_matches_regular_attention() -> None:
    q, k, v = _qkv()

    expected = attention(q, k, v)
    actual = memory_efficient_attention(
        q,
        k,
        v,
        q_bucket_size=3,
        k_bucket_size=4,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_chunked_attention_matches_regular_attention_with_mask() -> None:
    q, k, v = _qkv()
    mask = torch.tensor(
        [
            [True, True, True, False, True, False, True, True, False],
            [True, False, True, True, False, True, True, False, True],
        ]
    )

    expected = attention(q, k, v, mask=mask)
    actual = memory_efficient_attention(
        q,
        k,
        v,
        mask=mask,
        q_bucket_size=3,
        k_bucket_size=4,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_chunked_attention_matches_regular_causal_attention() -> None:
    torch.manual_seed(11)
    q = torch.randn(1, 2, 9, 4)
    k = torch.randn(1, 2, 9, 4)
    v = torch.randn(1, 2, 9, 4)

    expected = attention(q, k, v, causal=True)
    actual = memory_efficient_attention(
        q,
        k,
        v,
        causal=True,
        q_bucket_size=4,
        k_bucket_size=3,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_module_preserves_batch_token_shape() -> None:
    module = Attention(dim=16, heads=4, dim_head=4, memory_efficient=True)
    inputs = torch.randn(2, 13, 16)

    actual = module(inputs, inputs, inputs)

    assert actual.shape == (2, 13, 16)


def test_memory_efficient_transformer_preserves_shape() -> None:
    module = MemoryEffTransformer(
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 13, 16)

    actual = module(inputs)

    assert actual.shape == inputs.shape
    assert torch.isfinite(actual).all()


def test_attention_source_has_no_einops_dependency() -> None:
    source = inspect.getsource(attention_module)

    assert "einops" not in source
    assert "rearrange(" not in source
