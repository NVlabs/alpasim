# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _exists(value: object | None) -> bool:
    return value is not None


def _default(value: object | None, fallback: object) -> object:
    return value if _exists(value) else fallback


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    causal: bool = False,
    attn_bias: Tensor | None = None,
    **_: object,
) -> Tensor:
    q = q * (q.shape[-1] ** -0.5)
    similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k)
    if attn_bias is not None:
        similarity = similarity + attn_bias

    mask_value = -torch.finfo(similarity.dtype).max
    if mask is not None:
        similarity = similarity.masked_fill(~mask[:, None, None, :], mask_value)

    if causal:
        query_count, key_count = similarity.shape[-2:]
        causal_mask = torch.ones(
            query_count,
            key_count,
            device=q.device,
            dtype=torch.bool,
        ).triu(key_count - query_count + 1)
        similarity = similarity.masked_fill(causal_mask, mask_value)

    similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
    weights = similarity.softmax(dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", weights, v)


def _summarize_qkv_chunk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None,
    attn_bias_chunk: Tensor | None,
    causal: bool,
    qk_start_indices: tuple[int, int],
    dropout: float,
) -> tuple[Tensor, Tensor, Tensor]:
    q_start_index, k_start_index = qk_start_indices
    q_chunk_size = q.shape[-2]
    k_chunk_size = k.shape[-2]

    weight = torch.einsum("b h i d, b h j d -> b h i j", q, k)
    if attn_bias_chunk is not None:
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max
    if mask is not None:
        weight = weight.masked_fill(~mask[:, None, None, :], mask_value)

    if causal and q_start_index < k_start_index + k_chunk_size - 1:
        causal_mask = torch.ones(
            (q_chunk_size, k_chunk_size),
            dtype=torch.bool,
            device=q.device,
        ).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim=-1, keepdim=True).detach()
    exp_weight = (weight - weight_max).exp()
    exp_weight = F.dropout(exp_weight, p=dropout)
    weighted_value = torch.einsum("b h i j, b h j d -> b h i d", exp_weight, v)
    return exp_weight.sum(dim=-1), weighted_value, weight_max.squeeze(-1)


_checkpointed_summarize_qkv_chunk = partial(
    checkpoint,
    _summarize_qkv_chunk,
    use_reentrant=False,
)


def memory_efficient_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    causal: bool = False,
    attn_bias: Tensor | None = None,
    q_bucket_size: int = 512,
    k_bucket_size: int = 1024,
    eps: float = 1e-8,
    dropout: float = 0.0,
    training: bool = False,
) -> Tensor:
    q = q * (q.shape[-1] ** -0.5)
    summarize = (
        _checkpointed_summarize_qkv_chunk
        if q.requires_grad or k.requires_grad or v.requires_grad
        else _summarize_qkv_chunk
    )

    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)
    mask_chunks = (
        mask.split(k_bucket_size, dim=-1)
        if mask is not None
        else (None,) * len(k_chunks)
    )

    attn_bias_chunks: list[tuple[Tensor, ...]] | None = None
    if attn_bias is not None:
        attn_bias_chunks = [
            chunk.split(k_bucket_size, dim=-1)
            for chunk in attn_bias.split(q_bucket_size, dim=-2)
        ]

    output_chunks: list[Tensor] = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights: list[Tensor] = []
        weighted_values: list[Tensor] = []
        weight_maxes: list[Tensor] = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(
            zip(k_chunks, v_chunks, mask_chunks, strict=True)
        ):
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size
            if causal and k_start_index > q_start_index + q_chunk.shape[-2] - 1:
                continue

            bias_chunk = (
                attn_bias_chunks[q_index][k_index]
                if attn_bias_chunks is not None
                else None
            )
            exp_weight, weighted_value, weight_max = summarize(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                bias_chunk,
                causal,
                (q_start_index, k_start_index),
                dropout if training else 0.0,
            )
            exp_weights.append(exp_weight)
            weighted_values.append(weighted_value)
            weight_maxes.append(weight_max)

        stacked_maxes = torch.stack(weight_maxes, dim=-1)
        stacked_values = torch.stack(weighted_values, dim=-1)
        stacked_weights = torch.stack(exp_weights, dim=-1)
        global_max = stacked_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (stacked_maxes - global_max).exp().detach()
        stacked_weights = stacked_weights * renorm_factor
        stacked_values = stacked_values * renorm_factor.unsqueeze(-2)

        all_values = stacked_values.sum(dim=-1)
        all_weights = stacked_weights.sum(dim=-1)
        output_chunks.append(all_values / (all_weights.unsqueeze(-1) + eps))

    return torch.cat(output_chunks, dim=-2)


class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        memory_efficient: bool = False,
        q_bucket_size: int = 512,
        k_bucket_size: int = 1024,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.dropout = dropout
        self.memory_efficient = memory_efficient
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
        memory_efficient: bool | None = None,
        q_bucket_size: int | None = None,
        k_bucket_size: int | None = None,
    ) -> Tensor:
        use_chunked = bool(_default(memory_efficient, self.memory_efficient))
        query_bucket = int(_default(q_bucket_size, self.q_bucket_size))
        key_bucket = int(_default(k_bucket_size, self.k_bucket_size))
        heads = self.heads

        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        batch, query_tokens, inner_dim = q.shape
        dim_head = inner_dim // heads
        q = q.reshape(batch, query_tokens, heads, dim_head).permute(0, 2, 1, 3)
        k = k.reshape(batch, k.shape[1], heads, dim_head).permute(0, 2, 1, 3)
        v = v.reshape(batch, v.shape[1], heads, dim_head).permute(0, 2, 1, 3)

        attention_fn = memory_efficient_attention if use_chunked else attention
        output = attention_fn(
            q,
            k,
            v,
            mask=mask,
            attn_bias=attn_bias,
            causal=self.causal,
            q_bucket_size=query_bucket,
            k_bucket_size=key_bucket,
            dropout=self.dropout,
            training=self.training,
        )
        output = output.permute(0, 2, 1, 3).reshape(batch, query_tokens, inner_dim)
        return self.to_out(output)


class MemoryEffTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: object = F.relu,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(
            dim=d_model,
            heads=nhead,
            dim_head=d_model // nhead,
            memory_efficient=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self, x: Tensor | tuple[Tensor, Tensor, Tensor], need_mean: bool = False
    ) -> Tensor:
        if isinstance(x, tuple):
            q, k, v = x
        else:
            q = k = v = x
        residual = q
        attended = self.self_attn(q, k, v)
        if need_mean:
            num_query, embed_dims = q.shape[1:]
            batch = q.shape[0] // 2
            attended = attended.view(num_query, embed_dims, batch, 2).mean(-1)
            attended = attended.permute(2, 0, 1)
            residual = q[batch:]
        q = self.norm1(residual + self.dropout1(attended))
        feed_forward = self.linear2(self.dropout(self.activation(self.linear1(q))))
        return self.norm3(q + self.dropout3(feed_forward))
