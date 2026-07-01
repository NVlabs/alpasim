# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from typing import Any

import torch

LANELINE_MAP_KEYS = ("lanelines", "lane_lines")


def agent_center_z_from_nearest_lanelines(
    map_data: dict[str, Any] | None,
    agent_xy: torch.Tensor,
    *,
    agent_lwh: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    fallback_z: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return agent center z from nearest laneline z plus half the agent height."""
    if agent_xy.shape[-1] < 2:
        raise ValueError(f"agent_xy must have trailing xy dims, got {agent_xy.shape}")

    target_shape = agent_xy.shape[:-1]
    if fallback_z is None:
        out_z = agent_xy.new_zeros(target_shape)
    elif torch.is_tensor(fallback_z):
        out_z = fallback_z.to(device=agent_xy.device, dtype=agent_xy.dtype).clone()
        if out_z.shape != target_shape:
            out_z = out_z.expand(target_shape).clone()
    else:
        out_z = agent_xy.new_full(target_shape, float(fallback_z))

    segments = _laneline_segments_from_map(
        map_data,
        device=agent_xy.device,
        dtype=agent_xy.dtype,
    )
    if segments is None:
        return out_z

    flat_xy = agent_xy[..., :2].reshape(-1, 2)
    flat_out_z = out_z.reshape(-1)
    query_mask = torch.isfinite(flat_xy).all(dim=-1)
    if valid_mask is not None:
        query_mask = query_mask & valid_mask.to(
            device=agent_xy.device,
            dtype=torch.bool,
        ).reshape(-1)
    if not bool(query_mask.any().item()):
        return out_z

    flat_out_z[query_mask] = (
        _nearest_segment_z(
            flat_xy[query_mask],
            segments,
        )
        + _agent_half_height(
            agent_lwh,
            target_shape=target_shape,
            device=agent_xy.device,
            dtype=agent_xy.dtype,
        ).reshape(-1)[query_mask]
    )
    return out_z


def _agent_half_height(
    agent_lwh: torch.Tensor | None,
    *,
    target_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if agent_lwh is None:
        return torch.zeros(target_shape, device=device, dtype=dtype)
    half_height = agent_lwh.to(device=device, dtype=dtype)[..., 2] * 0.5
    while half_height.ndim < len(target_shape):
        half_height = half_height.unsqueeze(-1)
    return half_height.expand(target_shape)


def _laneline_segments_from_map(
    map_data: dict[str, Any] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if not map_data:
        return None

    segment_chunks: list[torch.Tensor] = []
    for map_key in LANELINE_MAP_KEYS:
        map_layer = map_data.get(map_key)
        if not isinstance(map_layer, dict):
            continue
        polylines = map_layer.get("polylines")
        if polylines is None:
            continue
        segment_chunks.extend(
            _segments_from_polylines(polylines, device=device, dtype=dtype)
        )

    if not segment_chunks:
        return None
    return torch.cat(segment_chunks, dim=0)


def _segments_from_polylines(
    polylines: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    if isinstance(polylines, (list, tuple)):
        chunks: list[torch.Tensor] = []
        for polyline in polylines:
            chunks.extend(
                _segments_from_polylines(polyline, device=device, dtype=dtype)
            )
        return chunks

    polyline_tensor = torch.as_tensor(polylines, device=device, dtype=dtype)
    if polyline_tensor.ndim == 2:
        polyline_tensor = polyline_tensor.unsqueeze(0)
    if polyline_tensor.ndim != 3 or polyline_tensor.shape[-1] < 3:
        return []
    if polyline_tensor.shape[1] < 2:
        return []

    start = polyline_tensor[:, :-1, :3].reshape(-1, 3)
    end = polyline_tensor[:, 1:, :3].reshape(-1, 3)
    segment_xy = end[:, :2] - start[:, :2]
    valid = (
        torch.isfinite(start).all(dim=-1)
        & torch.isfinite(end).all(dim=-1)
        & (segment_xy.square().sum(dim=-1) > 1e-12)
    )
    if not bool(valid.any().item()):
        return []
    return [torch.stack((start[valid], end[valid]), dim=1)]


def _nearest_segment_z(
    query_xy: torch.Tensor,
    segments: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> torch.Tensor:
    segment_start_xy = segments[:, 0, :2]
    segment_delta_xy = segments[:, 1, :2] - segment_start_xy
    segment_start_z = segments[:, 0, 2]
    segment_delta_z = segments[:, 1, 2] - segment_start_z
    segment_len_sq = segment_delta_xy.square().sum(dim=-1).clamp_min(1e-12)

    nearest_z = query_xy.new_empty((query_xy.shape[0],))
    for chunk_start in range(0, int(query_xy.shape[0]), int(chunk_size)):
        chunk_stop = min(chunk_start + int(chunk_size), int(query_xy.shape[0]))
        chunk_xy = query_xy[chunk_start:chunk_stop]
        rel_xy = chunk_xy[:, None, :] - segment_start_xy[None, :, :]
        t = (
            (rel_xy * segment_delta_xy[None, :, :]).sum(dim=-1)
            / segment_len_sq[None, :]
        ).clamp(0.0, 1.0)
        closest_xy = segment_start_xy[None, :, :] + (
            t[..., None] * segment_delta_xy[None, :, :]
        )
        closest_dist_sq = (chunk_xy[:, None, :] - closest_xy).square().sum(dim=-1)
        nearest_segment_idx = closest_dist_sq.argmin(dim=-1)
        nearest_t = t[
            torch.arange(chunk_xy.shape[0], device=query_xy.device),
            nearest_segment_idx,
        ]
        nearest_z[chunk_start:chunk_stop] = segment_start_z[nearest_segment_idx] + (
            nearest_t * segment_delta_z[nearest_segment_idx]
        )
    return nearest_z
