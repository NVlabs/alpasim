# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Collation helpers for variable-sized CATK inference batches."""

from typing import Any

import torch


def collate_model_inputs(
    input_data_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[int]]:
    """Collate variable-sized CATK graph inputs and return output split sizes."""
    if not input_data_items:
        raise ValueError("at least one CATK model input is required")

    # Accumulate each variable-sized field before concatenating the batch.
    device = input_data_items[0]["agent"]["id"].device
    agent_parts: dict[str, list[torch.Tensor]] = {}
    freeze_agent_parts: dict[str, list[torch.Tensor]] = {}
    map_triplets: list[torch.Tensor] = []
    map_thetas: list[torch.Tensor] = []
    polyline_parts: dict[str, list[torch.Tensor]] = {}
    freeze_masks: list[torch.Tensor] = []
    num_obstacles: list[int] = []
    non_ego_agent_counts: list[int] = []
    rb_polylines: list[torch.Tensor] = []
    rb_polylines_batch: list[int] = []

    for graph_idx, input_data in enumerate(input_data_items):
        # Collect active agents and assign them to their source graph.
        agent_data = input_data["agent"]
        agent_count = int(agent_data["id"].shape[0])
        non_ego_agent_counts.append(int((~agent_data["role"][:, 0]).sum().item()))
        num_obstacles.append(agent_count)
        for key, value in agent_data.items():
            if key == "batch":
                continue
            agent_parts.setdefault(key, []).append(value)
        agent_parts.setdefault("batch", []).append(
            torch.full(
                (agent_count,),
                graph_idx,
                dtype=torch.long,
                device=agent_data["id"].device,
            )
        )

        # Collect frozen agents and their masks separately from active agents.
        freeze_data = input_data["freeze_agent_data"]
        freeze_agents = freeze_data["agent"]
        freeze_agent_count = int(freeze_agents["id"].shape[0])
        for key, value in freeze_agents.items():
            if key == "batch":
                continue
            freeze_agent_parts.setdefault(key, []).append(value)
        freeze_agent_parts.setdefault("batch", []).append(
            torch.full(
                (freeze_agent_count,),
                graph_idx,
                dtype=torch.long,
                device=freeze_agents["id"].device,
            )
        )
        freeze_masks.append(input_data["freeze_agent_mask"])

        # Collect map polylines and assign them to their source graph.
        map_data = input_data["map"]
        map_triplets.append(map_data["triplets"])
        map_thetas.append(map_data["triplet_thetas"])
        polyline_extras = map_data["polyline_extras"]
        polyline_count = int(map_data["triplets"].shape[0])
        for key, value in polyline_extras.items():
            if key == "batch":
                continue
            polyline_parts.setdefault(key, []).append(value)
        polyline_parts.setdefault("batch", []).append(
            torch.full(
                (polyline_count,),
                graph_idx,
                dtype=torch.long,
                device=map_data["triplets"].device,
            )
        )

        # Preserve variable-length road-boundary polylines as a flat list.
        rb_data = map_data.get("rb_data")
        if rb_data is not None:
            for polyline in rb_data["rb_polylines"]:
                rb_polylines.append(polyline)
                rb_polylines_batch.append(graph_idx)

    # Concatenate tensor fields into the structure expected by CATK.
    collated = {
        "agent": {key: torch.cat(values, dim=0) for key, values in agent_parts.items()},
        "num_obstacles": torch.tensor(
            num_obstacles,
            dtype=torch.long,
            device=device,
        ),
        "num_graphs": len(input_data_items),
        "freeze_agent_data": {
            "agent": {
                key: torch.cat(values, dim=0)
                for key, values in freeze_agent_parts.items()
            },
            "num_obstacles": torch.tensor(
                [
                    int(item["freeze_agent_data"]["agent"]["id"].shape[0])
                    for item in input_data_items
                ],
                dtype=torch.long,
                device=device,
            ),
            "num_graphs": len(input_data_items),
        },
        "freeze_agent_mask": torch.cat(freeze_masks, dim=0),
        "map": {
            "triplets": torch.cat(map_triplets, dim=0),
            "triplet_thetas": torch.cat(map_thetas, dim=0),
            "polyline_extras": {
                key: torch.cat(values, dim=0) for key, values in polyline_parts.items()
            },
            "rb_data": {
                "rb_polylines": rb_polylines,
                "rb_polylines_batch": rb_polylines_batch,
            },
        },
    }
    return collated, non_ego_agent_counts
