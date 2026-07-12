# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any

import pytest
import torch
from alpasim_trafficsim.catk.batching import collate_model_inputs
from alpasim_trafficsim.grpc.catk_predictor import _InferenceBatcher, _PreparedInference


def _input_data(*, non_ego_agents: int, triplets: int) -> dict[str, Any]:
    total_agents = non_ego_agents + 1
    role = torch.zeros((total_agents, 3), dtype=torch.bool)
    role[-1, 0] = True
    role[:, 1:] = True
    agent = {
        "id": torch.arange(total_agents, dtype=torch.long),
        "role": role,
        "type": torch.zeros((total_agents,), dtype=torch.long),
        "valid_mask": torch.ones((total_agents, 2), dtype=torch.bool),
        "position": torch.zeros((total_agents, 2, 3)),
        "heading": torch.zeros((total_agents, 2)),
        "velocity": torch.zeros((total_agents, 2, 2)),
        "shape": torch.ones((total_agents, 3)),
        "batch": torch.zeros((total_agents,), dtype=torch.long),
    }
    freeze_agent = {key: value[-1:].clone() for key, value in agent.items()}
    freeze_agent["batch"].zero_()
    polyline_extras = {
        "type": torch.zeros((triplets,), dtype=torch.long),
        "pl_type": torch.zeros((triplets,), dtype=torch.long),
        "light_type": torch.zeros((triplets,), dtype=torch.long),
        "batch": torch.zeros((triplets,), dtype=torch.long),
    }
    return {
        "agent": agent,
        "num_graphs": 1,
        "freeze_agent_data": {
            "agent": freeze_agent,
            "num_obstacles": torch.ones((1,), dtype=torch.long),
            "num_graphs": 1,
        },
        "freeze_agent_mask": role[:, 0].clone(),
        "map": {
            "triplets": torch.zeros((triplets, 3, 2)),
            "triplet_thetas": torch.zeros((triplets,)),
            "polyline_extras": polyline_extras,
            "rb_data": {
                "rb_polylines": [torch.zeros((1, 3, 3))],
                "rb_polylines_batch": [0],
            },
        },
    }


class _ObservedFuture(Future[dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__()
        self.result_entered = threading.Event()

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        self.result_entered.set()
        return super().result(timeout)


def _prepared(input_data: dict[str, Any]) -> _PreparedInference:
    return _PreparedInference(
        input_data=input_data,
        future=_ObservedFuture(),
    )


def test_collate_model_inputs_remaps_graph_indices_and_split_sizes() -> None:
    first = _input_data(non_ego_agents=2, triplets=3)
    second = _input_data(non_ego_agents=1, triplets=2)

    collated, split_sizes = collate_model_inputs([first, second])

    assert split_sizes == [2, 1]
    assert collated["num_graphs"] == 2
    assert collated["num_obstacles"].tolist() == [3, 2]
    assert collated["agent"]["batch"].tolist() == [0, 0, 0, 1, 1]
    assert collated["freeze_agent_data"]["agent"]["batch"].tolist() == [0, 1]
    assert collated["freeze_agent_mask"].tolist() == [False, False, True, False, True]
    assert collated["map"]["polyline_extras"]["batch"].tolist() == [0, 0, 0, 1, 1]
    assert collated["map"]["rb_data"]["rb_polylines_batch"] == [0, 1]


def test_collate_model_inputs_requires_rb_polylines_when_rb_data_is_present() -> None:
    input_data = _input_data(non_ego_agents=1, triplets=2)
    del input_data["map"]["rb_data"]["rb_polylines"]

    with pytest.raises(KeyError, match="rb_polylines"):
        collate_model_inputs([input_data])


class _BlockingBatchModel:
    def __init__(self) -> None:
        self.prediction_steps = 5
        self.batch_sizes: list[int] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def inference(
        self,
        input_data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        self.batch_sizes.append(int(input_data["num_graphs"]))
        if len(self.batch_sizes) == 1:
            self.first_started.set()
            if not self.release_first.wait(timeout=3.0):
                raise TimeoutError("test did not release first batch")
        non_ego = ~input_data["agent"]["role"][:, 0]
        graph_ids = input_data["agent"]["batch"][non_ego].float()
        num_agents = int(graph_ids.shape[0])
        xyz = torch.zeros((num_agents, self.prediction_steps, 3))
        xyz[:, :, 0] = graph_ids.unsqueeze(1)
        return {
            "agent_future_xyz": xyz,
            "agent_future_heading": torch.zeros((num_agents, self.prediction_steps)),
            "agent_future_valid_mask": torch.ones(
                (num_agents, self.prediction_steps), dtype=torch.bool
            ),
            "agent_future_velocity": torch.zeros(
                (num_agents, self.prediction_steps, 2)
            ),
        }


def test_batcher_drains_all_available_requests() -> None:
    model = _BlockingBatchModel()
    batcher = _InferenceBatcher(model)
    requests = {
        "first": _prepared(_input_data(non_ego_agents=1, triplets=2)),
        "second": _prepared(_input_data(non_ego_agents=2, triplets=3)),
        "third": _prepared(_input_data(non_ego_agents=1, triplets=4)),
    }
    results: dict[str, dict[str, torch.Tensor]] = {}

    def submit(name: str) -> None:
        results[name] = batcher.submit(requests[name])

    first_thread = threading.Thread(target=submit, args=("first",))
    first_thread.start()
    assert model.first_started.wait(timeout=2.0)

    second_thread = threading.Thread(target=submit, args=("second",))
    third_thread = threading.Thread(target=submit, args=("third",))
    second_thread.start()
    third_thread.start()
    for name in ("second", "third"):
        future = requests[name].future
        assert isinstance(future, _ObservedFuture)
        assert future.result_entered.wait(timeout=2.0)

    model.release_first.set()
    for thread in (first_thread, second_thread, third_thread):
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    assert model.batch_sizes == [1, 2]
    assert results["first"]["agent_future_xyz"].shape[0] == 1
    assert results["second"]["agent_future_xyz"].shape[0] == 2
    assert results["third"]["agent_future_xyz"].shape[0] == 1
    second_ids = set(results["second"]["agent_future_xyz"][:, :, 0].unique().tolist())
    third_ids = set(results["third"]["agent_future_xyz"][:, :, 0].unique().tolist())
    assert len(second_ids) == 1
    assert len(third_ids) == 1
    assert second_ids | third_ids == {0.0, 1.0}
