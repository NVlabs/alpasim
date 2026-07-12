# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import copy
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch
from alpasim_trafficsim.catk.batching import collate_model_inputs
from alpasim_trafficsim.catk.model_adapter import CATK
from alpasim_trafficsim.grpc.config import CatkConfig
from alpasim_trafficsim.grpc.pipeline.env_builder import (
    backfill_static_agent_history,
    ensure_time_axis_length,
    static_agent_mask,
)
from alpasim_trafficsim.grpc.pipeline.laneline_elevation import (
    agent_center_z_from_nearest_lanelines,
)
from alpasim_trafficsim.grpc.service_structures import SessionState, SimEnvData


class CatkPredictionUnavailableError(RuntimeError):
    """Raised when CATK cannot produce predictions for the current request."""


def _actions_to_env_tensors(
    actions: dict[str, Any],
    env_data: SimEnvData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_xyz = (
        actions["agent_future_xyz"]
        .detach()
        .to(
            device=env_data["agents"]["xyz"].device,
            dtype=env_data["agents"]["xyz"].dtype,
        )
    )
    pred_heading = (
        actions["agent_future_heading"]
        .detach()
        .to(
            device=env_data["agents"]["heading"].device,
            dtype=env_data["agents"]["heading"].dtype,
        )
    )
    pred_valid = (
        actions["agent_future_valid_mask"]
        .detach()
        .to(
            device=env_data["agents"]["valid_mask"].device,
            dtype=env_data["agents"]["valid_mask"].dtype,
        )
    )
    return pred_xyz, pred_heading, pred_valid


def _copy_unpredicted_agents_forward(
    env_data: SimEnvData,
    *,
    total_agents: int,
    num_agents: int,
    step_idx: int,
    prev_step_idx: int,
) -> None:
    if total_agents <= num_agents:
        return
    env_data["agents"]["xyz"][num_agents:, step_idx, :] = env_data["agents"]["xyz"][
        num_agents:,
        prev_step_idx,
        :,
    ]
    env_data["agents"]["heading"][num_agents:, step_idx] = env_data["agents"][
        "heading"
    ][num_agents:, prev_step_idx]
    env_data["agents"]["valid_mask"][num_agents:, step_idx] = env_data["agents"][
        "valid_mask"
    ][num_agents:, prev_step_idx]


def _apply_z_correction(
    env_data: SimEnvData,
    *,
    total_agents: int,
    step_idx: int,
) -> None:
    if total_agents <= 0:
        return
    all_step_xyz = env_data["agents"]["xyz"][:total_agents, step_idx, :]
    all_step_valid = env_data["agents"]["valid_mask"][:total_agents, step_idx]
    all_step_xyz[:, 2] = agent_center_z_from_nearest_lanelines(
        env_data.get("map"),
        all_step_xyz[:, :2],
        agent_lwh=env_data["agents"]["lwh"][:total_agents],
        valid_mask=all_step_valid,
        fallback_z=all_step_xyz[:, 2],
    )


def _clear_invalid_step_values(
    step_xyz: torch.Tensor,
    step_heading: torch.Tensor,
    step_valid: torch.Tensor,
) -> None:
    invalid_mask = ~step_valid
    if bool(invalid_mask.any().item()):
        step_xyz.masked_fill_(invalid_mask.unsqueeze(-1), 0.0)
        step_heading.masked_fill_(invalid_mask, 0.0)


def _clone_env_data_for_model(env_data: SimEnvData) -> SimEnvData:
    model_env_data = dict(env_data)
    model_env_data["map"] = copy.deepcopy(env_data.get("map", {}))
    model_env_data["agents"] = dict(env_data["agents"])
    for key in ("xyz", "heading", "valid_mask"):
        model_env_data["agents"][key] = env_data["agents"][key].clone()
    return model_env_data


def _write_predictions_to_env(
    env_data: SimEnvData,
    *,
    future_step_indices: list[int],
    total_agents: int,
    num_agents: int,
    processed_xyz: torch.Tensor,
    processed_heading: torch.Tensor,
    processed_valid: torch.Tensor,
) -> None:
    for step_offset, step_idx in enumerate(future_step_indices):
        prev_step_idx = max(step_idx - 1, 0)
        _copy_unpredicted_agents_forward(
            env_data,
            total_agents=total_agents,
            num_agents=num_agents,
            step_idx=step_idx,
            prev_step_idx=prev_step_idx,
        )
        step_xyz = env_data["agents"]["xyz"][:num_agents, step_idx, :]
        step_heading = env_data["agents"]["heading"][:num_agents, step_idx]
        step_valid = env_data["agents"]["valid_mask"][:num_agents, step_idx]

        step_xyz[:] = processed_xyz[:, step_offset, :]
        step_heading[:] = processed_heading[:, step_offset]
        step_valid[:] = processed_valid[:, step_offset]
        _apply_z_correction(
            env_data,
            total_agents=total_agents,
            step_idx=step_idx,
        )
        _clear_invalid_step_values(step_xyz, step_heading, step_valid)


@dataclass
class _PreparedInference:
    input_data: dict[str, Any]
    future: Future[dict[str, Any]]


class _InferenceBatcher:
    """Work-conserving CATK inference worker with variable batch sizes."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._queue: queue.Queue[_PreparedInference] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="catk-inference-batcher",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: _PreparedInference) -> dict[str, Any]:
        self._queue.put(request)
        return request.future.result()

    def _collect_available_batch(self) -> list[_PreparedInference]:
        batch = [self._queue.get()]

        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                return batch

    def _run(self) -> None:
        while True:
            batch = self._collect_available_batch()
            try:
                self._execute(batch)
            except Exception as exc:  # noqa: BLE001 - deliver to every waiter
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)

    def _execute(self, batch: list[_PreparedInference]) -> None:

        collated_input, split_sizes = collate_model_inputs(
            [request.input_data for request in batch]
        )
        actions = self._model.inference(collated_input)

        offset = 0
        for request, split_size in zip(batch, split_sizes, strict=True):
            request_actions = {
                key: value[offset : offset + split_size]
                for key, value in actions.items()
            }
            offset += split_size
            request.future.set_result(request_actions)


class CATKTrafficPredictor:
    def __init__(self, catk_cfg: CatkConfig) -> None:
        self.cfg = catk_cfg
        self.predict_static = self.cfg.predict_static
        self.history_window_steps = self.cfg.loader.num_history_steps
        self.min_valid_history_steps = self.cfg.min_valid_history_steps
        self.model = self._build_model()
        self._batcher = _InferenceBatcher(self.model)

    def _build_model(self) -> Any:

        model_cfg = self.cfg.model
        return CATK(
            config_path=model_cfg.config_path,
            ckpt_path=model_cfg.ckpt_path,
            token_pkl_dir=model_cfg.token_pkl_dir,
            disable_sub_plyline_type=model_cfg.disable_sub_plyline_type,
            prediction_steps=self.cfg.loader.prediction_steps,
            use_downsampled_lines=model_cfg.use_downsampled_lines,
            device=self.cfg.device,
        )

    def run_inference(
        self,
        env_data: SimEnvData,
    ) -> dict[str, Any]:
        model_env_data = _clone_env_data_for_model(env_data)
        backfill_static_agent_history(
            model_env_data,
            curr_t=int(model_env_data["env"].get("curr_t", 0)),
            history_window_steps=self.history_window_steps,
            predict_static=self.predict_static,
        )
        model_input_result = self.model.create_model_input(
            model_env_data,
            filter_map_by_ego=True,
            filter_distance_th=self.cfg.filter_distance_th,
        )
        if model_input_result is None:
            raise CatkPredictionUnavailableError(
                "No usable map geometry was found within "
                f"{self.cfg.filter_distance_th:g} m of the current ego position; "
                "CATK cannot produce predictions"
            )
        future: Future[dict[str, Any]] = Future()
        return self._batcher.submit(
            _PreparedInference(
                input_data=model_input_result["input_data"],
                future=future,
            )
        )

    def apply_predictions_to_env(
        self,
        session_state: SessionState,
        *,
        future_step_indices: list[int],
        actions: dict[str, Any],
    ) -> list[int]:
        assert session_state.env_data is not None
        env_data = session_state.env_data
        if not future_step_indices:
            return []

        pred_xyz, pred_heading, pred_valid = _actions_to_env_tensors(actions, env_data)
        total_agents = env_data["agents"]["xyz"].shape[0]
        num_agents = min(total_agents, pred_xyz.shape[0])
        num_steps = min(len(future_step_indices), pred_xyz.shape[1])
        active_future_step_indices = future_step_indices[:num_steps]
        for step_idx in active_future_step_indices:
            ensure_time_axis_length(env_data, step_idx)
        if not active_future_step_indices:
            return []

        processed_xyz, processed_heading, processed_valid = (
            self._postprocess_predictions(
                env_data,
                future_step_indices=active_future_step_indices,
                pred_xyz=pred_xyz[:num_agents, :num_steps, :],
                pred_heading=pred_heading[:num_agents, :num_steps],
                pred_valid=pred_valid[:num_agents, :num_steps],
            )
        )
        _write_predictions_to_env(
            env_data,
            future_step_indices=active_future_step_indices,
            total_agents=total_agents,
            num_agents=num_agents,
            processed_xyz=processed_xyz,
            processed_heading=processed_heading,
            processed_valid=processed_valid,
        )
        return active_future_step_indices

    def _postprocess_predictions(
        self,
        env_data: SimEnvData,
        *,
        future_step_indices: list[int],
        pred_xyz: torch.Tensor,
        pred_heading: torch.Tensor,
        pred_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_xyz = pred_xyz.clone()
        processed_heading = pred_heading.clone()
        processed_valid = pred_valid.bool().clone()
        num_agents = int(processed_xyz.shape[0])
        num_steps = int(processed_xyz.shape[1])
        if num_agents == 0 or num_steps == 0 or not future_step_indices:
            return processed_xyz, processed_heading, processed_valid

        prev_step_idx = max(int(future_step_indices[0]) - 1, 0)
        prev_xyz = env_data["agents"]["xyz"][:num_agents, prev_step_idx, :]
        prev_heading = env_data["agents"]["heading"][:num_agents, prev_step_idx]
        prev_valid = env_data["agents"]["valid_mask"][:num_agents, prev_step_idx]
        history_beg = max(prev_step_idx - self.history_window_steps + 1, 0)
        history_valid_count = env_data["agents"]["valid_mask"][
            :num_agents, history_beg : prev_step_idx + 1
        ].sum(dim=1)
        sparse_history_mask = history_valid_count < self.min_valid_history_steps
        frozen_static_mask = (
            static_agent_mask(env_data, device=processed_valid.device)[:num_agents]
            if not self.predict_static
            else torch.zeros(
                (num_agents,),
                dtype=torch.bool,
                device=processed_valid.device,
            )
        )

        force_static_mask = frozen_static_mask | (sparse_history_mask & prev_valid)

        # Treat the current observed pose as time zero, then compute the most
        # recent valid source index for every agent and future step at once.
        # An index of -1 means that neither the previous pose nor any prediction
        # up to that point was valid; those invalid prediction values remain
        # untouched
        source_xyz = torch.cat((prev_xyz.unsqueeze(1), processed_xyz), dim=1)
        source_heading = torch.cat(
            (prev_heading.unsqueeze(1), processed_heading), dim=1
        )
        source_valid = torch.cat((prev_valid.unsqueeze(1), processed_valid), dim=1)
        source_step_indices = torch.arange(
            num_steps + 1,
            dtype=torch.long,
            device=processed_valid.device,
        ).unsqueeze(0)
        valid_source_indices = torch.where(
            source_valid,
            source_step_indices,
            torch.full_like(source_step_indices, -1),
        )
        last_valid_indices = valid_source_indices.cummax(dim=1).values[:, 1:]
        has_valid_source = last_valid_indices >= 0
        safe_indices = last_valid_indices.clamp_min(0)

        carried_xyz = source_xyz.gather(
            1, safe_indices.unsqueeze(-1).expand(-1, -1, source_xyz.shape[-1])
        )
        carried_heading = source_heading.gather(1, safe_indices)
        dynamic_mask = ~force_static_mask.unsqueeze(1)
        carry_mask = dynamic_mask & has_valid_source
        processed_xyz = torch.where(
            carry_mask.unsqueeze(-1), carried_xyz, processed_xyz
        )
        processed_heading = torch.where(carry_mask, carried_heading, processed_heading)
        processed_valid = torch.where(dynamic_mask, has_valid_source, processed_valid)

        # Frozen and sparse-history agents repeat their current observed state
        # across the complete prediction horizon.
        processed_xyz = torch.where(
            force_static_mask[:, None, None], prev_xyz[:, None, :], processed_xyz
        )
        processed_heading = torch.where(
            force_static_mask[:, None], prev_heading[:, None], processed_heading
        )
        processed_valid = torch.where(
            force_static_mask[:, None], prev_valid[:, None], processed_valid
        )

        return processed_xyz, processed_heading, processed_valid
