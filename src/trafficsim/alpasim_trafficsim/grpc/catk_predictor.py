# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import copy
import math
from typing import Any

import torch
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


def _fill_static_agent(
    processed_xyz: torch.Tensor,
    processed_heading: torch.Tensor,
    processed_valid: torch.Tensor,
    *,
    agent_idx: int,
    prev_xyz: torch.Tensor,
    prev_heading: torch.Tensor,
    prev_valid: torch.Tensor,
) -> None:
    processed_xyz[agent_idx, :, :] = prev_xyz
    processed_heading[agent_idx, :] = prev_heading
    processed_valid[agent_idx, :] = prev_valid


def _carry_invalid_predictions_forward(
    processed_xyz: torch.Tensor,
    processed_heading: torch.Tensor,
    processed_valid: torch.Tensor,
    *,
    agent_idx: int,
    prev_xyz: torch.Tensor,
    prev_heading: torch.Tensor,
    prev_valid: torch.Tensor,
) -> None:
    last_xyz = prev_xyz
    last_heading = prev_heading
    last_valid = bool(prev_valid.item())
    for step_offset in range(processed_xyz.shape[1]):
        if bool(processed_valid[agent_idx, step_offset].item()):
            last_xyz = processed_xyz[agent_idx, step_offset, :]
            last_heading = processed_heading[agent_idx, step_offset]
            last_valid = True
            continue
        if not last_valid:
            continue
        processed_xyz[agent_idx, step_offset, :] = last_xyz
        processed_heading[agent_idx, step_offset] = last_heading
        processed_valid[agent_idx, step_offset] = True


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


class CATKTrafficPredictor:
    def __init__(self, catk_cfg: CatkConfig) -> None:
        self.cfg = catk_cfg
        self.predict_static = self.cfg.predict_static
        self.history_window_steps = self.cfg.loader.num_history_steps
        self.min_valid_history_steps = self.cfg.min_valid_history_steps
        self.model = self._build_model()
        self._token_stride: int = self.model.model.encoder.agent_encoder.shift

    def _build_model(self) -> Any:
        from alpasim_trafficsim.catk.model_adapter import CATK

        model_cfg = self.cfg.model
        return CATK(
            config_path=model_cfg.config_path,
            ckpt_path=model_cfg.ckpt_path,
            token_pkl_dir=model_cfg.token_pkl_dir,
            disable_sub_plyline_type=model_cfg.disable_sub_plyline_type,
            use_downsampled_lines=model_cfg.use_downsampled_lines,
            device=self.cfg.device,
        )

    def run_inference(
        self,
        env_data: SimEnvData,
        *,
        predict_steps: int,
    ) -> dict[str, Any] | None:
        if self.model is None or predict_steps <= 0:
            return None
        # CATK emits full token strides; the caller crops to requested steps.
        model_steps = math.ceil(predict_steps / self._token_stride) * self._token_stride
        self.model.model_predict_step_num = model_steps
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
            return None
        return self.model.inference(model_input_result["input_data"])

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

        for agent_idx in range(num_agents):
            if bool(frozen_static_mask[agent_idx].item()):
                _fill_static_agent(
                    processed_xyz,
                    processed_heading,
                    processed_valid,
                    agent_idx=agent_idx,
                    prev_xyz=prev_xyz[agent_idx],
                    prev_heading=prev_heading[agent_idx],
                    prev_valid=prev_valid[agent_idx],
                )
                continue

            if bool(sparse_history_mask[agent_idx].item()) and bool(
                prev_valid[agent_idx].item()
            ):
                _fill_static_agent(
                    processed_xyz,
                    processed_heading,
                    processed_valid,
                    agent_idx=agent_idx,
                    prev_xyz=prev_xyz[agent_idx],
                    prev_heading=prev_heading[agent_idx],
                    prev_valid=prev_valid[agent_idx],
                )
                continue

            _carry_invalid_predictions_forward(
                processed_xyz,
                processed_heading,
                processed_valid,
                agent_idx=agent_idx,
                prev_xyz=prev_xyz[agent_idx],
                prev_heading=prev_heading[agent_idx],
                prev_valid=prev_valid[agent_idx],
            )

        return processed_xyz, processed_heading, processed_valid
