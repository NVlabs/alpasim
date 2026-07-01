# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
from typing import Any, Dict

import torch
from alpasim_trafficsim.catk.env_data_adapter import (
    build_freeze_agent_data,
    extract_agents_and_ego_data,
    extract_map_data,
    filter_map,
    load_model_config,
)
from alpasim_trafficsim.catk.smart import SMART
from loguru import logger


def _extract_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if "model_config" in config:
        return config["model_config"]
    if "model" in config and "model_config" in config["model"]:
        return config["model"]["model_config"]
    raise KeyError("CATK config must contain either model_config or model.model_config")


def _token_path(token_pkl_dir: str, token_file: str) -> str:
    return os.path.abspath(os.path.join(token_pkl_dir, os.path.basename(token_file)))


class _BatchDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class CATK:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        token_pkl_dir: str,
        disable_sub_plyline_type: bool,
        device: str,
        use_downsampled_lines: bool = False,
    ):
        self.model_config = _extract_model_config(load_model_config(config_path))
        self.ckpt_path = ckpt_path
        self.device = device

        self.model_input_step_num = self.model_config["decoder"]["num_historical_steps"]
        self.model_predict_step_num = 5
        self.delta_t = self.model_config["token_processor"]["time_step"]

        map_token_filename = self.model_config["token_processor"]["map_token_file"]
        agent_token_filename = self.model_config["token_processor"]["agent_token_file"]

        self.model_config["token_processor"]["map_token_file"] = _token_path(
            token_pkl_dir, map_token_filename
        )
        self.model_config["token_processor"]["agent_token_file"] = _token_path(
            token_pkl_dir, agent_token_filename
        )

        self.model = SMART(self.model_config).to(self.device)
        self.model.eval()

        state_dict = torch.load(
            self.ckpt_path, map_location=self.device, weights_only=False
        )["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)

        self.disable_sub_plyline_type = disable_sub_plyline_type
        self.use_downsampled_lines = use_downsampled_lines

    def create_model_input(
        self,
        env_data: dict[str, Any],
        filter_map_by_ego: bool,
        filter_distance_th: float,
    ) -> dict | None:
        ego_xy = env_data["ego"]["xyz"][env_data["env"]["curr_t"]]
        if filter_map_by_ego and filter_distance_th > 0:
            filter_map(
                env_data=env_data, center_xyz=ego_xy, distance_th=filter_distance_th
            )

        input_data = {
            "map": {},
            "agent": {},
            "num_graphs": 1,
        }

        input_step_num = self.model_input_step_num

        curr_t = env_data["env"]["curr_t"]
        t_end = curr_t + 1
        t_beg = t_end - input_step_num

        assert (
            t_beg >= 0
        ), f"not enough history and current data for model ({curr_t + 1})"

        logger.info(f"extract ego and agent data in t: [{t_beg}, {t_end}]")

        input_data["agent"] = extract_agents_and_ego_data(
            env_data,
            t_beg=t_beg,
            t_end=t_end,
            dt=self.delta_t,
            device=self.device,
        )

        assert curr_t >= 1
        logger.info(
            f"extract frozen agent data in t: [{curr_t}, {curr_t + 1 + self.model_predict_step_num}]"
        )
        freeze_agent_data, freeze_agent_mask = build_freeze_agent_data(
            env_data,
            curr_t=curr_t,
            target_steps=1 + self.model_predict_step_num,
            dt=self.delta_t,
            device=self.device,
        )
        input_data["freeze_agent_data"] = freeze_agent_data
        input_data["freeze_agent_mask"] = freeze_agent_mask

        triplets, triplet_thetas, polyline_extras, rb_data = extract_map_data(
            env_data,
            device=self.device,
            downsample_lines=self.use_downsampled_lines,
            disable_sub_plyline_type=self.disable_sub_plyline_type,
        )
        if triplets is None:
            logger.warning("CATK model input skipped because no map data is available")
            return None

        input_data["map"]["triplets"] = triplets
        input_data["map"]["triplet_thetas"] = triplet_thetas
        input_data["map"]["polyline_extras"] = polyline_extras
        input_data["map"]["rb_data"] = rb_data

        logger.info(f"[proxy] triplets #:{triplets.shape[0]}")
        return {"input_data": input_data}

    def inference(self, input_data: Dict[str, Any]):
        self.model.encoder.agent_encoder.num_future_steps = self.model_predict_step_num

        sampling_scheme = self.model.validation_rollout_sampling
        step_current_10hz = self.model.encoder.agent_encoder.num_historical_steps  # 10
        dt = self.delta_t

        with torch.no_grad():
            state = _BatchDict(
                {
                    "position": None,
                    "agent": input_data["agent"],
                    "num_obstacles": torch.tensor(
                        (input_data["agent"]["id"].shape[0],),
                        device=self.device,
                        dtype=torch.long,
                    ).reshape(1),
                    "traj_pos": None,
                    "traj_theta": None,
                    "map_save": {
                        "traj_pos": input_data["map"]["triplets"],
                        "traj_theta": input_data["map"]["triplet_thetas"],
                    },
                    "pt_token": input_data["map"]["polyline_extras"],
                    "rb_data": input_data["map"]["rb_data"],
                    "num_graphs": input_data["num_graphs"],
                }
            )
            tokenized_map, tokenized_agent, _, _ = self.model.token_processor(
                state,
                apply_heading_correction=True,
                apply_boundary_extrapolation=True,
            )
            map_feature = self.model.encoder.map_encoder(tokenized_map)

            is_ego = tokenized_agent["ego_mask"]

            # Tokenized known future for ego and static agents.
            fz_tokenized_agent, _ = self.model.token_processor.tokenize_agent(
                input_data["freeze_agent_data"],
                apply_heading_correction=True,
                apply_boundary_extrapolation=True,
            )

            output = self.model.encoder.agent_encoder.inference_with_mask(
                tokenized_agent=tokenized_agent,
                map_feature=map_feature,
                sampling_scheme=sampling_scheme,
                freeze_agent_future=True,
                freeze_agent_mask=input_data["freeze_agent_mask"],
                freeze_tokenized_agent=fz_tokenized_agent,
            )

            future_xyz = torch.cat(
                [output["pred_traj_10hz"], output["pred_z_10hz"].unsqueeze(-1)],
                dim=-1,
            )
            future_heading = output["pred_head_10hz"]
            future_valid = output["pred_valid_10hz"]

            pos_hist = state["agent"]["position"][:, :step_current_10hz]
            last_hist_pos = pos_hist[:, -1:, :]
            pos_all = torch.cat([last_hist_pos, future_xyz], dim=1)
            future_velocity = (pos_all[:, 1:] - pos_all[:, :-1]) / dt

            pnum = self.model_predict_step_num
            actions = {
                "agent_future_xyz": future_xyz[~is_ego, :pnum],
                "agent_future_heading": future_heading[~is_ego, :pnum],
                "agent_future_valid_mask": future_valid[~is_ego, :pnum],
                "agent_future_velocity": future_velocity[~is_ego, :pnum, :2],
            }
        return actions
