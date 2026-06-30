# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import os
import pickle
import random
from typing import Dict, Tuple

import torch
from alpasim_trafficsim.catk.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical


class TokenProcessor(torch.nn.Module):
    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
        time_step: float,  # 0.1 or 0.02
        map_dropout_prob: float = 0.0,
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5
        self.time_step = time_step
        self.map_dropout_prob = map_dropout_prob

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        module_dir = os.path.dirname(__file__)
        self.init_agent_token(os.path.join(module_dir, agent_token_file))
        self.init_map_token(os.path.join(module_dir, map_token_file))
        self.n_token_agent = self.agent_token_all_veh.shape[0]

    @torch.no_grad()
    def forward(
        self,
        data: Dict,
        apply_heading_correction: bool,
        apply_boundary_extrapolation: bool,
    ) -> Tuple[
        Dict[str, Tensor] | None,
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict | None,
    ]:
        tokenized_map, rb_data = self.tokenize_map(data)

        tokenized_agent, agent_data = self.tokenize_agent(
            data,
            apply_heading_correction=apply_heading_correction,
            apply_boundary_extrapolation=apply_boundary_extrapolation,
        )
        return tokenized_map, tokenized_agent, agent_data, rb_data

    def init_map_token(self, map_token_traj_path, argmin_sample_len=3) -> None:
        map_token_traj = pickle.load(open(map_token_traj_path, "rb"))["traj_src"]
        indices = torch.linspace(
            0, map_token_traj.shape[1] - 1, steps=argmin_sample_len
        ).long()

        self.register_buffer(
            "map_token_traj_src",
            torch.tensor(
                map_token_traj, dtype=torch.float32, device=self.device
            ).flatten(1, 2),
            persistent=False,
        )  # [n_token, 11*2]

        self.register_buffer(
            "map_token_sample_pt",
            torch.tensor(
                map_token_traj[:, indices], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            persistent=False,
        )  # [1, n_token, 3, 2]

    def init_agent_token(self, agent_token_path) -> None:
        agent_token_data = pickle.load(open(agent_token_path, "rb"))
        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32, device=self.device)
            # [n_token, 6, 4, 2], countour, 10 hz
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)

    def tokenize_map(
        self, data: Dict
    ) -> tuple[Dict[str, Tensor] | None, list[Tensor] | None]:
        traj_pos = data["map_save"]["traj_pos"]
        traj_theta = data["map_save"]["traj_theta"]
        polyline_extras = data["pt_token"]
        rb_data = data["rb_data"] if "rb_data" in data else None

        out_type = polyline_extras["type"].long()  # [n_pl]
        out_pl_type = polyline_extras["pl_type"].long()  # [n_pl]
        out_light_type = polyline_extras["light_type"].long()  # [n_pl]
        out_batch = polyline_extras["batch"]  # [n_pl]

        if (
            self.training
            and self.map_dropout_prob > 0.0
            and random.uniform(0, 1) < self.map_dropout_prob
        ):
            return None, None

        if traj_pos is None:
            return None, None

        traj_pos_local, _ = transform_to_local(
            pos_global=traj_pos,  # [n_pl, 3, 2]
            head_global=None,  # [n_pl, 1]
            pos_now=traj_pos[:, 0],  # [n_pl, 2]
            head_now=traj_theta,  # [n_pl]
        )
        # [1, n_token, 3, 2] - [n_pl, 1, 3, 2]
        dist = torch.sum(
            (self.map_token_sample_pt - traj_pos_local.unsqueeze(1)) ** 2,
            dim=(-2, -1),
        )  # [n_pl, n_token]

        if self.training and (self.map_token_sampling.num_k > 1):
            topk_dists, topk_indices = torch.topk(
                dist,
                self.map_token_sampling.num_k,
                dim=-1,
                largest=False,
                sorted=False,
            )  # [n_pl, K]

            topk_logits = (-1e-6 - topk_dists) / self.map_token_sampling.temp
            _samples = Categorical(logits=topk_logits).sample()  # [n_pl] in K
            token_idx = topk_indices[torch.arange(len(_samples)), _samples].contiguous()
        else:
            token_idx = torch.argmin(dist, dim=-1)

        tokenized_map = {
            "position": traj_pos[:, 0].contiguous(),  # [n_pl, 2]
            "orientation": traj_theta,  # [n_pl]
            "token_idx": token_idx,  # [n_pl]
            "token_traj_src": self.map_token_traj_src,  # [n_token, 11*2], [n_token, 3*2]
            "type": out_type,  # [n_pl]
            "pl_type": out_pl_type,  # [n_pl]
            "light_type": out_light_type,  # [n_pl]
            "batch": out_batch,  # [n_pl]
        }

        return tokenized_map, rb_data

    def tokenize_agent(
        self,
        data: Dict,
        apply_heading_correction: bool,
        apply_boundary_extrapolation: bool,
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Args: data["agent"]: Dict
            "valid_mask": [n_agent, n_step], bool
            "role": [n_agent, 3], bool
            "id": [n_agent], int64
            "type": [n_agent], uint8
            "position": [n_agent, n_step, 3], float32
            "heading": [n_agent, n_step], float32
            "velocity": [n_agent, n_step, 2], float32
            "shape": [n_agent, 3], float32
        """
        agent_data = data["agent"]
        num_graphs = (
            data.num_graphs if hasattr(data, "num_graphs") else data["num_graphs"]
        )

        # ! collate width/length, traj tokens for current batch
        agent_shape, real_agent_shape, token_traj_all, token_traj = (
            self._get_agent_shape_and_token_traj(
                agent_data["type"],
                agent_data["shape"],
            )
        )

        # ! get raw trajectory data
        valid = agent_data["valid_mask"].clone()  # [n_agent, n_step]
        heading = agent_data["heading"].clone()  # [n_agent, n_step]
        pos = (
            agent_data["position"][..., :2].contiguous().clone()
        )  # [n_agent, n_step, 2]
        vel = agent_data["velocity"].clone()  # [n_agent, n_step, 2]

        # ! agent, specifically vehicle's heading can be 180 degree off. We fix it here.
        if apply_heading_correction:
            heading = self._clean_heading(valid, heading)

        # ! extrapolate to previous motion boundary (5th step) because the valid mask may not align.
        if apply_boundary_extrapolation:
            valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
                valid,
                pos,
                heading,
                vel,
                time_step=self.time_step,
            )

        # ! prepare output dict
        tokenized_agent = {
            "num_graphs": num_graphs,
            "type": agent_data["type"],
            "shape": agent_data["shape"],
            "ego_mask": agent_data["role"][:, 0],  # [n_agent]
            "token_agent_shape": agent_shape,  # [n_agent, 2]
            "real_agent_shape": real_agent_shape,  # [n_agent, 2]
            "batch": agent_data["batch"],
            "token_traj_all": token_traj_all,  # [n_agent, n_token, 6, 4, 2]
            "token_traj": token_traj,  # [n_agent, n_token, 4, 2]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step]
        }

        # [n_token, 8]
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(
                self, f"agent_token_all_{k}"
            )[:, -1].flatten(1, 2)

        # ! match token for each agent
        if not self.training:
            # [n_agent]
            tokenized_agent["gt_z_raw"] = torch.zeros(
                (agent_data["position"].shape[0],),
                dtype=agent_data["position"].dtype,
                device=agent_data["position"].device,
            )

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
        )
        tokenized_agent.update(token_dict)
        return tokenized_agent, agent_data

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    ) -> Dict[str, Tensor]:
        """n_step_token=n_step//5
        n_step_token=18 for train with BC.
        n_step_token=2 for val/test and train with closed-loop rollout.
        Returns: Dict
            # ! action that goes from [(0->5), (5->10), ..., (85->90)]
            "valid_mask": [n_agent, n_step_token]
            "gt_idx": [n_agent, n_step_token]
            # ! at step [5, 10, 15, ..., 90]
            "gt_pos": [n_agent, n_step_token, 2]
            "gt_heading": [n_agent, n_step_token]
            # ! noisy sampling for training data augmentation
            "sampled_idx": [n_agent, n_step_token]
            "sampled_pos": [n_agent, n_step_token, 2]
            "sampled_heading": [n_agent, n_step_token]
        """
        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        out_dict = {
            "valid_mask": [],  # valid or not [n_agent, n_step_token]
            "gt_idx": [],  # matched token index [n_agent, n_step_token]
            "gt_pos": [],  # matched token's averaged position [n_agent, n_step_token, 2]
            "gt_heading": [],  # matched token's position [n_agent, n_step_token]
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
        }

        # matching start from the first token i=0
        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]
        prev_pos_sample, prev_head_sample = pos[:, 0], heading[:, 0]

        for i in range(
            self.shift, n_step, self.shift
        ):  # next token t=[5, 10, 15, ..., 90]
            _valid_mask = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid_mask = ~_valid_mask
            out_dict["valid_mask"].append(_valid_mask)

            # gt_contour: [n_agent, 4, 2] in global coord
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            # ! tokenize without sampling
            token_world_gt = transform_to_global(
                pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos,  # [n_agent, 2]
                head_now=prev_head,  # [n_agent]
            )[0].view(*token_traj.shape)

            token_idx_gt = torch.argmin(
                torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1), dim=-1
            )  # [n_agent]
            # [n_agent, 4, 2]
            token_contour_gt = token_world_gt[range_a, token_idx_gt]

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]
            prev_head[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid_mask]

            prev_pos = pos[:, i].clone()
            prev_pos[_valid_mask] = token_contour_gt.mean(1)[_valid_mask]

            # add to output dict
            out_dict["gt_idx"].append(token_idx_gt)
            out_dict["gt_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)
            )
            out_dict["gt_heading"].append(prev_head.masked_fill(_invalid_mask, 0))

            # ! tokenize from sampled rollout state
            if num_k == 1:  # K=1 means no sampling
                out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
                out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
                out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])
            else:
                # contour: [n_agent, n_token, 4, 2], 2HZ, global coord
                token_world_sample = transform_to_global(
                    pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
                    head_local=None,
                    pos_now=prev_pos_sample,  # [n_agent, 2]
                    head_now=prev_head_sample,  # [n_agent]
                )[0].view(*token_traj.shape)

                # dist: [n_agent, n_token]
                dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)
                topk_dists, topk_indices = torch.topk(
                    dist, num_k, dim=-1, largest=False, sorted=False
                )  # [n_agent, K]

                topk_logits = (-1.0 * topk_dists) / self.agent_token_sampling.temp
                _samples = Categorical(logits=topk_logits).sample()  # [n_agent] in K
                token_idx_sample = topk_indices[range_a, _samples]
                token_contour_sample = token_world_sample[range_a, token_idx_sample]

                # udpate prev_pos_sample, prev_head_sample
                prev_head_sample = heading[:, i].clone()
                dxy = token_contour_sample[:, 0] - token_contour_sample[:, 3]
                prev_head_sample[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[
                    _valid_mask
                ]
                prev_pos_sample = pos[:, i].clone()
                prev_pos_sample[_valid_mask] = token_contour_sample.mean(1)[_valid_mask]
                # add to output dict
                out_dict["sampled_idx"].append(token_idx_sample)
                out_dict["sampled_pos"].append(
                    prev_pos_sample.masked_fill(_invalid_mask.unsqueeze(1), 0.0)
                )
                out_dict["sampled_heading"].append(
                    prev_head_sample.masked_fill(_invalid_mask, 0.0)
                )
        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}

        return out_dict

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]
        return heading

    def _extrapolate_agent_to_prev_token_step(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        vel: Tensor,  # [n_agent, n_step, 2]
        time_step: float,  # 0.1 or 0.02
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # [n_agent], max will give the first True step
        first_valid_step = torch.max(valid, dim=1).indices

        for i, t in enumerate(first_valid_step):  # extrapolate to previous 5th step.
            n_step_to_extrapolate = t % self.shift
            # TODO(bivanovic): This is a magic number hack to ensure that the agent is always visible at the first token
            if (t == 15) and (not valid[i, 15 - self.shift]):
                # such that at least one token is valid in the history.
                n_step_to_extrapolate = self.shift

            if n_step_to_extrapolate > 0:
                vel[i, t - n_step_to_extrapolate : t] = vel[i, t]
                valid[i, t - n_step_to_extrapolate : t] = True
                heading[i, t - n_step_to_extrapolate : t] = heading[i, t]

                for j in range(n_step_to_extrapolate):
                    # TODO(bivanovic): This is a magic number hack (assuming 10 Hz uniformly).
                    pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * time_step

        return valid, pos, heading, vel

    def _get_agent_shape_and_token_traj(
        self, agent_type: Tensor, agent_shape: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        agent_shape: [n_agent, 2]
        token_traj_all: [n_agent, n_token, 6, 4, 2]
        token_traj: [n_agent, n_token, 4, 2]
        """
        agent_type_masks = {
            "veh": agent_type == 0,
            "ped": agent_type == 1,
            "cyc": agent_type == 2,
        }
        agent_shape_out = 0.0
        real_agent_shape = 0.0
        token_traj_all = 0.0
        real_length, real_width = agent_shape[:, 0], agent_shape[:, 1]
        for k, mask in agent_type_masks.items():
            if k == "veh":
                width = 2.0
                length = 4.8
            elif k == "cyc":
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0
            agent_shape_out += torch.stack([width * mask, length * mask], dim=-1)
            real_agent_shape += torch.stack(
                [real_width * mask, real_length * mask], dim=-1
            )

            token_traj_all += mask[:, None, None, None, None] * (
                getattr(self, f"agent_token_all_{k}").unsqueeze(0)
            )

        token_traj = token_traj_all[:, :, -1, :, :].contiguous()
        return agent_shape_out, real_agent_shape, token_traj_all, token_traj
