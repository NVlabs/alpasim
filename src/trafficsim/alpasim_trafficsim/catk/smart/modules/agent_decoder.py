# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Dict, Optional

import torch
import torch.nn as nn
from alpasim_trafficsim.catk.smart.layers import MLPLayer
from alpasim_trafficsim.catk.smart.layers.attention_layer import AttentionLayer
from alpasim_trafficsim.catk.smart.layers.fourier_embedding import (
    FourierEmbedding,
    MLPEmbedding,
)
from alpasim_trafficsim.catk.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)
from omegaconf import DictConfig
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph


class SMARTAgentDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        num_agent_types: int,
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(num_agent_types, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_emb_veh = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_ped = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_cyc = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
        )

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )
        self.apply(weight_init)

    def agent_token_embedding(
        self,
        agent_token_index,  # [n_agent, n_step]
        trajectory_token_veh,  # [n_token, 8]
        trajectory_token_ped,  # [n_token, 8]
        trajectory_token_cyc,  # [n_token, 8]
        pos_a,  # [n_agent, n_step, 2]
        head_vector_a,  # [n_agent, n_step, 2]
        agent_type,  # [n_agent]
        agent_shape,  # [n_agent, 3]
        inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2
        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]
        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a  # [n_agent, n_step, hidden_dim]

    def build_temporal_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2],
        mask,  # [n_agent, n_step]
        inference_mask=None,  # [n_agent, n_step]
    ):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        batch_s,  # [n_agent*n_step]
        mask,  # [n_agent, n_step]
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        pos_pl,  # [n_pl, 2]
        orient_pl,  # [n_pl]
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        mask,  # [n_agent, n_step]
        batch_s,  # [n_agent*n_step]
        batch_pl,  # [n_pl*n_step]
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        # ! get agent token embeddings
        feat_a = self.agent_token_embedding(
            agent_token_index=tokenized_agent["sampled_idx"],  # [n_ag, n_step]
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            agent_type=tokenized_agent["type"],  # [n_agent]
            agent_shape=tokenized_agent["shape"],  # [n_agent, 3]
        )  # feat_a: [n_agent, n_step, hidden_dim]

        # ! build temporal, interaction and map2agent edges
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_t: [2, n_edge_t], r_t: [n_edge_t, hidden_dim]

        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_agent*n_step]
        if map_feature is not None:
            batch_pl = torch.cat(
                [
                    map_feature["batch"] + tokenized_agent["num_graphs"] * t
                    for t in range(n_step)
                ],
                dim=0,
            )  # [n_pl*n_step]

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        if map_feature is not None:
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a=pos_a,  # [n_agent, n_step, 2]
                head_a=head_a,  # [n_agent, n_step]
                head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                mask=mask,  # [n_agent, n_step]
                batch_s=batch_s,  # [n_agent*n_step]
                batch_pl=batch_pl,  # [n_pl*n_step]
            )

        # ! attention layers
        # [n_step*n_pl, hidden_dim]
        if map_feature is not None:
            feat_map = (
                map_feature["pt_token"]
                .unsqueeze(0)
                .expand(n_step, -1, -1)
                .flatten(0, 1)
            )

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)  # [n_agent*n_step, hidden_dim]
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            # [n_step*n_agent, hidden_dim]
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)

            if map_feature is not None:
                feat_a = self.pt2a_attn_layers[i](
                    (feat_map, feat_a), r_pl2a, edge_index_pl2a
                )

            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)

        # ! final mlp to get outputs
        next_token_logits = self.token_predict_head(feat_a)

        return {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
        }

    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor] | None,
        sampling_scheme: DictConfig,
        fixed_agent_mask: torch.Tensor | None = None,
        ego_fixed: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Agent trajectory inference with optional ego trajectory conditioning.

        Args:
            tokenized_agent: Tokenized agent data
            map_feature: Encoded map features
            sampling_scheme: Token sampling configuration
            fixed_agent_mask: If True, use agent trajectory from tokenized_agent['gt_pos_raw']
                               and ['gt_head_raw'] instead of predicting agent motion. This enables
                               inference where certain agent trajectories are externally provided.
            ego_fixed: If True, use ego trajectory from tokenized_agent['gt_pos_raw']
                        and ['gt_head_raw'] instead of predicting ego motion.

        Returns:
            Dict containing predicted trajectories and other outputs
        """

        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps  # 80 or 45
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16 or 9

        step_current_10hz = self.num_historical_steps - 1  # 10 or 15
        step_current_2hz = step_current_10hz // self.shift  # 2 or 3

        # Validate fixed agent trajectory data is available if needed
        if fixed_agent_mask is not None or ego_fixed:
            required_keys = ["gt_pos_raw", "gt_head_raw", "gt_valid_raw"]
            for key in required_keys:
                if key not in tokenized_agent:
                    raise ValueError(
                        f"fixed_agent_mask not None but required key '{key}' not found in tokenized_agent!"
                    )

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        (
            feat_a,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb_veh,  # [n_agent, hidden_dim]
            agent_token_emb_ped,  # [n_agent, hidden_dim]
            agent_token_emb_cyc,  # [n_agent, hidden_dim]
            veh_mask,  # [n_agent]
            ped_mask,  # [n_agent]
            cyc_mask,  # [n_agent]
            categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        ) = self.agent_token_embedding(
            agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )

        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_valid_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=torch.bool, device=pos_a.device
            )

        # SMART expects this to also hold the valid mask for future steps.
        pred_valid = tokenized_agent["valid_mask"].clone()
        if pred_valid.shape[1] < n_step_future_2hz + step_current_2hz:
            pred_valid = torch.cat(
                [
                    pred_valid,
                    torch.repeat_interleave(
                        pred_valid[:, -1:], n_step_future_2hz, dim=1
                    ),
                ],
                dim=1,
            )

        pred_idx_list = []
        next_token_logits_list = []
        sample_logits_list = []
        feat_a_t_dict = {}
        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17

            if t == 0:  # init
                hist_step = step_current_2hz
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                if map_feature is not None:
                    batch_pl = torch.cat(
                        [
                            map_feature["batch"] + tokenized_agent["num_graphs"] * t
                            for t in range(hist_step)
                        ],
                        dim=0,
                    )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1
                batch_s = tokenized_agent["batch"]
                if map_feature is not None:
                    batch_pl = map_feature["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            # In the inference stage, we only infer the current stage for recurrent
            if map_feature is not None:
                edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                    pos_pl=map_feature["position"],  # [n_pl, 2]
                    orient_pl=map_feature["orientation"],  # [n_pl]
                    pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                    head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                    head_vector_a=head_vector_a[
                        :, -hist_step:
                    ],  # [n_agent, hist_step, 2]
                    mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                    batch_s=batch_s,  # [n_agent*hist_step]
                    batch_pl=batch_pl,  # [n_pl*hist_step]
                )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                batch_s=batch_s,  # [n_agent*hist_step]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
            )

            # ! attention layers
            for i in range(self.num_layers):
                # [n_agent, n_step, hidden_dim]
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:  # init, process hist_step together
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)

                    if map_feature is not None:
                        # [hist_step*n_pl, hidden_dim]
                        _feat_map = (
                            map_feature["pt_token"]
                            .unsqueeze(0)
                            .expand(hist_step, -1, -1)
                            .flatten(0, 1)
                        )
                        _feat_temporal = self.pt2a_attn_layers[i](
                            (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
                        )

                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(
                        0, 1
                    )
                    feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim]

                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal

                else:  # process one step
                    feat_a_now = self.t_attn_layers[i](
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]),
                        r_t,
                        edge_index_t,
                    )

                    if map_feature is not None:
                        feat_a_now = self.pt2a_attn_layers[i](
                            (map_feature["pt_token"], feat_a_now),
                            r_pl2a,
                            edge_index_pl2a,
                        )
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )

                    # [n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            # ! get outputs
            next_token_logits = self.token_predict_head(feat_a_now)
            next_token_logits_list.append(next_token_logits)  # [n_agent, n_token]

            sampling_args = dict(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                # ! for most-likely sampling
                next_token_logits=next_token_logits,
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )
            if sampling_scheme.criterium != "topk_prob":
                sampling_args.update(
                    dict(
                        pos_next_gt=tokenized_agent["gt_pos_raw"][
                            :, n_step
                        ],  # [n_agent, 2]
                        head_next_gt=tokenized_agent["gt_head_raw"][
                            :, n_step
                        ],  # [n_agent]
                        valid_next_gt=tokenized_agent["gt_valid_raw"][
                            :, n_step
                        ],  # [n_agent]
                        token_agent_shape=tokenized_agent[
                            "token_agent_shape"
                        ],  # [n_token, 2]
                    )
                )

            # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]
            next_token_idx, next_token_traj_all, sample_logits = sample_next_token_traj(
                **sampling_args
            )

            sample_logits_list.append(sample_logits)
            pred_idx_list.append(next_token_idx)

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if not self.training:
                pred_traj_10hz[:, t * 5 : (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2
                )
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )
                pred_valid_10hz[:, t * 5 : (t + 1) * 5] = pred_valid[
                    :, t_now
                ].unsqueeze(-1)

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]

            # ! Override positions/headings with provided trajectory if fixed agent mask is provided
            if ego_fixed:
                # n_step = token index in 2hz mode = 5 steps in 10hz mode
                ego_mask = tokenized_agent["ego_mask"]

                # Get ego position, heading, and validity from provided trajectory
                ego_pos_next = tokenized_agent["gt_pos_raw"][
                    ego_mask, n_step
                ]  # [n_ego, 2]
                ego_head_next = tokenized_agent["gt_head_raw"][
                    ego_mask, n_step
                ]  # [n_ego]
                ego_valid_next = tokenized_agent["gt_valid_raw"][
                    ego_mask, n_step
                ]  # [n_ego]

                # Override ego predictions with provided trajectory (for next step context)
                pos_a_next[ego_mask] = ego_pos_next
                head_a_next[ego_mask] = ego_head_next
                pred_valid[ego_mask, n_step] = ego_valid_next  # Use provided validity

                # ! CRITICAL FIX: Override token selection for ego to match provided trajectory
                # The token_processor already computed the correct gt_idx for ego
                if "gt_idx" in tokenized_agent:
                    # gt_idx shape: [n_agent, n_future_steps]
                    # Use the pre-computed token index from token_processor
                    next_token_idx[ego_mask] = tokenized_agent["gt_idx"][
                        ego_mask, n_step
                    ]

            # ! Override positions/headings with provided trajectory if fixed agent mask is provided
            if fixed_agent_mask is not None:
                # Get ego position, heading, and validity from provided trajectory
                agent_pos_next = tokenized_agent["gt_pos_raw"][
                    fixed_agent_mask, n_step
                ]  # [n_agent, 2]
                agent_head_next = tokenized_agent["gt_head_raw"][
                    fixed_agent_mask, n_step
                ]  # [n_agent]
                agent_valid_next = tokenized_agent["gt_valid_raw"][
                    fixed_agent_mask, n_step
                ]  # [n_agent]

                # Override ego predictions with provided trajectory (for next step context)
                pos_a_next[fixed_agent_mask] = agent_pos_next
                head_a_next[fixed_agent_mask] = agent_head_next
                pred_valid[fixed_agent_mask, n_step] = (
                    agent_valid_next  # Use provided validity
                )

                # ! CRITICAL FIX: Override token selection for fixed agents to match provided trajectory
                # The token_processor already computed the correct gt_idx for fixed agents
                if "gt_idx" in tokenized_agent:
                    # gt_idx shape: [n_agent, n_future_steps]
                    # Use the pre-computed token index from token_processor
                    next_token_idx[fixed_agent_mask] = tokenized_agent["gt_idx"][
                        fixed_agent_mask, n_step
                    ]

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(
                next_token_logits_list, dim=1
            ),  # [n_agent, n_step_future_2hz]
            "next_token_valid": pred_valid[:, step_current_2hz:],  # [n_agent, ]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # [n_agent, 16, n_token]
            "sample_logits": torch.stack(sample_logits_list, dim=1),
            "pred_idx": torch.stack(pred_idx_list, dim=1),  # [n_agent, 16]
        }

        if not self.training:  # 10hz predictions for runtime inference
            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])
            out_dict["pred_valid_10hz"] = pred_valid_10hz

        return out_dict

    def inference_with_mask(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor] | None,
        sampling_scheme: DictConfig,
        freeze_agent_future: bool = False,
        freeze_agent_mask: torch.Tensor | None = None,
        freeze_tokenized_agent: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Agent trajectory inference with optional ego trajectory conditioning.

        Args:
            tokenized_agent: Tokenized agent data
            map_feature: Encoded map features
            sampling_scheme: Token sampling configuration
            freeze_agents: If True, This enables inference where certain agent trajectories are externally provided.
            freeze_agents_mask: Use agent trajectory from
                            future_tokenized_agent['gt_pos_raw'],
                            future_tokenized_agent['gt_head_raw'],
                            future_tokenized_agent['gt_valid_raw'],
                            and future_tokenized_agent["gt_idx"],
                                   instead of predicting agent motion.
            future_tokenized_agent:

        Returns:
            Dict containing predicted trajectories and other outputs
        """
        n_agent = tokenized_agent["valid_mask"].shape[0]

        n_step_future_10hz = self.num_future_steps  # 80 or 45
        n_step_future_2hz = (
            n_step_future_10hz // self.shift
        )  # 16 or 9, or 1 if future=5

        # index
        step_current_10hz = self.num_historical_steps - 1  # 10 or 15
        step_current_2hz = step_current_10hz // self.shift  # 2 or 3

        # Validate fixed agent trajectory data is available if needed
        if freeze_agent_future and freeze_agent_mask is not None:
            # old usage: use tokenized_agent (this includes history, current, and future)
            # new usage: use future_tokenized_agent (this includes future only)
            required_keys = ["gt_pos_raw", "gt_head_raw", "gt_valid_raw", "gt_idx"]
            for key in required_keys:
                if key not in freeze_tokenized_agent:
                    raise ValueError(
                        f"freeze_agent_mask not None but required key '{key}' not found in tokenized_agent!"
                    )

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        token_idx = tokenized_agent["gt_idx"][:, :step_current_2hz]

        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        (
            feat_a,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb,  # [n_agent, step_current_2hz, hidden_dim]
            agent_token_emb_veh,  # [n_agent, hidden_dim]
            agent_token_emb_ped,  # [n_agent, hidden_dim]
            agent_token_emb_cyc,  # [n_agent, hidden_dim]
            veh_mask,  # [n_agent]
            ped_mask,  # [n_agent]
            cyc_mask,  # [n_agent]
            categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
        ) = self.agent_token_embedding(
            agent_token_index=token_idx,
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )

        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_valid_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=torch.bool, device=pos_a.device
            )

        # SMART expects this to also hold the valid mask for future steps.
        pred_valid = tokenized_agent["valid_mask"].clone()
        if pred_valid.shape[1] < n_step_future_2hz + step_current_2hz:
            pred_valid = torch.cat(
                [
                    pred_valid,
                    torch.repeat_interleave(
                        pred_valid[:, -1:], n_step_future_2hz, dim=1
                    ),
                ],
                dim=1,
            )

        pred_idx_list = []
        next_token_logits_list = []
        sample_logits_list = []
        feat_a_t_dict = {}
        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16 (token index)
            n_step = t_now + 1  # 2 -> 17 (next token index)

            if t == 0:  # init
                hist_step = step_current_2hz
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                if map_feature is not None:
                    batch_pl = torch.cat(
                        [
                            map_feature["batch"] + tokenized_agent["num_graphs"] * t
                            for t in range(hist_step)
                        ],
                        dim=0,
                    )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1
                batch_s = tokenized_agent["batch"]
                if map_feature is not None:
                    batch_pl = map_feature["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            # In the inference stage, we only infer the current stage for recurrent
            if map_feature is not None:
                edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                    pos_pl=map_feature["position"],  # [n_pl, 2]
                    orient_pl=map_feature["orientation"],  # [n_pl]
                    pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                    head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                    head_vector_a=head_vector_a[
                        :, -hist_step:
                    ],  # [n_agent, hist_step, 2]
                    mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                    batch_s=batch_s,  # [n_agent*hist_step]
                    batch_pl=batch_pl,  # [n_pl*hist_step]
                )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                batch_s=batch_s,  # [n_agent*hist_step]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
            )

            # ! attention layers
            for i in range(self.num_layers):
                # [n_agent, n_step, hidden_dim]
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:  # init, process hist_step together
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)

                    if map_feature is not None:
                        # [hist_step*n_pl, hidden_dim]
                        _feat_map = (
                            map_feature["pt_token"]
                            .unsqueeze(0)
                            .expand(hist_step, -1, -1)
                            .flatten(0, 1)
                        )
                        _feat_temporal = self.pt2a_attn_layers[i](
                            (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
                        )

                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(
                        0, 1
                    )
                    feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim]

                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal

                else:  # process one step
                    feat_a_now = self.t_attn_layers[i](
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]),
                        r_t,
                        edge_index_t,
                    )

                    if map_feature is not None:
                        feat_a_now = self.pt2a_attn_layers[i](
                            (map_feature["pt_token"], feat_a_now),
                            r_pl2a,
                            edge_index_pl2a,
                        )
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )

                    # [n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            # ! get outputs
            next_token_logits = self.token_predict_head(feat_a_now)
            next_token_logits_list.append(next_token_logits)  # [n_agent, n_token]

            sampling_args = dict(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                # ! for most-likely sampling
                next_token_logits=next_token_logits,
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )
            if sampling_scheme.criterium != "topk_prob":
                sampling_args.update(
                    dict(
                        pos_next_gt=tokenized_agent["gt_pos_raw"][
                            :, n_step
                        ],  # [n_agent, 2]
                        head_next_gt=tokenized_agent["gt_head_raw"][
                            :, n_step
                        ],  # [n_agent]
                        valid_next_gt=tokenized_agent["gt_valid_raw"][
                            :, n_step
                        ],  # [n_agent]
                        token_agent_shape=tokenized_agent[
                            "token_agent_shape"
                        ],  # [n_token, 2]
                    )
                )

            # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]
            next_token_idx, next_token_traj_all, sample_logits = sample_next_token_traj(
                **sampling_args
            )

            sample_logits_list.append(sample_logits)
            pred_idx_list.append(next_token_idx)

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if not self.training:
                pred_traj_10hz[:, t * 5 : (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2
                )
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )

                pred_valid_10hz[:, t * 5 : (t + 1) * 5] = pred_valid[
                    :, t_now
                ].unsqueeze(-1)

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]

            # ! Override positions/headings with provided trajectory if fixed agent mask is provided
            if freeze_agent_future:
                # Get ego position, heading, and validity from provided trajectory

                agent_pos_next = freeze_tokenized_agent["gt_pos_raw"][
                    :, t
                ]  # [n_agent, 2]
                agent_head_next = freeze_tokenized_agent["gt_head_raw"][
                    :, t
                ]  # [n_agent]
                agent_valid_next = freeze_tokenized_agent["gt_valid_raw"][
                    :, t
                ]  # [n_agent]
                agent_next_token_idx = freeze_tokenized_agent["gt_idx"][:, t]

                # Override ego predictions with provided trajectory (for next step context)
                pos_a_next[freeze_agent_mask] = agent_pos_next
                head_a_next[freeze_agent_mask] = agent_head_next

                next_token_idx[freeze_agent_mask] = agent_next_token_idx

                pred_valid[freeze_agent_mask, n_step] = agent_valid_next

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(next_token_logits_list, dim=1),
            "next_token_valid": pred_valid[:, step_current_2hz:],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # [n_agent, 16, n_token]
            "sample_logits": torch.stack(sample_logits_list, dim=1),
            "pred_idx": torch.stack(pred_idx_list, dim=1),  # [n_agent, 16]
        }

        if not self.training:  # 10hz predictions for runtime inference
            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])
            out_dict["pred_valid_10hz"] = pred_valid_10hz

        return out_dict
