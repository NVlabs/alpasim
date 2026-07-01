# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Dict

import torch
import torch.nn as nn
from alpasim_trafficsim.catk.smart.layers.attention_layer import AttentionLayer
from alpasim_trafficsim.catk.smart.layers.fourier_embedding import (
    FourierEmbedding,
    MLPEmbedding,
)
from alpasim_trafficsim.catk.smart.utils import (
    angle_between_2d_vectors,
    weight_init,
    wrap_angle,
)
from torch_cluster import radius_graph


class SMARTMapDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        num_polyline_types: int,
        num_polygon_types: int,
        num_light_types: int,
        num_polyline_points: int,
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.pl2pl_radius = pl2pl_radius
        self.num_layers = num_layers

        self.type_pt_emb = nn.Embedding(num_polyline_types, hidden_dim)
        self.polygon_type_emb = nn.Embedding(num_polygon_types, hidden_dim)
        self.light_pl_emb = nn.Embedding(num_light_types, hidden_dim)

        input_dim_r_pt2pt = 3
        self.r_pt2pt_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pt,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.pt2pt_layers = nn.ModuleList(
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

        # map_token_traj_src: [n_token, 11, 2].flatten(0,1)
        # old: 22=11*2
        input_dim = num_polyline_points * 2
        self.token_emb = MLPEmbedding(input_dim=input_dim, hidden_dim=hidden_dim)
        self.apply(weight_init)

    def forward(self, tokenized_map: Dict | None) -> Dict[str, torch.Tensor] | None:
        if tokenized_map is None:
            return None

        pos_pt = tokenized_map["position"]
        orient_pt = tokenized_map["orientation"]
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)
        pt_token_emb_src = self.token_emb(tokenized_map["token_traj_src"])
        x_pt = pt_token_emb_src[tokenized_map["token_idx"]]

        x_pt_categorical_embs = [
            self.type_pt_emb(tokenized_map["type"]),
            self.polygon_type_emb(tokenized_map["pl_type"]),
            self.light_pl_emb(tokenized_map["light_type"]),
        ]
        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)
        edge_index_pt2pt = radius_graph(
            x=pos_pt,
            r=self.pl2pl_radius,
            batch=tokenized_map["batch"],
            loop=False,
            max_num_neighbors=100,
        )
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        rel_orient_pt2pt = wrap_angle(
            orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]]
        )
        r_pt2pt = torch.stack(
            [
                torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                    nbr_vector=rel_pos_pt2pt[:, :2],
                ),
                rel_orient_pt2pt,
            ],
            dim=-1,
        )
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        return {
            "pt_token": x_pt,
            "position": pos_pt,
            "orientation": orient_pt,
            "batch": tokenized_map["batch"],
        }
