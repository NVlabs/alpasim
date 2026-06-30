# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Optional

import torch.nn as nn

from .agent_decoder import SMARTAgentDecoder
from .map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        n_token_agent: int,
        num_agent_types: int,
        num_polyline_types: int,
        num_polygon_types: int,
        num_light_types: int,
        num_polyline_points: int,
    ) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            num_polyline_types=num_polyline_types,
            num_polygon_types=num_polygon_types,
            num_light_types=num_light_types,
            num_polyline_points=num_polyline_points,
        )
        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            num_agent_types=num_agent_types,
        )
