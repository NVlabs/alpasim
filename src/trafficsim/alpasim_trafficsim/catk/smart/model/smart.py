# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from alpasim_trafficsim.catk.smart.modules.smart_decoder import SMARTDecoder
from alpasim_trafficsim.catk.smart.tokens.token_processor import TokenProcessor
from torch import nn


class SMART(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent
        )

        self.validation_rollout_sampling = model_config.validation_rollout_sampling
