# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

_STAGE_SPECS = {
    "V-99-eSE": {
        "stem": [64, 64, 128],
        "stage_conv_ch": [128, 160, 192, 224],
        "stage_out_ch": [256, 512, 768, 1024],
        "layer_per_block": 5,
        "block_per_stage": [1, 3, 9, 3],
        "eSE": True,
        "dw": False,
    }
}

_SUPPORTED_OUT_FEATURES = {"stage2", "stage3", "stage4", "stage5"}


def conv3x3(
    in_channels: int,
    out_channels: int,
    module_name: str,
    postfix: str | int,
    stride: int = 1,
    groups: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
) -> list[tuple[str, nn.Module]]:
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(
    in_channels: int,
    out_channels: int,
    module_name: str,
    postfix: str | int,
    stride: int = 1,
    groups: int = 1,
    kernel_size: int = 1,
    padding: int = 0,
) -> list[tuple[str, nn.Module]]:
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel: int, reduction: int = 4) -> None:
        super().__init__()
        del reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        inputs = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return inputs * x


class _OSA_module(nn.Module):
    def __init__(
        self,
        in_ch: int,
        stage_ch: int,
        concat_ch: int,
        layer_per_block: int,
        module_name: str,
        SE: bool = False,
        identity: bool = False,
        depthwise: bool = False,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        if depthwise:
            raise ValueError("the V-99-eSE graph does not use depthwise convolutions")

        self.with_cp = with_cp
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()

        in_channel = in_ch
        for index in range(layer_per_block):
            self.layers.append(
                nn.Sequential(
                    OrderedDict(conv3x3(in_channel, stage_ch, module_name, index))
                )
            )
            in_channel = stage_ch

        aggregate_channels = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(aggregate_channels, concat_ch, module_name, "concat"))
        )
        self.ese = eSEModule(concat_ch)

    def _forward(self, x: Tensor) -> Tensor:
        identity_feat = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        x = self.concat(x)
        x = self.ese(x)
        if self.identity:
            x = x + identity_feat
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.with_cp and self.training and x.requires_grad:
            return cp.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class _OSA_stage(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        stage_ch: int,
        concat_ch: int,
        block_per_stage: int,
        layer_per_block: int,
        stage_num: int,
        SE: bool = False,
        depthwise: bool = False,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        if stage_num != 2:
            self.add_module(
                "Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name,
            _OSA_module(
                in_ch,
                stage_ch,
                concat_ch,
                layer_per_block,
                module_name,
                SE,
                depthwise=depthwise,
                with_cp=with_cp,
            ),
        )

        for index in range(block_per_stage - 1):
            if index != block_per_stage - 2:
                SE = False
            module_name = f"OSA{stage_num}_{index + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise,
                    with_cp=with_cp,
                ),
            )


class VoVNet(nn.Module):
    def __init__(
        self,
        spec_name: str,
        input_ch: int = 3,
        out_features: list[str] | None = None,
        frozen_stages: int = -1,
        norm_eval: bool = True,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        if spec_name != "V-99-eSE":
            raise ValueError("spec_name must be 'V-99-eSE'")
        if input_ch != 3:
            raise ValueError("input_ch must be 3")
        if (
            not isinstance(out_features, list)
            or not out_features
            or not all(
                isinstance(feature, str) and feature in _SUPPORTED_OUT_FEATURES
                for feature in out_features
            )
            or len(out_features) != len(set(out_features))
        ):
            raise ValueError(
                "out_features must be a non-empty list of unique supported features"
            )

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        stage_specs = _STAGE_SPECS[spec_name]
        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        use_ese = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv3x3(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv3x3(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", nn.Sequential(OrderedDict(stem)))

        current_stride = 4
        self._out_feature_strides = {"stem": current_stride, "stage2": current_stride}
        self._out_feature_channels = {"stem": stem_ch[2]}
        in_ch_list = [stem_ch[2], *config_concat_ch[:-1]]

        self.stage_names: list[str] = []
        for index in range(4):
            name = f"stage{index + 2}"
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[index],
                    config_stage_ch[index],
                    config_concat_ch[index],
                    block_per_stage[index],
                    layer_per_block,
                    index + 2,
                    use_ese,
                    depthwise,
                    with_cp=with_cp,
                ),
            )
            self._out_feature_channels[name] = config_concat_ch[index]
            if index != 0:
                current_stride *= 2
                self._out_feature_strides[name] = current_stride

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, x: Tensor) -> list[Tensor]:
        outputs = []
        x = self.stem(x.flip(1))
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs.append(x)
        return outputs

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.stem.eval()
            for parameter in self.stem.parameters():
                parameter.requires_grad = False

        for index in range(1, self.frozen_stages + 1):
            stage = getattr(self, f"stage{index + 1}")
            stage.eval()
            for parameter in stage.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> VoVNet:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()
        return self
