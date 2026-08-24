# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import MemoryEffTransformer
from .backbone import TransfuserBackbone, VoVBackbone
from .config import GTRSDenseConfig

RELEASE_SCORER = "release"
NC_DAC_EP_SCORER = "nc_dac_ep"
SAFETY_GATE_EP_SCORER = "safety_gate_ep"
VALID_SCORER_MODES = frozenset(
    {RELEASE_SCORER, NC_DAC_EP_SCORER, SAFETY_GATE_EP_SCORER}
)
VALID_SPEED_PROXIES = frozenset({"longitudinal", "longitudinal_0p5s", "path_length"})


def _speed_reranked_scores(
    base_scores: Tensor,
    vocabulary: Tensor,
    *,
    top_k: int,
    speed_weight: float,
    speed_proxy: str = "longitudinal",
    curvature_weight: float = 0.0,
    heading_change_weight: float = 0.0,
) -> Tensor:
    if speed_proxy not in VALID_SPEED_PROXIES:
        raise ValueError(
            f"speed_proxy must be one of {sorted(VALID_SPEED_PROXIES)}; "
            f"got {speed_proxy!r}"
        )
    if speed_weight == 0.0 and curvature_weight == 0.0 and heading_change_weight == 0.0:
        return base_scores
    if top_k <= 0 or top_k > base_scores.shape[1]:
        raise ValueError("top_k must be between 1 and the vocabulary size")
    if vocabulary.ndim != 3 or vocabulary.shape[0] != base_scores.shape[1]:
        raise ValueError("vocabulary must align with the candidate scores")
    for name, weight in (
        ("speed_weight", speed_weight),
        ("curvature_weight", curvature_weight),
        ("heading_change_weight", heading_change_weight),
    ):
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    top_scores, top_indices = base_scores.topk(top_k, dim=1)
    adjusted = top_scores

    def standardized(metric: Tensor) -> Tensor:
        top_metric = metric[top_indices]
        centered = top_metric - top_metric.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).sqrt()
        return centered / scale.clamp_min(torch.finfo(base_scores.dtype).eps)

    if speed_weight > 0.0:
        if speed_proxy == "longitudinal":
            speed_metric = (vocabulary[:, -1, 0] - vocabulary[:, 0, 0]).clamp_min(0.0)
        elif speed_proxy == "longitudinal_0p5s":
            # Vocabulary poses start at 0.1 s, so index 4 is the 0.5 s pose.
            speed_metric = vocabulary[:, 4, 0].clamp_min(0.0)
        else:
            speed_metric = torch.linalg.vector_norm(
                torch.diff(vocabulary[..., :2], dim=1), dim=-1
            ).sum(dim=1)
        adjusted = adjusted + speed_weight * standardized(speed_metric)
    if curvature_weight > 0.0 or heading_change_weight > 0.0:
        delta_yaw = torch.diff(vocabulary[..., 2], dim=1)
        delta_yaw = torch.atan2(delta_yaw.sin(), delta_yaw.cos()).abs()
        if heading_change_weight > 0.0:
            adjusted = adjusted - heading_change_weight * standardized(
                delta_yaw.mean(dim=1)
            )
        if curvature_weight > 0.0:
            segment_length = torch.linalg.vector_norm(
                torch.diff(vocabulary[..., :2], dim=1), dim=-1
            )
            max_curvature = (delta_yaw / segment_length.clamp_min(0.1)).amax(dim=1)
            adjusted = adjusted - curvature_weight * standardized(max_curvature)

    scores = torch.full_like(base_scores, -torch.inf)
    return scores.scatter(1, top_indices, adjusted)


def _trajectory_scores(
    result: Mapping[str, Tensor],
    scorer_mode: str = RELEASE_SCORER,
    ep_exponent: float = 1.0,
) -> Tensor:
    if scorer_mode in {NC_DAC_EP_SCORER, SAFETY_GATE_EP_SCORER}:
        if not math.isfinite(ep_exponent) or ep_exponent <= 0.0 or ep_exponent > 20.0:
            raise ValueError("ep_exponent must be finite and in (0, 20]")
        return (
            result["no_at_fault_collisions"].sigmoid().log()
            + result["drivable_area_compliance"].sigmoid().log()
            + ep_exponent * result["ego_progress"].sigmoid().log()
        )
    if scorer_mode != RELEASE_SCORER:
        raise ValueError(
            f"scorer_mode must be one of {sorted(VALID_SCORER_MODES)}; "
            f"got {scorer_mode!r}"
        )
    return (
        0.03 * result["imi"].softmax(-1).log()
        + 0.1 * result["traffic_light_compliance"].sigmoid().log()
        + 0.1 * result["no_at_fault_collisions"].sigmoid().log()
        + 0.9 * result["drivable_area_compliance"].sigmoid().log()
        + 0.2 * result["driving_direction_compliance"].sigmoid().log()
        + 6.0
        * (
            7.0 * result["time_to_collision_within_bound"].sigmoid()
            + 7.0 * result["ego_progress"].sigmoid()
            + 3.0 * result["lane_keeping"].sigmoid()
        ).log()
    )


def _safety_gated_ep_scores(
    result: Mapping[str, Tensor], anchor_scores: Tensor
) -> Tensor:
    anchor_indices = anchor_scores.argmax(dim=1, keepdim=True)
    nc = result["no_at_fault_collisions"].sigmoid()
    dac = result["drivable_area_compliance"].sigmoid()
    eligible = (nc >= nc.gather(1, anchor_indices)) & (
        dac >= dac.gather(1, anchor_indices)
    )
    ep_scores = result["ego_progress"].sigmoid().log()
    return ep_scores.masked_fill(~eligible, -torch.inf)


class AgentHead(nn.Module):
    def __init__(self, num_agents: int, d_ffn: int, d_model: int) -> None:
        super().__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._mlp_states = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 5),
        )
        self._mlp_label = nn.Sequential(nn.Linear(d_model, 1))


class GTRSDenseModel(nn.Module):
    def __init__(
        self,
        config: GTRSDenseConfig,
        *,
        vocab: Tensor,
        scorer_mode: str = RELEASE_SCORER,
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
    ) -> None:
        super().__init__()
        expected_vocab_shape = (config.vocab_size, config.num_poses, 3)
        if tuple(vocab.shape) != expected_vocab_shape:
            raise ValueError(
                f"vocabulary must have shape {expected_vocab_shape}; got {tuple(vocab.shape)}"
            )
        if not vocab.is_floating_point():
            raise ValueError("vocabulary must use a floating-point dtype")

        self._query_splits = [config.num_bounding_boxes]
        self._config = config
        if config.backbone_type == "resnet":
            self._backbone = TransfuserBackbone(config)
            self._bev_downscale = nn.Conv2d(512 + 64, config.tf_d_model, 1)
            keyval_tokens = 4096
        else:
            self._backbone = VoVBackbone(config)
            self.downscale_layer = nn.Conv2d(
                self._backbone.img_feat_c, config.tf_d_model, 1
            )
            keyval_tokens = config.img_vert_anchors * config.img_horz_anchors
        self._keyval_embedding = nn.Embedding(keyval_tokens, config.tf_d_model)
        self._status_encoding = nn.Linear(8 * config.num_ego_status, config.tf_d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            config.num_bounding_boxes,
            config.tf_d_ffn,
            config.tf_d_model,
        )
        self._trajectory_head = GTRSDenseTrajectoryHead(
            num_poses=config.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab=vocab,
            config=config,
            scorer_mode=scorer_mode,
            ep_exponent=ep_exponent,
            speed_top_k=speed_top_k,
            speed_weight=speed_weight,
            speed_proxy=speed_proxy,
            curvature_weight=curvature_weight,
            heading_change_weight=heading_change_weight,
        )

    def _image_features(self, camera_feature: Tensor) -> Tensor:
        if self._config.backbone_type == "vov":
            image_features = self._backbone(camera_feature)
            return (
                self.downscale_layer(image_features)
                .flatten(-2, -1)
                .permute(0, 2, 1)
                .contiguous()
            )
        upscaled, bev_feature, _ = self._backbone(camera_feature)
        if upscaled is None:
            raise RuntimeError("GTRS-Dense requires the backbone top-down features")
        bev_feature = F.interpolate(
            bev_feature,
            size=upscaled.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        bev_feature = torch.cat((bev_feature, upscaled), dim=1)
        return (
            self._bev_downscale(bev_feature)
            .flatten(-2, -1)
            .permute(0, 2, 1)
            .contiguous()
        )

    def forward(
        self, features: Mapping[str, Tensor | list[Tensor]]
    ) -> dict[str, Tensor]:
        status_feature = features["status_feature"]
        if isinstance(status_feature, list):
            status_feature = status_feature[-1]
        camera_feature = features["camera_feature"]
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]

        status_encoding = self._status_encoding(status_feature)
        keyval = self._image_features(camera_feature)
        keyval = keyval + self._keyval_embedding.weight[None]
        return self._trajectory_head(keyval, status_encoding)


class GTRSDenseTrajectoryHead(nn.Module):
    def __init__(
        self,
        *,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        vocab: Tensor,
        config: GTRSDenseConfig,
        scorer_mode: str = RELEASE_SCORER,
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if scorer_mode not in VALID_SCORER_MODES:
            raise ValueError(
                f"scorer_mode must be one of {sorted(VALID_SCORER_MODES)}; "
                f"got {scorer_mode!r}"
            )
        self.config = config
        self.scorer_mode = scorer_mode
        self.ep_exponent = ep_exponent
        if not isinstance(speed_top_k, int) or speed_top_k < 0:
            raise ValueError("speed_top_k must be a non-negative integer")
        if speed_top_k > vocab.shape[0]:
            raise ValueError("speed_top_k cannot exceed the vocabulary size")
        if not math.isfinite(speed_weight) or speed_weight < 0.0:
            raise ValueError("speed_weight must be finite and non-negative")
        if speed_weight > 0.0 and speed_top_k == 0:
            raise ValueError(
                "speed_top_k must be positive when speed_weight is enabled"
            )
        nc_dac_ep_scorers = {NC_DAC_EP_SCORER, SAFETY_GATE_EP_SCORER}
        if speed_weight > 0.0 and scorer_mode not in nc_dac_ep_scorers:
            raise ValueError("speed reranking requires an NC/DAC/EP-based scorer")
        if speed_proxy not in VALID_SPEED_PROXIES:
            raise ValueError(
                f"speed_proxy must be one of {sorted(VALID_SPEED_PROXIES)}; "
                f"got {speed_proxy!r}"
            )
        for name, weight in (
            ("curvature_weight", curvature_weight),
            ("heading_change_weight", heading_change_weight),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (curvature_weight > 0.0 or heading_change_weight > 0.0) and speed_top_k == 0:
            raise ValueError("trackability reranking requires a positive speed_top_k")
        if (
            curvature_weight > 0.0 or heading_change_weight > 0.0
        ) and scorer_mode not in nc_dac_ep_scorers:
            raise ValueError(
                "trackability reranking requires an NC/DAC/EP-based scorer"
            )
        self.speed_top_k = speed_top_k
        self.speed_weight = speed_weight
        self.speed_proxy = speed_proxy
        self.curvature_weight = curvature_weight
        self.heading_change_weight = heading_change_weight
        self._num_poses = num_poses
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model,
                nhead,
                d_ffn,
                dropout=0.0,
                batch_first=True,
            ),
            nlayers,
        )
        self.vocab = nn.Parameter(vocab.detach().clone(), requires_grad=False)
        self.heads = nn.ModuleDict(
            {
                "no_at_fault_collisions": _score_head(d_model, d_ffn),
                "drivable_area_compliance": _score_head(d_model, d_ffn),
                "time_to_collision_within_bound": _score_head(d_model, d_ffn),
                "ego_progress": _score_head(d_model, d_ffn),
                "driving_direction_compliance": _score_head(d_model, d_ffn),
                "lane_keeping": _score_head(d_model, d_ffn),
                "traffic_light_compliance": _score_head(d_model, d_ffn),
                "imi": nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            }
        )
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0,
            )
        self.use_nerf = config.use_nerf
        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

    def forward(
        self, bev_feature: Tensor, status_encoding: Tensor
    ) -> dict[str, Tensor]:
        vocab = self.vocab.detach()
        vocab_count = vocab.shape[0]
        embedded_vocab = self.pos_embed(vocab.reshape(vocab_count, -1))[None]
        if self.normalize_vocab_pos:
            embedded_vocab = self.encoder(embedded_vocab)
        embedded_vocab = embedded_vocab.repeat(bev_feature.shape[0], 1, 1)
        trajectory_features = self.transformer(embedded_vocab, bev_feature)
        score_features = trajectory_features + status_encoding.unsqueeze(1)

        result = {
            name: head(score_features).squeeze(-1) for name, head in self.heads.items()
        }
        scores = _trajectory_scores(result, self.scorer_mode, self.ep_exponent)
        scores = _speed_reranked_scores(
            scores,
            vocab,
            top_k=self.speed_top_k,
            speed_weight=self.speed_weight,
            speed_proxy=self.speed_proxy,
            curvature_weight=self.curvature_weight,
            heading_change_weight=self.heading_change_weight,
        )
        if self.scorer_mode == SAFETY_GATE_EP_SCORER:
            scores = _safety_gated_ep_scores(result, scores)
        selected_indices = scores.argmax(1)
        result.update(
            {
                "dropout_indices": torch.arange(vocab_count, device=vocab.device),
                "trajectory_vocab_dropout": vocab,
                "trajectory": vocab[selected_indices],
                "trajectory_vocab": vocab,
                "selected_indices": selected_indices,
                "scores": scores,
            }
        )
        return result


def _score_head(d_model: int, d_ffn: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_ffn),
        nn.ReLU(),
        nn.Linear(d_ffn, 1),
    )
