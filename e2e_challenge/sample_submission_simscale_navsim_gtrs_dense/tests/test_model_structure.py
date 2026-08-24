# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from unittest import mock

import pytest
import timm
import torch
from navsim_gtrs_dense_challenge.simscale_gtrs_dense import backbone as backbone_module
from navsim_gtrs_dense_challenge.simscale_gtrs_dense.config import GTRSDenseConfig
from navsim_gtrs_dense_challenge.simscale_gtrs_dense.model import (
    NC_DAC_EP_SCORER,
    SAFETY_GATE_EP_SCORER,
    GTRSDenseModel,
    _safety_gated_ep_scores,
    _speed_reranked_scores,
    _trajectory_scores,
)
from torch import nn


def _model(vocab_size: int = 8, **config_kwargs: object) -> GTRSDenseModel:
    config = GTRSDenseConfig(vocab_size=vocab_size, **config_kwargs)
    return GTRSDenseModel(config, vocab=torch.zeros(vocab_size, 40, 3))


def test_config_defaults_to_release_resnet_architecture() -> None:
    config = GTRSDenseConfig(vocab_size=8)

    assert config.backbone_type == "resnet"
    assert config.image_architecture == "resnet34"
    assert config.lidar_architecture == "resnet34"
    assert config.latent is True
    assert config.lidar_seq_len == 4
    assert (config.tf_d_model, config.tf_d_ffn) == (256, 1024)

    assert GTRSDenseConfig(backbone_type="vov").backbone_type == "vov"
    with pytest.raises(ValueError, match="backbone_type"):
        GTRSDenseConfig(backbone_type="unknown")


def test_model_constructs_transfuser_encoders_offline() -> None:
    with mock.patch("timm.create_model", wraps=timm.create_model) as create_model:
        model = _model()

    assert isinstance(model._backbone, backbone_module.TransfuserBackbone)
    assert create_model.call_count == 2
    assert all(call.kwargs["pretrained"] is False for call in create_model.mock_calls)
    assert model._backbone.lidar_latent.shape == (1, 4, 256, 256)


def test_model_constructs_vov_encoder_offline() -> None:
    model = _model(backbone_type="vov")

    assert isinstance(model._backbone, backbone_module.VoVBackbone)
    assert model._backbone.img_feat_c == 1024
    assert model._keyval_embedding.weight.shape == (1024, 256)
    assert model.downscale_layer.weight.shape == (256, 1024, 1, 1)


def test_model_preserves_release_vocabulary_and_feature_shapes() -> None:
    model = _model()

    assert model._trajectory_head.vocab.shape == (8, 40, 3)
    assert model._trajectory_head.vocab.requires_grad is False
    assert model._keyval_embedding.weight.shape == (4096, 256)
    assert model._bev_downscale.weight.shape == (256, 576, 1, 1)


def test_image_features_are_contiguous_resnet_tokens() -> None:
    class StubBackbone(nn.Module):
        def forward(
            self, camera_feature: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, None]:
            batch = camera_feature.shape[0]
            return (
                torch.zeros(batch, 64, 64, 64),
                torch.zeros(batch, 512, 1, 1),
                None,
            )

    model = _model()
    model._backbone = StubBackbone()

    features = model._image_features(torch.zeros(2, 3, 8, 8))

    assert features.shape == (2, 4096, 256)
    assert features.is_contiguous()


def test_model_projects_features_to_configured_transformer_width() -> None:
    model = _model(tf_d_model=128)

    assert model._keyval_embedding.weight.shape == (4096, 128)
    assert model._bev_downscale.weight.shape == (128, 576, 1, 1)


def test_vov_image_features_are_contiguous_tokens() -> None:
    model = _model(backbone_type="vov")

    class StubBackbone(nn.Module):
        img_feat_c = 1024

        def forward(self, camera_feature: torch.Tensor) -> torch.Tensor:
            return torch.zeros(camera_feature.shape[0], 1024, 16, 64)

    model._backbone = StubBackbone()
    features = model._image_features(torch.zeros(2, 3, 8, 8))

    assert features.shape == (2, 1024, 256)
    assert features.is_contiguous()


def test_model_rejects_wrong_vocabulary_shape() -> None:
    config = GTRSDenseConfig(vocab_size=8)

    with torch.no_grad(), pytest.raises(ValueError, match=r"\(8, 40, 3\)"):
        GTRSDenseModel(config, vocab=torch.zeros(8, 8, 3))


def test_model_retains_checkpoint_visible_unused_heads() -> None:
    keys = _model().state_dict()

    assert "_agent_head._mlp_states.0.weight" in keys
    assert "_agent_head._mlp_label.0.weight" in keys
    assert "_tf_decoder.layers.0.self_attn.in_proj_weight" in keys


def test_model_has_all_release_scoring_heads() -> None:
    assert set(_model()._trajectory_head.heads) == {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "time_to_collision_within_bound",
        "ego_progress",
        "driving_direction_compliance",
        "lane_keeping",
        "traffic_light_compliance",
        "imi",
    }


def _score_results() -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor([[0.0, 1.0, -1.0]])
        for name in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "driving_direction_compliance",
            "lane_keeping",
            "traffic_light_compliance",
            "imi",
        )
    }


def test_nc_dac_ep_scorer_is_stable_equal_product() -> None:
    result = _score_results()
    scores = _trajectory_scores(result, NC_DAC_EP_SCORER)
    product = (
        result["no_at_fault_collisions"].sigmoid()
        * result["drivable_area_compliance"].sigmoid()
        * result["ego_progress"].sigmoid()
    )

    torch.testing.assert_close(scores, product.log())
    assert scores.argmax(1).item() == product.argmax(1).item()


def test_nc_dac_ep_scorer_ignores_omitted_heads() -> None:
    result = _score_results()
    expected = _trajectory_scores(result, NC_DAC_EP_SCORER)
    for name in (
        "time_to_collision_within_bound",
        "driving_direction_compliance",
        "lane_keeping",
        "traffic_light_compliance",
        "imi",
    ):
        result[name] = torch.full_like(result[name], 1000.0)

    torch.testing.assert_close(_trajectory_scores(result, NC_DAC_EP_SCORER), expected)


def test_nc_dac_ep_exponent_increases_progress_preference() -> None:
    result = _score_results()
    base = _trajectory_scores(result, NC_DAC_EP_SCORER, 1.0)
    aggressive = _trajectory_scores(result, NC_DAC_EP_SCORER, 10.0)

    torch.testing.assert_close(
        aggressive - base,
        9.0 * result["ego_progress"].sigmoid().log(),
    )


def test_safety_gate_rejects_candidates_below_anchor_nc_or_dac() -> None:
    result = {
        name: torch.zeros(1, 4)
        for name in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "driving_direction_compliance",
            "lane_keeping",
            "traffic_light_compliance",
            "imi",
        )
    }
    result["no_at_fault_collisions"] = torch.tensor([[1.0, 0.0, 2.0, 2.0]])
    result["drivable_area_compliance"] = torch.tensor([[1.0, 2.0, 0.0, 2.0]])
    result["ego_progress"] = torch.tensor([[0.0, 20.0, 30.0, 5.0]])
    anchor_scores = torch.tensor([[10.0, 0.0, 0.0, 0.0]])

    scores = _safety_gated_ep_scores(result, anchor_scores)

    assert scores.argmax(dim=1).item() == 3
    assert torch.isfinite(scores[0, 0])
    assert torch.isneginf(scores[0, 1])
    assert torch.isneginf(scores[0, 2])
    assert torch.isfinite(scores[0, 3])


def test_safety_gate_scorer_uses_nc_dac_ep_base_for_anchor() -> None:
    result = _score_results()

    torch.testing.assert_close(
        _trajectory_scores(result, SAFETY_GATE_EP_SCORER, 3.0),
        _trajectory_scores(result, NC_DAC_EP_SCORER, 3.0),
    )


def test_speed_reranking_cannot_select_outside_safe_top_k() -> None:
    base_scores = torch.tensor([[3.0, 2.0, 1.0]])
    vocabulary = torch.zeros(3, 4, 3)
    vocabulary[:, -1, 0] = torch.tensor([0.0, 10.0, 100.0])

    scores = _speed_reranked_scores(
        base_scores,
        vocabulary,
        top_k=2,
        speed_weight=10.0,
    )

    assert scores.argmax(1).item() == 1
    assert torch.isneginf(scores[0, 2])


def test_half_second_longitudinal_proxy_rewards_immediate_motion() -> None:
    base_scores = torch.tensor([[3.0, 3.0]])
    vocabulary = torch.zeros(2, 40, 3)
    vocabulary[0, 4, 0] = 4.0
    vocabulary[0, -1, 0] = 5.0
    vocabulary[1, 4, 0] = 1.0
    vocabulary[1, -1, 0] = 10.0

    scores = _speed_reranked_scores(
        base_scores,
        vocabulary,
        top_k=2,
        speed_weight=1.0,
        speed_proxy="longitudinal_0p5s",
    )

    assert scores.argmax(1).item() == 0


def test_path_length_proxy_rewards_motion_through_turns() -> None:
    base_scores = torch.tensor([[3.0, 3.0, 1.0]])
    vocabulary = torch.zeros(3, 4, 3)
    vocabulary[0, :, 0] = torch.tensor([0.0, 3.0, 6.0, 10.0])
    vocabulary[1, :, :2] = torch.tensor(
        [[0.0, 0.0], [0.0, 5.0], [4.0, 5.0], [8.0, 0.0]]
    )

    longitudinal_scores = _speed_reranked_scores(
        base_scores,
        vocabulary,
        top_k=2,
        speed_weight=1.0,
    )
    path_length_scores = _speed_reranked_scores(
        base_scores,
        vocabulary,
        top_k=2,
        speed_weight=1.0,
        speed_proxy="path_length",
    )

    assert longitudinal_scores.argmax(1).item() == 0
    assert path_length_scores.argmax(1).item() == 1
    assert torch.isneginf(path_length_scores[0, 2])


def _trackability_vocabulary() -> torch.Tensor:
    vocabulary = torch.zeros(4, 5, 3)
    vocabulary[0, :, 0] = torch.arange(5, dtype=torch.float32) * 2.5
    vocabulary[1, :, 0] = vocabulary[0, :, 0]
    vocabulary[1, :, 2] = torch.tensor([0.0, 0.4, -0.4, 0.4, -0.4])
    vocabulary[2, :, 0] = torch.arange(5, dtype=torch.float32) * 100.0
    vocabulary[3, :, 0] = vocabulary[0, :, 0]
    vocabulary[3, :, 2] = torch.tensor([0.0, 0.02, -0.02, 0.02, -0.02])
    return vocabulary


def test_heading_change_penalty_prefers_smooth_candidate_inside_top_k() -> None:
    scores = _speed_reranked_scores(
        torch.tensor([[3.0, 3.0, 2.0, 2.0]]),
        _trackability_vocabulary(),
        top_k=2,
        speed_weight=0.0,
        heading_change_weight=1.0,
    )

    assert scores.argmax(1).item() == 0
    assert torch.isneginf(scores[0, 2])


def test_curvature_penalty_prefers_smooth_candidate_inside_top_k() -> None:
    vocabulary = _trackability_vocabulary()
    vocabulary[1, :, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3, 2.5])
    scores = _speed_reranked_scores(
        torch.tensor([[3.0, 3.0, 2.0, 2.0]]),
        vocabulary,
        top_k=2,
        speed_weight=0.0,
        curvature_weight=1.0,
    )

    assert scores.argmax(1).item() == 0
    assert torch.isneginf(scores[0, 2])


def test_zero_speed_weight_preserves_base_scores() -> None:
    base_scores = torch.tensor([[3.0, 2.0, 1.0]])
    vocabulary = torch.zeros(3, 4, 3)
    vocabulary[:, -1, 0] = torch.tensor([0.0, 10.0, 100.0])

    scores = _speed_reranked_scores(
        base_scores,
        vocabulary,
        top_k=2,
        speed_weight=0.0,
    )

    torch.testing.assert_close(scores, base_scores)


def test_zero_trackability_weights_preserve_base_scores() -> None:
    base_scores = torch.tensor([[3.0, 2.0, 1.0]])
    scores = _speed_reranked_scores(
        base_scores,
        torch.zeros(3, 4, 3),
        top_k=2,
        speed_weight=0.0,
        curvature_weight=0.0,
        heading_change_weight=0.0,
    )

    torch.testing.assert_close(scores, base_scores)


def test_model_configures_speed_reranking_on_trajectory_head() -> None:
    config = GTRSDenseConfig(vocab_size=8)

    model = GTRSDenseModel(
        config,
        vocab=torch.zeros(8, 40, 3),
        scorer_mode=NC_DAC_EP_SCORER,
        ep_exponent=3.0,
        speed_top_k=4,
        speed_weight=0.1,
    )

    assert model._trajectory_head.speed_top_k == 4
    assert model._trajectory_head.speed_weight == pytest.approx(0.1)


def test_model_configures_safety_gate_anchor_reranking() -> None:
    model = GTRSDenseModel(
        GTRSDenseConfig(vocab_size=8),
        vocab=torch.zeros(8, 40, 3),
        scorer_mode=SAFETY_GATE_EP_SCORER,
        ep_exponent=3.0,
        speed_top_k=4,
        speed_weight=3.0,
    )

    assert model._trajectory_head.scorer_mode == SAFETY_GATE_EP_SCORER
    assert model._trajectory_head.speed_top_k == 4
    assert model._trajectory_head.speed_weight == pytest.approx(3.0)
