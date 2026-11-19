# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Exercise real SDK/Transformers APIs with tiny CPU models and no downloads."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3VLConfig


def _vlm_config(attn_implementation="sdpa"):
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [],
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    config._attn_implementation = attn_implementation
    return config


def _action_config(package):
    return {
        "action_space_cfg": {
            "_target_": f"{package}.action_space.UnicycleAccelCurvatureActionSpace",
            "n_waypoints": 4,
        },
        "diffusion_cfg": {
            "_target_": f"{package}.diffusion.flow_matching.FlowMatching",
            "int_method": "euler",
        },
        "action_in_proj_cfg": {
            "_target_": f"{package}.models.action_in_proj.PerWaypointActionInProjV2",
            "hidden_size": 32,
            "num_fourier_feats": 4,
            "num_enc_layers": 1,
        },
        "action_out_proj_cfg": {"_target_": "torch.nn.Linear"},
    }


@pytest.fixture
def tiny_model(request, monkeypatch):
    package, model_name, config_name = request.param
    # Only config/tokenizer/processor downloads are replaced. Model construction,
    # checkpoint loading, generation, vision, attention and cache APIs are real.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"token_{i}": i for i in range(128)}, unk_token="token_1")
        ),
        pad_token="token_0",
        unk_token="token_1",
    )
    tokenizer.traj_token_start_idx = 100
    tokenizer.traj_token_ids = {}
    module = importlib.import_module(f"{package}.models.{package}")
    config_cls = getattr(importlib.import_module(f"{package}.config"), config_name)
    cls = getattr(module, model_name)
    if package in ("alpamayo_r1", "alpamayo1_5"):
        base = importlib.import_module(f"{package}.models.base_model")
        monkeypatch.setattr(
            base.ReasoningVLAConfig,
            "_build_processor",
            lambda self: SimpleNamespace(tokenizer=tokenizer),
        )
        monkeypatch.setattr(
            base.ReasoningVLA, "_build_tokenizer", lambda self, config: tokenizer
        )
        monkeypatch.setattr(
            base.Qwen3VLConfig,
            "from_pretrained",
            classmethod(
                lambda cls, *args, **kwargs: _vlm_config(kwargs["attn_implementation"])
            ),
        )
        config = config_cls(
            vlm_name_or_path=None,
            model_dtype="float32",
            attn_implementation="sdpa",
            **_action_config(package),
        )
        config.traj_token_start_idx = 100
        config.traj_token_ids = {}
        config.vocab_size = 128
    else:
        monkeypatch.setattr(
            module, "build_alpamayo2_super_tokenizer", lambda *args: tokenizer
        )
        config = config_cls(
            vlm_config=_vlm_config().to_dict(),
            vlm_class="Qwen3VLForConditionalGeneration",
            dtype="float32",
            hist_traj_tokenizer_cfg={
                "_target_": f"{package}.models.delta_tokenizer.DeltaTrajectoryTokenizer"
            },
            future_traj_tokenizer_cfg={
                "_target_": f"{package}.models.delta_tokenizer.DeltaTrajectoryTokenizer"
            },
            expert_config={
                "llm_config": _vlm_config().text_config.to_dict(),
                **_action_config(package),
            },
            traj_ids={},
        )
        config.vlm_name_or_path = "tiny-offline"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = cls(config).eval()
        yield model
    finally:
        torch.set_num_threads(previous_threads)


MODELS = [
    ("alpamayo_r1", "AlpamayoR1", "AlpamayoR1Config"),
    ("alpamayo1_5", "Alpamayo1_5", "Alpamayo1_5Config"),
    ("alpamayo2_super", "Alpamayo2Super", "Alpamayo2SuperConfig"),
]


def _write_checkpoint(model, tmp_path):
    model.config.to_json_file(tmp_path / "config.json", use_diff=False)
    save_file(
        {
            key: value.detach().contiguous().clone()
            for key, value in model.state_dict().items()
        },
        tmp_path / "model.safetensors",
        metadata={"format": "pt"},
    )


@pytest.mark.parametrize("tiny_model", MODELS, indirect=True)
def test_checkpoint_loading_and_cached_expert(tiny_model, tmp_path):
    model = tiny_model
    _write_checkpoint(model, tmp_path)
    restored = type(model).from_pretrained(tmp_path, local_files_only=True).eval()
    torch.testing.assert_close(
        restored.state_dict(), model.state_dict(), rtol=0, atol=0
    )
    # RoPE frequencies are non-persistent buffers, so loading must regenerate them.
    torch.testing.assert_close(
        dict(restored.named_buffers()), dict(model.named_buffers()), rtol=0, atol=0
    )
    with torch.inference_mode():
        output = restored.vlm.generate(
            input_ids=torch.tensor([[3, 122, 120, 4]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            pixel_values=torch.randn(4, 3 * 2 * 16 * 16),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            max_new_tokens=2,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )
        assert output.sequences.shape == (1, 6)
        cache = output.past_key_values
        prefix_length = cache.get_seq_length()
        expert = restored.expert
        if model.config.model_type == "alpamayo2_super":
            expert = expert.expert
        positions = torch.arange(prefix_length, prefix_length + 4)
        positions = positions.view(1, 1, -1).expand(3, 1, -1)
        # Successive diffusion steps reuse the same VLM prefix cache.
        for _ in range(2):
            result = expert(
                inputs_embeds=torch.randn(1, 4, 32),
                position_ids=positions,
                past_key_values=cache,
                attention_mask=torch.zeros(1, 1, 4, prefix_length + 4),
                use_cache=True,
                is_causal=False,
            )
            assert result.last_hidden_state.shape == (1, 4, 32)
            assert torch.isfinite(result.last_hidden_state).all()
            cache.crop(prefix_length)
            assert cache.get_seq_length() == prefix_length


@pytest.mark.parametrize("tiny_model", [MODELS[0]], indirect=True)
def test_alpamayo1_adapter_overrides_checkpoint_flash_attention(
    tiny_model, tmp_path, monkeypatch
):
    from alpasim_driver.models import alpamayo1_model

    tiny_model.config.attn_implementation = "flash_attention_2"
    _write_checkpoint(tiny_model, tmp_path)
    monkeypatch.setattr(
        alpamayo1_model.helper, "get_processor", lambda tokenizer: SimpleNamespace()
    )
    adapter = alpamayo1_model.Alpamayo1Model(
        str(tmp_path), torch.device("cpu"), ["camera_front_wide_120fov"]
    )
    assert adapter._model.config.attn_implementation == "sdpa"
    assert adapter._model.vlm.config._attn_implementation == "sdpa"
