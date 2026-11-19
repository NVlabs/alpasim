# Alpamayo SDK patches for Transformers 5

Alpamayo 1, 1.5, and 2 track upstream `main` in the root `pyproject.toml`.
These Apache-2.0 patches modify the installed SDK source so all drivers can
share Edge's Transformers 5.17 runtime. The root dependency override handles
the SDKs' existing Transformers 4 pins. Alpamayo 2 Edge targets Transformers 5
upstream and needs no patch.

Validated against these upstream `main` commits on 2026-10-30:

| SDK | Repository / commit |
| --- | --- |
| Alpamayo 1 | [NVlabs/alpamayo — `11a0e01c13a5622377c45ee37d653351453ec43b`](https://github.com/NVlabs/alpamayo/commit/11a0e01c13a5622377c45ee37d653351453ec43b) |
| Alpamayo 1.5 | [NVlabs/alpamayo1.5 — `36aeb4c5938cbc2eb2aed33b22434773da4ab639`](https://github.com/NVlabs/alpamayo1.5/commit/36aeb4c5938cbc2eb2aed33b22434773da4ab639) |
| Alpamayo 2 | [NVlabs/alpamayo2 — `59776920bbd981f2c989213bba5ee8594e71ed1d`](https://github.com/NVlabs/alpamayo2/commit/59776920bbd981f2c989213bba5ee8594e71ed1d) |

## Rationale and references

- All three SDKs delegate weight tying to the superclass with the new arguments.
  The [v5.17 API](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/model#transformers.PreTrainedModel.tie_weights)
  adds `missing_keys` and `recompute_mapping` for checkpoint loading.
- All three SDKs' read-only prompt cache layer for the CUDA-graph diffusion expert
  implements the
  [v5.17 `CacheLayerMixin`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
  interface: `get_max_length()` replaces `get_max_cache_shape()`, and
  `get_mask_sizes()` takes the query length instead of the cache positions.
  Without it, every inference with `diffusion_expert_cuda_graph: true` fails.
- Alpamayo 2 passes the processor's `mm_token_type_ids` to its shared prompt
  prefill. The v5.17 Qwen3-VL model requires them to compute multimodal RoPE
  whenever image inputs are passed.
- Alpamayo 1/1.5 use `vlm.get_input_embeddings()` to access the existing embedding
  through its public interface.
- Alpamayo 1 declares SDPA support for its nested Qwen backbone; the driver
  explicitly selects SDPA to preserve the previous image's effective backend.
  In `alpasim-base:0.159.0` (Transformers 4.57.1, same SDK commit listed above),
  `flash-attn` was absent. Transformers 4's config loader replaced the checkpoint's
  `attn_implementation="flash_attention_2"` with `None` when the caller omitted
  that argument, allowing Qwen to select SDPA. Transformers 5 preserves the
  checkpoint value and consequently requests the missing FlashAttention package.
  A probe in the previous image using the real checkpoint config and a tiny Qwen
  model confirmed SDPA selection for both text and vision backbones. The SDK flag
  lets Transformers accept the explicit selection; it adds no attention kernels.
  This preserves the backend choice, not guaranteed identical model outputs
  across Transformers versions.
- Alpamayo 1.5 regenerates its non-persistent Fourier-frequency and action-normalization buffers in
  `_init_weights`. Transformers 5 materializes these buffers during loading;
  they need their original logarithmic frequencies restored, not empty storage.
  The migration guide's custom-model initialization section covers this hook.
- Alpamayo 1 creates its Beta-distribution constants on CPU so construction
  succeeds under meta-device checkpoint loading. They are used for training
  timestep sampling, not the inference sampler.
- Alpamayo 2 calls `post_init()` on its wrapper and expert to collect composite
  model metadata and initialize modules. See the [API reference](https://huggingface.co/docs/transformers/v5.17.0/en/main_classes/model#transformers.PreTrainedModel.post_init)
  and the [v5 migration guide](https://github.com/huggingface/transformers/blob/v5.17.0/MIGRATION_GUIDE_V5.md#custom-pretrained-models).

These fixes restore API compatibility. They do not establish numerical parity
between Transformers 4 and 5; initialization, tokenization, and generation
changes still warrant comparisons using identical weights and inputs.

## Applying and updating

`setup_local_env.sh`, Docker builds, and driver CI tests apply the patches after
installing dependencies. After a manual installation/update, run:

```bash
uv sync --extra all
uv run --no-sync python third_party/alpamayo-transformers5/apply_patches.py
```

To refresh an existing environment to upstream `main`:

```bash
uv sync --extra all --upgrade-package alpamayo-r1 \
  --upgrade-package alpamayo1-5 --upgrade-package alpamayo2-super
uv run --no-sync python third_party/alpamayo-transformers5/apply_patches.py
```

`uv.lock` records the resolved commits. The patch script prints each installed
commit, checks all patches before changing files, and accepts already-applied
patches. If upstream changes no longer match, it fails for review.
It edits installed packages, not upstream checkouts or uv's cache. After these
fixes land upstream, remove the patch step and the obsolete dependency overrides.

Run the CPU compatibility checks without checkpoint downloads:

```bash
uv run --no-sync pytest src/driver/src/alpasim_driver/tests/test_transformers_compatibility.py
```

They exercise tiny-model checkpoint loading, exact parameter/buffer restoration,
image-conditioned generation, and repeated expert-cache reuse. Full-size model
inference remains a separate validation step.
