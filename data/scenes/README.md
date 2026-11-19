# Scene Data

This directory contains public clip catalogs and scene suites.

## Files

- `clip_catalog.csv` — Public 26.01 and 26.04 artifact metadata (2,523 clips)
- `clip_catalog_2505.csv` — Public 25.05 artifact metadata used by `public_2507`
- [`scene_suites/`](scene_suites/README.md) — Ordered, UUID-pinned scene suites
- [`ego_textures/`](/data/scenes/ego_textures/README.md) — Bundled carline textures for all public ClipGT clips

A clip catalog maps each `(clip_id, uuid)` pair to one USDZ artifact. The
`artifact_repository` and `hf_revision` columns identify its storage location and
pinned Hugging Face revision.

See the [scene data model](../../docs/DESIGN.md#scene-data-model) for the clip,
data-source, scene, and rollout definitions.

## Public suites

All public artifacts are hosted in
[nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec).

Use `public_2601` for new runs. Use `public_2604` when you need the broader
26.04 catalog, and `public_2507` only for historical reproduction.

The table below summarizes the checked-in CSV files. `Suite` is the CSV filename
stem; it is not a column in the CSV. See the
[three-column scene-suite schema](scene_suites/README.md#scene-suites).

| Suite | Clips | NRE | HF revision | Description |
| --- | ---: | --- | --- | --- |
| `public_2601` | 913 | 26.1.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01) | Recommended public NRE suite containing 913 of the 916 artifacts in the 26.01 release. Of these, 729 include ClipGT map inputs; the remaining 184 are XODR-only. Requires sensorsim NRE-GA 26.02 or later. |
| `public_2601_eval_filtered` | 875 | 26.1.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01) | `public_2601` without the 38 scenes whose recorded drive leaves the road when AlpaSim replays it. The closed-loop suite of the [Alpamayo 2 Edge model card](https://huggingface.co/nvidia/Alpamayo2-Edge). |
| `public_2601_video_model` | 729 | 26.1.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01) | UUID-pinned subset compatible with the current single-view [video-model path](../../docs/VIDEO_MODEL.md#scene-data). |
| `public_2604` | 1,606 | 26.4.x | [26.04](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.04) | Public NRE suite containing 1,606 of the 1,607 artifacts in the 26.04 release. All suite artifacts are compatible with the current single-view [video-model path](../../docs/VIDEO_MODEL.md#scene-data). Mostly new clips: only 159 overlap with `public_2601`. |
| `public_2507` | 910 | 25.7.x | [25.05](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/25.05) | Legacy public NRE clips from the 25.07 release, hosted on the 25.05 Hugging Face revision. |
| `curated_train` | 1,761 | 26.1.x and 26.4.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01), [26.04](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.04) | Curated challenge training split (`nurec_curated_train` on the challenge branch). See [split details](scene_suites/README.md#curated-nurec-split). |
| `curated_val` | 441 | 26.1.x and 26.4.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01), [26.04](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.04) | Curated challenge validation split (`nurec_curated_val` on the challenge branch). See [split details](scene_suites/README.md#curated-nurec-split). |

> :warning: Artifacts in the 26.01 and 26.04 revisions were replaced in place
> on Hugging Face. If you may have cached either release previously, delete the
> AlpaSim scene cache at
> `data/nre-artifacts` (or your configured `scenes.scene_cache`) before running
> a 26.01 or 26.04 suite, even if you are unsure whether your files are stale.
> This does not affect the separate Hugging Face model cache.
