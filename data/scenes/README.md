# Test Suites

This directory contains public scene and test suite definitions for Alpasim.

## Files

- `sim_scenes.csv` - Public scene artifact metadata for all current releases
  (26.01 and 26.04)
- `sim_suites.csv` - Public suite-to-artifact mappings for all current releases
- `sim_scenes_2505.csv` / `sim_suites_2505.csv` - Legacy public 25.07 release
  (not loaded by default)

Each suite row includes a readable scene ID and the UUID of the exact artifact.
By contrast, `scenes.scene_ids=[...]` selects the most recently modified artifact
for each scene ID.

### Artifact Repositories

The `artifact_repository` column in the scene CSVs indicates where scene files
are stored:
- `huggingface` - HuggingFace Hub

## Available Test Suites

All public scenes are hosted in the
[nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
dataset on Hugging Face; the `hf_revision` column pins each scene to a dataset
revision.

Use `public_2601` for new runs. Use `public_2604` when you need the broader
26.04 catalog, and `public_2507` only for historical reproduction.

| Suite ID | Scenes | NRE | HF revision | Description |
|----------|--------|-----|-------------|-------------|
| `public_2601` | 913 | 26.1.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01) | Recommended public NRE suite containing 913 of the 916 artifacts in the 26.01 release. Of these, 729 include ClipGT map inputs; the remaining 184 are XODR-only. Requires sensorsim NRE-GA 26.02 or later. |
| `public_2601_video_model` | 729 | 26.1.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01) | UUID-pinned subset compatible with the current single-view [video-model path](../../docs/VIDEO_MODEL.md#scene-data). |
| `public_2604` | 1606 | 26.4.x | [26.04](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.04) | Public NRE suite containing 1,606 of the 1,607 artifacts in the 26.04 release. All suite artifacts are compatible with the current single-view [video-model path](../../docs/VIDEO_MODEL.md#scene-data). Mostly new scenarios: only 159 scenes overlap with `public_2601`. |
| `public_2507` | 910 | 25.7.x | [25.05](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/25.05) | Legacy public NRE scenes from the 25.07 release, hosted on the 25.05 Hugging Face revision. |

> :warning: Artifacts in the 26.01 and 26.04 revisions were replaced in place
> on Hugging Face. If you may have cached either release previously, delete the
> Alpasim scene cache at
> `data/nre-artifacts` (or your configured `scenes.scene_cache`) before running
> a 26.01 or 26.04 suite, even if you are unsure whether your files are stale.
> This does not affect the separate Hugging Face model cache.

## Managing Scenes

Use `alpasim-scenes-validate` to validate CSV files.
