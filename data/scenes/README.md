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

| Suite ID | Scenes | NRE | HF revision | Description |
|----------|--------|-----|-------------|-------------|
| `public_2604` | 1606 | 26.4.x | [26.04](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.04) | Public NRE scenes from the 26.04 release, excluding known-invalid scenes. Mostly new scenarios: only 159 scenes overlap with `public_2601`. |
| `public_2601` | 913 | 26.1.x | [26.01](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/26.01) | Public NRE scenes from the 26.01 release, excluding known-invalid scenes. Re-renders scenarios from `public_2507` with newer NRE. Requires sensorsim NRE-GA 26.02 or later. |
| `public_2507` | 910 | 25.7.x | [25.05](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/25.05) | Legacy public NRE scenes from the 25.07 release, hosted on the 25.05 Hugging Face revision. |

## Managing Scenes

Use `alpasim-scenes-validate` to validate CSV files.
