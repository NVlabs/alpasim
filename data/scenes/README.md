# Test Suites

This directory contains public scene and test suite definitions for Alpasim.

## Files

- `sim_scenes.csv` - Current public scene artifact metadata
  (uuid, scene_id, NRE version, path, artifact_repository, hf_revision)
- `sim_suites.csv` - Current public suite-to-scene mappings
- `sim_scenes_2505.csv` - Legacy public 25.07/25.05 scene artifact metadata
- `sim_suites_2505.csv` - Legacy public 25.07/25.05 suite-to-scene mappings

### Artifact Repositories

The `artifact_repository` column in the scene CSVs indicates where scene files
are stored:
- `huggingface` - HuggingFace Hub

## Available Test Suites

| Suite ID | Scenes | Description |
|----------|--------|-------------|
| `public_2601` | 916 | All public NRE scenes from the 26.01 release. Requires sensorsim NRE-GA 26.02 or later. |
| `public_2507` | 910 | Legacy public NRE scenes from the 25.07 release, hosted on the 25.05 Hugging Face revision. |

## Managing Scenes

Use `alpasim-scenes-validate` to validate CSV files.
