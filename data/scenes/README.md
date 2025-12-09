# Test Suites

This directory contains scene and test suite definitions for Alpasim.

## Files

- `sim_scenes.csv` - Scene artifact metadata (uuid, scene_id, NRE version, path, artifact_repository)
- `sim_suites.csv` - Suite-to-scene mappings (which scenes belong to which test suites)

### Artifact Repositories

The `artifact_repository` column in `sim_scenes.csv` indicates where scene files are stored:
- `swiftstack` - SwiftStack/S3 storage (current default)
- `huggingface` - HuggingFace Hub (planned for future support)

## Available Test Suites

| Suite ID | Scenes | Creator | Description |
|----------|--------|---------|-------------|
| `dev.alpasim.unit_tests.v0` | 3 | @alpasim-team | CI smoke test suite with minimal scenes for quick validation. Uses stripped USDZ files containing only metadata (fast to download). |
| `public_2507_all_ex3` | 920 | @migl | All public NRE scenes (date 05. Dez 2025) excluding those with missing maps. |
| `cle_hcm_8251` | 8249 | @migl | Human driven clips from CLE. |
| `cle_nudge_1432_v2` | 1424 | @migl | Nudging clips from CLE. |
| `interactive0_0` | 63 | @migl | 3DGS AND NeRF artifacts for interactive0. |

## Managing Scenes

For documentation on how to add, migrate, or validate scenes, see the CLI tools documentation:

**[Scene Management CLI Tools](../../src/wizard/alpasim_wizard/scenes/cli/README.md)**

Quick reference:
- `alpasim-scenes-populate` - Add scenes from SwiftStack
- `alpasim-scenes-migrate` - Migrate scenes from Kratos DB
- `alpasim-scenes-validate` - Validate CSV files
- `alpasim-scenes-car2sim` - Import Car2Sim workflow scenes
