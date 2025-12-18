# Test Suites

This directory contains scene and test suite definitions for Alpasim.

## Files

- `sim_scenes.csv` - Scene artifact metadata (uuid, scene_id, NRE version, path, artifact_repository)
- `sim_suites.csv` - Suite-to-scene mappings (which scenes belong to which test suites)

### Artifact Repositories

The `artifact_repository` column in `sim_scenes.csv` indicates where scene files are stored:
- `swiftstack` - SwiftStack/S3 storage (unavailable in OSS version)
- `huggingface` - HuggingFace Hub

## Available Test Suites

| Suite ID | Scenes | Creator | Description |
|----------|--------|---------|-------------|
| `public_2507_ex_failures` | 910 | @mwatson | All public NRE scenes (date 10. Dec 2025) excluding those with map issues. |
