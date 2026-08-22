# DiffusionDrive assets

The only supported release asset is the SimScale DiffusionDrive NAVHARD
checkpoint. Download and stage it from the repository root:

```bash
DOWNLOAD_DIR="${TMPDIR:-/tmp}/simscale-checkpoints"
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
bash e2e_challenge/sample_submission_simscale_navsim_diffusiondrive/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt"
```

The installed filename is
`diffusiondrive_sim_navhard.ckpt` and its required identity is:

```text
size:   243,596,717 bytes
sha256: 8fdbdb3fdfa7b496e7d7a438efbb5c2022377e59cbfd7095270d89623c5d963f
```
