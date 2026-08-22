# LTF Assets

Download the official checkpoint and stage it from the repository root:

```bash
DOWNLOAD_DIR="${TMPDIR:-/tmp}/simscale-checkpoints"
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/LTF/ltf_sim_navtest.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
bash e2e_challenge/sample_submission_simscale_navsim_transfuser/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/LTF/ltf_sim_navtest.ckpt"
```

Docker expects the staged checkpoint at `ltf/ltf_sim_navtest.ckpt`. Do not
commit the checkpoint.

Expected SHA256: `9c1a17651bb2cd8e2edf006ea45634432c38554a8f44e0714f64d11ea31f2c69`.
