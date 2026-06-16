# VAVAM Assets

Model weights are local build inputs and should not be committed.

Run:

```bash
bash e2e_challenge/sample_submission_vavam/scripts/prepare_assets.sh /path/to/vavam_weights
```

Expected source files:

```text
VAM_width_1024_pretrained_139k.pt
VQ_ds16_16384_llamagen_encoder.jit
```

The script copies them into `e2e_challenge/sample_submission_vavam/assets/vavam/`
for the Docker build.
