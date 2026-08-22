# GTRS-Dense assets

The canonical asset directory supports these fixed checkpoint identities:

- ResNet reward NAVHARD: `gtrs_dense_resnet_sim_reward_navhard.ckpt` from
  `exp/_ckpts/weights/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt`,
  exactly 269,095,388 bytes with SHA-256
  `8dad0395332ccd844785cbfc7c9e24cb3f8d8dbf5cb9ca7f8f8dc75478fcf409`.
- ResNet expert NAVHARD: `gtrs_dense_resnet_sim_expert_navhard.ckpt` from
  `exp/_ckpts/weights/GTRS_Dense/gtrs_dense_resnet_sim_expert_navhard.ckpt`,
  exactly 269,095,388 bytes with SHA-256
  `2496b82f5f256d7de09fca656c7634967b8660eb12e5c10386a587283629a7ff`.
- VoV reward NAVHARD: `gtrs_dense_vov_sim_reward_navhard.ckpt` from
  `exp/_ckpts/weights/GTRS_Dense/gtrs_dense_vov_sim_reward_navhard.ckpt`,
  exactly 332,348,155 bytes with SHA-256
  `7567d269bd8d0757cf906c30612bf1ad167ac7310e8af0ead74dc7798fe54c99`.
- VoV expert NAVHARD: `gtrs_dense_vov_sim_expert_navhard.ckpt` from
  `exp/_ckpts/weights/GTRS_Dense/gtrs_dense_vov_sim_expert_navhard.ckpt`,
  exactly 332,348,155 bytes with SHA-256
  `badcf3e7c3e2ecc1d7ecb9fc744c78420c368f96e47b89d1681ade7833cd5e57`.

Download the default reward checkpoint from the official
`OpenDriveLab/SimScale` Hugging Face dataset and stage it from the repository
root:

```bash
DOWNLOAD_DIR="${TMPDIR:-/tmp}/simscale-checkpoints"
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
bash e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_reward_navhard.ckpt"
```

The expert checkpoint is optional. To build that variant, download and stage it
separately:

```bash
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_expert_navhard.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
GTRS_METHOD=expert \
  bash e2e_challenge/sample_submission_simscale_navsim_gtrs_dense/scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_expert_navhard.ckpt"
```

The canonical scripts default to ResNet reward. Select expert directly with:

```bash
GTRS_METHOD=expert bash scripts/prepare_assets.sh /path/to/gtrs_dense_resnet_sim_expert_navhard.ckpt
GTRS_METHOD=expert bash scripts/build_image.sh
```

For VoV, download the corresponding official file and select the backbone for
both staging and building:

```bash
uv run --with huggingface-hub hf download \
  OpenDriveLab/SimScale \
  SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_reward_navhard.ckpt \
  --repo-type dataset --local-dir "$DOWNLOAD_DIR"
GTRS_BACKBONE=vov bash scripts/prepare_assets.sh \
  "$DOWNLOAD_DIR/SimScale_ckpts/GTRS_Dense/gtrs_dense_vov_sim_reward_navhard.ckpt"
GTRS_BACKBONE=vov bash scripts/build_image.sh
```

Use `GTRS_BACKBONE=vov GTRS_METHOD=expert` with
`gtrs_dense_vov_sim_expert_navhard.ckpt` for VoV expert. `GTRS_ASSET_DIR` and
`GTRS_ASSET_PATH` overrides remain available. An image's checkpoint identity is fixed at build time
together with its backbone; the runtime does not select or switch checkpoints.
Checkpoints remain excluded from Git.

The default release uses tracked `gtrs_dense/navsim_16384.npy`, exactly
7,864,448 bytes with SHA-256
`e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d`,
shape `(16384, 40, 3)` in `float32`. With official runtime `4d352ba`, inference
uses same-timestamp RPC dynamics when present and falls back to pose history
only when the RPC state is missing. The tracked `gtrs_dense/navhard_8192.npy`
remains an explicit rollback option; it is exactly 3,932,288 bytes with SHA-256
`cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35`.
The release checkpoint remains unchanged; only its persisted vocabulary entry
is replaced in memory before the complete strict FP32 load.
