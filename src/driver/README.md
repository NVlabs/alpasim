# Alpasim driver

## Alpamayo 2 Edge

`driver=alpamayo2_edge` uses six cameras, 22 sampled images over two seconds,
20 ego poses, and turn-by-turn text. It returns 64 poses at 10 Hz, retaining the
checkpoint's body-yaw rotations. The preset uses CPU image decoding, SDPA,
eager inference, and a 2.2-second warmup.

All drivers share Transformers 5.17 and the standard simulator image. The
Alpamayo SDKs, including [NVlabs/alpamayo2-edge](https://github.com/NVlabs/alpamayo2-edge),
track upstream `main`; Alpamayo 1, 1.5, and 2 need [local compatibility patches](/third_party/alpamayo-transformers5/README.md).
Local setup and Docker builds apply these patches after installing the SDKs.
After a manual `uv sync`, run the patch command documented there.
The [Alpamayo tutorial](/docs/TUTORIAL.md#alpamayo) shows how to download
weights and run Edge.
Model weights are downloaded separately.

Download weights on the host before launching:

```bash
./data/auto-init.sh  # Select A2 Edge; deselect VaVAM-B if it is not needed.
```

For a direct download:

```bash
uv run hf download nvidia/Alpamayo2-Edge
```

These commands download the complete checkpoint, including weights, tokenizer,
processor, and configuration files, into the Hugging Face cache. The wizard
mounts that cache into the driver container; `HF_HOME` selects a different host
cache directory. The checkpoint is public, so no token is needed.

For an offline checkpoint, add `--local-dir data/drivers/Alpamayo2-Edge` to the
direct download command, then run with
`driver.model.checkpoint_path=/mnt/drivers/Alpamayo2-Edge`.
Remote deployments use the same simulator image through `defines.base_image`.
