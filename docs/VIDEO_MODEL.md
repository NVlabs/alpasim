# Video Model Renderer

Alpasim includes a built-in client for using an external chunked video model as
the renderer. The wizard config entry point is:

```bash
uv run --project src/wizard alpasim_wizard \
  deploy=external_video_model \
  topology=1gpu \
  driver=<driver> \
  +chunking=<8frame|12frame|16frame> \
  wizard.external_services.renderer='["<host>:50051"]' \
  wizard.log_dir=$PWD/outputs/video-model-run
```

The driver config owns the camera rig and rectification calibration. Avoid
adding a separate `+cameras=...` override unless the driver config was also
updated to match.

## Scene Data

The default config uses the public OmniDreams scene pack. Download it with:

```bash
bash data/auto-init.sh
```

Then select the `OmniDreams Scenes` option. The deploy config points at:

```text
data/omni-dreams-scenes/scenes
```

Default scene:

```text
clipgt-01d503d4-449b-46fc-8d78-9085e70d3554
```

Other scene IDs in the public pack:

```text
clipgt-065dcac9-ee67-4434-a835-c6b816c88e48
clipgt-0b10bce8-61f1-4350-8577-cf3c9493ffc3
clipgt-0d1fcd2c-ed47-4c72-b756-8e24bce0b9f4
clipgt-0d76134f-350d-44b5-a694-208e9dab9600
```

## Server Examples

The external video model server lives in the Flashdreams repository. Run the
server commands below from your Flashdreams checkout.

One-view server:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m integrations.alpadreams.alpadreams.grpc.server \
  --n_cameras 1 --num_frames_per_block <8|12|16> \
  --local_attn_size 6 --sink_size 0 \
  --output_format jpeg --jpeg_quality 90
```

Four-view server:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m integrations.alpadreams.alpadreams.grpc.server \
  --n_cameras 4 --num_frames_per_block 16 \
  --encode_with_pixel_shuffle \
  --output_format jpeg --jpeg_quality 90
```

## Driver Notes

VAVAM uses a single latest image (`context_length=1`), so no image-history
subsampling is needed.

Alpamayo uses a four-frame image history at 10Hz. The video model emits frames
at 30Hz, so Alpamayo video-model runs should use driver-side subsampling:

```bash
driver.inference.subsample_factor=3
```

This keeps the renderer forwarding all frames while the driver cache selects
the policy's expected input cadence.
