# Video Model Renderer

Alpasim can use a video model as the renderer, hooking [NVIDIA OmniDreams](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/) into the same `renderer` endpoint as the default [NuRec](https://docs.nvidia.com/nurec/). This mode achieves better visual quality, especially for dynamic and non-rigid objects and allows for prompt-based editing.

## How It Works

Alpasim can use the video model as a stateful renderer. Unlike NuRec, which
renders individual frames from a scene request, the video model opens a rollout
session and then generates video in chunks.

At session start, Alpasim sends the video model the static conditioning for the
scene:

- HD map parquet data extracted from the selected USDZ artifact.
- Recorded first-frame JPEGs from the USDZ, one per requested camera.
- Camera intrinsics and rig-to-camera calibration parsed from the USDZ.
- Positive and negative text prompts from `runtime.renderer.video_model_config`.

Internally, the video model is conditioned on camera-view 2D renders of the HD
map, even though the HD map source data is 3D. Dynamic actors are represented
separately as cuboids sampled along their trajectories. The generated RGB video
is conditioned on these HD map renders, actor cuboids, camera calibration, text
prompt, and the recorded first frame.

The image below shows a camera frame with the HD map conditioning render
alpha-composited on top:

![Camera frame with HD map conditioning render overlaid](assets/images/frame-with-hdmap-render-overlaid.png)

The recorded first frame is the visual anchor for generation. This is why the
first frame, camera calibration, and HD map render must all agree. If a local
camera override changes the camera pose while the first-frame JPEG still comes
from the recorded USDZ camera, the HD map conditioning render no longer lines
up with the visual seed frame, and generated video can drift or misalign.

During rollout, Alpasim does not request one frame at a time. Instead, it
prefetches chunks. For each chunk, the runtime samples the ego trajectory at the
video-model frame rate, builds dynamic actor conditioning for those timestamps,
and sends both to the video model server. The server returns a block of RGB
frames for each camera, and can optionally return HD map debug frames. These
debug frames are the camera-view HD map conditioning renders used by the video
model, which makes them useful for checking alignment.

The runtime schedules the returned frames back into the simulation event loop
at their requested timestamps. Driver policies receive the generated camera
frames as normal image observations, while the next chunk is requested as the
ego trajectory advances.

Chunk size is configured with presets such as `+chunking=8frame`. The preset
keeps the video-model settings and simulation timing aligned: `chunk_frames`
controls the regular generated block size, `first_chunk_frames` controls the
short initial block after the recorded first frame, and
`runtime.simulation_config.control_timestep_us` must match the duration of a
regular chunk.

## Wizard-Managed FlashDreams Renderer
The suggested entry point is `deploy=managed_flashdreams`, which runs
OmniDreams inference using the [FlashDreams](https://github.com/NVIDIA/flashdreams) acceleration framework as a wizard-managed service. In this mode, the wizard starts the FlashDreams container automatically and hooks it up to the Alpasim runtime.

This deployment mode requires 48GB of VRAM for the default VaVam policy and
96GB of VRAM for Alpamayo1.5. If you do not have enough resources, consider
[launching the video model on a separate machine](#alternative-deployment-external-flashdreams-server).

### Building docker images

FlashDreams publishes `Dockerfile`s but not pre-built images. We need to build them ourselves.


1. Build the FlashDreams base image from your FlashDreams checkout:

```bash
cd ../flashdreams
docker build -t flashdreams-base:local -f docker/Dockerfile .
```

2. Then build the Alpasim-ready FlashDreams image. This bakes the workspace source
and installs the OmniDreams package into the locked uv environment:

```bash
docker build -t flashdreams-alpasim:local -f docker/Dockerfile.alpasim .
```

> :warning: Build with the same Docker context that will run Alpasim. You can check with `docker context show`; Docker Compose only sees images in that active daemon.

### Deploy
Run Alpasim with the managed FlashDreams deploy preset. For the default VaVam
policy:

```bash
cd ../alpasim
uv run --project src/wizard alpasim_wizard \
  deploy=managed_flashdreams \
  topology=1gpu \
  driver=vavam_video_model \
  +chunking=8frame \
  wizard.log_dir=$PWD/outputs/managed-flashdreams-vavam-run
```

`driver=vavam_video_model` uses the same VaVam model and camera selection as
`driver=vavam`, but skips the local camera calibration override that is only
valid for the default NuRec renderer. Video-model sessions use the recorded
USDZ calibration for their first-frame JPEGs.

For Alpamayo1.5, use the matching single-camera driver preset:

```bash
cd ../alpasim
uv run --project src/wizard alpasim_wizard \
  deploy=managed_flashdreams \
  topology=1gpu \
  driver=alpamayo1_5_1cam \
  +chunking=8frame \
  wizard.log_dir=$PWD/outputs/managed-flashdreams-run
```

The preset uses `services.renderer.image=flashdreams-alpasim:local`,
`external_image=true`, and `pull_policy=never`, so Docker Compose uses the local image tag instead of trying to pull from a registry. The wizard starts a `renderer-0` container and connects Alpasim services to it.

The managed container mounts persistent host caches for Hugging Face, Torch, and FlashDreams. Override these host paths if needed:

```bash
defines.hf_cache=/path/to/hf-cache \
defines.torch_cache=/path/to/torch-cache \
defines.flashdreams_cache=/path/to/flashdreams-cache
```

`HF_TOKEN` is passed through from the host environment when present.

> :book: If a Hugging Face download fails with a 401/403, authenticate locally and make sure your account has access to the asset named in the error:

```bash
uv run --project src/wizard huggingface-cli login
```

## Scene Data

The video-model deploy config uses the same default 26.01 Hugging Face scene
catalog as the sensorsim renderer. If necessary, the wizard downloads the
selected USDZ artifacts into the configured scene cache.

Set `HF_TOKEN` as described in [ONBOARDING.md](ONBOARDING.md) before running if
the scene is not already cached.

## Driver And Timing Notes

The driver config owns the camera rig and rectification calibration. Prefer a
driver preset that matches the video-model view count instead of injecting
`+cameras=...` by hand.

The OmniDreams recipe documented here supports only single-view generation. Use
a front-wide camera driver preset with the 8-frame timing preset:

```bash
driver=vavam_video_model +chunking=8frame
# or
driver=alpamayo1_5_1cam +chunking=8frame
```

VAVAM uses a single latest image (`context_length=1`), so no image-history
subsampling is needed.

Alpamayo uses a four-frame image history at 10 Hz. The video model emits frames
at 30 Hz, so Alpamayo video-model runs should use driver-side subsampling. The
`driver=alpamayo1_5_1cam` preset sets:

```bash
driver.inference.subsample_factor=3
```

This keeps the renderer forwarding all frames while the driver cache selects
the policy's expected input cadence.

## Alternative deployment: external FlashDreams server

In some cases it may be convenient to run FlashDreams server manually. This is especially useful to run the video model on a cloud/cluster machine while keeping the rest of Alpasim on a local workstation.
This can be achieved with the external
video-model deploy preset and `wizard.external_services.renderer` setting.

1. Start the OmniDreams gRPC server from a FlashDreams checkout:

```bash
cd path/to/flashdreams
uv run --package flashdreams-omnidreams torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  -m omnidreams.grpc.server \
  --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
  --host 0.0.0.0 \
  --port 50051 \
  --output_format jpeg \
  --jpeg_quality 90
```

2. Note the external IP of your FlashDreams server (e.g. using `hostname -I`)

3. Run Alpasim with the external video-model deploy preset:

```bash
cd path/to/alpasim
uv run --project src/wizard alpasim_wizard \
  deploy=external_video_model \
  topology=1gpu \
  driver=alpamayo1_5_1cam \
  +chunking=8frame \
  'wizard.external_services.renderer=["<flashdreams-ip>:50051"]' \
  wizard.log_dir=$PWD/outputs/video-model-run
```

## HDMap Debug Frames

To request HDMap debug frames from the video model server, use Hydra's `+`
syntax because these fields are optional in the deploy preset:

```bash
+runtime.renderer.video_model_config.return_hdmap_frames=true
```
