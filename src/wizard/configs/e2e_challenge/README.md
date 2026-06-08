# E2E Presets

## Dev

Run trusted AlpaSim against an already-running contestant driver:

```bash
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev \
  wizard.log_dir=./runs/e2e_challenge_smoke
```

Starter driver:

```bash
docker build -f e2e_challenge/starter_kit/Dockerfile \
  -t alpasim-e2e-starter-driver:latest .
docker run --rm -p 6789:6789 alpasim-e2e-starter-driver:latest
```

Scene data stays on the trusted side:

```bash
scenes.scene_cache=/mnt/alpasim-scene-cache
scenes.local_usdz_dir=/mnt/challenge-usdzs
```

Result:

```text
<wizard.log_dir>/aggregate/metrics_results.json
```

## Direct EC2

Organizer/backend preset for generating an image-only Docker Compose run
directory from the trusted AlpaSim image. The direct p5 runner starts the
contestant containers separately and passes their external driver endpoints to
the wizard.

The EC2 preset selects the current production topology, `8gpu_32rollouts`. The
direct runner supplies the complete external driver endpoint list and any
production-only renderer image/entrypoint overrides for official evaluation.

```bash
docker run --rm --gpus all \
  -e ALPASIM_IMAGE \
  -e NRE_IMAGE \
  -e CONTESTANT_IMAGE \
  -e ALPASIM_RUN_DIR \
  -e ALPASIM_DATA_DIR \
  -e SCENE_SET_ID \
  -v "$ALPASIM_RUN_DIR:$ALPASIM_RUN_DIR" \
  -v "$ALPASIM_DATA_DIR:$ALPASIM_DATA_DIR:ro" \
  "$ALPASIM_IMAGE" \
  bash -lc 'cd /repo && uv run alpasim_wizard +e2e_challenge=ec2 \
    scenes.local_usdz_dir="$ALPASIM_DATA_DIR/nre-artifacts/local-usdz/$SCENE_SET_ID" \
    scenes.scene_ids=null scenes.test_suite_id=local'
```

Then run on the EC2 host:

```bash
cd "$ALPASIM_RUN_DIR"
docker compose -f docker-compose.yaml up --force-recreate --exit-code-from runtime-0
```
