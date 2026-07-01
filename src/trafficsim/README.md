# alpasim-trafficsim

CATK-backed traffic gRPC service for AlpaSim. The runtime starts a session with
logged ego and agent trajectories, then calls `simulate` to get closed-loop
contender trajectory updates.

## Service Flow

- `start_session` loads the requested USDZ scene, builds CATK env data, and
  seeds `SessionState.closed_loop_trajectories` from logged trajectories.
- Requests at or before `handover_time_us` use logged replay.
- Requests after handover resample the latest closed-loop history into a CATK
  input window, condition CATK on ego future poses, run world-model inference,
  and return contender updates at the requested timestamp plus available
  forecast points.
- CATK must produce predictions after handover. If inference cannot produce
  actions, `simulate` returns `FAILED_PRECONDITION` rather than fabricating a
  static fallback.

## Layout

```text
alpasim_trafficsim/
  grpc/
    catk_trafficsim.py     # Hydra entrypoint and gRPC server bootstrap
    servicer.py            # TrafficService RPC implementation
    catk_predictor.py      # CATK inference and prediction post-processing
    session/               # SessionState construction and trajectory history
    pipeline/              # Env-data transforms and response building
  catk/
    scene_adapter.py       # USDZ scene to CATK env data
    map_adapter.py         # Vector map conversion
    model_adapter.py       # CATK world-model wrapper
    env_data_adapter.py    # Env data to SMART tensors
    smart/                 # SMART model implementation
  config/server.yaml
```

## Run

For local development, start the CATK server from a Hydra YAML config:

```bash
uv run catk_trafficsim_server \
  --config-path=/path/to/config-dir \
  --config-name=trafficsim-config.yaml \
  server.port=6003 \
  catk.loader.usdz_folder=/path/to/scenesets
```

The wizard writes `trafficsim-config.yaml` for docker-compose runs. Static
settings, including model paths, live in that file; the service port and active
sceneset path are passed as launch-time overrides.
