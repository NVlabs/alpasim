# Interactive Web UI

This is a lightweight browser UI for the interactive runtime work.

The first version is intentionally split into:

- a static browser frontend
- a tiny Python HTTP gateway
- an adapter layer that can target either mock data or the interactive runtime

This keeps the page usable while the gRPC control plane is still under active
development, while still allowing real runtime integration.

## What it includes

- ego-centric scene map rendered from artifact vector map data
- camera view panel with sensor switching
- recent speed trace panel
- control panel with session create/start/pause/resume/step actions

## Current backend status

The page is wired to HTTP endpoints that mirror the intended interactive
session flow.

There are now two gateway backends:

- `MockInteractiveApiAdapter` for local UI work
- `GrpcInteractiveApiAdapter` for a real runtime daemon

The gRPC adapter calls:

- `CreateSession`
- `GetSessionState`
- `ListSensors`
- `StepSession`
- `PauseSession`
- `ResumeSession`
- `GetFrame`

from the interactive runtime.

The map endpoint is local to the HTTP gateway. It resolves the scene artifact
from `--usdz-glob`, loads the vector map or XODR-backed map through
`alpasim_utils.artifact.Artifact`, and serializes a lightweight line-layer
payload for the browser.

## Run it

From the repository root, mock mode:

```bash
uv run --project src/runtime python -m alpasim_runtime.web_debugger --host 127.0.0.1 --port 8080
```

Against a real runtime daemon:

```bash
uv run --project src/runtime python -m alpasim_runtime.web_debugger \
  --host 127.0.0.1 \
  --port 8080 \
  --runtime-address 127.0.0.1:50051 \
  --usdz-glob "$PWD/data/nre-artifacts/all-usdzs/**/*.usdz"
```

Then open:

```text
http://127.0.0.1:8080
```

## Runtime behavior

When a session is in `RUNNING` state, the page polls session state and sensor
frames automatically so pause/resume can be observed without manual refresh.
