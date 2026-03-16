# alpamayo-physics

This repo contains the code for the Physics micro-service of the Alpamayo project, see [design doc](https://docs.google.com/document/d/1JFGuGnFFASgHbDlO6a8L1GttR-feAO6fRpTifebX5NU/edit?usp=sharing). Please contact [Riccardo de Lutio](mailto:rdelutio@nvidia.com) for any questions/suggestions/inquiries.

## Environment Setup

```bash
uv sync
```

## Running the Sim

### gRPC server

Start the physics server via the registered entry point:

```bash
uv run physics_server --artifact-glob '/path/to/artifacts/**/*.usdz' --host 0.0.0.0 --port 8080
```

Additional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--artifact-glob` | *(required)* | Glob expression to find artifacts (must match `.usdz` files) |
| `--host` | `localhost` | Host to bind the gRPC server to |
| `--port` | `8080` | Port to bind the gRPC server to |
| `--cache-size` | `16` | Number of scene backends to keep in LRU cache |
| `--use-ground-mesh` | `False` | Use ground mesh for scene loading |
| `--visualize` | `False` | Enable Polyscope visualization (requires `vis` extra) |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### gRPC client (gradio)

A simple gradio-based client is available (requires the `local` extra):

```bash
uv run --extra local python client.py
```

## TODO

- [ ] Include semantic mesh
- [ ] Further tests with multiple assets
- [ ] Speedups
- [ ] Collisions
- [ ] Ego vehicle dynamics
- [ ] Refine gRPC API, use request to read environment mesh
