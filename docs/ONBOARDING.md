# Onboarding

Alpasim depends on access to the following:

- Hugging Face access
  - Used for downloading simulation artifacts
  - Data is
    [here](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/tree/main/sample_set/25.07_release)
  - See info on data
    [here](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/blob/main/README.md#dataset-format)
    for more information on the contents of artifacts used to define scenes
  - You will need to create a free Hugging Face account if you do not already have one and create an
    access token with read access. See [access tokens](https://huggingface.co/settings/tokens).
  - **You must also request access** to the gated dataset at
    [nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec).
    Without this, scene downloads will fail with a `GatedRepoError`.
  - Once you have the token, set it as an environment variable: `export HF_TOKEN=<token>`
- A version of `uv` installed (see [here](https://docs.astral.sh/uv/getting-started/installation/))
  - Example installation command for Ubuntu: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Rust toolchain (`cargo`) for building `utils_rs`, a compiled extension that accelerates trajectory
  transformations and interpolations in the runtime. Install via
  [rustup](https://rustup.rs/) or interactively let `setup_local_env.sh` install it for you.
- Docker installed (see [setup instructions](https://docs.docker.com/engine/install/ubuntu/))
- Docker compose installed (see
  [setup instructions](https://docs.docker.com/compose/install/linux/))
  - The wizard needs `docker`, `docker-compose-plugin`, and `docker-buildx-plugin`
  - Docker needs to be able to run without `sudo`. If you see a permission error when running
    `docker` commands, add yourself to the docker group: `sudo usermod -aG docker $USER`
- CUDA 12.8 or greater installed (see [here](https://developer.nvidia.com/cuda-downloads) for
  instructions)
  - The NRE container uses CUDA 12.8, so your host NVIDIA driver must support it (driver version >=
    570.x).
- Install the NVIDIA Container Toolkit (see
  [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

## Dependency management

The repo is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/). All packages under `src/` and `plugins/` are workspace members sharing a single lockfile (`uv.lock`). The root `pyproject.toml` has empty `dependencies`, so a bare `uv sync` installs nothing -- this is intentional to avoid pulling heavy dependencies (torch, warp-lang) by default.

Each workspace member is exposed as a named optional dependency extra, enabling composable installs from the repo root:

```bash
# Recommended for local development (compiles protos, installs all core + transfuser_driver plugin)
source setup_local_env.sh

# Or install selectively:
uv sync --extra wizard                           # wizard + transitive deps only
uv sync --extra all                              # all core packages
uv sync --extra all --extra transfuser_driver    # core + transfuser_driver plugin

# Single-package install from a subdirectory also works:
cd src/wizard && uv sync
```

Use `uv run` to execute commands in the workspace environment:

```bash
uv run pytest                                # run tests
uv run alpasim_wizard deploy=local topology=1gpu driver=vavam ...  # run the wizard
uv run --project src/runtime python -c "..." # run in a sub-project context
```

All members share one dependency resolution; there is no per-member version isolation. See [Plugin System](PLUGIN_SYSTEM.md) for how plugins integrate with the workspace.

## Key Dependencies

### Trajdata (Unified Data Loading)

AlpaSim uses [trajdata](https://github.com/NVlabs/trajdata) for unified scene data loading across
different autonomous driving datasets (USDZ, NuPlan, Waymo, etc.). The dependency is automatically
installed via `uv` from the `alpasim` branch:

```toml
# src/runtime/pyproject.toml
[tool.uv.sources.trajdata-alpasim]
git = "https://github.com/NVlabs/trajdata"
branch = "alpasim"  # Custom branch with AlpaSim-specific extensions
```

**You don't need to manually install trajdata** - it's handled automatically when you run
`setup_local_env.sh` or `uv sync --extra runtime`.

The `alpasim` branch includes:
- USDZ data format support
- Custom trajectory smoothing
- Performance optimizations for simulation workloads

## Troubleshooting

**`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` during NRE warmup**
Your NVIDIA driver is too old for the CUDA version in the NRE container. Upgrade to driver >= 570.x
(CUDA 12.8 support). Check your current version with `nvidia-smi`.

**`GatedRepoError` when downloading scenes**
You need to request access to the gated HuggingFace dataset. Visit
[nvidia/PhysicalAI-Autonomous-Vehicles-NuRec](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec)
and request access.

**`HF_TOKEN` not set / authentication failures**
Set your HuggingFace token: `export HF_TOKEN=<your-token>`. You can create a token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**gRPC connectivity errors between services**
Ensure Docker networking is working and all containers are on the same network. Check
`docker compose logs` for port binding conflicts. If using `debug_flags.use_localhost: true`,
ensure no other process is using the assigned ports.

**`setup_local_env.sh` fails silently in non-interactive environments**
The script only installs Rust when a TTY is detected. To install Rust manually,
use: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`

## Next steps

Once you have access to the above, please follow instructions in the [tutorial](TUTORIAL.md)
to get started running Alpasim.
