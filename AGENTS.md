# Repository Guidelines

## Overview
Alpasim is a lightweight, data-driven research simulator for autonomous vehicle testing using a microservice architecture with gRPC communication. The runtime orchestrates physics simulation, traffic, neural rendering, and ego vehicle policy evaluation.

## Codebase Overview

Alpasim is structured around a **Runtime** orchestrator with gRPC microservices for driver, controller, physics, traffic, and sensorsim. **Wizard** generates service configs and deployment artifacts (SLURM/Docker Compose). **Eval** computes metrics from ASL logs and runtime data.

**Key architectural patterns**:
- Two-phase rollout initialization: `UnboundRollout` (validation) → `BoundRollout` (execution)
- Worker pool parallelism: subprocess workers (W>1) or inline mode (W=1)
- Queue-based service pools with atomic acquisition
- Message broadcasting for logging and in-runtime evaluation
- Coordinate frame transforms: DS rig ↔ AABB center ↔ local/ENU

**Stack**: Python 3.11-3.12, gRPC/protobuf, OmegaConf/Hydra, uv
**Structure**: `src/<module>` packages with colocated tests; `docs/`, `tools/`, and `cicd/` support developer workflows

For detailed architecture, data flow diagrams, and navigation guides, see [docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md).

## Project Structure & Module Organization

- Core packages live in `src/<module>`; each module (e.g., `runtime`, `wizard`, `grpc`, `utils`,
  `ddb`, `physics`, `avmf`, `tools`, `controller`, `driver`, `eval`) bundles its code with a
  colocated `tests` folder.
- Shared docs in `docs/` (onboarding, design), assets & sample data in `data/`, CI wiring in
  `cicd/`.
- Proto files compile under `src/grpc`; generated stubs feed other packages.
- Scripts and helpers sit in `tools/` (includes `buildauth`, map utilities). Keep temp artifacts
  under `tmp/` or gitignored caches.

## Build, Test, and Development Commands

- Create/update a local env: `./setup_local_env.sh` (uses `uv` to create `.venv`, install editable
  packages, compile gRPC stubs, and register `pre-commit` hooks).
- Run commands via `uv run …` (no need to activate the venv).
- Run the fast test bundle: `uv run pytest` (respects default `-m 'not manual'` marker). Target a
  module with `uv run pytest src/runtime/tests`.
- Static checks: `pre-commit run --all-files` covers `black`, `ruff`, import sorting, and basic
  lint. Type-check runtime-heavy code via `uv run mypy src/runtime`.

## Coding Style & Naming Conventions

- Python 3.12+, 4-space indentation, limit files to UTF-8 ASCII unless data demands otherwise.
- Auto-format with `black`; keep imports sorted by the hooks. Use `ruff` to satisfy lint warnings
  before pushing.
- Follow PEP 8 naming plus domain hints: prefix vectors/poses with frames (`pose_local_to_rig`,
  `velocity_vehicle_in_local`) to avoid ambiguity in physics/AV math.
- Do not use TYPE_CHECKING to conditionally import packages for type checking.
- Do not use on-demand or runtime imports; always place all imports at the top of the file.
- Import Rust extension types (`Pose`, `Trajectory`, `Polyline`, etc.) via `alpasim_utils.geometry`, **not** directly from `utils_rs`. The `utils_rs` package is a private implementation detail.
- Document complex flows with concise comments; prefer dataclasses and type hints for public APIs.

## Testing Guidelines

- Place tests next to their module under `src/<module>/tests` and name files `test_*.py`.
- Default pytest config skips `@pytest.mark.manual` suites; mark long-running or cluster-dependent
  cases accordingly.
- Use fixtures over hard-coded paths; when acting on sample assets, reference `data/` or create
  temporary files.
- Extend async tests with `pytest-asyncio`; keep gRPC client stubs isolated to avoid network
  side-effects.

## Architecture & Data Flow
- Services: **Runtime**, **Physics**, **Controller**, **Eval**, **DDB**, **Wizard**, **GRPC**, **Utils** (see `src/<service>/`).
- All services communicate via gRPC messages defined in `src/grpc/alpasim_grpc/v0/`.
- Wizard generates configs, Runtime orchestrates loop, services exchange data, Runtime logs to ASL, Eval computes metrics, DDB indexes outputs.

## Environment Variables
- Required: `ALPAMAYO_S3_SECRET`.
- Optional: `SLURM_JOB_ID`, `UV_EXTRA_INDEX_URL`.

## Common Issues & Solutions
- Use `uv sync` instead of `pip install` for dependency resolution.
- After changing `.proto`: run `uv run compile-protos` in `src/grpc`.

## External Repositories
- Alpamayo policy: https://gitlab-master.nvidia.com/alpamayo/alpamayo
- NRE: https://gitlab-master.nvidia.com/Toronto_DL_Lab/nre
- Trafficsim: https://gitlab-master.nvidia.com/alpamayo/trafficsim

## Commit & Pull Request Guidelines
- Keep commits focused and imperative (`runtime: guard invalid rig transforms`). Avoid `wip` in final history.
- Rebase onto `main` before submitting; force-pushes are expected after rebases due to the linear history requirement.
- Pipelines auto-bump versions for touched packages; allow the bot-generated commit to land and re-trigger CI if needed.
- PRs should explain scenario impact, reference issue IDs, and attach logs/screens for wizard/runtime regressions. Confirm tests and `pre-commit` pass before requesting review.
- When pushing to a brach that has the auto-bump commit "Alpasim automatic
  version bump", force push over it (if that's the only commit you'd overwrite).
  Similarly, if you pulled that commit, delete it before pushing. This commit is
  automatically created by the CI pipeline after every push and should only
  exist once.
- Do not manually update the docker container versions - this is done by the
  "Alpasim automatic version bump" commit that is created by the CI pipeline.

- Keep commits focused and imperative (`runtime: guard invalid rig transforms`). Avoid `wip` in
  final history.
- Rebase onto `main` before submitting; force-pushes are expected after rebases due to the linear
  history requirement.
- Pipelines auto-bump versions for touched packages; allow the bot-generated commit to land and
  re-trigger CI if needed.
- PRs should explain scenario impact, reference issue IDs, and attach logs/screens for
  wizard/runtime regressions. Confirm tests and `pre-commit` pass before requesting review.

## Other conventions

- Conventions on the used coordinate frames can be found in `CONTRIBUTING.md`


## MCP Servers
When asked to access any of the following services, check if you have access to
the corresponding MCP server:
- Linar
- Gitlab. This is especially relevant for MRs.
