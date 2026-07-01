# Repository Guidelines

## Overview

Alpasim is a lightweight, data-driven research simulator for autonomous vehicle testing using a microservice architecture with gRPC communication. The runtime orchestrates physics simulation, traffic, neural rendering, and ego vehicle policy evaluation.

## Documentation

- **User docs** → [docs/ONBOARDING.md](docs/ONBOARDING.md), [docs/TUTORIAL.md](docs/TUTORIAL.md), [docs/OPERATIONS.md](docs/OPERATIONS.md)
- **Design and data** → [docs/DESIGN.md](docs/DESIGN.md), [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)
- **Coordinate frames, coding style, contributing** → [CONTRIBUTING.md](CONTRIBUTING.md)

## Build and run (quick reference)

- **Environment**: `source setup_local_env.sh` (or `./setup_local_env.sh`). The project uses **uv** for dependencies and scripts; use `uv run` for commands (e.g. `uv run pytest`, `uv run alpasim_wizard ...`).
- **Internal configs**: before using `plugins/internal` configs such as `deploy=iad`, run `uv sync --all-packages --extra internal` from the repo root. Prefer repo-root commands like `uv run alpasim_wizard ...`; if using `uv run --project src/wizard`, make sure that project env can see the `alpasim-internal` entry point, otherwise Hydra will not find internal config groups.
- **After changing `.proto` files**: `cd src/grpc && uv run compile-protos`
- **Run simulation locally** (from repo root or `src/wizard`): `uv run alpasim_wizard deploy=local topology=1gpu driver=vavam wizard.log_dir=./my_run` (deploy configs live in `src/wizard/configs/deploy/`.)
- **Tests**: `uv run pytest` (e.g. `uv run pytest src/runtime/tests`)
- **Static checks**: `pre-commit run --all-files`

## Coding principles

- Prefer readability over flexibility. Add an abstraction only when it reduces
  net complexity or removes meaningful duplication.
- Keep one canonical path for each behavior. Avoid compatibility shims, silent
  fallbacks, aliases, and parallel old/new code unless there is a current
  requirement.
- Keep behavior local. Avoid one-line wrappers, gratuitous delegation, and class
  hierarchies unless the extra hop makes the code easier to understand or
  enables real reuse.
- Fail fast on unexpected input. Prefer direct required access over defensive
  `getattr` / `.get(..., default)` patterns that hide broken invariants.
- Reuse existing project helpers and upstream libraries before adding local
  equivalents.
- Keep dependency direction clear: generic/shared modules must not import from
  specific deployment, policy, or test modules.
- Comments and docstrings must describe the current code. Do not add
  change-history comments, restatements of obvious code, or speculative notes
  about future support.
- Tests should prove behavior and regressions, not mirror implementation details
  or lock down defaults already obvious at the call site.

## Environment and external repos

Environment variables and external repository URLs are project- or environment-specific; see CONTRIBUTING and your setup for local development.

## Commit and pull request guidelines

- Keep commits focused and imperative. Rebase onto `main` before submitting; force-pushes are expected after rebases.
- Changelogs are append-only: add new entries, but never change previous entries.
- Pipelines auto-bump versions for touched packages; allow the bot-generated commit to land and re-trigger CI if needed.
- PRs should explain scenario impact, reference issue IDs, and attach logs/screens for wizard/runtime regressions. Confirm tests and `pre-commit` pass before requesting review.
- When pushing to a branch that has the auto-bump commit "Alpasim automatic version bump", force push over it if that's the only commit you'd overwrite. Do not manually update docker container versions; that is done by the CI pipeline.

## Other conventions

- Coordinate frame conventions: [CONTRIBUTING.md](CONTRIBUTING.md)

## MCP Servers

When asked to access any of the following services, check if you have access to the corresponding MCP server:

- Linar
- Gitlab (especially relevant for MRs)
