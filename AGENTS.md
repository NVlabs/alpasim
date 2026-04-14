# Repository Guidelines

## Overview

Alpasim is a lightweight, data-driven research simulator for autonomous vehicle testing using a microservice architecture with gRPC communication. The runtime orchestrates physics simulation, traffic, neural rendering, and ego vehicle policy evaluation.

## Documentation

- **User docs** → [docs/ONBOARDING.md](docs/ONBOARDING.md), [docs/TUTORIAL.md](docs/TUTORIAL.md), [docs/OPERATIONS.md](docs/OPERATIONS.md)
- **Design and data** → [docs/DESIGN.md](docs/DESIGN.md), [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)
- **Coordinate frames, coding style, contributing** → [CONTRIBUTING.md](CONTRIBUTING.md)

## Build and run (quick reference)

- **Environment**: `source setup_local_env.sh` (or `./setup_local_env.sh`). The project uses **uv** for dependencies and scripts; use `uv run` for commands (e.g. `uv run pytest`, `uv run alpasim_wizard ...`).
- **After changing `.proto` files**: `cd src/grpc && uv run compile-protos`
- **Run simulation locally** (from repo root or `src/wizard`): `uv run alpasim_wizard +deploy=local wizard.log_dir=./my_run` (deploy configs live in `src/wizard/configs/deploy/`.)
- **Tests**: `uv run pytest` (e.g. `uv run pytest src/runtime/tests`)
- **Static checks**: `pre-commit run --all-files`

## Environment and external repos

Environment variables and external repository URLs are project- or environment-specific; see CONTRIBUTING and your setup for local development.

## Commit and pull request guidelines

- Keep commits focused and imperative. Rebase onto `main` before submitting; force-pushes are expected after rebases.
- Pipelines auto-bump versions for touched packages; allow the bot-generated commit to land and re-trigger CI if needed.
- PRs should explain scenario impact, reference issue IDs, and attach logs/screens for wizard/runtime regressions. Confirm tests and `pre-commit` pass before requesting review.
- When pushing to a branch that has the auto-bump commit "Alpasim automatic version bump", force push over it if that's the only commit you'd overwrite. Do not manually update docker container versions; that is done by the CI pipeline.

## Other conventions

- Coordinate frame conventions: [CONTRIBUTING.md](CONTRIBUTING.md)

## MCP Servers

When asked to access any of the following services, check if you have access to the corresponding MCP server:

- Linar
- Gitlab (especially relevant for MRs)

## Skills

A skill is a set of local instructions stored in a `SKILL.md` file.

### Project skills

- alpasim-data-access: Comprehensive data access guide for AlpaSim scene/state/map access. Use when implementing or reviewing ego state queries, traffic object access, traffic rule extraction, or map/lane lookups. (file: `.codeartsdoer/skills/alpasim-data-access/SKILL.md`)
