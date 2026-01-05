# Repository Guidelines

## Project Structure & Module Organization
- Core packages live in `src/<module>`; each module (e.g., `runtime`, `wizard`, `grpc`, `utils`, `ddb`, `physics`, `avmf`, `tools`, `controller`, `driver`, `eval`) bundles its code with a colocated `tests` folder.
- Shared docs in `docs/` (onboarding, design), assets & sample data in `data/`, CI wiring in `cicd/`.
- Proto files compile under `src/grpc`; generated stubs feed other packages.
- Scripts and helpers sit in `tools/` (includes `buildauth`, map utilities). Keep temp artifacts under `tmp/` or gitignored caches.

## Build, Test, and Development Commands
- Create/update a local env: `./setup_local_env.sh` (uses `uv` to create `.venv`, install editable packages, compile gRPC stubs, and register `pre-commit` hooks).
- Activate tooling with `source .venv/bin/activate`, then run module tasks using `uv run …`.
- Run the fast test bundle: `uv run pytest` (respects default `-m 'not manual'` marker). Target a module with `uv run pytest src/runtime/tests`.
- Static checks: `pre-commit run --all-files` covers `black`, `ruff`, import sorting, and basic lint. Type-check runtime-heavy code via `uv run mypy src/runtime`.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indentation, limit files to UTF-8 ASCII unless data demands otherwise.
- Auto-format with `black`; keep imports sorted by the hooks. Use `ruff` to satisfy lint warnings before pushing.
- Follow PEP 8 naming plus domain hints: prefix vectors/poses with frames (`pose_local_to_rig`, `velocity_vehicle_in_local`) to avoid ambiguity in physics/AV math.
- Document complex flows with concise comments; prefer dataclasses and type hints for public APIs.

## Testing Guidelines
- Place tests next to their module under `src/<module>/tests` and name files `test_*.py`.
- Default pytest config skips `@pytest.mark.manual` suites; mark long-running or cluster-dependent cases accordingly.
- Use fixtures over hard-coded paths; when acting on sample assets, reference `data/` or create temporary files.
- Extend async tests with `pytest-asyncio`; keep gRPC client stubs isolated to avoid network side-effects.

## Commit & Pull Request Guidelines
- Keep commits focused and imperative (`runtime: guard invalid rig transforms`). Avoid `wip` in final history.
- Rebase onto `main` before submitting; force-pushes are expected after rebases due to the linear history requirement.
- Pipelines auto-bump versions for touched packages; allow the bot-generated commit to land and re-trigger CI if needed.
- PRs should explain scenario impact, reference issue IDs, and attach logs/screens for wizard/runtime regressions. Confirm tests and `pre-commit` pass before requesting review.


## Other conventions
- Conventions on the used coordinate frames can be found in `CONTRIBUTING.md`
