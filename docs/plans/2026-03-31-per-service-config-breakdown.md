# Per-Service Config Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the `stable_manifest` config group entirely. Replace it with `${defines.base_image}` (derived from `pyproject.toml`) for alpasim-base services, and per-service config groups for external services. Each service's configuration becomes orthogonal and independently selectable.

**Architecture:** The `alpasim-base` image tag derives from `pyproject.toml` version + a registry prefix (auto-set by the internal plugin). Image defaults and the OSS sensorsim pin move into `base_config.yaml`. External services (controller, trafficsim, sensorsim) each get their own optional Hydra config group for internal overrides.

**Tech Stack:** Hydra/OmegaConf config composition, YAML, existing SearchPathPlugin

---

## Context

### Problem

`stable_manifest/internal.yaml` conflates three concerns:

1. Image version pins (sensorsim, physics, runtime)
2. Service launch configuration (controller command/args, trafficsim volumes/workdir/command/gpus)
3. Behavioral config override (`override /trafficsim: defaults`)

Meanwhile `oss_gitlab.yaml` duplicates the trafficsim block and is the only manifest CI actually bumps. `internal.yaml` is stale at `alpasim-base:0.41.0` vs `0.50.0`.

The `controller_ndas/` directory name doesn't match its Hydra group references (`/controller:` in `sim/force_gt.yaml`, `model/alpamayo-cat-k.yaml`, etc.), and its configs state coupling with `stable_manifest/internal.yaml` in comments.

### Decisions

| Question | Decision |
|----------|----------|
| Controller | Optional. OSS base_config provides the default. Internal users add `controller=ndas`. |
| Sensorsim | Config group (`sensorsim=internal_nre`). |
| `oss_gitlab.yaml` | Remove. HuggingFace cache handled via `defines.hf_cache`. Missing plugins mount fixed in base_config. |
| `oss.yaml` | Remove. Merge into base_config. All service images set directly (sensorsim pin + `${defines.base_image}`). |
| Scenes | Use the OSS scene everywhere. Default scene_ids in base_config. |
| MR 544 dependency | None. Replicate the `${defines.base_image}` approach directly. |

### Target state

| Service      | Where config lives | Selection | Notes |
|-------------|-------------------|-----------|-------|
| Controller  | `controller/` config group | `controller=ndas`, `controller=ndas_noisefree`, etc. | Optional; renamed from `controller_ndas/` |
| Trafficsim  | `trafficsim/` config group | `trafficsim=internal` | Optional; extends existing group |
| Sensorsim   | `sensorsim/` config group | `sensorsim=internal_nre` | Optional; base_config pins OSS NRE |
| Physics     | `base_config.yaml` | `${defines.base_image}` | No per-environment override needed |
| Runtime     | `base_config.yaml` | `${defines.base_image}` | No per-environment override needed |
| Driver      | `driver/` config group | `driver=dino`, `driver=alpamayo1`, etc. | Unchanged |

No `stable_manifest` config group at all.

### Invocation after (internal)

```
uv run alpasim_wizard \
  deploy=ord \
  topology=8gpu_12rollouts \
  driver=alpamayo1 \
  controller=ndas \
  trafficsim=internal \
  wizard.log_dir=...
```

### Invocation after (OSS) — unchanged

```
uv run alpasim_wizard \
  deploy=local \
  topology=single_gpu \
  driver=vavam \
  wizard.log_dir=...
```

---

## Tasks

### Task 1: Introduce `${defines.base_image}`, merge `oss.yaml` into base_config, fix driver volumes

Derive the `alpasim-base` image tag from `pyproject.toml` version + registry prefix. Move all image definitions and the default scene_id from `oss.yaml` into `base_config.yaml`. Fix the missing `plugins` mount in the driver service and extract the HuggingFace cache path into `defines.hf_cache`.

**Files:**
- Modify: `src/wizard/alpasim_wizard/setup_omegaconf.py` — add `repo-version` resolver
- Modify: `src/wizard/configs/base_config.yaml` — add defines, set service images, add default scene, fix driver volumes, remove `stable_manifest` from defaults
- Delete: `src/wizard/configs/stable_manifest/oss.yaml`
- Create: `plugins/internal/configs/image_defaults/internal.yaml` — set `defines.image_registry`
- Modify: `plugins/internal/configs/deploy/ord.yaml` — add `defines.hf_cache` override
- Modify: `plugins/internal/configs/deploy/iad.yaml` — add `defines.hf_cache` override
- Modify: `src/wizard/configs/deploy/docker_build_only.yaml` — remove `/stable_manifest/oss` default

**Step 1:** Add `_read_repo_version()` to `setup_omegaconf.py` and register the `repo-version` resolver:

```python
def _read_repo_version() -> str:
    """Read the project version from the root pyproject.toml."""
    import tomllib
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]

OmegaConf.register_new_resolver("repo-version", _read_repo_version)
```

**Step 2:** Update `base_config.yaml` `defines` section:

```yaml
defines:
  filesystem: ???
  driver_model: ???
  mpc_implementation: linear
  image_registry: ""
  base_image: "${defines.image_registry}alpasim-base:${repo-version:}"
  hf_cache: "${oc.env:HF_HOME,${oc.env:HOME}/.cache/huggingface}"

  # defaults will work out of the box
  drivers: "${defines.filesystem}/drivers"
  sensordata: "${defines.filesystem}/nre-artifacts"
  alpackages: "${defines.filesystem}/alpackages"
  trafficsim_map_cache: "${defines.filesystem}/trafficsim/unified_data_cache"
  # ... rest unchanged
```

**Step 3:** Set service images directly in `base_config.yaml` (replace `???` with actual values):

```yaml
services:
  sensorsim:
    image: nvcr.io/nvidia/nre/nre-ga:26.02
    external_image: true
    # ... rest unchanged

  driver:
    image: ${defines.base_image}
    volumes:
      - "${defines.drivers}:/mnt/drivers"
      - "${wizard.log_dir}:/mnt/output"
      - "${repo-relative:'src'}:/repo/src"
      - "${repo-relative:'plugins'}:/repo/plugins"        # was missing
      - "${defines.hf_cache}:/root/.cache/huggingface"     # was inline oc.env
    # ... rest unchanged

  physics:
    image: ${defines.base_image}
    # ... rest unchanged

  trafficsim:
    image: ${defines.base_image}
    # ... rest unchanged

  controller:
    image: ${defines.base_image}
    # ... rest unchanged

  runtime:
    image: ${defines.base_image}
    # ... rest unchanged
```

**Step 4:** Add default scene_ids to `base_config.yaml`:

```yaml
scenes:
  scene_ids:
    - clipgt-01d503d4-449b-46fc-8d78-9085e70d3554
```

(Was in `oss.yaml`, now moved here. The `optional scenes_catalog: internal` still extends CSV paths when the internal plugin is installed.)

**Step 5:** Remove `stable_manifest: oss` from `base_config.yaml` defaults list. Add `optional image_defaults: internal`:

```yaml
defaults:
  - config_schema
  - _self_
  - deploy: null
  - driver: null
  - topology: null
  - trafficsim: null
  - optional scenes_catalog: internal
  - optional image_defaults: internal
```

**Step 6:** Create `plugins/internal/configs/image_defaults/internal.yaml`:

```yaml
# @package _global_
# Auto-discovered when internal plugin is installed.
# Configures the Docker registry prefix for pre-built images.
defines:
  image_registry: "nvcr.io/nvidian/alpamayo/"
```

**Step 7:** Delete `src/wizard/configs/stable_manifest/oss.yaml`.

**Step 8:** Remove `- /stable_manifest/oss` from `deploy/docker_build_only.yaml` defaults (no longer needed — images are in base_config).

**Step 9:** Add `defines.hf_cache` override to `deploy/ord.yaml` and `deploy/iad.yaml`:

```yaml
defines:
  filesystem: "/lustre/..."   # existing
  hf_cache: "${defines.filesystem}/huggingface"
```

**Step 10:** Verify: `uv run pytest src/wizard/tests -v`

**Step 11:** Commit: `refactor: merge oss.yaml into base_config and derive image versions from pyproject.toml`

---

### Task 2: Rename `controller_ndas/` to `controller/` and add base `ndas.yaml`

**Files:**
- Rename: `plugins/internal/configs/controller_ndas/` -> `plugins/internal/configs/controller/`
- Create: `plugins/internal/configs/controller/ndas.yaml`
- Modify: all existing files in the renamed directory (add `ndas` to defaults, remove stale comments)
- Modify: `src/wizard/configs/base_config.yaml` — add `controller: null` to defaults

**Step 1:** `git mv plugins/internal/configs/controller_ndas plugins/internal/configs/controller`

This fixes existing broken `/controller:` references in `sim/force_gt.yaml`, `model/alpamayo-cat-k.yaml`, `model/forcegt.yaml`, and `exp/catk_no_recovery.yaml`.

**Step 2:** Create `plugins/internal/configs/controller/ndas.yaml` (extracted from `internal.yaml`):

```yaml
# @package _global_
# NDAS controller service configuration.
# Usage: controller=ndas

services:
  controller:
    image: nvcr.io/nvidian/alpamayo/ndas-controller:0.1.6
    command:
      - "tools/pacsim/alpasim_vdc/py_alpasim_vdc"
      - "--port={port}"
      - "--log_dir=/mnt/output"
      - "--enable_noise"
    external_image: true
```

**Step 3:** Update each tuning variant to include `ndas` as a default. Example for `ndas_noisefree.yaml`:

```yaml
# @package _global_
# NDAS controller with noise disabled.
# Usage: controller=ndas_noisefree

defaults:
  - ndas
  - _self_

services:
  controller:
    command:
      - "tools/pacsim/alpasim_vdc/py_alpasim_vdc"
      - "--port={port}"
      - "--log_dir=/mnt/output"
```

Apply the same pattern to `ndas_noisefree_no_latency.yaml`, `ndas_no_latency.yaml`, `ndas_tuned_response.yaml`, `perfect_control.yaml`. Remove the "only works with stable_manifest/internal.yaml" comments from all.

`use_oss_controller.yaml` has been removed — its only meaningful addition (`--mpc-implementation`) is now in `base_config.yaml`'s default controller command.

**Step 4:** Add `controller: null` to `base_config.yaml` defaults list (after `trafficsim`). This makes it optional — when unset, the base_config controller definition applies.

**Step 5:** Verify: `uv run pytest src/wizard/tests plugins/internal/tests -v`

**Step 6:** Commit: `refactor: rename controller_ndas to controller and extract ndas base config`

---

### Task 3: Create `trafficsim/internal.yaml`

**Files:**
- Create: `plugins/internal/configs/trafficsim/internal.yaml`

**Step 1:** Create `plugins/internal/configs/trafficsim/internal.yaml`:

```yaml
# @package _global_
# Internal trafficsim service and behavioral configuration.
# Usage: trafficsim=internal
#
# Loads behavioral params from trafficsim/defaults.yaml and configures
# the trafficsim Docker container.

defaults:
  - defaults    # behavioral params (goes into trafficsim.* via default package)
  - _self_

services:
  trafficsim:
    image: nvcr.io/nvidian/alpamayo/alpasim-trafficsim:25_03_10_v4
    external_image: true
    volumes:
      - "${defines.trafficsim_map_cache}:/mnt/map-data"
      - "${wizard.log_dir}:/mnt/log_dir"
      - "${scenes.scene_cache}:/mnt/nre-data"
    workdir: /workspace/trafficsim
    command:
      - "/opt/conda/bin/python -m trafficsim.api"
      - "--config-path=/mnt/log_dir"
      - "--config-name=trafficsim-config.yaml"
      - "server.host=0.0.0.0"
      - "server.port={port}"
      - "api_data.unified_data_cache_path=/mnt/map-data"
      - "api_data.usdz_glob=/mnt/nre-data/{sceneset}/**/*.usdz"
      - "api_data.data_dirs.mads_v2=/tmp/extracted_data/mads_v2"
    gpus: [1]
```

`defaults.yaml` has no `@package` directive, so Hydra places it under `trafficsim.*`. This file has `@package _global_`, so `services.trafficsim.*` writes to the root. Both compose correctly.

**Step 2:** Verify: `uv run pytest src/wizard/tests -v`

**Step 3:** Commit: `feat: add trafficsim/internal config for trafficsim service`

---

### Task 4: Create `sensorsim/internal_nre.yaml`

**Files:**
- Create: `plugins/internal/configs/sensorsim/internal_nre.yaml`
- Modify: `src/wizard/configs/base_config.yaml` — add `sensorsim: null` to defaults

**Step 1:** Create `plugins/internal/configs/sensorsim/internal_nre.yaml`:

```yaml
# @package _global_
# Internal NRE sensorsim image.
# Usage: sensorsim=internal_nre

services:
  sensorsim:
    image: nvcr.io/nvidian/alpamayo/nre_run:26.3.79-7869d378
```

**Step 2:** Add `sensorsim: null` to `base_config.yaml` defaults list (after `controller`).

**Step 3:** Verify: `uv run pytest src/wizard/tests -v`

**Step 4:** Commit: `feat: add sensorsim config group for NRE image override`

---

### Task 5: Update CICD Hydra configs

Replace `stable_manifest=internal` / `stable_manifest=oss_gitlab` with per-service composition.

**Files:**
- Modify: `plugins/internal/configs/cicd/slurm_internal.yaml`
- Modify: `plugins/internal/configs/cicd/docker_internal.yaml`
- Modify: `plugins/internal/configs/cicd/slurm_oss.yaml`
- Modify: `plugins/internal/configs/cicd/docker_oss.yaml`

**Step 1:** Update `cicd/slurm_internal.yaml`:

```yaml
defaults:
  - common
  - override /controller: ndas
  - override /trafficsim: internal
  - override /sensorsim: internal_nre
  - override /driver: alpamayo1
  - _self_
```

Remove `override /stable_manifest: internal`.

**Step 2:** Update `cicd/docker_internal.yaml` the same way.

**Step 3:** Update `cicd/slurm_oss.yaml` — remove `override /stable_manifest: oss_gitlab`. Add `override /trafficsim: internal` if the OSS CI test needs a real trafficsim.

**Step 4:** Update `cicd/docker_oss.yaml` similarly.

**Step 5:** Verify: `uv run pytest src/wizard/tests plugins/internal/tests -v`

**Step 6:** Commit: `refactor: update CICD configs to per-service composition`

---

### Task 6: Update CI scripts for `${defines.base_image}`

The CI scripts parse YAML manifests with `yq`. With `${defines.base_image}`, base-image services no longer have manifest entries. External service images move to per-service configs.

**Files:**
- Modify: `plugins/internal/cicd/gitlab/scripts/bump-versions.sh`
- Modify: `plugins/internal/cicd/gitlab/scripts/prepare-images.sh`
- Modify: `plugins/internal/cicd/gitlab/scripts/squash-external-service.sh`
- Modify: `plugins/internal/cicd/gitlab/scripts/common.sh`

**Step 1: `bump-versions.sh`** — Remove manifest update logic.

The script bumps `pyproject.toml` versions (keep this), then updates `oss_gitlab.yaml` via `update_manifest_version()` (remove this). With `${defines.base_image}`, the manifest is no longer the source of truth.

- Remove the `MANIFEST_PATH` variable (line 34)
- Remove the `update_manifest_version()` function (lines 47-55)
- Remove all calls to `update_manifest_version` (line 222)
- Remove `"$MANIFEST_PATH"` from `VERSION_FILES` (line 273)
- Remove the manifest entry from the commit message logic (lines 289-290)

The rest of the script (pyproject.toml bumping, version comparison, commit/push) stays unchanged.

**Step 2: `prepare-images.sh`** — Derive stable images from `pyproject.toml`.

For the "stable" branch (Priority 3, lines 131-154), replace the `oss_gitlab.yaml` read with version derivation:

```bash
# Priority 3: No changes, no override - derive stable image from pyproject version
log "INFO" "  -> No changes, using stable image derived from pyproject.toml"
echo "${SERVICE_UPPER}_SOURCE=stable" >> build.env.new

if service_uses_base_image "$service"; then
    local version=$(yq -oy '.project.version' "pyproject.toml")
    IMAGE_TAG="nvcr.io/nvidian/alpamayo/alpasim-base:${version}"
else
    local version=$(yq -oy '.project.version' "src/$service/pyproject.toml")
    IMAGE_TAG="nvcr.io/nvidian/alpamayo/alpasim-${service}:${version}"
fi

log "INFO" "  -> Stable image: $IMAGE_TAG"
echo "${SERVICE_UPPER}_IMAGE=$IMAGE_TAG" >> build.env.new
```

Remove the `oss_gitlab.yaml` file existence check and `yq` read.

**Step 3: `squash-external-service.sh`** — Read from per-service configs.

Replace the manifest reads (lines 56-81) with per-service config reads:

```bash
# Priority 2: Get image from per-service config
IMAGE=""
CONFIG_FILE=""
case "$SERVICE" in
    sensorsim)  CONFIG_FILE="$CI_PROJECT_DIR/plugins/internal/configs/sensorsim/internal_nre.yaml" ;;
    controller) CONFIG_FILE="$CI_PROJECT_DIR/plugins/internal/configs/controller/ndas.yaml" ;;
    trafficsim) CONFIG_FILE="$CI_PROJECT_DIR/plugins/internal/configs/trafficsim/internal.yaml" ;;
esac

if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    IMAGE=$(yq ".services.${SERVICE}.image // \"\"" "$CONFIG_FILE")
fi
```

Remove the `internal.yaml` and `oss_gitlab.yaml` reads and the version-mismatch check between them.

**Step 4: `common.sh`** — Remove `internal.yaml` parsing from `build_wizard_overrides()`.

Replace the `yq` parsing of `internal.yaml` for `external_image: true` services (lines 176-195) with an explicit list:

```bash
local external_services=("sensorsim" "controller" "trafficsim")
```

**Step 5:** Verify: manual review and/or CI dry run.

**Step 6:** Commit: `refactor: update CI scripts to derive images from pyproject.toml and per-service configs`

---

### Task 7: Remove `internal.yaml`, `oss_gitlab.yaml`, and `stable_manifest/` directory

**Files:**
- Delete: `plugins/internal/configs/stable_manifest/internal.yaml`
- Delete: `plugins/internal/configs/stable_manifest/oss_gitlab.yaml`
- Delete: `src/wizard/configs/stable_manifest/` (directory, now empty after Task 1 deleted `oss.yaml`)
- Modify: `plugins/internal/tests/test_internal_plugin.py` — update expected config list
- Update: deploy config comments (`iad.yaml`, `ord.yaml`), `submit.sh` comments

**Step 1:** Delete the manifests and the empty stable_manifest directories:

```
git rm plugins/internal/configs/stable_manifest/internal.yaml
git rm plugins/internal/configs/stable_manifest/oss_gitlab.yaml
```

**Step 2:** Update `test_internal_plugin.py` — remove `stable_manifest/internal.yaml` from expected configs. Add:
- `controller/ndas.yaml`
- `trafficsim/internal.yaml`
- `sensorsim/internal_nre.yaml`
- `image_defaults/internal.yaml`

**Step 3:** Update deploy config comments to remove `stable_manifest=internal`:

- `deploy/iad.yaml` line 3: change to `deploy=iad topology=8gpu_12rollouts driver=alpamayo1 controller=ndas trafficsim=internal`
- `deploy/ord.yaml` line 3: same
- `src/tools/run-on-slurm/submit.sh` line 13: same

**Step 4:** Verify: `uv run pytest src/wizard/tests plugins/internal/tests -v && pre-commit run --all-files`

**Step 5:** Commit: `refactor: remove stable_manifest config group`

---

## `base_config.yaml` defaults after all tasks

```yaml
defaults:
  - config_schema
  - _self_
  - deploy: null           # Required (e.g., deploy=local)
  - driver: null           # Required (e.g., driver=vavam)
  - topology: null         # Required (e.g., topology=single_gpu)
  - trafficsim: null       # Optional (e.g., trafficsim=internal)
  - controller: null       # Optional (e.g., controller=ndas)
  - sensorsim: null        # Optional (e.g., sensorsim=internal_nre)
  - optional scenes_catalog: internal
  - optional image_defaults: internal
```

No `stable_manifest` in the chain. Service images come from base_config directly, overridden by per-service config groups or `image_defaults`.

## CI pipeline changes summary

| Script | Change | Risk |
|--------|--------|------|
| `bump-versions.sh` | Remove `update_manifest_version()` and `MANIFEST_PATH`. Pyproject.toml bumping unchanged. | Low — removing dead code path |
| `prepare-images.sh` | Derive stable image tags from `pyproject.toml` version instead of reading `oss_gitlab.yaml`. | Medium — verify tag format matches existing images |
| `squash-external-service.sh` | Read external service images from per-service configs instead of manifests. | Medium — verify yq paths work with new file structure |
| `common.sh` | Remove `internal.yaml` parsing from `build_wizard_overrides()`. Use explicit external service list. | Low — simplification |

## Rollback plan

Tasks 1-4 are additive (new files/groups) and independently committable. They can be merged even if Tasks 5-7 are deferred. Tasks 5-7 (CICD updates and manifest removal) should land together.
