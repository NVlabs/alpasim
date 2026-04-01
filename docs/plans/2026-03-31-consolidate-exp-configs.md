# Consolidate exp/experiment/model/sim Config Groups

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge the `exp/`, `experiment/`, `model/`, and `sim/` config groups into a single `exp/` group with subfolders, so all append-only override bundles live under one namespace.

**Architecture:** Move files into `exp/{presets,model,sim,nre,scenes}/` subfolders. OSS configs stay in `src/wizard/configs/exp/`, internal configs in `plugins/internal/configs/exp/`. Hydra merges both search paths into one `exp` group. Update all cross-group references.

**Tech Stack:** Hydra/OmegaConf, YAML, git mv

---

## Target structure

**OSS (`src/wizard/configs/exp/`):**
```
exp/
  presets/
    vavam_4hz.yaml
    vavam_4hz_eco.yaml
  sim/
    force_gt.yaml              # moved from src/wizard/configs/sim/
```

**Internal (`plugins/internal/configs/exp/`):**
```
exp/
  presets/
    catk_no_recovery.yaml      # moved from exp/
  model/
    alpamayo-cat-k.yaml        # moved from model/
    alpamayo-cat-k-eval.yaml
    forcegt.yaml
    hydra-mdp.yaml
    recovery_heuristic.yaml
  sim/
    20s_at_30Hz.yaml           # moved from sim/
  nre/
    fixer.yaml                 # moved from experiment/nre/
  scenes/
    interactive_fail_8.yaml    # moved from experiment/scenes/
```

**CLI usage changes:**
- `+model=alpamayo-cat-k` -> `+exp/model=alpamayo-cat-k`
- `+model=forcegt` -> `+exp/model=forcegt`
- `+sim=force_gt` -> `+exp/sim=force_gt`
- `+exp=catk_no_recovery` -> `+exp/presets=catk_no_recovery`
- `+exp=vavam_4hz` -> `+exp/presets=vavam_4hz`
- `+experiment/nre=fixer` -> `+exp/nre=fixer`
- `+experiment/scenes=interactive_fail_8` -> `+exp/scenes=interactive_fail_8`

---

## References to update

### Hydra defaults lists (in the moved configs themselves)

| File (after move) | Current reference | New reference | Why |
|---|---|---|---|
| `exp/presets/catk_no_recovery.yaml` | `- /model: alpamayo-cat-k` | `- /exp/model: alpamayo-cat-k` | Cross-group ref; `model` is now `exp/model` |
| `exp/model/alpamayo-cat-k.yaml` | `- /sim: 20s_at_30Hz` | `- /exp/sim: 20s_at_30Hz` | Cross-group ref; `sim` is now `exp/sim` |
| `exp/model/forcegt.yaml` | `- alpamayo-cat-k` | no change | Same-group relative ref (both in `exp/model/`) |
| `exp/model/alpamayo-cat-k-eval.yaml` | `- alpamayo-cat-k` | no change | Same-group relative ref |
| `exp/presets/vavam_4hz_eco.yaml` | `- vavam_4hz` | no change | Same-group relative ref (both in `exp/presets/`) |
| `exp/sim/force_gt.yaml` | `- /physics: disabled`, `- /controller: perfect_control` | no change | `physics/` and `controller/` are separate groups, not moving |
| `exp/model/forcegt.yaml` | `- /physics: disabled`, `- override /controller: perfect_control` | no change | Same reason |

### Tests

| File | Current | New |
|---|---|---|
| `test_per_service_config_regressions.py:51` | `"+model=alpamayo-cat-k"` | `"+exp/model=alpamayo-cat-k"` |
| `test_per_service_config_regressions.py:52` | `"+exp=catk_no_recovery"` | `"+exp/presets=catk_no_recovery"` |
| `test_per_service_config_regressions.py:53` | `"+sim=force_gt"` | `"+exp/sim=force_gt"` |
| `test_per_service_config_regressions.py:54` | `"+model=forcegt"` | `"+exp/model=forcegt"` |
| `test_internal_plugin.py` | `"exp/catk_no_recovery.yaml"`, `"experiment/nre/fixer.yaml"`, etc. | Updated to new paths |

### Other references

| File | Current | New |
|---|---|---|
| `catk_pipeline_config_example.yaml:39` | `+model=alpamayo-cat-k` | `+exp/model=alpamayo-cat-k` |
| `docs/OPERATIONS.md:202` | reference to `sim/20s_at_30Hz.yaml` | `exp/sim/20s_at_30Hz.yaml` |

---

## Tasks

### Task 1: Move files to new structure

**Step 1:** Move OSS files:
```bash
mkdir -p src/wizard/configs/exp/presets src/wizard/configs/exp/sim
git mv src/wizard/configs/exp/vavam_4hz.yaml src/wizard/configs/exp/presets/
git mv src/wizard/configs/exp/vavam_4hz_eco.yaml src/wizard/configs/exp/presets/
git mv src/wizard/configs/sim/force_gt.yaml src/wizard/configs/exp/sim/
```

**Step 2:** Move internal files:
```bash
mkdir -p plugins/internal/configs/exp/presets plugins/internal/configs/exp/model plugins/internal/configs/exp/sim plugins/internal/configs/exp/nre plugins/internal/configs/exp/scenes
git mv plugins/internal/configs/exp/catk_no_recovery.yaml plugins/internal/configs/exp/presets/
git mv plugins/internal/configs/model/*.yaml plugins/internal/configs/exp/model/
git mv plugins/internal/configs/sim/20s_at_30Hz.yaml plugins/internal/configs/exp/sim/
git mv plugins/internal/configs/experiment/nre/fixer.yaml plugins/internal/configs/exp/nre/
git mv plugins/internal/configs/experiment/scenes/interactive_fail_8.yaml plugins/internal/configs/exp/scenes/
```

**Step 3:** Clean up empty directories (git handles this automatically, but verify).

### Task 2: Update Hydra defaults references in moved configs

**Step 1:** In `plugins/internal/configs/exp/presets/catk_no_recovery.yaml`, change:
```
- /model: alpamayo-cat-k  ->  - /exp/model: alpamayo-cat-k
```

**Step 2:** In `plugins/internal/configs/exp/model/alpamayo-cat-k.yaml`, change:
```
- /sim: 20s_at_30Hz  ->  - /exp/sim: 20s_at_30Hz
```

### Task 3: Update test references

**Step 1:** Update `plugins/internal/tests/test_per_service_config_regressions.py`:
```
"+model=alpamayo-cat-k"  ->  "+exp/model=alpamayo-cat-k"
"+exp=catk_no_recovery"  ->  "+exp/presets=catk_no_recovery"
"+sim=force_gt"          ->  "+exp/sim=force_gt"
"+model=forcegt"         ->  "+exp/model=forcegt"
```

**Step 2:** Update `plugins/internal/tests/test_internal_plugin.py` expected paths:
```
"exp/catk_no_recovery.yaml"              ->  "exp/presets/catk_no_recovery.yaml"
"experiment/nre/fixer.yaml"              ->  "exp/nre/fixer.yaml"
"experiment/scenes/interactive_fail_8.yaml"  ->  "exp/scenes/interactive_fail_8.yaml"
"model/alpamayo-cat-k-eval.yaml"         ->  "exp/model/alpamayo-cat-k-eval.yaml"
"model/alpamayo-cat-k.yaml"              ->  "exp/model/alpamayo-cat-k.yaml"
"model/forcegt.yaml"                     ->  "exp/model/forcegt.yaml"
"model/hydra-mdp.yaml"                   ->  "exp/model/hydra-mdp.yaml"
"model/recovery_heuristic.yaml"          ->  "exp/model/recovery_heuristic.yaml"
"sim/20s_at_30Hz.yaml"                   ->  "exp/sim/20s_at_30Hz.yaml"
```

### Task 4: Update other references

**Step 1:** Update `catk_pipeline_config_example.yaml`:
```
+model=alpamayo-cat-k  ->  +exp/model=alpamayo-cat-k
```

**Step 2:** Update `docs/OPERATIONS.md` reference to `sim/20s_at_30Hz.yaml`.

### Task 5: Verify and commit

**Step 1:** Run: `uv run pytest src/wizard/tests plugins/internal/tests -v`

**Step 2:** Commit: `refactor: consolidate exp, experiment, model, and sim into unified exp/ group`
