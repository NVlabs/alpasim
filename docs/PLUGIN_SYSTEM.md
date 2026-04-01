# Plugin System

Alpasim uses Python [entry points](https://packaging.python.org/en/latest/specifications/entry-points/) to discover components at runtime. Any installed package can register models, configs, or tools without modifying the core codebase.

## How it works

A package registers a component by declaring an entry point in its `pyproject.toml`. For example, the transfuser driver plugin (`plugins/transfuser_driver/`) registers a model:

```toml
[project.entry-points."alpasim.models"]
transfuser = "alpasim_transfuser.transfuser_model:TransfuserModel"
```

At runtime, the `PluginRegistry` (in `src/plugins/`) scans all installed packages for matching entry-point groups, loads them lazily on first access, and makes them available by name:

```python
from alpasim_plugins import models

models.get_names()                  # → ['alpamayo1', 'alpamayo1_5', 'manual', 'transfuser', 'vam']
models.get("transfuser")            # → <class TransfuserModel>
```

Hydra config directories work the same way. The transfuser plugin registers its config package:

```toml
[project.entry-points."alpasim.configs"]
transfuser = "alpasim_transfuser.configs"
```

At startup the wizard's `AlpasimConfigDiscoveryPlugin` discovers all `alpasim.configs` entry points and adds `pkg://alpasim_transfuser.configs` to Hydra's search path. YAML files inside that package are then available by config group — for example `driver=transfuser` resolves to `alpasim_transfuser/configs/driver/transfuser.yaml` without any manual search-path overrides.

Run `uv run alpasim-info` to see all currently registered plugins.

The available entry-point groups are:

| Group | Purpose |
|---|---|
| `alpasim.models` | Trajectory prediction models |
| `alpasim.mpc` | MPC controller implementations |
| `alpasim.configs` | Hydra config directories (auto-discovered by the wizard) |
| `alpasim.scorers` | Evaluation metric scorers |
| `alpasim.tools` | CLI tools |

---

## Workspace structure

See [Onboarding — Dependency management](ONBOARDING.md#dependency-management) for a full description of the uv workspace and install commands.

Plugins are discovered via the `plugins/*` glob in `[tool.uv.workspace] members`. Each plugin has a corresponding extra in the root `pyproject.toml` (e.g. `internal`, `transfuser`). To install core packages plus a plugin:

```bash
uv sync --extra all --extra internal      # core + internal plugin
uv sync --extra all --extra transfuser    # core + transfuser plugin
```

---

## Creating a new plugin

To add a plugin you need two things: a class that implements the right interface, and a `pyproject.toml` that registers it as an entry point. The transfuser driver plugin (`plugins/transfuser_driver/`) is a complete example — it registers both a model and Hydra configs:

```toml
[project]
name = "alpasim_transfuser"
dependencies = ["alpasim_plugins", "alpasim_driver", "torch", ...]

[project.entry-points."alpasim.models"]
transfuser = "alpasim_transfuser.transfuser_model:TransfuserModel"

[project.entry-points."alpasim.configs"]
transfuser = "alpasim_transfuser.configs"
```

`TransfuserModel` extends `BaseTrajectoryModel` (defined in `src/driver/src/alpasim_driver/models/base.py`). See that base class for the methods you need to implement (`from_config`, `predict`, `camera_ids`, etc.). The same pattern applies to other groups — implement the base class, register the entry point.

Once your package is ready, install and verify:

```bash
uv pip install -e path/to/my-plugin    # or add to plugins/ and uv sync --extra all
uv run alpasim-info                    # should list your new component
```

Model plugins can then be referenced by name in driver configs (e.g. `model_type: transfuser`).

### Adding Hydra configs

Plugins can ship Hydra config files alongside their code. Place YAML files under a Python package with an `__init__.py`, using subdirectories that match the Hydra config group (e.g. `driver/`). Register an `alpasim.configs` entry point pointing to the package:

```toml
[project.entry-points."alpasim.configs"]
transfuser = "alpasim_transfuser.configs"
```

The transfuser plugin's config layout:

```
plugins/transfuser_driver/
  alpasim_transfuser/
    configs/
      __init__.py
      driver/
        transfuser.yaml
        transfuser_configs.yaml
```

The wizard discovers all `alpasim.configs` entry points at startup and adds `pkg://<value>` to Hydra's search path automatically. When the transfuser plugin is installed, `driver=transfuser` resolves from the plugin's config directory without any manual search-path overrides.

---

## Guidelines

- Plugins extend available components; they do not replace or override built-in ones.
- Avoid registering names that conflict with names in the main packages.
- Plugin packages must depend on `alpasim_plugins` and on the package that defines their interface.
