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

models.get_names()                  # → ['alpamayo1_5_recipes_sft', 'transfuser', 'vam', ...]
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
| `alpasim.route_generators` | Runtime route generators |

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

### Adding route generators

A route plugin provides a `RouteGenerator` instance via a
`from_context(recorded_waypoints_in_local, vector_map, *, route_start_offset_m=0.0)`
classmethod. The recorded waypoints are an `(N, 3)` array in the local frame.
`vector_map` is a `VectorMap` or `None` when the scene has no map; plugins that
require map data must handle that case. The returned generator implements the
runtime's `generate_route(timestamp_us, pose_local_to_rig)` interface.

For example, this minimal plugin reuses recorded routing:

```python
# my_routes.py
from alpasim_runtime.route_generator import RouteGeneratorRecorded


class CustomRoutes(RouteGeneratorRecorded):
    @classmethod
    def from_context(
        cls, recorded_waypoints_in_local, vector_map, *, route_start_offset_m=0.0
    ):
        return cls(recorded_waypoints_in_local, route_start_offset_m)
```

Declare dependencies on `alpasim_plugins` and `alpasim-runtime` in your package,
and register the class:

```toml
[project.entry-points."alpasim.route_generators"]
custom = "my_routes:CustomRoutes"
```

Install the package in the **runtime service's environment**, then select it in
the runtime user configuration:

```yaml
simulation_config:
  route_generator_plugin: custom
  route_start_offset_m: 0.0
```

When set, `route_generator_plugin` overrides `route_generator_type`, including
`NONE`. An unknown name raises `PluginNotFoundError`; it does not fall back to a
built-in generator. When unset or `null`, the existing `MAP`/`RECORDED`/`NONE`
selection is used and the runtime does not need `alpasim_plugins` for routing.

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
