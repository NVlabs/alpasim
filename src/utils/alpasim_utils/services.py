# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Service-naming + renderer-plugin helpers shared by wizard and runtime.

Lives in ``alpasim_utils`` because both the wizard (config generation) and the
runtime (worker dispatch) need these primitives. ``alpasim_utils`` is the
lowest-level workspace package both depend on, so placing them here avoids a
wizard -> runtime dependency edge.
"""

from typing import cast

from omegaconf import OmegaConf

CORE_SERVICE_NAMES = (
    "driver",
    "sensorsim",
    "physics",
    "trafficsim",
    "controller",
)

DEFAULT_RENDERER_TYPE = "sensorsim"


def normalize_renderer_type(renderer_type: str | None) -> str:
    """Normalize renderer identifiers to the lowercase plugin registry form."""
    if renderer_type is None:
        return DEFAULT_RENDERER_TYPE
    return str(renderer_type).lower()


def validate_renderer_config(
    renderer_type: str | None,
    renderer_config: dict | None,
) -> dict | None:
    """Validate ``renderer_config`` against the active renderer plugin's schema.

    The plugin's service class (registered under ``alpasim.services``) exposes
    its typed config dataclass via ``get_config_schema()``.  Validation
    catches two classes of errors at submit time (rather than minutes later
    at worker startup):

    - unknown top-level keys (typos like ``chunk_frams``), via an explicit
      key-set check against the schema's declared fields,
    - type mismatches on known fields (e.g. ``fps: "not-an-int"``), via
      OmegaConf's structured-config merge.

    The unknown-key check is done manually rather than with
    ``OmegaConf.set_struct(schema, True)`` because struct mode propagates
    into nested ``Dict[str, X]`` fields, which legitimately accept arbitrary
    user-provided keys.

    Returns a dict that conforms to the schema.  Sensorsim and unknown /
    not-locally-installed renderers pass through unvalidated -- the runtime
    container where the plugin lives will still catch errors at worker
    startup.
    """
    renderer_type = normalize_renderer_type(renderer_type)
    if renderer_type == DEFAULT_RENDERER_TYPE:
        return renderer_config

    if renderer_config is None:
        # No user-provided config; let the plugin fall back to its own defaults
        # at worker-start time rather than synthesizing a defaults dict here.
        return None

    try:
        from alpasim_plugins import PluginRegistry  # type: ignore[import-not-found]
    except ImportError:
        return renderer_config

    try:
        service_cls = PluginRegistry("alpasim.services").get(renderer_type)
    except Exception:  # noqa: BLE001 -- registry may raise various types
        # Plugin not installed locally; defer validation to the runtime
        # container that actually has the plugin.
        return renderer_config

    schema_fn = getattr(service_cls, "get_config_schema", None)
    if schema_fn is None:
        return renderer_config
    schema_cls = schema_fn()
    if schema_cls is None:
        return renderer_config

    schema = OmegaConf.structured(schema_cls)
    user_cfg = OmegaConf.create(renderer_config or {})

    # OmegaConf.keys() returns a wide union type; renderer_config is always
    # string-keyed so coerce to str to keep mypy happy on `sorted()` below.
    known_fields = {str(k) for k in schema.keys()}
    provided_fields = {str(k) for k in user_cfg.keys()}
    unknown = provided_fields - known_fields
    if unknown:
        raise ValueError(
            f"Unknown field(s) in renderer_config for {renderer_type!r}: "
            f"{sorted(unknown)}. Valid fields: {sorted(known_fields)}"
        )

    merged = OmegaConf.merge(schema, user_cfg)
    return cast("dict | None", OmegaConf.to_container(merged, resolve=True))
