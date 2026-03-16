# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Transfuser Hydra configs.

When this plugin is installed, its configs are automatically added to
Hydra's search path via the ``alpasim.configs`` entry point (registered in
this package's ``pyproject.toml``).  The wizard ships a Hydra
``SearchPathPlugin`` (in ``hydra_plugins/alpasim_config_discovery/``) that
discovers all ``alpasim.configs`` entry points at startup and adds
``pkg://<value>`` to the search path for each one.

No manual ``hydra.searchpath=[pkg://alpasim_transfuser.configs]`` override is
needed — just install the plugin and the configs are available.
"""
