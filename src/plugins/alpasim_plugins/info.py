# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""CLI to print available Alpasim plugins (models, MPCs, scorers, tools, configs)."""

from __future__ import annotations

from alpasim_plugins.plugins import get_plugin_info


def main() -> None:
    """Print a summary of all installed plugins to stdout."""
    info = get_plugin_info()
    labels = {
        "alpasim.models": "Models",
        "alpasim.mpc": "MPC",
        "alpasim.scorers": "Scorers",
        "alpasim.tools": "Tools",
        "alpasim.configs": "Configs",
    }
    for group, names in info.items():
        label = labels.get(group, group)
        line = f"{label}: {', '.join(names)}" if names else f"{label}: (none)"
        print(line)


if __name__ == "__main__":
    main()
