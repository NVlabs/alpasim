# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sim_stats(
    stats: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    show: bool = True,
    close: bool = True,
):
    fig, axes = plt.subplots(nrows=len(stats), ncols=2, figsize=(4, 8))

    axes[0, 0].set_title("Ground Truth")
    axes[0, 1].set_title("Simulated")

    for row, scene_stats in enumerate(stats.values()):
        histogram, bins = scene_stats["gt"]
        axes[row, 0].hist(histogram, bins, linewidth=0.5, edgecolor="white")

        histogram, bins = scene_stats["sim"]
        axes[row, 1].hist(histogram, bins, linewidth=0.5, edgecolor="white")

    plt.tight_layout()

    if show:
        plt.show()

    if close:
        plt.close(fig)
