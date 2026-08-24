# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE_ROOT))
