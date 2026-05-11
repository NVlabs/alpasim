"""Launch the MTGS-backed AlpaSim SensorsimService.

This module lives in the AlpaSim source tree so the wizard can deploy it as a
normal sensorsim service. It delegates the renderer implementation to the MTGS
Python package, which must still be present in the runtime image or mounted into
the container because the trained model, Nerfstudio config, and CUDA kernels are
MTGS-owned assets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_path(path: str | Path | None) -> None:
    if path is None:
        return
    path = str(path)
    if path and path not in sys.path:
        sys.path.insert(0, path)


def configure_import_paths() -> None:
    """Make AlpaSim grpc and MTGS imports available inside the sensorsim image."""
    repo_root = Path(__file__).resolve().parents[3]
    _prepend_path(repo_root / "src" / "grpc")

    # Container default used by local_mtgs_sensorsim.yaml.
    _prepend_path(os.environ.get("MTGS_REPO", "/mnt/mtgs"))


def main() -> None:
    configure_import_paths()
    from mtgs_sensorsim_server import main as mtgs_server_main

    mtgs_server_main()


if __name__ == "__main__":
    main()
