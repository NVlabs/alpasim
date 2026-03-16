# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import pytest
from alpasim_utils.artifact import Artifact


@pytest.fixture(scope="session")
def sample_artifact():
    usdz_file = "tests/data/mock/dataset-clipgt-1ea7dc88-88ed-4c91-81fe-b6eb489cfa71_runid-2lwtrh0z/last.usdz"
    artifact = Artifact(source=usdz_file)
    return artifact
