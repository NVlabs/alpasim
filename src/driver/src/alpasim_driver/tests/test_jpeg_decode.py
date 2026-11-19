# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""The GPU has to decode JPEGs exactly as libjpeg does."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch
from PIL import Image
from torchvision.io import decode_jpeg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_gpu_decodes_jpegs_exactly_as_libjpeg_does() -> None:
    """Without the video driver capability the GPU's JPEG engine is hidden,
    and nvJPEG silently truncates where libjpeg rounds, half a level dark.
    The probe is 4:4:4 like the renderer's frames, because subsampled chroma
    is upsampled differently by each decoder even on a healthy GPU.
    """
    generator = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8, generator=generator)
    buffer = BytesIO()
    Image.fromarray(pixels.numpy()).save(buffer, format="JPEG", subsampling=0)
    payload = torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)

    assert torch.equal(
        decode_jpeg(payload, device="cuda").cpu(), decode_jpeg(payload, device="cpu")
    )
