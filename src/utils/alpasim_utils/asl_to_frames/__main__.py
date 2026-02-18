# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
A CLI utility for extracting camera frames from alpasim logs (.asl files) and saving as video or
individual images. The results for each discovered file `path/to/<name>.asl` will be saved at
`path/to/<name>_asl_frames/<camera_id>/<mp4_or_jpegs_or_pngs>`.
The necessary dependencies for this script can be installed with the optional dependency: eg
`pip install alpasim_grpc_protobuf4[asl_to_frames]`.
"""

import argparse
import asyncio
import glob
import logging
from typing import Literal, Optional, TypeAlias

import aiofiles
import numpy as np
from aiofiles import os as aios
from alpasim_grpc.v0.egodriver_pb2 import DriveSessionRequest, RolloutCameraImage
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_utils.logs import async_read_pb_log

try:
    import imageio.v3 as iio
except ImportError:
    raise ImportError(
        "This script requires additionally installing imageio[ffmpeg] "
        + "or installing with the [asl_to_frames} optional dependency."
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

SaveFormat: TypeAlias = Literal["mp4", "frames"]


def pad_to_divisible_by_16(image: np.ndarray) -> np.ndarray:
    """
    Pads an image (h, w, 3) with zeros so that height and width are divisible by 16.

    Parameters:
    image (numpy.ndarray): Input image array of shape (h, w, 3).

    Returns:
    numpy.ndarray: Padded image with dimensions divisible by 16.
    """
    h, w, _ = image.shape

    # Calculate the padding required for height and width
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    # Calculate padding amounts
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the image
    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    return padded_image


async def convert_single_log(
    log_path: str,
    save_dir: str,
    format: SaveFormat,
) -> None:
    frames_by_camera: dict[str, list[RolloutCameraImage.CameraImage]] = {}

    rollout_metadata: Optional[RolloutMetadata] = None
    drive_session_request: Optional[DriveSessionRequest] = None

    async for message in async_read_pb_log(log_path):
        if message.WhichOneof("log_entry") == "driver_session_request":
            drive_session_request = message.driver_session_request
        elif message.WhichOneof("log_entry") == "rollout_metadata":
            rollout_metadata = message.rollout_metadata
        elif message.WhichOneof("log_entry") == "driver_camera_image":
            image: RolloutCameraImage.CameraImage = (
                message.driver_camera_image.camera_image
            )
            frames_by_camera.setdefault(image.logical_id, []).append(image)

    if rollout_metadata is None:
        raise ValueError("RolloutMetadata not found in log; unknown rollout index.")

    if drive_session_request is None:
        raise ValueError("DriveSessionRequest not found in log; unknown camera IDs.")

    await aios.makedirs(save_dir, exist_ok=True)

    for camera_logical_id, images in frames_by_camera.items():
        camera_name = camera_logical_id
        save_path = f"{save_dir}/{camera_name}"
        images = sorted(images, key=lambda frame: frame.frame_start_us)
        timestamps_us = np.array([frame.frame_start_us for frame in images])

        match format:
            case "mp4":
                await frames_to_mp4(images, timestamps_us, save_path)
            case "frames":
                await save_frames_as_files(images, timestamps_us, save_path)
            case _:
                raise TypeError(f"Unknown {format=}")


async def frames_to_mp4(
    images: list[RolloutCameraImage.CameraImage],
    timestamps_us: np.ndarray,
    save_path: str,
) -> None:
    average_fps = 1 / (np.diff(timestamps_us).mean() / 1e6)

    bitmaps = [
        pad_to_divisible_by_16(iio.imread(image.image_bytes)) for image in images
    ]

    duration_us = timestamps_us[-1] - timestamps_us[0]
    duration_s = duration_us / 1e6

    save_path = f"{save_path}.mp4"

    logger.info(f"Saving {save_path=} (duration {duration_s:.2f}s).")
    iio.imwrite(
        save_path,
        image=bitmaps,
        extension=".mp4",
        fps=average_fps,
    )


async def _write_image(content: bytes, path: str) -> None:
    """Detects the format (JPG or PNG) and writes `content` to disk."""
    format: str
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        format = "png"
    elif content[:3] == b"\xff\xd8\xff":
        format = "jpg"
    else:
        raise ValueError("A frame could not be identified as either png or jpg")

    async with aiofiles.open(f"{path}.{format}", "wb") as file:
        await file.write(content)


async def save_frames_as_files(
    images: list[RolloutCameraImage.CameraImage],
    timestamps_us: np.ndarray,
    save_path: str,
) -> None:
    await aios.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving {save_path=} as frames.")
    await asyncio.gather(
        *[
            _write_image(image.image_bytes, f"{save_path}/{timestamp_us}")
            for image, timestamp_us in zip(images, timestamps_us)
        ]
    )


def determine_save_dir(log_path: str, log_save_dir: Optional[str]) -> str:
    if log_save_dir is None:
        log_save_name = log_path.removesuffix(".asl")
        return f"{log_save_name}_asl_frames"
    else:
        # Note(mwatson): This code is taken from an earlier version of the kpi codebase.
        log_save_name = "/".join(
            log_path.removesuffix(".asl").split("/")[-3:]
        )  # clipgt/batch/rollout
        return f"{log_save_dir}/{log_save_name}"


async def convert_multiple_logs(
    asl_glob: str,
    format: SaveFormat,
    log_save_dir: Optional[str] = None,
) -> None:
    assert asl_glob.endswith(".asl"), asl_glob

    log_paths = glob.glob(asl_glob, recursive=True)

    logger.info(f"Found {len(log_paths)} log files for conversion in {asl_glob=}.")

    async def convert_log_with_exception_handling(log_path: str, save_dir: str) -> None:
        try:
            await convert_single_log(
                log_path=log_path, save_dir=save_dir, format=format
            )
        except Exception as e:
            logger.error(f"Exception {e}, skipping {log_path=}.")

    tasks = []
    for (
        log_path
    ) in log_paths:  # tqdm is added in alpasim-kpi... should be added here too?
        save_dir = determine_save_dir(log_path, log_save_dir)

        tasks.append(convert_log_with_exception_handling(log_path, save_dir))

    await asyncio.gather(*tasks)

    logger.info(f"Converted {len(log_paths)} logs to {format=} in {log_save_dir=}.")


EXAMPLE_USAGE = (
    'Example usage: python -m alpasim_utils.asl_to_frames "path/to/logs/**/*.asl"'
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "asl_glob",
        type=str,
        help="Glob to find the asl files for conversion. To prevent expansion in shell quote it.",
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "frames"],
        default="mp4",
    )
    parser.add_argument(
        "--log-save-dir",
        type=str,
        default=None,
        help="Optional output directory. If not provided, saves alongside the .asl files.",
    )
    args = parser.parse_args()

    asyncio.run(
        convert_multiple_logs(
            args.asl_glob, format=args.format, log_save_dir=args.log_save_dir
        )
    )
