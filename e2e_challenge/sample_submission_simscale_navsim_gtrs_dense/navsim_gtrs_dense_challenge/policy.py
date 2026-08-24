# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .preprocessing import CAMERA_IDS, build_status_feature, preprocess_images
from .simscale_gtrs_dense.config import GTRSDenseConfig
from .simscale_gtrs_dense.model import RELEASE_SCORER, GTRSDenseModel

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "agent.model."
VOCABULARY_KEY = f"{CHECKPOINT_PREFIX}_trajectory_head.vocab"
NAVHARD_VOCABULARY_SHAPE = (8192, 40, 3)
CHECKPOINT_VOCABULARY_SHAPE = (16384, 40, 3)
OFFICIAL_VOCABULARIES = {
    3_932_288: (
        "cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35",
        NAVHARD_VOCABULARY_SHAPE,
    ),
    7_864_448: (
        "e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d",
        CHECKPOINT_VOCABULARY_SHAPE,
    ),
}


@dataclass(frozen=True)
class InferenceInput:
    images: Mapping[str, np.ndarray]
    command_one_hot: np.ndarray
    velocity_xy: np.ndarray
    acceleration_xy: np.ndarray

    def __post_init__(self) -> None:
        missing = [
            camera_id for camera_id in CAMERA_IDS if camera_id not in self.images
        ]
        if missing:
            raise ValueError(f"missing required cameras: {', '.join(missing)}")


@dataclass(frozen=True)
class Prediction:
    trajectory: np.ndarray


def _validate_vocabulary(
    vocabulary: Tensor,
    *,
    expected_shape: tuple[int, int, int],
    source: str,
) -> None:
    if tuple(vocabulary.shape) != expected_shape:
        raise ValueError(
            f"{source} vocabulary must have shape {expected_shape}; "
            f"got {tuple(vocabulary.shape)}"
        )
    if vocabulary.dtype != torch.float32:
        raise ValueError(f"{source} vocabulary must use dtype float32")
    if not torch.isfinite(vocabulary).all():
        raise ValueError(f"{source} vocabulary contains non-finite values")


def load_navhard_vocabulary(path: str | PathLike[str]) -> Tensor:
    vocabulary_path = Path(path)
    try:
        size = vocabulary_path.stat().st_size
    except OSError as exc:
        raise ValueError(
            f"NAVHARD vocabulary is unavailable: {vocabulary_path}"
        ) from exc
    vocabulary_spec = OFFICIAL_VOCABULARIES.get(size)
    if vocabulary_spec is None:
        raise ValueError(
            "official vocabulary size "
            f"{size}, expected one of {sorted(OFFICIAL_VOCABULARIES)}"
        )
    expected_sha, expected_shape = vocabulary_spec

    digest = hashlib.sha256()
    with vocabulary_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "official vocabulary SHA256 " f"{actual_sha}, expected {expected_sha}"
        )

    try:
        array = np.load(vocabulary_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("NAVHARD vocabulary is not a valid NPY array") from exc
    vocabulary = torch.from_numpy(np.asarray(array).copy())
    _validate_vocabulary(
        vocabulary,
        expected_shape=expected_shape,
        source="official",
    )
    return vocabulary


def normalized_checkpoint(
    payload: object,
    inference_vocabulary: Tensor,
) -> dict[str, Tensor]:
    if not isinstance(payload, dict) or set(payload) != {"state_dict"}:
        raise ValueError(
            "checkpoint payload must be a dict containing only 'state_dict'"
        )
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint state_dict must be a dict")

    normalized: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.startswith(CHECKPOINT_PREFIX):
            raise ValueError(f"unexpected checkpoint key: {key}")
        if not isinstance(value, Tensor):
            raise ValueError(f"checkpoint value for {key} must be a tensor")
        normalized[key.removeprefix(CHECKPOINT_PREFIX)] = value

    checkpoint_vocabulary = state_dict.get(VOCABULARY_KEY)
    if not isinstance(checkpoint_vocabulary, Tensor):
        raise ValueError("checkpoint vocabulary tensor is missing")
    _validate_vocabulary(
        checkpoint_vocabulary,
        expected_shape=CHECKPOINT_VOCABULARY_SHAPE,
        source="checkpoint",
    )
    inference_shape = tuple(inference_vocabulary.shape)
    supported_shapes = {spec[1] for spec in OFFICIAL_VOCABULARIES.values()}
    if inference_shape not in supported_shapes:
        raise ValueError(
            "inference vocabulary must have a supported shape; "
            f"got {inference_shape}"
        )
    _validate_vocabulary(
        inference_vocabulary,
        expected_shape=inference_shape,
        source="inference",
    )
    normalized["_trajectory_head.vocab"] = inference_vocabulary
    return normalized


class GTRSDensePolicy:
    def __init__(
        self,
        checkpoint_path: str | PathLike[str],
        vocabulary_path: str | PathLike[str],
        device: str | torch.device = "cuda",
        scorer_mode: str = RELEASE_SCORER,
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
        warm_up: bool = True,
        backbone_type: str = "resnet",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but CUDA is not available"
            )
        self.dtype = torch.float32

        vocabulary = load_navhard_vocabulary(vocabulary_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = normalized_checkpoint(payload, vocabulary)
        config = GTRSDenseConfig(
            vocab_size=int(vocabulary.shape[0]), backbone_type=backbone_type
        )
        self.model = GTRSDenseModel(
            config,
            vocab=vocabulary,
            scorer_mode=scorer_mode,
            ep_exponent=ep_exponent,
            speed_top_k=speed_top_k,
            speed_weight=speed_weight,
            speed_proxy=speed_proxy,
            curvature_weight=curvature_weight,
            heading_change_weight=heading_change_weight,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.state_dict_count = len(state_dict)
        LOGGER.info(
            "Loaded %d tensors from checkpoint %s with scorer_mode=%s "
            "ep_exponent=%s speed_top_k=%s speed_weight=%s "
            "speed_proxy=%s curvature_weight=%s heading_change_weight=%s",
            self.state_dict_count,
            checkpoint_path,
            scorer_mode,
            ep_exponent,
            speed_top_k,
            speed_weight,
            speed_proxy,
            curvature_weight,
            heading_change_weight,
        )
        self.model.to(self.device).eval()

        if warm_up:
            zero_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.predict_batch(
                [
                    InferenceInput(
                        images={camera_id: zero_image for camera_id in CAMERA_IDS},
                        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
                        velocity_xy=np.zeros(2, dtype=np.float32),
                        acceleration_xy=np.zeros(2, dtype=np.float32),
                    )
                ]
            )
            LOGGER.info("GTRS-Dense policy warm-up completed")

    def predict_batch(self, requests: list[InferenceInput]) -> list[Prediction]:
        if not requests:
            return []

        camera_feature = torch.stack(
            [
                preprocess_images(request.images, dtype=torch.float32)
                for request in requests
            ]
        ).to(device=self.device, dtype=self.dtype)
        status_feature = torch.stack(
            [
                build_status_feature(
                    command_one_hot=request.command_one_hot,
                    velocity_xy=request.velocity_xy,
                    acceleration_xy=request.acceleration_xy,
                )
                for request in requests
            ]
        ).to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            output = self.model(
                {
                    "camera_feature": camera_feature,
                    "status_feature": status_feature,
                }
            )

        trajectory = output["trajectory"].detach().float().cpu().numpy()
        expected_shape = (len(requests), 40, 3)
        if trajectory.shape != expected_shape:
            raise ValueError(
                f"model trajectory must have shape {expected_shape}; "
                f"got {trajectory.shape}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError("model trajectory contains non-finite values")
        return [Prediction(trajectory=sample.copy()) for sample in trajectory]
