# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Decoder-only KV-cache ONNX trajectory policy (rmind PatchPolicy exports).

The model sees one camera frame per tick plus ego speed and route waypoints,
and attends to earlier frames through a KV cache that the host carries per
session.  It drives on the trajectory head; ``policy.joint_actions`` is
ignored.  Input conventions follow ``rsim_port``.

Model contract (rsim ``OnnxDecoderAgent``, rmind ``decoder_only_kv_cache.md``):

* Camera: the raw ``cam_front_left`` fisheye at 1920x1080, defished with the
  fleet ``cam-90-deg.json`` calibration, then cropped and resized.
* Speed: km/h.
* Trajectory: ``(Q, P, 3)`` hypotheses x poses x (x, y, theta) in metres,
  absolute from t0, poses ``1/3`` s apart starting at t0 + 1/3 s.  Exports
  trained before rmind 827d5fd emit ``/100``-scaled xy and are not supported.
  The driven hypothesis is the argmax of the mode logits.  Headings come from
  the xy path; theta is not used.
* Rate: one inference every 1/3 s.  The graph must run every tick to keep
  the cache consistent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.models.rig_frame import (
    route_in_rig_at_t0,
    trajectory_from_pose_messages,
)
from alpasim_driver.schema import ModelConfig

from .rsim_port import (
    FLEET_CALIBRATION_PATH,
    SPEED_INPUT,
    TRAJECTORY_MODE_LOGITS_OUTPUT,
    TRAJECTORY_OUTPUT,
    WAYPOINTS_INPUT,
    DecoderGraphSpec,
    Defisher,
    FisheyeCalibration,
    KvCacheRing,
    model_xy_to_rig,
    preprocess_image,
    route_to_model_waypoints,
)

logger = logging.getLogger(__name__)

CAMERA_ID = "cam_front_left"
MPS_TO_KMH = 3.6
KV_RING_KEY = "kv_ring"


class OnnxTrajModel(BaseTrajectoryModel):
    """Runs a decoder-only KV-cache ONNX export with onnxruntime."""

    OUTPUT_FREQUENCY_HZ = 3

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> OnnxTrajModel:
        if context_length not in (None, 1):
            raise ValueError(
                f"OnnxTrajModel carries history in its KV cache; "
                f"context_length must be 1, got {context_length}"
            )
        return cls(
            checkpoint_path=model_cfg.checkpoint_path,
            device=device,
            camera_ids=camera_ids,
        )

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        camera_ids: list[str],
    ):
        import onnxruntime as ort

        if camera_ids != [CAMERA_ID]:
            raise ValueError(
                f"OnnxTrajModel needs exactly [{CAMERA_ID!r}], got {camera_ids}"
            )
        model_path = Path(checkpoint_path)
        if device.type == "cuda":
            providers: list = [
                ("CUDAExecutionProvider", {"device_id": device.index or 0})
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        active = self._session.get_providers()
        if device.type == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"onnxruntime could not use the GPU (active providers: {active}); "
                "check that onnxruntime-gpu matches the installed CUDA driver"
            )

        self._spec = DecoderGraphSpec.from_session(self._session)
        if list(self._spec.camera_inputs) != camera_ids:
            raise ValueError(
                f"{model_path.name} expects cameras {list(self._spec.camera_inputs)}, "
                f"configured {camera_ids}"
            )
        self._camera_ids = camera_ids
        self._output_index = {
            o.name: i for i, o in enumerate(self._session.get_outputs())
        }
        self._defisher = Defisher(FisheyeCalibration.from_json(FLEET_CALIBRATION_PATH))
        self._step_us = 1_000_000 // self.OUTPUT_FREQUENCY_HZ

        logger.info(
            "Loaded %s: providers=%s image_hw=%s cache_frames=%d tokens_per_frame=%d "
            "hypotheses=%d poses=%d",
            model_path,
            active,
            self._spec.image_hw,
            self._spec.cache_frames,
            self._spec.tokens_per_frame,
            self._spec.num_hypotheses,
            self._spec.num_poses,
        )

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return 1

    @property
    def output_frequency_hz(self) -> int:
        return self.OUTPUT_FREQUENCY_HZ

    def _encode_command(self, command: DriveCommand) -> None:
        """The model takes route waypoints instead of a command."""
        return None

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)
        frames = prediction_input.camera_images[CAMERA_ID]
        if len(frames) != 1:
            raise ValueError(f"OnnxTrajModel expects 1 frame, got {len(frames)}")
        t0_us, image = frames[0]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        pose_local_to_rig_t0 = trajectory_from_pose_messages(
            prediction_input.ego_pose_history
        ).interpolate_pose(np.uint64(t0_us))
        if prediction_input.route is None:
            route_xy_rig = np.zeros((0, 2))
        else:
            route_xy_rig = route_in_rig_at_t0(
                prediction_input.route,
                prediction_input.ego_pose_history,
                pose_local_to_rig_t0,
            )[:, :2]
        waypoints = route_to_model_waypoints(route_xy_rig, self._spec.num_waypoints)

        ring = prediction_input.session_state.get(KV_RING_KEY)
        if ring is None:
            ring = KvCacheRing(self._spec)
            prediction_input.session_state[KV_RING_KEY] = ring

        rectilinear = self._defisher.apply(image)
        feed = {
            self._spec.camera_inputs[CAMERA_ID]: preprocess_image(
                rectilinear, self._spec.image_hw
            )[np.newaxis, np.newaxis],
            SPEED_INPUT: np.array(
                [[[prediction_input.speed * MPS_TO_KMH]]], dtype=np.float32
            ),
            WAYPOINTS_INPUT: waypoints[np.newaxis, np.newaxis],
            **ring.feed(),
        }
        start = time.perf_counter()
        outputs = self._session.run(None, feed)
        latency_ms = (time.perf_counter() - start) * 1e3
        ring.advance(
            outputs[self._output_index["new_k"]], outputs[self._output_index["new_v"]]
        )

        trajectory = outputs[self._output_index[TRAJECTORY_OUTPUT]][0]  # (Q, P, 3)
        logits = outputs[self._output_index[TRAJECTORY_MODE_LOGITS_OUTPUT]][0]
        selected = int(np.argmax(logits))
        modes_xy_rig = model_xy_to_rig(trajectory[..., :2].astype(np.float64))
        headings = self._compute_headings_from_trajectory_batch(modes_xy_rig)
        candidates = [
            ModelPrediction.from_planar(xy, heading)
            for xy, heading in zip(modes_xy_rig, headings, strict=True)
        ]
        logger.debug(
            "frame=%d latency=%.1fms speed=%.1fkm/h mode=%d logits=%s end_xy=%s",
            ring.frame_index,
            latency_ms,
            prediction_input.speed * MPS_TO_KMH,
            selected,
            np.round(logits, 2).tolist(),
            np.round(modes_xy_rig[selected, -1], 2).tolist(),
        )

        return ModelPrediction(
            candidate_positions=np.concatenate(
                [c.candidate_positions for c in candidates]
            ),
            candidate_rotations=np.concatenate(
                [c.candidate_rotations for c in candidates]
            ),
            selected_index=selected,
            model_t0_us=t0_us,
            pose_local_to_rig_t0=pose_local_to_rig_t0,
            waypoint_timestamps_us=(
                t0_us
                + np.arange(1, self._spec.num_poses + 1, dtype=np.uint64)
                * self._step_us
            ),
        )
