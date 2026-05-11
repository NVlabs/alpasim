#!/usr/bin/env python3
"""Serve MTGS rendering behind the AlpaSim sensorsim gRPC API.

This is a compatibility shim: AlpaSim still talks to a SensorsimService, but the
implementation renders RGB images from an MTGS/Nerfstudio checkpoint instead of
NuRec. The first version focuses on camera RGB rendering; dynamic object pose
overrides are intentionally not applied yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import grpc
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps

DEFAULT_ALPASIM_GRPC = Path("/home/one/src/alpasim/src/grpc")
if DEFAULT_ALPASIM_GRPC.exists() and str(DEFAULT_ALPASIM_GRPC) not in sys.path:
    sys.path.append(str(DEFAULT_ALPASIM_GRPC))

from alpasim_grpc import API_VERSION_MESSAGE  # noqa: E402
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc  # noqa: E402
from mtgs.tools.render import eval_setup  # noqa: E402


LOGGER = logging.getLogger("mtgs_sensorsim")
OPENCV_TO_NERFSTUDIO = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def pose_proto_from_se3(matrix: np.ndarray) -> common_pb2.Pose:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return common_pb2.Pose(
        vec=common_pb2.Vec3(
            x=float(matrix[0, 3]),
            y=float(matrix[1, 3]),
            z=float(matrix[2, 3]),
        ),
        quat=common_pb2.Quat(
            w=float(quat_xyzw[3]),
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
        ),
    )


def se3_from_pose_proto(pose: common_pb2.Pose) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
    matrix[:3, :3] = Rotation.from_quat(
        [pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w]
    ).as_matrix()
    return matrix


def opencv_distortion_to_proto(distortion: list[float]) -> tuple[list[float], list[float]]:
    """Map OpenCV k1,k2,p1,p2,k3 into sensorsim's radial/tangential layout."""

    if len(distortion) >= 5:
        radial = [distortion[0], distortion[1], distortion[4], 0.0, 0.0, 0.0]
        tangential = [distortion[2], distortion[3]]
    elif len(distortion) >= 4:
        radial = [distortion[0], distortion[1], 0.0, 0.0, 0.0, 0.0]
        tangential = [distortion[2], distortion[3]]
    else:
        radial = [0.0] * 6
        tangential = [0.0] * 2
    return radial, tangential


def rgb_tensor_to_encoded_bytes(
    rgb: torch.Tensor,
    image_format: int,
    quality: float,
) -> bytes:
    rgb_np = (
        torch.clamp(rgb.detach(), 0.0, 1.0).mul(255.0).byte().cpu().numpy()
    )
    if rgb_np.ndim != 3 or rgb_np.shape[-1] < 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {rgb_np.shape}")
    rgb_np = rgb_np[..., :3]

    if image_format == sensorsim_pb2.ImageFormat.PNG:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    else:
        jpeg_quality = int(np.clip(quality if quality else 95, 1, 100))
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
    if not ok:
        raise RuntimeError("OpenCV image encoding failed")
    return encoded.tobytes()


def configure_cuda_extension_arch() -> None:
    """Avoid compiling CUDA extensions for every visible architecture on first render."""

    if os.environ.get("TORCH_CUDA_ARCH_LIST") or not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    LOGGER.info("Set TORCH_CUDA_ARCH_LIST=%s for first-render CUDA extension builds", arch)


class MTGSSensorsimServicer(sensorsim_pb2_grpc.SensorsimServiceServicer):
    def __init__(
        self,
        config_path: Path,
        artifact_dir: Path,
        scene_id: Optional[str],
        travel_id: int,
        native_height: int,
        native_width: int,
        eval_num_rays_per_chunk: Optional[int],
        background_rgb: tuple[float, float, float],
    ):
        self.config_path = config_path
        self.artifact_dir = artifact_dir
        self.rig_json = json.loads((artifact_dir / "rig_trajectories.json").read_text())
        self.scene_id = scene_id or self.rig_json["rig_trajectories"][0]["sequence_id"]
        self.travel_id = travel_id
        self.native_height = native_height
        self.native_width = native_width

        rig = self.rig_json["rig_trajectories"][0]
        self.min_timestamp_us = int(min(rig["T_rig_world_timestamps_us"]))
        self.max_timestamp_us = int(max(rig["T_rig_world_timestamps_us"]))

        configure_cuda_extension_arch()
        LOGGER.info("Loading MTGS pipeline from %s", config_path)
        _config, pipeline, checkpoint_path, step = eval_setup(
            config_path,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
            test_mode="test",
        )
        self.pipeline = pipeline
        self.pipeline.eval()
        self.device = pipeline.device
        self.render_lock = asyncio.Lock()
        self.server: Optional[grpc.aio.Server] = None
        LOGGER.info("Loaded checkpoint %s at step %s on %s", checkpoint_path, step, self.device)

        model = self.pipeline.model
        if hasattr(model, "set_background"):
            model.set_background(torch.tensor(background_rgb, dtype=torch.float32, device=self.device))

    def _normalised_time(self, timestamp_us: int) -> float:
        denom = max(1, self.max_timestamp_us - self.min_timestamp_us)
        return float(np.clip((int(timestamp_us) - self.min_timestamp_us) / denom, 0.0, 1.0))

    def _available_camera_from_calibration(
        self,
        unique_id: str,
        calibration: dict,
    ) -> sensorsim_pb2.AvailableCamerasReturn.AvailableCamera:
        params = calibration["camera_model"]["parameters"]
        intr = np.asarray(params["intrinsics"], dtype=np.float64)
        distortion = [float(v) for v in params.get("distortion", [])]
        radial, tangential = opencv_distortion_to_proto(distortion)

        spec = sensorsim_pb2.CameraSpec(
            logical_id=calibration["logical_sensor_name"],
            resolution_h=self.native_height,
            resolution_w=self.native_width,
            shutter_type=sensorsim_pb2.ShutterType.GLOBAL,
        )
        pinhole = spec.opencv_pinhole_param
        pinhole.principal_point_x = float(intr[0, 2])
        pinhole.principal_point_y = float(intr[1, 2])
        pinhole.focal_length_x = float(intr[0, 0])
        pinhole.focal_length_y = float(intr[1, 1])
        pinhole.radial_coeffs.extend(radial)
        pinhole.tangential_coeffs.extend(tangential)
        pinhole.thin_prism_coeffs.extend([0.0, 0.0, 0.0, 0.0])

        pose = pose_proto_from_se3(np.asarray(calibration["T_sensor_rig"], dtype=np.float32))
        return sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
            intrinsics=spec,
            rig_to_camera=pose,
            logical_id=calibration["logical_sensor_name"],
        )

    def _intrinsics_from_request(self, request: sensorsim_pb2.RGBRenderRequest) -> tuple[float, float, float, float]:
        spec = request.camera_intrinsics
        model = spec.WhichOneof("camera_param")
        if model != "opencv_pinhole_param":
            raise ValueError(
                f"Only opencv_pinhole_param is supported by the first MTGS sensorsim shim, got {model}"
            )

        pinhole = spec.opencv_pinhole_param
        native_w = float(spec.resolution_w or request.resolution_w or self.native_width)
        native_h = float(spec.resolution_h or request.resolution_h or self.native_height)
        scale_x = float(request.resolution_w) / native_w
        scale_y = float(request.resolution_h) / native_h
        return (
            float(pinhole.focal_length_x) * scale_x,
            float(pinhole.focal_length_y) * scale_y,
            float(pinhole.principal_point_x) * scale_x,
            float(pinhole.principal_point_y) * scale_y,
        )

    def _camera_from_request(self, request: sensorsim_pb2.RGBRenderRequest) -> Cameras:
        height = int(request.resolution_h)
        width = int(request.resolution_w)
        if height <= 0 or width <= 0:
            height = int(request.camera_intrinsics.resolution_h or self.native_height)
            width = int(request.camera_intrinsics.resolution_w or self.native_width)

        camera_to_world_opencv = se3_from_pose_proto(request.sensor_pose.start_pose)
        camera_to_world_ns = camera_to_world_opencv @ OPENCV_TO_NERFSTUDIO
        fx, fy, cx, cy = self._intrinsics_from_request(request)

        timestamp_us = int(request.frame_start_us)
        camera = Cameras(
            fx=torch.tensor([fx], dtype=torch.float32),
            fy=torch.tensor([fy], dtype=torch.float32),
            cx=torch.tensor([cx], dtype=torch.float32),
            cy=torch.tensor([cy], dtype=torch.float32),
            height=torch.tensor([height], dtype=torch.int32),
            width=torch.tensor([width], dtype=torch.int32),
            camera_to_worlds=torch.from_numpy(camera_to_world_ns[:3, :4])[None, ...].float(),
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor([self._normalised_time(timestamp_us)], dtype=torch.float32),
            metadata={
                "travel_id": self.travel_id,
                "multicolor_travel_id": self.travel_id,
                "linear_velocity": np.zeros(3, dtype=np.float32),
                "angular_velocity": np.zeros(3, dtype=np.float32),
            },
        )
        return camera.to(self.device)

    def _warmup_request(self, height: int, width: int) -> sensorsim_pb2.RGBRenderRequest:
        rig = self.rig_json["rig_trajectories"][0]
        camera_calibration = next(iter(self.rig_json["camera_calibrations"].values()))
        camera = self._available_camera_from_calibration("warmup", camera_calibration)

        rig_to_world = np.asarray(rig["T_rig_worlds"][0], dtype=np.float32)
        rig_to_camera = np.asarray(camera_calibration["T_sensor_rig"], dtype=np.float32)
        camera_to_world = rig_to_world @ rig_to_camera
        pose = pose_proto_from_se3(camera_to_world)

        intrinsics = sensorsim_pb2.CameraSpec()
        intrinsics.CopyFrom(camera.intrinsics)
        return sensorsim_pb2.RGBRenderRequest(
            scene_id=self.scene_id,
            resolution_h=height,
            resolution_w=width,
            camera_intrinsics=intrinsics,
            frame_start_us=int(rig["T_rig_world_timestamps_us"][0]),
            frame_end_us=int(rig["T_rig_world_timestamps_us"][0]),
            sensor_pose=sensorsim_pb2.PosePair(start_pose=pose, end_pose=pose),
            image_format=sensorsim_pb2.ImageFormat.JPEG,
            image_quality=85,
        )

    def warmup(self, num_renders: int, height: int, width: int) -> None:
        if num_renders <= 0:
            return
        request = self._warmup_request(height=height, width=width)
        LOGGER.info("Warming MTGS renderer with %d render(s) at %dx%d", num_renders, width, height)
        for idx in range(num_renders):
            started = time.perf_counter()
            camera = self._camera_from_request(request)
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(camera)
            rgb = outputs.get("rgb")
            if rgb is None:
                keys = ", ".join(outputs.keys())
                raise RuntimeError(f"MTGS model output has no 'rgb'. Available outputs: {keys}")
            encoded = rgb_tensor_to_encoded_bytes(
                colormaps.apply_colormap(rgb),
                request.image_format,
                request.image_quality,
            )
            torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
            LOGGER.info(
                "Warmup render %d/%d completed in %.3fs, image_bytes=%d",
                idx + 1,
                num_renders,
                time.perf_counter() - started,
                len(encoded),
            )

    async def get_version(self, request, context):
        LOGGER.info("RPC get_version peer=%s", context.peer())
        return common_pb2.VersionId(
            version_id="mtgs-sensorsim-local",
            git_hash="unknown",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    async def get_available_scenes(self, request, context):
        LOGGER.info("RPC get_available_scenes peer=%s", context.peer())
        return common_pb2.AvailableScenesReturn(scene_ids=[self.scene_id])

    async def get_available_cameras(self, request, context):
        LOGGER.info("RPC get_available_cameras scene_id=%s peer=%s", request.scene_id, context.peer())
        if request.scene_id and request.scene_id != self.scene_id:
            LOGGER.warning("Requested scene_id=%s, serving %s", request.scene_id, self.scene_id)
        response = sensorsim_pb2.AvailableCamerasReturn()
        for unique_id, calibration in self.rig_json["camera_calibrations"].items():
            response.available_cameras.append(
                self._available_camera_from_calibration(unique_id, calibration)
            )
        return response

    async def get_available_ego_masks(self, request, context):
        LOGGER.info("RPC get_available_ego_masks peer=%s", context.peer())
        return sensorsim_pb2.AvailableEgoMasksReturn()

    async def get_available_trajectories(self, request, context):
        LOGGER.info("RPC get_available_trajectories peer=%s", context.peer())
        return sensorsim_pb2.AvailableTrajectoriesReturn()

    async def render_lidar(self, request, context):
        LOGGER.info("RPC render_lidar scene_id=%s peer=%s", request.scene_id, context.peer())
        return sensorsim_pb2.LidarRenderReturn(num_points=0)

    async def render_rgb(self, request, context):
        request_started = time.perf_counter()
        LOGGER.info(
            "RPC render_rgb received scene_id=%s size=%dx%d frame_start_us=%s peer=%s",
            request.scene_id,
            int(request.resolution_w),
            int(request.resolution_h),
            int(request.frame_start_us),
            context.peer(),
        )
        if request.dynamic_objects:
            LOGGER.debug(
                "Ignoring %d dynamic object overrides in first MTGS sensorsim shim",
                len(request.dynamic_objects),
            )

        camera_started = time.perf_counter()
        camera = self._camera_from_request(request)
        camera_elapsed = time.perf_counter() - camera_started
        async with self.render_lock:
            render_started = time.perf_counter()
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs_for_camera(camera)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            render_elapsed = time.perf_counter() - render_started
        rgb = outputs.get("rgb")
        if rgb is None:
            keys = ", ".join(outputs.keys())
            raise RuntimeError(f"MTGS model output has no 'rgb'. Available outputs: {keys}")

        encode_started = time.perf_counter()
        encoded = rgb_tensor_to_encoded_bytes(
            colormaps.apply_colormap(rgb),
            request.image_format,
            request.image_quality,
        )
        encode_elapsed = time.perf_counter() - encode_started
        LOGGER.info(
            "render_rgb scene_id=%s camera=%s size=%dx%d frame_start_us=%s timings camera=%.3fs render=%.3fs encode=%.3fs total=%.3fs bytes=%d",
            request.scene_id,
            request.camera_intrinsics.logical_id,
            int(request.resolution_w),
            int(request.resolution_h),
            int(request.frame_start_us),
            camera_elapsed,
            render_elapsed,
            encode_elapsed,
            time.perf_counter() - request_started,
            len(encoded),
        )
        return sensorsim_pb2.RGBRenderReturn(image_bytes=encoded)

    async def render_aggregated(self, request, context):
        LOGGER.info("RPC render_aggregated rgb_requests=%d peer=%s", len(request.rgb_requests), context.peer())
        response = sensorsim_pb2.AggregatedRenderReturn()
        for rgb_request in request.rgb_requests:
            response.rgb_returns.append(await self.render_rgb(rgb_request, context))
        return response

    async def shut_down(self, request, context):
        LOGGER.info("Received shut_down request")
        if self.server is not None:
            asyncio.create_task(self.server.stop(grace=1.0))
        return common_pb2.Empty()


async def serve(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    servicer = MTGSSensorsimServicer(
        config_path=args.config,
        artifact_dir=args.artifact_dir,
        scene_id=args.scene_id,
        travel_id=args.travel_id,
        native_height=args.native_height,
        native_width=args.native_width,
        eval_num_rays_per_chunk=args.eval_num_rays_per_chunk,
        background_rgb=tuple(args.background_rgb),
    )
    servicer.warmup(
        num_renders=args.warmup_renders,
        height=args.warmup_height,
        width=args.warmup_width,
    )
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", args.max_message_bytes),
            ("grpc.max_receive_message_length", args.max_message_bytes),
        ]
    )
    sensorsim_pb2_grpc.add_SensorsimServiceServicer_to_server(servicer, server)
    servicer.server = server
    listen_addr = f"{args.host}:{args.port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    LOGGER.info("MTGS sensorsim server listening on %s for scene_id=%s", listen_addr, servicer.scene_id)
    await server.wait_for_termination()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an MTGS-backed AlpaSim sensorsim gRPC service.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/main_mt/MTGS/road_block-331220_4690660_331190_4690710/config.yml"),
        help="MTGS/Nerfstudio config.yml that points to the trained checkpoint directory.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("exports/alpasim/road_block-331220_4690660_331190_4690710"),
        help="AlpaSim artifact directory containing rig_trajectories.json.",
    )
    parser.add_argument("--scene-id", default=None, help="Override scene_id reported to AlpaSim.")
    parser.add_argument("--travel-id", type=int, default=7, help="MTGS traversal id used for time/color conditioning.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50053)
    parser.add_argument("--native-height", type=int, default=1080)
    parser.add_argument("--native-width", type=int, default=1920)
    parser.add_argument("--eval-num-rays-per-chunk", type=int, default=None)
    parser.add_argument("--max-message-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--background-rgb", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--warmup-renders",
        type=int,
        default=0,
        help="Run this many local renders before accepting gRPC requests. This pays first-render CUDA JIT cost at startup.",
    )
    parser.add_argument("--warmup-height", type=int, default=180)
    parser.add_argument("--warmup-width", type=int, default=320)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
