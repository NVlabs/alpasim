# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Video model gRPC service wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from alpasim_grpc.v0 import sensorsim_pb2, video_model_pb2_grpc
from alpasim_grpc.v0.common_pb2 import Pose, Trajectory
from alpasim_grpc.v0.logging_pb2 import LogEntry
from alpasim_grpc.v0.video_model_pb2 import (
    DebugOptions,
    DynamicWorldState,
    Image,
    ImageFormat,
    SessionCloseRequest,
    SessionId,
    SessionRequest,
    StaticWorldMap,
    TextPrompt,
    VideoChunkRequest,
    VideoChunkReturn,
)
from alpasim_runtime.camera_catalog import CameraCatalog, CameraDefinition
from alpasim_runtime.config import SimulationConfig, VideoModelConfig
from alpasim_runtime.services.service_base import ServiceBase, SessionInfo
from alpasim_runtime.services.session_configs import RendererSessionConfig
from alpasim_runtime.telemetry.rpc_wrapper import profiled_rpc_call
from alpasim_runtime.types import RuntimeCamera
from alpasim_runtime.video_model.usdz_calibration import parse_cameras_from_usdz
from alpasim_runtime.video_model.utils import (
    build_camera_specs_and_initial_frames,
    extract_hdmap_for_video_model,
)
from alpasim_utils.types import ImageWithMetadata
from omegaconf import OmegaConf

import grpc

logger = logging.getLogger(__name__)

MAX_GRPC_MESSAGE_BYTES = 64 * 1024 * 1024


@dataclass
class ChunkResult:
    """Result of a video model chunk request."""

    rgb_frames_per_camera: dict[str, list[ImageWithMetadata]] = field(
        default_factory=dict
    )
    hdmap_frames_per_camera: dict[str, list[ImageWithMetadata]] = field(
        default_factory=dict
    )
    raw_request: VideoChunkRequest | None = None
    raw_response: VideoChunkReturn | None = None

    @property
    def rgb_frames(self) -> list[ImageWithMetadata]:
        return [f for frames in self.rgb_frames_per_camera.values() for f in frames]

    @property
    def hdmap_frames(self) -> list[ImageWithMetadata]:
        return [f for frames in self.hdmap_frames_per_camera.values() for f in frames]


class VideoModelService(ServiceBase[video_model_pb2_grpc.WorldModelServiceStub]):
    """gRPC client for the video model renderer.

    Unlike sensorsim (stateless, per-frame), the video model maintains a
    session and renders frames in chunks.
    """

    def __init__(
        self,
        address: str,
        config: VideoModelConfig,
        skip: bool = False,
        camera_catalog: CameraCatalog | None = None,
    ):
        """Create a video-model service client for one renderer endpoint."""
        super().__init__(address, skip)
        self.config = self._coerce_config(config)
        self._camera_catalog = camera_catalog
        self._session_id: SessionId | None = None
        self._is_first_chunk = True
        self._frame_interval_us = 1_000_000 // self.config.fps
        self._runtime_cameras: list[RuntimeCamera] = []

    @classmethod
    def from_config(
        cls,
        raw_config: dict[str, Any],
        address: str,
        skip: bool = False,
        *,
        camera_catalog: CameraCatalog | None = None,
    ) -> VideoModelService:
        """Factory used by the core worker to build the built-in renderer."""
        config = VideoModelConfig(**raw_config) if raw_config else VideoModelConfig()
        return cls(
            address=address,
            config=config,
            skip=skip,
            camera_catalog=camera_catalog,
        )

    @staticmethod
    def _coerce_config(config: Any) -> VideoModelConfig:
        if isinstance(config, VideoModelConfig):
            return config
        values = OmegaConf.to_container(config, resolve=True)
        if not isinstance(values, dict):
            raise TypeError(
                f"video_model_config must be a mapping, got {type(values).__name__}"
            )
        return VideoModelConfig(**values)

    @property
    def stub_class(self) -> type[video_model_pb2_grpc.WorldModelServiceStub]:
        """Return the generated gRPC stub class for this renderer service."""
        return video_model_pb2_grpc.WorldModelServiceStub

    async def _open_connection(self) -> None:
        """Open gRPC connection with larger video chunk message limits."""
        if self.skip:
            return
        self.channel = grpc.aio.insecure_channel(
            self.address,
            options=[
                ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_BYTES),
                ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_BYTES),
            ],
        )
        self.stub = self.stub_class(self.channel)

    def make_initial_render_event(self, **kwargs: Any) -> Any:
        """Create the initial video-model prefetch event for a rollout."""
        from alpasim_runtime.events.video_model.prefetch import (
            make_initial_video_model_render_event,
        )

        return make_initial_video_model_render_event(
            renderer_service=self,
            **kwargs,
        )

    def validate_timing_alignment(self, simulation_config: SimulationConfig) -> None:
        """Validate rollout timing against the video-model chunk cadence."""
        force_gt_duration_us = simulation_config.force_gt_duration_us
        control_timestep_us = simulation_config.control_timestep_us

        if force_gt_duration_us < 0:
            raise ValueError(
                f"force_gt_duration_us must be >= 0, got {force_gt_duration_us}."
            )
        if force_gt_duration_us == 0:
            return

        first_chunk_duration_us = (
            self.config.first_chunk_frames * self.frame_interval_us
        )
        regular_chunk_duration_us = self.config.chunk_frames * self.frame_interval_us

        if control_timestep_us != regular_chunk_duration_us:
            raise ValueError(
                "For video_model_config, control_timestep_us "
                f"({control_timestep_us}) must equal chunk_frames * "
                f"frame_interval_us ({regular_chunk_duration_us})."
            )
        minimum_force_gt_us = first_chunk_duration_us + control_timestep_us
        if force_gt_duration_us < minimum_force_gt_us:
            raise ValueError(
                "For video_model_config, force_gt_duration_us "
                f"({force_gt_duration_us}) must cover the first chunk plus at "
                f"least one regular chunk ({minimum_force_gt_us})."
            )
        remainder = force_gt_duration_us - first_chunk_duration_us
        if remainder % control_timestep_us != 0:
            raise ValueError(
                "For video_model_config, force_gt_duration_us must "
                "equal first_chunk_frames * frame_interval_us + N * "
                "control_timestep_us."
            )

    def required_policy_start_timestmap_us(
        self,
        render_start_timestamp_us: int,
    ) -> int:
        """Start policy after the video model's short initial chunk."""
        return (
            render_start_timestamp_us
            + self.config.first_chunk_frames * self.frame_interval_us
        )

    @property
    def chunk_size(self) -> int:
        """Return the chunk size for the next video-model request."""
        return (
            self.config.first_chunk_frames
            if self._is_first_chunk
            else self.config.chunk_frames
        )

    @property
    def frame_interval_us(self) -> int:
        """Return the video-model frame interval in microseconds."""
        return self._frame_interval_us

    async def start_session(
        self,
        hdmap_bytes: bytes,
        camera_specs: list[sensorsim_pb2.CameraSpec],
        rig_to_camera: list[Pose],
        initial_frames: list[tuple[bytes, ImageFormat]],
        debug_options: DebugOptions | None = None,
        text_prompt_positive: str | None = None,
    ) -> SessionId:
        """Start a remote video-model session for the current rollout."""
        if self.skip:
            logger.info("Skip mode: video model returning mock session")
            return SessionId(session_id="mock-session")

        if len(camera_specs) != len(initial_frames):
            raise ValueError(
                f"camera_specs ({len(camera_specs)}) and initial_frames "
                f"({len(initial_frames)}) must have the same length"
            )
        if len(camera_specs) != len(rig_to_camera):
            raise ValueError(
                f"camera_specs ({len(camera_specs)}) and rig_to_camera "
                f"({len(rig_to_camera)}) must have the same length"
            )

        positive_prompt = (
            text_prompt_positive
            if text_prompt_positive is not None
            else self.config.text_prompt_positive
        )
        image_protos = [Image(data=data, format=fmt) for data, fmt in initial_frames]
        request = SessionRequest(
            static_world_map=StaticWorldMap(hdmap_parquets=hdmap_bytes),
            text_prompt=TextPrompt(
                positive=positive_prompt,
                negative=self.config.text_prompt_negative,
            ),
            camera_specs=camera_specs,
            rig_to_camera=rig_to_camera,
            initial_frames=image_protos,
        )

        if debug_options is not None:
            request.debug_options.CopyFrom(debug_options)
        elif self.config.return_hdmap_frames:
            request.debug_options.CopyFrom(
                DebugOptions(
                    return_hdmap_frames=self.config.return_hdmap_frames,
                )
            )

        logger.info(
            "Starting video model session with %d camera(s)...", len(camera_specs)
        )
        await self._broadcast(LogEntry(video_model_session_request=request))
        self._session_id = await profiled_rpc_call(
            "start_session",
            "video_model",
            self.stub.start_session,
            request,
        )
        await self._broadcast(LogEntry(video_model_session_id=self._session_id))
        logger.info("Video model session started: %s", self._session_id.session_id)
        return self._session_id

    async def render_chunk(
        self,
        trajectory_local_to_rig: Trajectory,
        dynamic_state: DynamicWorldState,
    ) -> ChunkResult:
        """Render one chunk along ``trajectory_local_to_rig``.

        The trajectory contains ego-rig poses in the rollout local frame at the
        timestamps requested from the server. Returned frames are timestamped
        from this request trajectory; the optional trajectory in the response is
        retained only for wire compatibility with existing servers.
        """
        if self.skip:
            result = self._placeholder_chunk_result(trajectory_local_to_rig)
            self._is_first_chunk = False
            return result

        if self._session_id is None:
            raise RuntimeError("Video model session not started.")

        request = VideoChunkRequest(
            session_id=self._session_id,
            rig_trajectory=trajectory_local_to_rig,
        )
        request.dynamic_state.CopyFrom(dynamic_state)

        logger.info(
            "Requesting video chunk: chunk_size=%d, is_first=%s, dynamic_actors=%d",
            self.chunk_size,
            self._is_first_chunk,
            len(request.dynamic_state.actors),
        )

        await self._broadcast(LogEntry(video_model_chunk_request=request))
        response: VideoChunkReturn = await profiled_rpc_call(
            "render_video_chunk",
            "video_model",
            self.stub.render_video_chunk,
            request,
        )
        await self._broadcast(LogEntry(video_model_chunk_return=response))

        rgb_frames_per_camera: dict[str, list[ImageWithMetadata]] = {}
        hdmap_frames_per_camera: dict[str, list[ImageWithMetadata]] = {}

        for camera_output in response.camera_outputs:
            cam_id = camera_output.camera_logical_id
            cam_rgb: list[ImageWithMetadata] = []
            cam_hdmap: list[ImageWithMetadata] = []

            for i, rgb_frame in enumerate(camera_output.rgb_frames):
                timestamp_us = self._request_frame_timestamp(i, trajectory_local_to_rig)
                cam_rgb.append(
                    ImageWithMetadata(
                        start_timestamp_us=timestamp_us,
                        end_timestamp_us=timestamp_us,
                        image_bytes=rgb_frame.data,
                        camera_logical_id=cam_id,
                    )
                )

            rgb_frames_per_camera[cam_id] = cam_rgb

            for i, hdmap_frame in enumerate(camera_output.hdmap_condition_frames):
                ts = (
                    cam_rgb[i].start_timestamp_us
                    if i < len(cam_rgb)
                    else self._request_frame_timestamp(i, trajectory_local_to_rig)
                )
                cam_hdmap.append(
                    ImageWithMetadata(
                        start_timestamp_us=ts,
                        end_timestamp_us=ts,
                        image_bytes=hdmap_frame.data,
                        camera_logical_id=f"hdmap_{cam_id}",
                    )
                )

            if cam_hdmap:
                hdmap_frames_per_camera[cam_id] = cam_hdmap

        self._is_first_chunk = False

        return ChunkResult(
            rgb_frames_per_camera=rgb_frames_per_camera,
            hdmap_frames_per_camera=hdmap_frames_per_camera,
            raw_request=request,
            raw_response=response,
        )

    def _request_frame_timestamp(
        self,
        frame_index: int,
        request_trajectory: Trajectory,
    ) -> int:
        if frame_index < len(request_trajectory.poses):
            return request_trajectory.poses[frame_index].timestamp_us
        raise RuntimeError(
            f"Frame {frame_index} has no timestamp in request trajectory"
        )

    def _placeholder_chunk_result(
        self,
        request_trajectory: Trajectory,
    ) -> ChunkResult:
        """Return timestamped empty frames for skip-mode rollout timing."""
        rgb_frames_per_camera: dict[str, list[ImageWithMetadata]] = {}
        for camera in self._runtime_cameras:
            rgb_frames_per_camera[camera.logical_id] = [
                ImageWithMetadata(
                    start_timestamp_us=pose.timestamp_us,
                    end_timestamp_us=pose.timestamp_us,
                    image_bytes=b"",
                    camera_logical_id=camera.logical_id,
                )
                for pose in request_trajectory.poses
            ]
        return ChunkResult(rgb_frames_per_camera=rgb_frames_per_camera)

    def reset_session_state(self) -> None:
        self._session_id = None
        self._is_first_chunk = True
        self._runtime_cameras = []

    async def _broadcast(self, entry: LogEntry) -> None:
        if self.session_info is None:
            return
        await self.session_info.broadcaster.broadcast(entry)

    async def _initialize_session(
        self, session_info: SessionInfo, **kwargs: Any
    ) -> None:
        """Initialize a video-model rollout session.

        Registers synthetic camera definitions in ``camera_catalog`` for this
        scene, extracts the HD map and first frames from the USDZ artifact,
        and opens a session with the remote video-model server.
        """
        await super()._initialize_session(session_info=session_info)
        self.reset_session_state()

        cfg = session_info.session_config
        if not isinstance(cfg, RendererSessionConfig):
            # Direct-RPC/test caller; skip session bootstrap.
            return

        self._runtime_cameras = list(cfg.runtime_cameras)

        if self.skip:
            return

        if self._camera_catalog is None:
            raise RuntimeError(
                "VideoModelService requires a CameraCatalog to bootstrap a session"
            )

        usdz_path = self._usdz_path_from_data_source(cfg)
        scene_id = cfg.data_source.scene_id

        self._register_scene_cameras(scene_id, usdz_path)

        camera_specs, rig_to_camera, initial_frames = (
            build_camera_specs_and_initial_frames(
                runtime_cameras=cfg.runtime_cameras,
                camera_catalog=self._camera_catalog,
                scene_id=scene_id,
                usdz_path=usdz_path,
            )
        )
        hdmap_bytes = extract_hdmap_for_video_model(usdz_path)
        await self.start_session(
            hdmap_bytes=hdmap_bytes,
            camera_specs=camera_specs,
            rig_to_camera=rig_to_camera,
            initial_frames=initial_frames,
        )

    @staticmethod
    def _usdz_path_from_data_source(cfg: RendererSessionConfig) -> str:
        usdz_path = str(getattr(cfg.data_source, "source", "") or "")
        if not usdz_path:
            raise RuntimeError(
                "VideoModelService requires an artifact-backed SceneDataSource "
                "with a source USDZ path"
            )
        return usdz_path

    def _register_scene_cameras(self, scene_id: str, usdz_path: str) -> None:
        """Populate ``CameraCatalog._scene_definitions`` from the USDZ.

        Parses ``clipgt/calibration_estimate.parquet`` to recover the real
        ftheta intrinsics and rig-to-camera extrinsics that the recorded
        scene was captured with, then applies any per-camera local overrides
        (``resolution_hw`` only) on top.

        Cameras present in ``CameraCatalog._local_overrides`` must exist in
        the USDZ calibration. Subset selection should be done via
        ``simulation_config.cameras``; local camera definitions are overrides,
        so an unknown ID is a configuration error.
        """
        assert self._camera_catalog is not None

        scene_defs = parse_cameras_from_usdz(usdz_path)

        for logical_id, override in self._camera_catalog.get_local_overrides().items():
            self._validate_camera_override(override)
            base = scene_defs.get(logical_id)
            if base is None:
                raise ValueError(
                    f"Local camera definition {logical_id!r} has no matching "
                    "camera in USDZ calibration for video_model_config. "
                    "Use simulation_config.cameras to select a subset of the "
                    f"available cameras: {sorted(scene_defs.keys())}."
                )
            scene_defs[logical_id] = self._apply_override(base, override)

        self._camera_catalog.register_scene_definitions(scene_id, scene_defs)

    @staticmethod
    def _apply_override(base: CameraDefinition, override: Any) -> CameraDefinition:
        """Apply a partial ``CameraDefinitionConfig`` on top of a USDZ camera.

        Video-model camera definitions start from the recorded USDZ calibration.
        The initial JPEG frames in the USDZ are tied to that calibration, so
        only resolution overrides are allowed; the server scales the seed image
        to the requested resolution.
        """
        VideoModelService._validate_camera_override(override)

        intrinsics = sensorsim_pb2.CameraSpec()
        intrinsics.CopyFrom(base.intrinsics)

        if override.resolution_hw is not None:
            intrinsics.resolution_h = override.resolution_hw[0]
            intrinsics.resolution_w = override.resolution_hw[1]

        return CameraDefinition(
            logical_id=base.logical_id,
            intrinsics=intrinsics,
            rig_to_camera=base.rig_to_camera.clone(),
        )

    @staticmethod
    def _validate_camera_override(override: Any) -> None:
        disallowed_fields = [
            field
            for field in ("rig_to_camera", "intrinsics", "shutter_type")
            if getattr(override, field, None) is not None
        ]
        if not disallowed_fields:
            return

        raise ValueError(
            "For video_model_config, camera definition override "
            f"for {override.logical_id!r} may only set logical_id and "
            "resolution_hw. The USDZ first-frame JPEGs are tied to the "
            "recorded camera calibration; unsupported override field(s): "
            f"{', '.join(disallowed_fields)}."
        )

    async def _cleanup_session(self, session_info: SessionInfo, **kwargs: Any) -> None:
        if self.skip:
            self.reset_session_state()
            return
        if self._session_id is not None and self.stub is not None:
            request = SessionCloseRequest(session_id=self._session_id.session_id)
            try:
                await self._broadcast(
                    LogEntry(video_model_session_close_request=request)
                )
                await profiled_rpc_call(
                    "close_session",
                    "video_model",
                    self.stub.close_session,
                    request,
                )
            except grpc.aio.AioRpcError as e:
                logger.warning("Failed to close video model session: %s", e)
        self.reset_session_state()
