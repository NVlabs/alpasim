# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import math
import threading
from pathlib import Path
from typing import Any, Optional

from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc
from alpasim_runtime.errors import UnknownSceneError
from alpasim_runtime.scene_loader import ArtifactSceneProvider, SceneLoader
from alpasim_trafficsim.catk.scene_adapter import (
    CATKSceneAdapter,
    preprocess_runtime_map,
)
from alpasim_trafficsim.grpc import API_VERSION_MESSAGE, VersionId
from alpasim_trafficsim.grpc.catk_predictor import CATKTrafficPredictor
from alpasim_trafficsim.grpc.config import CatkConfig, CatkLoaderConfig
from alpasim_trafficsim.grpc.pipeline.env_builder import (
    InsufficientEgoTrajectoryError,
    ensure_time_axis_length,
    populate_ego_future_from_trajectory,
)
from alpasim_trafficsim.grpc.pipeline.response_builder import build_simulation_response
from alpasim_trafficsim.grpc.service_structures import SessionState, SimEnvData
from alpasim_trafficsim.grpc.session import factory
from alpasim_trafficsim.grpc.session.history import (
    build_resampled_env_data,
    merge_env_step_trajectories,
    merge_object_trajectory_updates,
)
from alpasim_utils.geometry import trajectory_from_grpc
from loguru import logger

import grpc


def _gap_seconds(later_us: int | None, earlier_us: int | None) -> float | None:
    if later_us is None or earlier_us is None:
        return None
    return (int(later_us) - int(earlier_us)) / 1e6


def _prediction_step_count(actions: dict[str, Any], *, fallback: int) -> int:
    pred_xyz = actions.get("agent_future_xyz")
    if pred_xyz is None or not hasattr(pred_xyz, "shape") or len(pred_xyz.shape) < 2:
        return fallback
    return max(int(pred_xyz.shape[1]), fallback)


def preprocess_session_map(
    env_data: SimEnvData,
    *,
    loader_cfg: CatkLoaderConfig,
    minimum_history_length: int,
) -> None:
    """Apply TarCache-style runtime map preprocessing in place."""
    curr_t = int(env_data["env"].get("curr_t", minimum_history_length - 1))
    curr_t = max(curr_t, 0)
    preprocess_runtime_map(
        env_data["map"],
        ego_xyz=env_data["ego"]["xyz"][: curr_t + 1],
        agents_xyz=env_data["agents"]["xyz"][:, : curr_t + 1],
        agents_valid_mask=env_data["agents"]["valid_mask"][:, : curr_t + 1],
        map_element_names=loader_cfg.map_element_names,
        map_polyline_filter_mode=loader_cfg.map_polyline_filter_mode,
        map_max_pts_to_ego_distance=loader_cfg.map_max_pts_to_ego_distance,
        map_polyline_number_control_mode=loader_cfg.map_polyline_number_control_mode,
        map_adv_max_lane_polylines_num=loader_cfg.map_adv_max_lane_polylines_num,
        map_adv_max_road_boundary_num=loader_cfg.map_adv_max_road_boundary_num,
        map_adv_max_other_polylines_num=loader_cfg.map_adv_max_other_polylines_num,
    )


class CatkPredictionUnavailableError(RuntimeError):
    """Raised when CATK cannot produce predictions for the current request."""


class TrafficServiceServicer(traffic_pb2_grpc.TrafficServiceServicer):
    def __init__(
        self,
        server: Optional[grpc.Server] = None,
        *,
        catk_config: CatkConfig,
        usdz_folder: str | Path,
        service_version: str = "simple-traffic-service",
    ) -> None:
        self._server = server
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}
        self._service_version = service_version
        self._time_step_s = catk_config.loader.time_step
        self._dt_us = int(round(self._time_step_s * 1e6))
        self._loader_cfg = catk_config.loader
        self._minimum_future_steps = self._loader_cfg.minimum_future_steps
        self._minimum_history_length = self._loader_cfg.num_history_steps
        self._scene_loader = SceneLoader(
            ArtifactSceneProvider.from_path(
                usdz_folder,
                smooth_trajectories=False,
            )
        )
        self._scene_adapter = CATKSceneAdapter(
            num_history_steps=self._loader_cfg.num_history_steps,
            motion_stepsize=self._loader_cfg.time_step,
            map_distance_x=self._loader_cfg.map_distance_x,
            map_distance_y=self._loader_cfg.map_distance_y,
            map_polyline_length_k=self._loader_cfg.map_polyline_length_k,
            map_resample_interval_m=self._loader_cfg.map_resample_interval_m,
        )
        self._catk_predictor = CATKTrafficPredictor(catk_config)

    def _preprocess_session_map(self, env_data: SimEnvData) -> None:
        preprocess_session_map(
            env_data,
            loader_cfg=self._loader_cfg,
            minimum_history_length=self._minimum_history_length,
        )

    def _future_step_indices_from_history_window(
        self,
        *,
        current_step_idx: int,
        current_ts_us: int,
        query_ts_us: int,
    ) -> list[int]:
        if query_ts_us <= current_ts_us:
            return []
        num_future_steps = math.ceil((query_ts_us - current_ts_us) / self._dt_us)
        return list(
            range(current_step_idx + 1, current_step_idx + num_future_steps + 1)
        )

    def start_session(
        self,
        request: traffic_pb2.TrafficSessionRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.SessionRequestStatus:
        if not request.session_uuid:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("session_uuid is required")
            return common_pb2.SessionRequestStatus()
        if not request.scene_id:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("scene_id is required")
            return common_pb2.SessionRequestStatus()
        if not request.logged_object_trajectories:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("at least one logged_object_trajectory is required")
            return common_pb2.SessionRequestStatus()

        try:
            data_source = self._scene_loader.get_data_source(request.scene_id)
            base_env_data = self._scene_adapter.load(data_source)
        except (KeyError, ValueError, FileNotFoundError, UnknownSceneError) as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(
                f"scene_id {request.scene_id!r} could not be loaded: {exc}"
            )
            return common_pb2.SessionRequestStatus()

        try:
            session_state = factory.build_session_state(
                request,
                base_env_data=base_env_data,
                dt_us=self._dt_us,
                minimum_history_length=self._minimum_history_length,
            )
            env_data = session_state.env_data
            self._preprocess_session_map(env_data)
        except (KeyError, ValueError) as exc:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"could not build session from request: {exc}")
            return common_pb2.SessionRequestStatus()

        first_ego_pose_ts_us = factory.first_ego_pose_ts_us(
            session_state.closed_loop_trajectories
        )
        initial_ts_us = session_state.current_ts_us
        num_static_agents = sum(
            bool(v) for v in env_data["env"].get("agent_is_static", [])
        )

        with self._lock:
            self._sessions[request.session_uuid] = session_state

        logger.info(
            "start_session: session={} scene_id={} initial_ts_us={} first_ego_pose_ts_us={} "
            "handover_time_us={} handover_minus_first_ego_s={} "
            "handover_minus_initial_s={} initial_minus_first_ego_s={} "
            "random_seed={} num_logged_objects={} num_static_agents={}",
            request.session_uuid,
            request.scene_id,
            initial_ts_us,
            first_ego_pose_ts_us,
            session_state.handover_time_us,
            _gap_seconds(session_state.handover_time_us, first_ego_pose_ts_us),
            _gap_seconds(session_state.handover_time_us, initial_ts_us),
            _gap_seconds(initial_ts_us, first_ego_pose_ts_us),
            request.random_seed,
            len(request.logged_object_trajectories),
            num_static_agents,
        )
        return common_pb2.SessionRequestStatus()

    def close_session(
        self,
        request: traffic_pb2.TrafficSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> common_pb2.Empty:
        with self._lock:
            removed = self._sessions.pop(request.session_uuid, None)

            if removed is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Unknown session_uuid: {request.session_uuid}")
                return common_pb2.Empty()

        logger.info("close_session: session={}", request.session_uuid)
        return common_pb2.Empty()

    def _apply_model_predictions(
        self,
        *,
        session_uuid: str,
        query_ts_us: int,
        session_state: SessionState,
        future_step_indices: list[int],
    ) -> list[int]:
        if not future_step_indices:
            return []

        env_data = session_state.env_data
        current_step_idx = int(env_data["env"].get("curr_t", 0))
        predict_steps = max(len(future_step_indices), self._minimum_future_steps)
        actions = self._catk_predictor.run_inference(
            env_data,
            predict_steps=predict_steps,
        )
        if actions is None:
            raise CatkPredictionUnavailableError(
                "CATK did not produce predictions for the current request"
            )

        forecast_steps = _prediction_step_count(actions, fallback=predict_steps)
        forecast_step_indices = list(
            range(current_step_idx + 1, current_step_idx + forecast_steps + 1)
        )
        for step_idx in forecast_step_indices:
            ensure_time_axis_length(env_data, step_idx)
        applied_step_indices = self._catk_predictor.apply_predictions_to_env(
            session_state,
            future_step_indices=forecast_step_indices,
            actions=actions,
        )
        if applied_step_indices is None:
            applied_step_indices = forecast_step_indices
        env_data["env"]["curr_t"] = future_step_indices[-1]
        return list(applied_step_indices)

    def _simulate_logged_replay(
        self,
        *,
        session_state: SessionState,
        session_uuid: str,
        query_ts_us: int,
    ) -> traffic_pb2.TrafficReturn:
        env_data = build_resampled_env_data(
            session_state,
            end_ts_us=query_ts_us,
            history_steps=self._minimum_history_length,
            dt_us=self._dt_us,
        )
        session_state.env_data = env_data
        result, _response_current_ts_us = build_simulation_response(
            session_uuid=session_uuid,
            env_data=env_data,
            query_ts_us=query_ts_us,
            future_step_indices=[],
            dt_us=self._dt_us,
            minimum_history_length=self._minimum_history_length,
        )
        session_state.current_ts_us = int(query_ts_us)
        return result

    def simulate(
        self,
        request: traffic_pb2.TrafficRequest,
        context: grpc.ServicerContext,
    ) -> traffic_pb2.TrafficReturn:
        with self._lock:
            query_ts_us = request.time_query_us
            session_state = self._sessions.get(request.session_uuid)
            if session_state is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Unknown session_uuid: {request.session_uuid}")
                return traffic_pb2.TrafficReturn()

            request_update_trajectories = {
                update.object_id: trajectory_from_grpc(update.trajectory)
                for update in request.object_trajectory_updates
            }
            merge_object_trajectory_updates(
                session_state.closed_loop_trajectories,
                request.object_trajectory_updates,
            )

            if query_ts_us <= session_state.handover_time_us:
                return self._simulate_logged_replay(
                    session_state=session_state,
                    session_uuid=request.session_uuid,
                    query_ts_us=query_ts_us,
                )

            history_end_ts_us = int(session_state.current_ts_us)
            if history_end_ts_us < session_state.handover_time_us < int(query_ts_us):
                history_end_ts_us = session_state.handover_time_us

            try:
                env_data = build_resampled_env_data(
                    session_state,
                    end_ts_us=history_end_ts_us,
                    history_steps=self._minimum_history_length,
                    dt_us=self._dt_us,
                )
                session_state.env_data = env_data
                current_step_idx = self._minimum_history_length - 1

                future_step_indices = self._future_step_indices_from_history_window(
                    current_step_idx=current_step_idx,
                    current_ts_us=history_end_ts_us,
                    query_ts_us=query_ts_us,
                )
                ego_trajectory = (
                    request_update_trajectories.get("EGO")
                    or session_state.closed_loop_trajectories["EGO"]
                )
                populate_ego_future_from_trajectory(
                    env_data,
                    ego_trajectory,
                    current_step_idx=current_step_idx,
                    requested_timestamp_us=query_ts_us,
                    future_step_indices=future_step_indices,
                    dt_us=self._dt_us,
                )

                forecast_step_indices = self._apply_model_predictions(
                    session_uuid=request.session_uuid,
                    query_ts_us=query_ts_us,
                    session_state=session_state,
                    future_step_indices=future_step_indices,
                )

                result, _response_current_ts_us = build_simulation_response(
                    session_uuid=request.session_uuid,
                    env_data=env_data,
                    query_ts_us=query_ts_us,
                    future_step_indices=future_step_indices,
                    forecast_step_indices=forecast_step_indices,
                    dt_us=self._dt_us,
                    minimum_history_length=self._minimum_history_length,
                )
                merge_env_step_trajectories(
                    session_state.closed_loop_trajectories,
                    env_data,
                    step_indices=future_step_indices,
                    dt_us=self._dt_us,
                    include_ego=True,
                )
                merge_env_step_trajectories(
                    session_state.closed_loop_trajectories,
                    env_data,
                    step_indices=forecast_step_indices,
                    dt_us=self._dt_us,
                    include_ego=False,
                )
                merge_object_trajectory_updates(
                    session_state.closed_loop_trajectories,
                    result.object_trajectory_updates,
                )
            except CatkPredictionUnavailableError as exc:
                logger.warning(
                    "simulate rejected: session={} query_ts_us={} reason={}",
                    request.session_uuid,
                    query_ts_us,
                    exc,
                )
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(str(exc))
                return traffic_pb2.TrafficReturn()
            except InsufficientEgoTrajectoryError as exc:
                logger.warning(
                    "simulate rejected: session={} query_ts_us={} reason={}",
                    request.session_uuid,
                    query_ts_us,
                    exc,
                )
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(str(exc))
                return traffic_pb2.TrafficReturn()
            except Exception as exc:  # noqa: BLE001 - surface as a gRPC status
                logger.exception(
                    "simulate failed: session={} query_ts_us={}",
                    request.session_uuid,
                    query_ts_us,
                )
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"simulation step failed: {exc}")
                return traffic_pb2.TrafficReturn()

            session_state.current_ts_us = int(query_ts_us)
            return result

    def get_metadata(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> traffic_pb2.TrafficModuleMetadata:
        del request, context
        return traffic_pb2.TrafficModuleMetadata(
            version_id=VersionId(
                version_id=self._service_version,
                git_hash="",
                grpc_api_version=API_VERSION_MESSAGE,
            ),
            minimum_history_length_us=self._minimum_history_length * self._dt_us,
        )

    def get_available_scenes(
        self,
        request: common_pb2.Empty,
        context: grpc.ServicerContext,
    ) -> common_pb2.AvailableScenesReturn:
        del request, context
        return common_pb2.AvailableScenesReturn(scene_ids=self._scene_loader.scene_ids)

    def shut_down(self, request, context):
        context.add_callback(self._shut_down)
        return common_pb2.Empty()

    def _shut_down(self):
        self._server.stop(0)
