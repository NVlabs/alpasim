# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import yaml
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.config import (
    PhysicsUpdateMode,
    RuntimeCameraConfig,
    SimulationConfig,
    VehicleConfig,
)
from alpasim_runtime.ego_masks import load_ego_mask_overrides
from alpasim_runtime.errors import InvalidSceneError
from alpasim_runtime.events.camera import make_initial_sensorsim_render_event
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.types import RuntimeCamera
from alpasim_runtime.unbound_rollout import UnboundRollout
from alpasim_utils.geometry import Pose, Trajectory, trajectory_to_grpc
from alpasim_utils.scenario import AABB, CameraId, Rig, TrafficObject, TrafficObjects
from alpasim_utils.types import T0_AUTO, ClipDataSource, Scene


def _scene(
    policy_start_us: int | Literal["AUTO"] = T0_AUTO,
    *,
    clip_id: str = "scene",
) -> Scene:
    return Scene(ClipDataSource(clip_id, "artifact-uuid"), policy_start_us)


def _pose_at_x(x: float) -> Pose:
    return Pose(
        np.array([x, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def _trajectory(timestamps_us: list[int]) -> Trajectory:
    return Trajectory.from_poses(
        timestamps=np.array(timestamps_us, dtype=np.uint64),
        poses=[_pose_at_x(float(ts) / 1e6) for ts in timestamps_us],
    )


def _artifact(timestamps_us: list[int] | None = None) -> SimpleNamespace:
    if timestamps_us is None:
        timestamps_us = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
    rig = Rig(
        sequence_id="clipgt-test",
        trajectory=_trajectory(timestamps_us),
        camera_ids=[
            CameraId("camera_front", 0, "clipgt-test", "unique-front"),
            CameraId("camera_left", 0, "clipgt-test", "unique-left"),
        ],
        camera_frame_timestamps_us={
            "unique-front": [200_000, 300_000],
            "unique-left": [150_000, 250_000],
        },
        camera_frame_ranges_us={
            "unique-front": [range(170_000, 200_000), range(270_000, 300_000)],
            "unique-left": [range(120_000, 150_000), range(220_000, 250_000)],
        },
        world_to_nre=np.eye(4),
        vehicle_config=VehicleConfig(),
    )
    traffic_objects = TrafficObjects(
        actor=TrafficObject(
            track_id="actor",
            aabb=AABB(1.0, 1.0, 1.0),
            trajectory=_trajectory(timestamps_us),
            is_static=False,
            label_class="vehicle",
        )
    )
    metadata = SimpleNamespace(
        logger=SimpleNamespace(run_id="run-1"),
        version_string="0.0.0-test",
        uuid="artifact-uuid",
    )
    return SimpleNamespace(
        source_id="scene",
        rig=rig,
        traffic_objects=traffic_objects,
        source="artifact.usdz",
        metadata=metadata,
        map=None,
    )


def _simulation_config(**overrides) -> SimulationConfig:
    cfg = SimulationConfig(
        n_sim_steps=2,
        n_rollouts=1,
        control_timestep_us=100_000,
        force_gt_duration_us=100_000,
        skip_driver_during_force_gt=True,
        min_traffic_duration_us=0,
        cameras=[
            RuntimeCameraConfig(logical_id="camera_front"),
            RuntimeCameraConfig(logical_id="camera_left"),
        ],
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _sensorsim_renderer() -> SensorsimService:
    return SensorsimService(
        address="localhost:0",
        skip=True,
        camera_catalog=SimpleNamespace(),
    )


@pytest.mark.parametrize("policy_start_us", [T0_AUTO, 300_000])
def test_external_clip_masks_preserve_defaults_and_scene_identity(
    tmp_path, policy_start_us
):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "first": {
                    "rig_config_id": "bank-first",
                    "camera_logical_ids": ["camera_front", "camera_rear"],
                },
                "second": {
                    "rig_config_id": "bank-second",
                    "camera_logical_ids": ["camera_left"],
                },
            }
        )
    )
    cfg = _simulation_config(
        ego_mask_rig_config_id="stock", ego_mask_overrides_file=str(path)
    )
    overrides = load_ego_mask_overrides(str(path))
    path.unlink()  # Rollout construction must not reopen the startup input.
    artifact = _artifact()
    for clip, bank, cameras in [
        ("first", "bank-first", ["camera_front"]),
        ("unlisted", "stock", None),
        ("second", "bank-second", ["camera_left"]),
        ("artifact-uuid", "stock", None),
    ]:
        scene = _scene(policy_start_us, clip_id=clip)
        rollout = UnboundRollout.create(
            cfg,
            scene,
            RolloutMetadata.VersionIds(),
            artifact,
            str(tmp_path),
            _sensorsim_renderer(),
            ego_mask_overrides=overrides,
        )
        assert rollout.scene == scene
        assert rollout.nre_uuid == "artifact-uuid"
        assert rollout.ego_mask_rig_config_id == bank
        assert rollout.ego_mask_camera_logical_ids == cameras
        assert rollout.camera_configs == cfg.cameras
    assert cfg.ego_mask_rig_config_id == "stock"


@pytest.mark.parametrize("default", [None, "stock"])
@pytest.mark.parametrize(
    "mapping",
    [
        None,
        {},
        {"other": {"rig_config_id": "bank", "camera_logical_ids": ["camera_front"]}},
    ],
)
def test_absent_clip_uses_configured_default(tmp_path, mapping, default):
    path = tmp_path / "overrides.yaml"
    if mapping is not None:
        path.write_text(yaml.safe_dump(mapping))
    cfg = _simulation_config(
        ego_mask_rig_config_id=default,
        ego_mask_overrides_file=str(path) if mapping is not None else None,
    )
    rollout = UnboundRollout.create(
        cfg,
        _scene(),
        RolloutMetadata.VersionIds(),
        _artifact(),
        str(tmp_path),
        _sensorsim_renderer(),
        ego_mask_overrides=(
            load_ego_mask_overrides(str(path)) if mapping is not None else None
        ),
    )
    assert rollout.ego_mask_rig_config_id == default
    assert rollout.ego_mask_camera_logical_ids is None


def test_override_with_no_configured_cameras_warns_and_applies_no_patch(
    tmp_path, caplog
):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        yaml.safe_dump(
            {"scene": {"rig_config_id": "bank", "camera_logical_ids": ["camera_rear"]}}
        )
    )
    rollout = UnboundRollout.create(
        _simulation_config(
            ego_mask_rig_config_id="stock", ego_mask_overrides_file=str(path)
        ),
        _scene(),
        RolloutMetadata.VersionIds(),
        _artifact(),
        str(tmp_path),
        _sensorsim_renderer(),
        ego_mask_overrides=load_ego_mask_overrides(str(path)),
    )
    assert "listed cameras ('camera_rear',) do not intersect" in caplog.text
    assert "Ego mask override for scene (bank bank) applies no patch" in caplog.text
    assert rollout.ego_mask_rig_config_id == "bank"
    assert rollout.ego_mask_camera_logical_ids == []


def _ego_history_bytes(history: Trajectory) -> bytes:
    return trajectory_to_grpc(history).SerializeToString()


def test_create_uses_rig_start_for_context_and_closed_loop_after_force_gt(
    tmp_path,
) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.egomotion_context_start_us == 0
    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 250_000
    assert rollout.end_timestamp_us == 350_000
    log_metadata = rollout.get_log_metadata()
    assert log_metadata.start_timestamp_us == 0
    assert log_metadata.render_start_timestamp_us == 150_000
    assert not log_metadata.scene.HasField("policy_start_us")
    assert rollout.force_gt_period == range(150_000, 250_001)
    assert rollout.first_camera_frame_ranges_us["camera_front"] == range(
        170_000, 200_000
    )
    assert rollout.gt_ego_trajectory.time_range_us.start == 0
    assert rollout.traffic_objs["actor"].trajectory.time_range_us.start == 0


def test_create_propagates_disabled_asl_logging(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(write_asl=False),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.write_asl is False


def test_create_explicit_policy_t0_is_not_moved_by_camera_alignment(tmp_path) -> None:
    simulation_config = _simulation_config(
        cameras=[
            RuntimeCameraConfig(
                logical_id="camera_front",
                shutter_duration_us=17_000,
            ),
            RuntimeCameraConfig(
                logical_id="camera_left",
                shutter_duration_us=30_000,
            ),
        ]
    )
    rollout = UnboundRollout.create(
        simulation_config=simulation_config,
        scene=_scene(300_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    # The recorded first frames close at 150k and 200k. Numeric T0 anchors both
    # runtime cameras at policy_start - force_gt = 200k.
    assert rollout.render_start_timestamp_us == 200_000
    assert rollout.closed_loop_start_us == 300_000
    assert rollout.first_camera_frame_ranges_us == {
        "camera_front": range(183_000, 200_000),
        "camera_left": range(170_000, 200_000),
    }
    assert rollout.egomotion_context_start_us == 170_000
    assert rollout.gt_ego_trajectory.time_range_us.start == 170_000
    assert rollout.traffic_objs["actor"].trajectory.time_range_us.start == 170_000

    runtime_cameras = [
        RuntimeCamera.from_camera_config(
            camera_config,
            rollout.first_camera_frame_ranges_us[camera_config.logical_id],
        )
        for camera_config in simulation_config.cameras
    ]
    camera_events = make_initial_sensorsim_render_event(
        scene_start_us=rollout.egomotion_context_start_us,
        render_start_timestamp_us=rollout.render_start_timestamp_us,
        closed_loop_start_us=rollout.closed_loop_start_us,
        simulation_end_us=rollout.end_timestamp_us,
        control_timestep_us=rollout.control_timestep_us,
        runtime_cameras=runtime_cameras,
        renderer_service=_sensorsim_renderer(),
        driver=SimpleNamespace(),
        frame_quality=SimpleNamespace(),
        broadcaster=SimpleNamespace(),
    )
    assert [event.timestamp_us for event in camera_events] == [200_000, 200_000]
    assert [event.trigger.time_range_us for event in camera_events] == [
        range(183_000, 200_000),
        range(170_000, 200_000),
    ]


def test_create_explicit_policy_t0_is_exact_handoff(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(cameras=[]),
        scene=_scene(300_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.egomotion_context_start_us == 200_000
    assert rollout.render_start_timestamp_us == 200_000
    assert rollout.first_policy_timestamp_us == 200_000
    assert rollout.closed_loop_start_us == rollout.scene.policy_start_us
    assert rollout.end_timestamp_us == 400_000
    assert rollout.n_sim_steps == 2
    log_metadata = rollout.get_log_metadata()
    assert log_metadata.scene.policy_start_us == 300_000
    # GT and traffic are trimmed to begin at the offset.
    assert rollout.gt_ego_trajectory.time_range_us.start == 200_000
    assert rollout.traffic_objs["actor"].trajectory.time_range_us.start == 200_000


def test_create_fork_uses_parent_ego_history_for_warmup(tmp_path) -> None:
    history = Trajectory.from_poses(
        np.array([0, 100_000, 170_000, 200_000, 300_000], dtype=np.uint64),
        [_pose_at_x(x) for x in [-3.0, -2.0, -1.0, 40.0, 42.0]],
    )
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            force_gt_frame_cache=SimpleNamespace(
                enabled=True,
                cache_dir="/tmp/cache",
                extra_key=None,
            )
        ),
        scene=_scene(300_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
        ego_history_bytes=_ego_history_bytes(history),
    )

    assert rollout.render_start_timestamp_us == 200_000
    assert rollout.closed_loop_start_us == 300_000
    assert rollout.fork_ego_history is not None
    assert rollout.fork_ego_history.timestamps_us.tolist() == [
        0,
        100_000,
        170_000,
        183_000,
        200_000,
        300_000,
    ]
    assert rollout.fork_ego_history.last_pose.vec3[0] == pytest.approx(42.0)
    assert not rollout.use_cached_frames_during_force_gt


def test_create_fork_requires_post_handoff_control_step(tmp_path) -> None:
    history = _trajectory([100_000, 120_000, 150_000, 250_000, 350_000])

    with pytest.raises(InvalidSceneError, match="post-handoff control step"):
        UnboundRollout.create(
            simulation_config=_simulation_config(force_gt_duration_us=200_000),
            scene=_scene(350_000),
            version_ids=RolloutMetadata.VersionIds(),
            data_source=_artifact(),
            rollouts_dir=str(tmp_path),
            renderer_service=_sensorsim_renderer(),
            ego_history_bytes=_ego_history_bytes(history),
        )


def test_create_fork_near_recording_end_stops_at_recording_end(tmp_path) -> None:
    history = _trajectory([0, 100_000, 200_000, 300_000])

    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(n_sim_steps=5),
        scene=_scene(300_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
        ego_history_bytes=_ego_history_bytes(history),
    )

    assert rollout.closed_loop_start_us == 300_000
    assert rollout.n_sim_steps == 3
    assert rollout.end_timestamp_us == 500_000
    assert rollout.get_log_metadata().n_sim_steps == 3


def test_create_fork_near_recording_end_requires_warmup_history(tmp_path) -> None:
    history = _trajectory([200_000, 300_000])

    with pytest.raises(InvalidSceneError, match="does not cover the warm-up"):
        UnboundRollout.create(
            simulation_config=_simulation_config(n_sim_steps=5),
            scene=_scene(300_000),
            version_ids=RolloutMetadata.VersionIds(),
            data_source=_artifact(),
            rollouts_dir=str(tmp_path),
            renderer_service=_sensorsim_renderer(),
            ego_history_bytes=_ego_history_bytes(history),
        )


def test_create_explicit_policy_t0_uses_recorded_shutter_duration_for_zero_config(
    tmp_path,
) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            cameras=[
                RuntimeCameraConfig(
                    logical_id="camera_front",
                    shutter_duration_us=0,
                ),
                RuntimeCameraConfig(
                    logical_id="camera_left",
                    shutter_duration_us=17_000,
                ),
            ]
        ),
        scene=_scene(300_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    # camera_front's recorded first exposure is 30k, so a zero configured shutter
    # duration uses 30k at the numeric anchor.
    assert rollout.first_camera_frame_ranges_us == {
        "camera_front": range(170_000, 200_000),
        "camera_left": range(183_000, 200_000),
    }
    assert rollout.egomotion_context_start_us == 170_000
    assert rollout.closed_loop_start_us == 300_000


def test_create_explicit_policy_t0_is_relative_to_recording_start(tmp_path) -> None:
    recording_start_us = 1_000_000
    artifact = _artifact(
        [
            recording_start_us + offset_us
            for offset_us in (0, 100_000, 200_000, 300_000, 400_000, 500_000)
        ]
    )

    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(),
        scene=_scene(300_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=artifact,
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.egomotion_context_start_us == recording_start_us + 183_000
    assert rollout.render_start_timestamp_us == recording_start_us + 200_000
    assert rollout.closed_loop_start_us == recording_start_us + 300_000
    assert rollout.n_sim_steps == 2


def test_create_explicit_policy_t0_does_not_require_a_recorded_frame_at_t0(
    tmp_path,
) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(),
        scene=_scene(350_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.render_start_timestamp_us == 250_000
    assert rollout.closed_loop_start_us == 350_000
    assert rollout.first_camera_frame_ranges_us["camera_left"] == range(
        233_000, 250_000
    )


def test_create_explicit_policy_t0_requires_camera_exposure_history(tmp_path) -> None:
    with pytest.raises(InvalidSceneError, match="camera exposure history"):
        UnboundRollout.create(
            simulation_config=_simulation_config(),
            scene=_scene(100_000),
            version_ids=RolloutMetadata.VersionIds(),
            data_source=_artifact(),
            rollouts_dir=str(tmp_path),
            renderer_service=_sensorsim_renderer(),
        )


def test_create_clips_n_sim_steps_to_complete_policy_steps(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(n_sim_steps=10),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.closed_loop_start_us == 250_000
    assert rollout.n_sim_steps == 3
    assert rollout.end_timestamp_us == 450_000
    assert rollout.get_log_metadata().n_sim_steps == 3


def test_create_explicit_t0_requires_full_horizon_even_at_recording_start(
    tmp_path,
) -> None:
    with pytest.raises(InvalidSceneError, match="full configured horizon"):
        UnboundRollout.create(
            simulation_config=_simulation_config(n_sim_steps=10, cameras=[]),
            scene=_scene(100_000),
            version_ids=RolloutMetadata.VersionIds(),
            data_source=_artifact(),
            rollouts_dir=str(tmp_path),
            renderer_service=_sensorsim_renderer(),
        )


def test_create_allows_force_gt_over_entire_rollout(tmp_path) -> None:
    force_gt_duration_us = 100_000_000_000_000
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            force_gt_duration_us=force_gt_duration_us,
            n_sim_steps=2,
        ),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 150_000 + force_gt_duration_us
    assert rollout.end_timestamp_us == 350_000
    assert rollout.closed_loop_start_us > rollout.end_timestamp_us
    assert rollout.n_sim_steps == 2
    assert 250_000 in rollout.force_gt_period
    assert 350_000 in rollout.force_gt_period


def test_create_synchronizes_cameras_for_zero_decision_delay(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(assert_zero_decision_delay=True),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.first_camera_frame_ranges_us["camera_front"] == range(
        133_000, 150_000
    )
    assert rollout.first_camera_frame_ranges_us["camera_left"] == range(
        133_000, 150_000
    )
    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 250_000


def test_create_allows_physics_with_zero_force_gt_duration(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            force_gt_duration_us=0,
            physics_update_mode=PhysicsUpdateMode.EGO_ONLY,
        ),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.render_start_timestamp_us == 150_000
    assert rollout.first_policy_timestamp_us == 150_000
    assert rollout.closed_loop_start_us == 150_000
    assert rollout.force_gt_period == range(150_000, 150_001)
    assert rollout.physics_update_mode == PhysicsUpdateMode.EGO_ONLY


def test_create_keeps_synthetic_first_exposure_inside_rollout_window(
    tmp_path,
) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(
            assert_zero_decision_delay=True,
            cameras=[
                RuntimeCameraConfig(
                    logical_id="camera_front",
                    shutter_duration_us=100_000,
                ),
                RuntimeCameraConfig(
                    logical_id="camera_left",
                    shutter_duration_us=100_000,
                ),
            ],
        ),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.first_camera_frame_ranges_us["camera_front"] == range(
        50_000, 150_000
    )
    assert rollout.first_camera_frame_ranges_us["camera_left"] == range(50_000, 150_000)
    assert rollout.egomotion_context_start_us == 0
    assert rollout.traffic_objs["actor"].trajectory.time_range_us.start == 0


@pytest.mark.parametrize(
    ("exception_type", "expected_type"),
    [
        pytest.param(ValueError, InvalidSceneError, id="invalid-scene"),
        pytest.param(OSError, OSError, id="transient-io"),
        pytest.param(MemoryError, MemoryError, id="transient-memory"),
    ],
)
def test_create_classifies_vector_map_failures(
    tmp_path,
    exception_type: type[Exception],
    expected_type: type[Exception],
) -> None:
    artifact = _artifact()

    class BrokenMapDataSource:
        def __getattr__(self, name: str):
            return getattr(artifact, name)

        @property
        def map(self):
            raise exception_type("map-loading failure")

    with pytest.raises(expected_type):
        UnboundRollout.create(
            simulation_config=_simulation_config(),
            scene=_scene(clip_id="bad-scene"),
            version_ids=RolloutMetadata.VersionIds(),
            data_source=BrokenMapDataSource(),
            rollouts_dir=str(tmp_path),
            renderer_service=_sensorsim_renderer(),
        )


def _nav_route_library(tmp_path, clip_ids: dict[str, str]):
    import pickle

    from alpasim_runtime.nav_route import (
        NavRouteLibrary,
        RouteInstruction,
        SessionRouteIndex,
    )

    route_dir = tmp_path / "routes"
    route_dir.mkdir(exist_ok=True)
    for clip_id in clip_ids:
        # The recording runs 0..0.5 m over 0..500_000 us; the turn sits at 0.4 m.
        route_index = SessionRouteIndex(
            timestamps=np.array([0, 200_000, 400_000], dtype=np.int64),
            cumulative_distances=np.array([10.0, 10.2, 10.4]),
            instructions=[RouteInstruction(10.4, "TURN_LEFT", "Turn left")],
        )
        with open(route_dir / f"{clip_id.removeprefix('clipgt-')}.pkl", "wb") as f:
            pickle.dump(route_index, f)
    return NavRouteLibrary.build(str(route_dir), set(clip_ids))


def test_create_anchors_the_clip_route_for_navigation(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
        nav_routes=_nav_route_library(tmp_path, {"scene": "session-1"}),
    )

    assert rollout.route_navigator is not None
    instruction = rollout.route_navigator.lookup(np.array([0.1, 0.0, 0.0]))
    assert instruction.maneuver == "TURN_LEFT"
    assert instruction.remaining_distance_m == pytest.approx(0.3, abs=1e-5)


def test_explicit_t0_navigation_is_anchored_over_the_whole_recording(
    tmp_path,
) -> None:
    import pickle

    from alpasim_runtime.nav_route import (
        NavRouteLibrary,
        RouteInstruction,
        SessionRouteIndex,
    )

    # Route distances that are not proportional to the recording's arc length, so a
    # trace cropped at the explicit T0 would extrapolate a different distance before
    # the crop than the recording's own anchors give.
    route_dir = tmp_path / "routes"
    route_dir.mkdir()
    with open(route_dir / "scene.pkl", "wb") as f:
        pickle.dump(
            SessionRouteIndex(
                timestamps=np.array([0, 200_000, 400_000], dtype=np.int64),
                cumulative_distances=np.array([10.0, 10.05, 10.4]),
                instructions=[RouteInstruction(10.4, "TURN_LEFT", "Turn left")],
            ),
            f,
        )
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(),
        scene=_scene(300_000),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
        nav_routes=NavRouteLibrary.build(str(route_dir), {"scene"}),
    )

    assert rollout.gt_ego_trajectory.time_range_us.start > 100_000
    assert rollout.route_navigator is not None
    instruction = rollout.route_navigator.lookup(np.array([0.1, 0.0, 0.0]))
    assert instruction.remaining_distance_m == pytest.approx(0.375, abs=1e-5)


def test_create_rejects_a_clip_without_a_route_index(tmp_path) -> None:
    with pytest.raises(InvalidSceneError, match="navigation instructions unavailable"):
        UnboundRollout.create(
            simulation_config=_simulation_config(),
            scene=_scene(),
            version_ids=RolloutMetadata.VersionIds(),
            data_source=_artifact(),
            rollouts_dir=str(tmp_path),
            renderer_service=_sensorsim_renderer(),
            nav_routes=_nav_route_library(tmp_path, {"other-clip": "session-1"}),
        )


def test_create_without_navigation_leaves_no_navigator(tmp_path) -> None:
    rollout = UnboundRollout.create(
        simulation_config=_simulation_config(),
        scene=_scene(),
        version_ids=RolloutMetadata.VersionIds(),
        data_source=_artifact(),
        rollouts_dir=str(tmp_path),
        renderer_service=_sensorsim_renderer(),
    )

    assert rollout.route_navigator is None
