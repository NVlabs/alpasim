# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Navigation conditioning in the Alpamayo prediction path.

Uses a stub model so the route hand-off and the generation budget can be tested
without a checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model
from alpasim_driver.models.alpamayo_base import AlpamayoBaseModel, route_in_rig_at_t0
from alpasim_driver.models.base import DriveCommand, PredictionInput
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3
from alpasim_grpc.v0.egodriver_pb2 import Route
from alpasim_utils.geometry import Pose as GeometryPose

CAMERA_ID = "camera_front_wide_120fov"
NUM_WAYPOINTS = 20
SPEED_M_S = 10.0


class _StubAlpamayo(AlpamayoBaseModel):
    """Alpamayo variant recording what its message and sampler receive."""

    def __init__(self) -> None:
        self.nav_text: str | None = None
        self.sampler_kwargs: dict = {}
        self._init_common(
            model=SimpleNamespace(
                action_space=SimpleNamespace(
                    get_action_space_dims=lambda: (NUM_WAYPOINTS, 3), dt=0.1
                ),
                sample_trajectories_from_data_with_vlm_rollout=self._sample,
            ),
            processor=SimpleNamespace(apply_chat_template=lambda *args, **kwargs: {}),
            helper_module=SimpleNamespace(to_device=lambda inputs, device: inputs),
            device=torch.device("cpu"),
            camera_ids=[CAMERA_ID],
            context_length=1,
        )

    def _create_chat_message(
        self, image_frames: torch.Tensor, nav_text: str | None
    ) -> list:
        self.nav_text = nav_text
        return []

    def _sample(self, **kwargs) -> tuple:
        self.sampler_kwargs = kwargs
        positions = torch.zeros(1, 1, 1, NUM_WAYPOINTS, 3)
        positions[..., 0] = torch.arange(1, NUM_WAYPOINTS + 1)
        rotations = torch.eye(3).expand(1, 1, 1, NUM_WAYPOINTS, 3, 3)
        return positions, rotations, {}


def _poses(latest_us: int) -> list[PoseAtTime]:
    """Ego poses driving straight along the local x axis."""
    return [
        PoseAtTime(
            timestamp_us=timestamp_us,
            pose=Pose(
                vec=Vec3(x=SPEED_M_S * timestamp_us / 1e6, y=0.0, z=0.0),
                quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        )
        for timestamp_us in range(0, latest_us + 1, 100_000)
    ]


def _route(timestamp_us: int, waypoints: np.ndarray) -> Route:
    return Route(
        timestamp_us=timestamp_us,
        waypoints=[Vec3(x=x, y=y, z=z) for x, y, z in waypoints],
    )


def _prediction_input(latest_us: int, route: Route | None) -> PredictionInput:
    return PredictionInput(
        camera_images={CAMERA_ID: [(latest_us, np.zeros((8, 8, 3), dtype=np.uint8))]},
        command=DriveCommand.STRAIGHT,
        speed=SPEED_M_S,
        acceleration=0.0,
        ego_pose_history=_poses(latest_us),
        inference_seed=0,
        previous_plan=None,
        route=route,
    )


def _turn_left_route() -> np.ndarray:
    """Route running 16 m ahead, then turning left."""
    ahead = [(x, 0.0, 0.0) for x in np.arange(0.0, 20.0, 4.0)]
    left = [(16.0, y, 0.0) for y in np.arange(4.0, 24.0, 4.0)]
    return np.array(ahead + left)


def test_route_reaches_the_model_as_navigation_text() -> None:
    model = _StubAlpamayo()

    model.predict(_prediction_input(2_000_000, _route(2_000_000, _turn_left_route())))

    assert model.nav_text == "Turn left in 16m"


def test_without_a_route_the_model_runs_unconditioned() -> None:
    model = _StubAlpamayo()

    model.predict(_prediction_input(2_000_000, route=None))

    assert model.nav_text is None


def test_a_route_without_waypoints_leaves_the_model_unconditioned() -> None:
    model = _StubAlpamayo()

    model.predict(_prediction_input(2_000_000, _route(2_000_000, np.zeros((0, 3)))))

    assert model.nav_text is None


def test_route_is_re_expressed_at_the_prediction_time() -> None:
    """A route older than t0 is corrected for the ego motion since it arrived."""
    latest_us = 2_000_000
    route = _route(latest_us - 1_000_000, _turn_left_route())

    model = _StubAlpamayo()
    model.predict(_prediction_input(latest_us, route))

    # The ego covered 10 m in the second between the route and t0, so the turn
    # that was 16 m ahead is now 6 m ahead.
    assert model.nav_text == "Turn left in 7m"


def test_route_waypoints_move_with_the_ego() -> None:
    waypoints = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    latest_us = 2_000_000

    in_rig_t0 = route_in_rig_at_t0(
        _route(latest_us - 500_000, waypoints),
        _poses(latest_us),
        pose_local_to_rig_t0=GeometryPose(
            np.array([SPEED_M_S * latest_us / 1e6, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
    )

    # Half a second of travel at 10 m/s pulls the waypoints 5 m closer.
    np.testing.assert_allclose(in_rig_t0[:, 0], [5.0, 15.0], atol=1e-3)


def test_chain_of_thought_budget_is_not_left_to_the_checkpoint() -> None:
    """The fallback budget truncates the reasoning before the trajectory."""
    model = _StubAlpamayo()

    model.predict(_prediction_input(2_000_000, route=None))

    assert model.sampler_kwargs["max_generation_length"] == 256


class _StubAlpamayo15(Alpamayo15Model):
    """Alpamayo 1.5 without a checkpoint, recording which sampler runs."""

    def __init__(self, cfg_guidance_weight: float | None) -> None:
        self._cfg_guidance_weight = cfg_guidance_weight
        self._top_p = 0.98
        self._temperature = 0.6
        self._num_traj_samples = 1
        self.guided_kwargs: dict | None = None
        self._model = SimpleNamespace(
            sample_trajectories_from_data_with_vlm_rollout=lambda **kwargs: "unguided",
            sample_trajectories_from_data_with_vlm_rollout_cfg_nav=self._guided,
        )

    def _guided(self, **kwargs) -> str:
        self.guided_kwargs = kwargs
        return "guided"


def test_navigation_guidance_passes_its_weight_to_the_sampler() -> None:
    model = _StubAlpamayo15(cfg_guidance_weight=1.5)

    assert model._run_inference({}, nav_text="Turn left in 16m") == "guided"
    assert model.guided_kwargs["diffusion_kwargs"] == {
        "use_classifier_free_guidance": True,
        "inference_guidance_weight": 1.5,
    }


def test_navigation_guidance_needs_an_instruction_to_guide_towards() -> None:
    model = _StubAlpamayo15(cfg_guidance_weight=1.5)

    assert model._run_inference({}, nav_text=None) == "unguided"


def test_guidance_is_off_by_default() -> None:
    model = _StubAlpamayo15(cfg_guidance_weight=None)

    assert model._run_inference({}, nav_text="Turn left in 16m") == "unguided"
