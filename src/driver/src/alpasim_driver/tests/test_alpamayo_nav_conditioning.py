# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Navigation conditioning in the Alpamayo prediction path.

Uses a stub model so the instruction hand-off and the generation budget can be
tested without a checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model
from alpasim_driver.models.alpamayo_base import AlpamayoBaseModel
from alpasim_driver.models.base import DriveCommand, PredictionInput
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat, Vec3
from alpasim_grpc.v0.egodriver_pb2 import NavigationInstruction, Route

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


def _prediction_input(
    latest_us: int,
    route: Route | None,
    navigation: NavigationInstruction | None = None,
) -> PredictionInput:
    return PredictionInput(
        camera_images={CAMERA_ID: [(latest_us, np.zeros((8, 8, 3), dtype=np.uint8))]},
        command=DriveCommand.STRAIGHT,
        speed=SPEED_M_S,
        acceleration=0.0,
        ego_pose_history=_poses(latest_us),
        inference_seed=0,
        previous_plan=None,
        route=route,
        navigation=navigation,
    )


def _turn_left_route() -> np.ndarray:
    """Route running 16 m ahead, then turning left."""
    ahead = [(x, 0.0, 0.0) for x in np.arange(0.0, 20.0, 4.0)]
    left = [(16.0, y, 0.0) for y in np.arange(4.0, 24.0, 4.0)]
    return np.array(ahead + left)


def test_the_navigation_instruction_reaches_the_model_as_text() -> None:
    """The runtime's instruction is already in the trained wording."""
    model = _StubAlpamayo()
    navigation = NavigationInstruction(
        timestamp_us=2_000_000,
        instruction_text="Turn right onto Oak Ave in 42m",
        debug_info=NavigationInstruction.DebugInfo(
            remaining_distance_m=42.0, status=NavigationInstruction.DebugInfo.UPCOMING
        ),
    )

    model.predict(
        _prediction_input(2_000_000, _route(2_000_000, _turn_left_route()), navigation)
    )

    assert model.nav_text == "Turn right onto Oak Ave in 42m"


def test_a_route_alone_leaves_the_model_unconditioned() -> None:
    """Route geometry is turned into text by the runtime, never by the driver."""
    model = _StubAlpamayo()

    model.predict(_prediction_input(2_000_000, _route(2_000_000, _turn_left_route())))

    assert model.nav_text is None


def test_without_navigation_the_model_runs_unconditioned() -> None:
    model = _StubAlpamayo()

    model.predict(_prediction_input(2_000_000, route=None))

    assert model.nav_text is None


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
