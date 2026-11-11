# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Check the calibration delivered to drivers against the MTGS render input."""

from types import SimpleNamespace

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2, egodriver_pb2_grpc, sensorsim_pb2
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.session_configs import DriverSessionConfig
from alpasim_utils.geometry import Pose
from alpasim_utils.scenario import AABB

import grpc

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pyquaternion")

from alpasim_mtgs.server.artifact_adapter import (  # noqa: E402
    get_available_cameras_from_data_source,
    rgb_render_request_to_render_state,
)


def _asset(distortion, *, colmap=True, flat=False):
    calibration = {
        "cam_intrinsic": np.array([[910.0, 0, 960.0], [0, 920.0, 540.0], [0, 0, 1]]),
        "distortion": np.asarray(distortion),
    }
    camera = {
        **calibration,
        "sensor2ego_rotation": [1.0, 0.0, 0.0, 0.0],
        "sensor2ego_translation": [0.0, 0.0, 0.0],
        "height": 1080,
        "width": 1920,
    }
    if colmap:
        camera["colmap_param"] = calibration
        # A distinct raw calibration catches mixing raw and refined K/D.
        camera["cam_intrinsic"] = np.eye(3)
        camera["distortion"] = np.zeros(5)
    video = {"frame_infos": [{"cams": {"CAM_F0": camera}}]}
    return SimpleNamespace(video_scene_dict=video if flat else {"video": video})


class _Driver(egodriver_pb2_grpc.EgodriverServiceServicer):
    async def start_session(self, request, context):
        self.request = request
        return common_pb2.SessionRequestStatus()

    async def close_session(self, request, context):
        return common_pb2.Empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("colmap", [False, True])
@pytest.mark.parametrize("count", [4, 5, 8, 12])
async def test_driver_calibration_matches_renderer(colmap, count):
    # Nonzero k3 and higher terms catch incorrect OpenCV/protobuf ordering.
    distortion = np.array(
        [
            -0.12,
            0.035,
            0.001,
            -0.002,
            0.004,
            0.005,
            -0.006,
            0.007,
            0.0001,
            -0.0002,
            0.0003,
            -0.0004,
        ]
    )[:count]
    asset = _asset(distortion, colmap=colmap)
    request = sensorsim_pb2.RGBRenderRequest()
    request.ego_pose.start_pose.quat.w = 1.0
    render_camera = rgb_render_request_to_render_state(request, asset)["cameras"][
        "CAM_F0"
    ]
    np.testing.assert_array_equal(render_camera["distortion"], distortion)

    response = sensorsim_pb2.AvailableCamerasReturn(
        available_cameras=get_available_cameras_from_data_source(asset)
    )
    response = sensorsim_pb2.AvailableCamerasReturn.FromString(
        response.SerializeToString()
    )
    catalog = CameraCatalog([])
    await catalog.merge_local_and_sensorsim_cameras("scene", response.available_cameras)
    config = DriverSessionConfig(
        sensorsim_cameras=[
            c.as_proto() for c in catalog.get_camera_definitions("scene").values()
        ],
        ego_aabb=AABB(4.0, 2.0, 1.5),
        rig_to_ego_aabb=Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
        scene_id="scene",
    )
    driver = _Driver()
    server = grpc.aio.server()
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(driver, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        service = DriverService(f"127.0.0.1:{port}")
        async with service.rollout_session("session", MessageBroadcaster(), config):
            spec = driver.request.rollout_spec.vehicle.available_cameras[0].intrinsics
    finally:
        await server.stop(None)

    assert (spec.resolution_h, spec.resolution_w) == (1080, 1920)
    param = spec.opencv_pinhole_param
    intrinsic = np.array(
        [
            [param.focal_length_x, 0, param.principal_point_x],
            [0, param.focal_length_y, param.principal_point_y],
            [0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(intrinsic, render_camera["intrinsic"])
    assert len(param.radial_coeffs) == 6
    assert len(param.tangential_coeffs) == 2
    assert len(param.thin_prism_coeffs) == 4
    k1, k2, k3, k4, k5, k6 = param.radial_coeffs
    p1, p2 = param.tangential_coeffs
    received_distortion = np.array(
        [k1, k2, p1, p2, k3, k4, k5, k6, *param.thin_prism_coeffs]
    )
    np.testing.assert_array_equal(received_distortion[:count], distortion)
    np.testing.assert_array_equal(received_distortion[count:], 0)

    points = np.array([[-1.0, -0.5, 2.0], [0.8, 0.6, 1.0], [0.1, -0.2, 3.0]])
    expected, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), render_camera["intrinsic"], distortion
    )
    actual, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), intrinsic, received_distortion
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_flat_video_and_explicit_zero_distortion():
    cameras = get_available_cameras_from_data_source(_asset(np.zeros(5), flat=True))
    assert len(cameras) == 1
    param = cameras[0].intrinsics.opencv_pinhole_param
    assert list(param.radial_coeffs) == [0.0] * 6
    assert list(param.tangential_coeffs) == [0.0] * 2
    assert list(param.thin_prism_coeffs) == [0.0] * 4


@pytest.mark.parametrize(
    "distortion", [[], [1.0] * 6, [0.0] * 14, [0, 0, 0, np.nan], [0, 0, 0, np.inf]]
)
def test_unrepresentable_distortion_is_rejected(distortion):
    with pytest.raises(ValueError, match="distortion"):
        get_available_cameras_from_data_source(_asset(distortion))


def test_missing_colmap_distortion_does_not_use_raw_coefficients():
    asset = _asset(np.zeros(5))
    del asset.video_scene_dict["video"]["frame_infos"][0]["cams"]["CAM_F0"][
        "colmap_param"
    ]["distortion"]
    with pytest.raises(KeyError, match="distortion"):
        get_available_cameras_from_data_source(asset)
