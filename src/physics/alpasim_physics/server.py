# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import argparse
import functools
import logging
from concurrent import futures

from alpasim_grpc.v0.common_pb2 import (
    AvailableScenesReturn,
    Empty,
    Pose,
    PoseAtTime,
    Trajectory,
    VersionId,
)
from alpasim_grpc.v0.physics_pb2 import (
    PhysicsGroundIntersectionRequest,
    PhysicsGroundIntersectionReturn,
)
from alpasim_grpc.v0.physics_pb2_grpc import (
    PhysicsServiceServicer,
    add_PhysicsServiceServicer_to_server,
)
from alpasim_physics import VERSION_MESSAGE
from alpasim_physics.backend import PhysicsBackend
from alpasim_physics.utils import (
    aabb_to_ndarray,
    ndarray_to_vec3,
    pose_grpc_to_ndarray,
    pose_status_to_grpc,
    scipy_to_quat,
)
from alpasim_utils.artifact import Artifact
from scipy.spatial.transform import Rotation as R

import grpc

logger = logging.getLogger(__name__)


class PhysicsSimService(PhysicsServiceServicer):
    def __init__(
        self,
        server: grpc.Server,
        artifact_glob: str,
        cache_size: int = 2,
        use_ground_mesh: bool = False,
        visualize: bool = False,
    ) -> None:
        self.server = server
        self.artifacts = Artifact.discover_from_glob(
            artifact_glob, use_ground_mesh=use_ground_mesh
        )

        self.visualize = visualize
        logger.info(f"Available scenes: {list(self.artifacts.keys())}.")

        # instantiate the method here to avoid caching `self`
        @functools.lru_cache(maxsize=cache_size)
        def get_backend(scene_id: str) -> PhysicsBackend:
            if scene_id not in self.artifacts:
                raise KeyError(f"Scene {scene_id=} not available.")

            artifact = self.artifacts[scene_id]
            logger.info(f"Cache miss, loading {artifact.scene_id=}")
            mesh_ply = artifact.mesh_ply

            # don't keep the .ply file once the backend is evicted from the lru_cache
            artifact.clear_cache()

            return PhysicsBackend(
                mesh_ply,
                visualize=self.visualize,
            )

        self.get_backend = get_backend

    def ground_intersection(
        self, request: PhysicsGroundIntersectionRequest, context: grpc.ServicerContext
    ) -> PhysicsGroundIntersectionReturn:
        logger.debug(f"Received request for scene_id={request.scene_id}")
        logger.debug(f"full request={request}")
        try:
            backend = self.get_backend(request.scene_id)

            other_updates = []
            for other in request.other_objects:
                other_updates.append(
                    backend.update_pose(
                        pose_grpc_to_ndarray(other.pose_pair.future_pose),
                        aabb_to_ndarray(other.aabb),
                        request.future_us,
                    )
                )

            if request.HasField("ego_data"):
                ego_aabb = aabb_to_ndarray(request.ego_data.aabb)
                corrected_poses = []
                ego_statuses = []
                for pat in request.ego_data.ego_trajectory_aabb.poses:
                    pose_nd = pose_grpc_to_ndarray(pat.pose)
                    updated_pose, status = backend.update_pose(
                        pose_nd, ego_aabb, pat.timestamp_us
                    )
                    quat = R.from_matrix(updated_pose[:3, :3]).as_quat(canonical=False)
                    corrected_poses.append(
                        PoseAtTime(
                            pose=Pose(
                                vec=ndarray_to_vec3(updated_pose[:3, 3]),
                                quat=scipy_to_quat(quat),
                            ),
                            timestamp_us=pat.timestamp_us,
                        )
                    )
                    ego_statuses.append(status.to_grpc())

                response = PhysicsGroundIntersectionReturn(
                    ego_trajectory_aabb=Trajectory(poses=corrected_poses),
                    ego_status=ego_statuses,
                    other_poses=[
                        pose_status_to_grpc(pose, status)
                        for pose, status in other_updates
                    ],
                )
            else:
                response = PhysicsGroundIntersectionReturn(
                    other_poses=[
                        pose_status_to_grpc(pose, status)
                        for pose, status in other_updates
                    ],
                )
            logger.debug("sending response")
            return response
        except Exception as e:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details(str(e))
            raise

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        logger.info("get_version")
        try:
            return VERSION_MESSAGE
        except Exception as e:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details(str(e))
            raise

    def get_available_scenes(
        self, request: Empty, context: grpc.ServicerContext
    ) -> AvailableScenesReturn:
        logger.info("get_available_scenes")
        return AvailableScenesReturn(scene_ids=list(self.artifacts.keys()))

    def shut_down(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        logger.info("shut_down")
        context.add_callback(self._shut_down)
        return Empty()

    def _shut_down(self) -> None:
        self.server.stop(0)


def parse_args(
    arg_list: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-glob",
        type=str,
        help="Glob expression to find artifacts. Must end in .usdz to find relevant files.",
        required=True,
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--use-ground-mesh", type=bool, default=False)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=16,
        help="Number of scene backends to keep in LRU cache. Set to match or exceed concurrent scenes.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args, overrides = parser.parse_known_args(arg_list)
    return args, overrides


def main(arg_list: list[str] | None = None) -> None:
    args, _ = parse_args(arg_list)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info(f"Identifying as\n{VERSION_MESSAGE}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)

    service = PhysicsSimService(
        server,
        args.artifact_glob,
        cache_size=args.cache_size,
        use_ground_mesh=args.use_ground_mesh,
        visualize=args.visualize,
    )
    add_PhysicsServiceServicer_to_server(service, server)

    logger.info(f"Serving on {address}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
