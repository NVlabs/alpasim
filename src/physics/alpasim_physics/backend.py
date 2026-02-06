# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging
from enum import IntEnum, unique

import numpy as np
import scipy.spatial.transform as scipy_trans
import warp as wp
from alpasim_grpc.v0.physics_pb2 import PhysicsGroundIntersectionReturn
from alpasim_physics.ply_io import load_mesh_vf
from alpasim_physics.utils import batch_so3_trans_2_se3, so3_trans_2_se3

try:
    import polyscope as ps
except ImportError:
    ps = None

logger = logging.getLogger(__name__)


@wp.kernel
def z_ray_mesh_intersections_warp(
    mesh_id: wp.uint64,
    start_pos: wp.array(dtype=wp.vec3),
    dir: wp.array(dtype=wp.vec3),
    max_t: wp.float32,
    return_pos: wp.array(dtype=wp.vec3),
    return_success: wp.array(dtype=wp.bool),
) -> None:
    tid = wp.tid()
    start_i = start_pos[tid]
    dir_i = dir[tid]
    result = wp.mesh_query_ray(mesh_id, start_i, dir_i, max_t)

    return_success[tid] = result.result
    if result.result:
        return_pos[tid] = wp.mesh_eval_position(
            mesh_id, result.face, result.u, result.v
        )
    else:
        return_pos[tid] = wp.vec3(0.0, 0.0, 0.0)


class InsufficientPointsFitPlane(Exception):
    """Raised when there are not enough points to fit a plane.
    Example: 2 points are passed and we're trying to fit a 3D plane."""

    pass


class HighTranslation(Exception):
    """Raised when an unlikely high translation is found."""

    pass


class HighRotation(Exception):
    """Raised when an unlikely high rotation is found."""

    pass


@unique
class GroundIntersectionStatus(IntEnum):
    """Enum to keep track ground intersection status"""

    SUCCESSFUL_UPDATE = 1
    INSUFFICIENT_POINTS_FITPLANE = 2
    HIGH_TRANSLATION = 3
    HIGH_ROTATION = 4

    def to_grpc(self) -> PhysicsGroundIntersectionReturn.Status:
        match self.value:
            case GroundIntersectionStatus.SUCCESSFUL_UPDATE:
                return PhysicsGroundIntersectionReturn.Status.SUCCESSFUL_UPDATE
            case GroundIntersectionStatus.HIGH_ROTATION:
                return PhysicsGroundIntersectionReturn.Status.HIGH_ROTATION
            case GroundIntersectionStatus.HIGH_TRANSLATION:
                return PhysicsGroundIntersectionReturn.Status.HIGH_TRANSLATION
            case GroundIntersectionStatus.INSUFFICIENT_POINTS_FITPLANE:
                return (
                    PhysicsGroundIntersectionReturn.Status.INSUFFICIENT_POINTS_FITPLANE
                )
            case _:
                return PhysicsGroundIntersectionReturn.Status.UNKNOWN


class PhysicsBackend:
    def __init__(
        self,
        env_mesh_ply: bytes,
        visualize: bool = False,
        profile: bool = False,
    ) -> None:
        wp.init()

        self.env_mesh_points, self.env_mesh_indices = load_mesh_vf(env_mesh_ply)
        self.env_mesh = wp.Mesh(
            points=wp.array(self.env_mesh_points, dtype=wp.vec3),
            velocities=None,
            indices=wp.array(self.env_mesh_indices.flatten(), dtype=int),
        )

        self.profile = profile
        self.visualize = visualize
        if self.visualize:

            self.viz_point_radius = 0.0005
            ps.init()
            ps.set_up_dir("z_up")
            ps.set_front_dir("neg_x_front")

            ps.register_surface_mesh(
                "environment mesh", self.env_mesh_points, self.env_mesh_indices
            )

        self.num_random_points = 16
        self.min_intersections = 6
        self.translation_threshold_m = 1.5
        self.rotation_threshold_deg = 10.0

    def update_ego_others(
        self,
        ego_request: dict[str, np.ndarray],
        others_request: list[dict[str, np.ndarray]] = [],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        return self.update_pose(**ego_request)[0], [
            self.update_pose(**other)[0] for other in others_request
        ]

    def update_pose(
        self,
        predicted_pose: np.ndarray,
        aabb: np.ndarray,
        timestamp: int,
    ) -> tuple[np.ndarray, GroundIntersectionStatus]:
        with wp.ScopedTimer("update_pose", active=self.profile, use_nvtx=True):
            if self.visualize:
                ps.register_point_cloud(
                    f"request_pose_{timestamp}",
                    predicted_pose[:3, 3:].T,
                    radius=self.viz_point_radius,
                    color=(0.8, 0, 0),
                )
            # get random points from the bottom of the aabb
            with wp.ScopedTimer(
                "get_random_points_aabb", active=self.profile, use_nvtx=True
            ):
                aabb_bottom_points = self.get_aabb_bottom_points(
                    aabb, self.num_random_points
                )
                aabb_bottom_poses = batch_so3_trans_2_se3(trans=aabb_bottom_points)
                posed_bottom = predicted_pose @ aabb_bottom_poses
                bottom_positions = posed_bottom[:, :3, 3]

            # get closest intersection with mesh on z axis from these points
            with wp.ScopedTimer(
                "ray_mesh_intersection", active=self.profile, use_nvtx=True
            ):
                mesh_id = wp.uint64(self.env_mesh.id)
                start_points = wp.array(
                    np.concatenate((bottom_positions, bottom_positions)), dtype=wp.vec3
                )
                rays_d = wp.array(
                    np.concatenate(
                        (
                            np.repeat(
                                np.array([0.0, 0.0, -1.0])[None, :],
                                num_points := bottom_positions.shape[0],
                                axis=0,
                            ),
                            np.repeat(
                                np.array([0.0, 0.0, 1.0])[None, :],
                                num_points := bottom_positions.shape[0],
                                axis=0,
                            ),
                        )
                    ),
                    dtype=wp.vec3,
                )
                max_t = wp.float32(100.0)

                return_success = wp.array(
                    shape=(2 * num_points), dtype=wp.bool, device="cuda"
                )
                return_pos = wp.array(
                    shape=(2 * num_points), dtype=wp.vec3f, device="cuda"
                )
                wp.launch(
                    z_ray_mesh_intersections_warp,
                    dim=2 * num_points,
                    inputs=[mesh_id, start_points, rays_d, max_t],
                    outputs=[return_pos, return_success],
                )

                mask = return_success.numpy()
                returned_pos_np = return_pos.numpy()

                mask_down = mask[:num_points]
                mask_up = mask[num_points:]

                # grab/combine the up/down points that intersected with the mesh
                ray_intersections = np.concatenate(
                    (
                        returned_pos_np[num_points:][mask_up],
                        returned_pos_np[:num_points][mask_down],
                    )
                )
                bottom_positions_filtered = np.concatenate(
                    (bottom_positions[mask_up], bottom_positions[mask_down])
                )

                try:
                    with wp.ScopedTimer(
                        "fit_planes", active=self.profile, use_nvtx=True
                    ):
                        # fit planes to both sets of points and get transl and rot between them
                        transl, rot = self.fit_planes_get_transl_rot(
                            ray_intersections, bottom_positions_filtered
                        )
                except (HighRotation, HighTranslation, InsufficientPointsFitPlane) as e:
                    # predicted rotation too high
                    updated_pose = predicted_pose
                    logger.exception(e)

                    match e:
                        case HighRotation():
                            status = GroundIntersectionStatus.HIGH_ROTATION
                        case HighTranslation():
                            status = GroundIntersectionStatus.HIGH_TRANSLATION
                        case InsufficientPointsFitPlane():
                            status = (
                                GroundIntersectionStatus.INSUFFICIENT_POINTS_FITPLANE
                            )
                else:
                    # if no exception proceed with update
                    with wp.ScopedTimer(
                        "rotate_and_translate", active=self.profile, use_nvtx=True
                    ):
                        # rotate around predicted pose
                        t = so3_trans_2_se3(trans=predicted_pose[:3, 3], so3=np.eye(3))
                        inv_t = so3_trans_2_se3(
                            trans=-predicted_pose[:3, 3], so3=np.eye(3)
                        )
                        local_rot = t @ rot @ inv_t

                        updated_pose = transl @ local_rot @ predicted_pose
                        status = GroundIntersectionStatus.SUCCESSFUL_UPDATE
                if self.visualize:
                    ps.register_point_cloud(
                        f"returned_pose_{timestamp}",
                        updated_pose[:3, 3:].T,
                        radius=self.viz_point_radius,
                        color=(0.0, 0.8, 0),
                    )
                    ps.show()

        # Physics updates can result in lateral "drift" of the car in some instances.
        # As a workaround, we allow the ground intersection to only
        # affect the z position of the ego vehicle, as well as the rotation.
        updated_pose[:2, 3] = predicted_pose[:2, 3]

        return updated_pose, status

    def get_aabb_bottom_points(
        self, aabb_size: np.ndarray, n_points: int
    ) -> np.ndarray:
        n_points_per_dim = int(np.ceil(n_points ** (1 / 2)))
        p = np.linspace(0, 1, n_points_per_dim)
        X, Y = np.meshgrid(p, p)
        points = np.vstack(
            (
                X.ravel() * aabb_size[0] - aabb_size[0] * 0.5,
                Y.ravel() * aabb_size[1] - aabb_size[1] * 0.5,
                np.full_like(X.ravel(), fill_value=0 - aabb_size[2] * 0.5),
            )
        )

        return points.T

    def fit_planes_get_transl_rot(
        self, ray_intersections: np.ndarray, bottom_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        point_o, normal_o = self.fit_plane(bottom_positions.T)
        point, normal = self.fit_plane(ray_intersections.T)

        displacement = point - point_o

        rot = scipy_trans.Rotation.align_vectors(normal, normal_o)[0]

        if (translation := np.linalg.norm(displacement)) > self.translation_threshold_m:
            raise HighTranslation(
                f"Predicted translation of {translation} meters but maximum "
                f"plausible translation set at {self.translation_threshold_m}"
            )
        if (
            rotation_max := np.abs(rot.as_euler("xyz", degrees=True).max())
        ) > self.rotation_threshold_deg:
            raise HighRotation(
                f"Predicted rotation of {rotation_max} degrees but maximum "
                f"plausible rotation set at {self.rotation_threshold_deg}"
            )

        return so3_trans_2_se3(trans=displacement, so3=np.eye(3)), so3_trans_2_se3(
            trans=np.zeros((3,)), so3=rot.as_matrix()
        )

    @staticmethod
    def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        p, n = fit_plane(points)

        Given an array, points, of shape (d,...)
        representing points in d-dimensional space,
        fit an d-dimensional plane to the points.
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
        """
        points = np.reshape(
            points, (np.shape(points)[0], -1)
        )  # Collapse trailing dimensions

        if not points.shape[0] <= points.shape[1]:
            raise InsufficientPointsFitPlane(
                f"There are only {points.shape[1]} points in {points.shape[0]} dimensions."
            )

        ctr = points.mean(axis=1)
        x = points - ctr[:, np.newaxis]
        M = np.dot(x, x.T)
        return ctr, np.linalg.svd(M)[0][:, -1]
