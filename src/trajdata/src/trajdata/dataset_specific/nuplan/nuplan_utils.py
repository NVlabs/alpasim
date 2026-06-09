# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import glob
import logging
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Final, Generator, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.scene_metadata import Scene
from trajdata.maps import TrafficLightStatus, VectorMap
from trajdata.maps.vec_map import split_lane_segments
from trajdata.maps.vec_map_elements import (
    MapElementType,
    PedCrosswalk,
    PedWalkway,
    Polyline,
    RoadArea,
    RoadLane,
)
from trajdata.utils import map_utils

logger = logging.getLogger(__name__)

NUPLAN_DT: Final[float] = 0.05
NUPLAN_FULL_MAP_NAME_DICT: Final[Dict[str, str]] = {
    "boston": "us-ma-boston",
    "singapore": "sg-one-north",
    "las_vegas": "us-nv-las-vegas-strip",
    "pittsburgh": "us-pa-pittsburgh-hazelwood",
}
_NUPLAN_SQL_MAP_FRIENDLY_NAMES_DICT: Final[Dict[str, str]] = {
    "us-ma-boston": "boston",
    "sg-one-north": "singapore",
    "las_vegas": "las_vegas",
    "us-pa-pittsburgh-hazelwood": "pittsburgh",
}
NUPLAN_LOCATIONS: Final[Tuple[str, str, str, str]] = tuple(
    NUPLAN_FULL_MAP_NAME_DICT.keys()
)
NUPLAN_MAP_VERSION: Final[str] = "nuplan-maps-v1.0"

NUPLAN_TRAFFIC_STATUS_DICT: Final[Dict[str, TrafficLightStatus]] = {
    "green": TrafficLightStatus.GREEN,
    "red": TrafficLightStatus.RED,
    "unknown": TrafficLightStatus.UNKNOWN,
}
# Lidar-to-ego extrinsics for the NuPlan real split (trainval/test).
# Source: NuPlan devkit sensor calibration; verify against your dataset version
# if using a different split or sensor configuration.
NUPLAN_REAL_LIDAR2EGO_ROTATION = [
    -0.0016505558783280307,
    -0.00023289146777086609,
    0.003725490480134295,
    0.9999916710390838,
]
NUPLAN_REAL_LIDAR2EGO_TRANSLATION = [1.5185133218765259, 0.0, 1.6308990716934204]


class NuPlanObject:
    def __init__(
        self,
        dataset_path: Path,
        subfolder: str,
        central_tokens_config: Optional[List[Dict[str, Any]]] = None,
        num_timesteps_before: int = 30,
        num_timesteps_after: int = 80,
    ) -> None:
        """
        Args:
            dataset_path: Root path of the NuPlan dataset.
            subfolder: Subfolder name (e.g. "test", "train").
            central_tokens_config: Optional list of central tokens configurations.
            num_timesteps_before: Default number of timesteps before the central token.
            num_timesteps_after: Default number of timesteps after the central token.
        """
        self.base_path: Path = dataset_path / subfolder

        self.connection: sqlite3.Connection = None
        self.cursor: sqlite3.Cursor = None

        self.num_timesteps_before = num_timesteps_before
        self.num_timesteps_after = num_timesteps_after
        self.central_tokens_config: List[Dict[str, Any]] = central_tokens_config or []

        # Load scenes based on mode
        if self.use_central_tokens:
            self.scenes: List[Dict[str, str]] = self._load_scenes_from_central_tokens()
        else:
            self.scenes: List[Dict[str, str]] = self._load_scenes()

    @property
    def use_central_tokens(self) -> bool:
        return bool(self.central_tokens_config)

    def open_db(self, db_filename: str) -> None:
        self.connection = sqlite3.connect(str(self.base_path / db_filename))
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

    def execute_query_one(
        self, query_text: str, query_params: Optional[Iterable] = None
    ) -> sqlite3.Row:
        self.cursor.execute(
            query_text, query_params if query_params is not None else []
        )
        return self.cursor.fetchone()

    def execute_query_all(
        self, query_text: str, query_params: Optional[Iterable] = None
    ) -> List[sqlite3.Row]:
        self.cursor.execute(
            query_text, query_params if query_params is not None else []
        )
        return self.cursor.fetchall()

    def execute_query_iter(
        self, query_text: str, query_params: Optional[Iterable] = None
    ) -> Generator[sqlite3.Row, None, None]:
        self.cursor.execute(
            query_text, query_params if query_params is not None else []
        )

        for row in self.cursor:
            yield row

    def _load_scenes_from_central_tokens(self) -> List[Dict[str, str]]:
        """
        Create scenes based on a central token and a range of timesteps before and after it.
        central_tokens_config format: [
            {
                "central_token": "hex_string",  # central token as hex string
                "logfile": "log_filename",      # corresponding log file name (without .db extension)
            },
            ...
        ]
        """
        scenes: List[Dict[str, str]] = []

        for config in self.central_tokens_config:
            central_token_hex = config["central_token"]
            logfile = config["logfile"]

            # Convert hex string to bytearray.
            central_token = bytearray.fromhex(central_token_hex)

            # Open the corresponding database.
            db_path = self.base_path / f"{logfile}.db"
            if not db_path.exists():
                print(f"Warning: Database file {db_path} not found, skipping...")
                continue

            self.open_db(f"{logfile}.db")

            # Query the timestamp and location corresponding to the central token.
            central_token_query = """
            SELECT  lpc.timestamp,
                    log.location,
                    log.logfile
            FROM lidar_pc AS lpc
            LEFT JOIN scene AS sc ON lpc.scene_token = sc.token
            LEFT JOIN log ON sc.log_token = log.token
            WHERE lpc.token = ?
            """
            central_row = self.execute_query_one(central_token_query, (central_token,))

            if central_row is None:
                print(
                    f"Warning: Central token {central_token_hex} not found in {logfile}.db, skipping..."
                )
                self.close_db()
                continue

            central_timestamp = central_row["timestamp"]
            location = central_row["location"]
            logfile_name = central_row["logfile"]

            # Query all lidar_pc_tokens in the specified range.
            # First get all lidar_pc timestamps and then filter those within the range.
            range_query = """
            SELECT  lpc.token,
                    lpc.timestamp
            FROM lidar_pc AS lpc
            LEFT JOIN scene AS sc ON lpc.scene_token = sc.token
            LEFT JOIN log ON sc.log_token = log.token
            WHERE log.logfile = ?
            ORDER BY lpc.timestamp ASC
            """
            all_frames = self.execute_query_all(range_query, (logfile_name,))

            # Find the position of the central token in the list.
            central_idx = None
            for idx, row in enumerate(all_frames):
                if row["token"] == central_token:
                    central_idx = idx
                    break

            if central_idx is None:
                print(
                    f"Warning: Could not find central token in ordered list, skipping..."
                )
                self.close_db()
                continue

            # Compute the range and store it in the config.
            start_idx = max(0, central_idx - self.num_timesteps_before)
            end_idx = min(len(all_frames), central_idx + self.num_timesteps_after + 1)

            # Store start_idx and end_idx in the config.
            config["start_idx"] = start_idx
            config["end_idx"] = end_idx

            num_timesteps = end_idx - start_idx

            # Create scene name using the central token format.
            scene_name = f"{logfile_name}-{central_token_hex}"

            scenes.append(
                {
                    "name": scene_name,
                    "location": _NUPLAN_SQL_MAP_FRIENDLY_NAMES_DICT.get(
                        location, location
                    ),
                    "num_timesteps": num_timesteps,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

            self.close_db()

        return scenes

    def _load_scenes(self) -> List[Dict[str, str]]:
        scene_info_query = """
        SELECT  sc.token AS scene_token,
                log.location,
                log.logfile,
                (
                    SELECT count(*)
                    FROM lidar_pc AS lpc
                    WHERE lpc.scene_token = sc.token
                ) AS num_timesteps
        FROM scene AS sc
        LEFT JOIN log ON sc.log_token = log.token
        """
        scenes: List[Dict[str, str]] = []

        for log_filename in glob.glob(str(self.base_path / "*.db")):
            self.open_db(log_filename)

            for row in self.execute_query_iter(scene_info_query):
                scenes.append(
                    {
                        "name": f"{row['logfile']}={row['scene_token'].hex()}",
                        "location": _NUPLAN_SQL_MAP_FRIENDLY_NAMES_DICT[
                            row["location"]
                        ],
                        "num_timesteps": row["num_timesteps"],
                    }
                )

            self.close_db()

        return scenes

    def get_scene_frames(self, scene: Scene) -> pd.DataFrame:
        """
        Get scene frames. Automatically detects scene type and calls the appropriate method.

        Scene naming conventions:
        - Old format (full scene): "logfile=scene_token"
        - New format (central token): "logfile-central_token"
        """
        # Detect scene type based on naming convention
        if "=" in scene.name:
            # Old format: full scene
            return self._get_scene_frames_full(scene)
        else:
            # New format: central token
            # Check if this is a central token scene by looking for start_idx/end_idx
            if hasattr(scene, "data_access_info") and isinstance(
                scene.data_access_info, dict
            ):
                if (
                    "start_idx" in scene.data_access_info
                    or "end_idx" in scene.data_access_info
                ):
                    return self._get_scene_frames_with_central_token(scene)

            # Also check central_tokens_config
            if self.use_central_tokens:
                return self._get_scene_frames_with_central_token(scene)
            else:
                # Fallback to full scene (in case scene uses "-" but is not central token)
                return self._get_scene_frames_full(scene)

    def _get_scene_frames_full(self, scene: Scene) -> pd.DataFrame:
        """
        Get all frames for a complete scene (original/legacy method).
        Scene name format: "logfile=scene_token"
        """
        query = """
        SELECT  lpc.token AS lpc_token,
                ep.x AS ego_x,
                ep.y AS ego_y,
                ep.z AS ego_z,
                ep.qw AS ego_qw,
                ep.qx AS ego_qx,
                ep.qy AS ego_qy,
                ep.qz AS ego_qz,
                ep.vx AS ego_vx,
                ep.vy AS ego_vy,
                ep.acceleration_x AS ego_ax,
                ep.acceleration_y AS ego_ay
        FROM lidar_pc AS lpc
        LEFT JOIN ego_pose AS ep ON lpc.ego_pose_token = ep.token
        WHERE scene_token = ?
        ORDER BY lpc.timestamp ASC;
        """
        # Parse scene name: "logfile=scene_token"
        log_filename, scene_token_str = scene.name.split("=")
        scene_token = bytearray.fromhex(scene_token_str)

        return pd.read_sql_query(
            query, self.connection, index_col="lpc_token", params=(scene_token,)
        )

    def _get_scene_frames_with_central_token(self, scene: Scene) -> pd.DataFrame:
        """
        Get frames for a central token scene (new method).
        Scene name format: "logfile-central_token"
        """
        log_filename, scene_token_str = scene.name.rsplit("-", 1)
        scene_token = bytearray.fromhex(scene_token_str)

        # Look up the corresponding start_idx and end_idx from the config.
        start_idx = None
        end_idx = None

        # Find the matching config.
        for config in self.central_tokens_config:
            if (
                config.get("central_token") == scene_token_str
                and config.get("logfile") == log_filename
            ):
                start_idx = config.get("start_idx")
                end_idx = config.get("end_idx")
                break

        # If not found, try to get them from scene.data_access_info (if stored previously).
        if start_idx is None or end_idx is None:
            if hasattr(scene, "data_access_info") and isinstance(
                scene.data_access_info, dict
            ):
                start_idx = scene.data_access_info.get("start_idx")
                end_idx = scene.data_access_info.get("end_idx")

        # If still not found, raise an error.
        if start_idx is None or end_idx is None:
            raise ValueError(
                f"Could not find start_idx and end_idx for scene {scene.name}. "
                f"Please ensure the scene was created with central token configuration."
            )

        range_query = """
            SELECT  lpc.token,
                    lpc.timestamp
            FROM lidar_pc AS lpc
            LEFT JOIN scene AS sc ON lpc.scene_token = sc.token
            LEFT JOIN log ON sc.log_token = log.token
            WHERE log.logfile = ?
            ORDER BY lpc.timestamp ASC
            """

        all_frames = self.execute_query_all(range_query, (log_filename,))
        target_tokens = [row["token"] for row in all_frames[start_idx:end_idx]]

        if not target_tokens:
            raise ValueError(f"No tokens found in range [{start_idx}, {end_idx})")

        # Query frame data corresponding to these tokens.
        query = f"""
        SELECT  lpc.token AS lpc_token,
                ep.x AS ego_x,
                ep.y AS ego_y,
                ep.z AS ego_z,
                ep.qw AS ego_qw,
                ep.qx AS ego_qx,
                ep.qy AS ego_qy,
                ep.qz AS ego_qz,
                ep.vx AS ego_vx,
                ep.vy AS ego_vy,
                ep.acceleration_x AS ego_ax,
                ep.acceleration_y AS ego_ay
        FROM lidar_pc AS lpc
        LEFT JOIN ego_pose AS ep ON lpc.ego_pose_token = ep.token
        WHERE lpc.token IN ({('?,'*len(target_tokens))[:-1]})
        ORDER BY lpc.timestamp ASC;
        """

        return pd.read_sql_query(
            query, self.connection, index_col="lpc_token", params=target_tokens
        )

    def get_detected_agents(self, binary_lpc_tokens: List[bytearray]) -> pd.DataFrame:
        query = f"""
        SELECT  lb.lidar_pc_token,
                lb.track_token,
                (SELECT category.name FROM category WHERE category.token = tr.category_token) AS category_name,
                tr.width,
                tr.length,
                tr.height,
                lb.x,
                lb.y,
                lb.z,
                lb.vx,
                lb.vy,
                lb.yaw
        FROM lidar_box AS lb
        LEFT JOIN track AS tr ON lb.track_token = tr.token

        WHERE lidar_pc_token IN ({('?,'*len(binary_lpc_tokens))[:-1]}) AND category_name IN ('vehicle', 'bicycle', 'pedestrian')
        """
        return pd.read_sql_query(query, self.connection, params=binary_lpc_tokens)

    def get_traffic_light_status(
        self, binary_lpc_tokens: List[bytearray]
    ) -> pd.DataFrame:
        query = f"""
        SELECT  tls.lidar_pc_token AS lidar_pc_token,
                tls.lane_connector_id AS lane_id,
                tls.status AS raw_status
        FROM traffic_light_status AS tls
        WHERE lidar_pc_token IN ({('?,'*len(binary_lpc_tokens))[:-1]});
        """
        df = pd.read_sql_query(query, self.connection, params=binary_lpc_tokens)
        df["status"] = df["raw_status"].map(NUPLAN_TRAFFIC_STATUS_DICT)
        df["lane_id"] = df["lane_id"].astype(str)
        return df.drop(columns=["raw_status"])

    def get_sensor_calibration(self, log_filename: str) -> Dict[str, Any]:
        """
        Extract camera and lidar calibration information from the NuPlan database.

        Args:
            log_filename: Log file name (without .db extension).

        Returns:
            Dict containing 'cameras' and 'lidar' calibration info.
        """
        sensor_calib: Dict[str, Any] = {
            "cameras": {},
            "lidar": {
                "channel": "LIDAR_TOP",
                "sensor2ego_rotation": np.array(NUPLAN_REAL_LIDAR2EGO_ROTATION),
                "sensor2ego_translation": np.array(NUPLAN_REAL_LIDAR2EGO_TRANSLATION),
            },
        }
        try:
            import pickle

            # NuPlan stores these camera calibration fields as pickle-backed SQL
            # columns in the official ORM. See nuplan.database.common.sql_types
            # SimplePickleType and nuplan.database.nuplan_db_orm.camera.Camera.
            # This mirrors the official deserialization path, so the NuPlan SQLite
            # DB is treated as trusted input and must not be an arbitrary DB file.
            # Check whether the database connection is already open.
            if not self.connection.in_transaction:
                db_path = self.base_path / f"{log_filename}.db"
                if not db_path.exists():
                    return sensor_calib
                self.open_db(f"{log_filename}.db")
                should_close = True
            else:
                should_close = False

            # Query log token.
            log_query = "SELECT token FROM log WHERE logfile = ?"
            log_row = self.execute_query_one(log_query, (log_filename,))
            if log_row is None:
                if should_close:
                    self.close_db()
                return sensor_calib

            log_token = log_row["token"]

            # Query the camera table (if it exists).
            try:
                camera_query = """
                SELECT channel, translation, rotation, intrinsic, distortion, height, width
                FROM camera
                WHERE log_token = ?
                """
                camera_rows = self.execute_query_all(camera_query, (log_token,))

                for row in camera_rows:
                    cam_name = row["channel"]

                    # Parse pickle serialized data.
                    try:
                        # translation is nuplan.database.common.data_types.Translation.
                        trans_obj = pickle.loads(row["translation"])
                        if hasattr(trans_obj, "__iter__") and not isinstance(
                            trans_obj, str
                        ):
                            translation = np.array(list(trans_obj))
                        else:
                            translation = (
                                np.array([trans_obj.x, trans_obj.y, trans_obj.z])
                                if hasattr(trans_obj, "x")
                                else None
                            )
                    except Exception:
                        logger.warning(
                            "Failed to parse translation for camera %s in %s",
                            cam_name,
                            log_filename,
                        )
                        translation = None

                    try:
                        # rotation is nuplan.database.common.data_types.Rotation (quaternion).
                        rot_obj = pickle.loads(row["rotation"])
                        if hasattr(rot_obj, "__iter__") and not isinstance(
                            rot_obj, str
                        ):
                            rotation = np.array(list(rot_obj))
                        elif hasattr(rot_obj, "w"):
                            rotation = np.array(
                                [rot_obj.w, rot_obj.x, rot_obj.y, rot_obj.z]
                            )
                        elif hasattr(rot_obj, "quaternion"):
                            q = rot_obj.quaternion
                            rotation = (
                                np.array([q.w, q.x, q.y, q.z])
                                if hasattr(q, "w")
                                else None
                            )
                        else:
                            rotation = None
                    except Exception:
                        logger.warning(
                            "Failed to parse rotation for camera %s in %s",
                            cam_name,
                            log_filename,
                        )
                        rotation = None

                    try:
                        # intrinsic is nuplan.database.common.data_types.CameraIntrinsic.
                        intrinsic_obj = pickle.loads(row["intrinsic"])
                        if isinstance(intrinsic_obj, np.ndarray):
                            intrinsic = intrinsic_obj
                        elif hasattr(intrinsic_obj, "__iter__") and not isinstance(
                            intrinsic_obj, str
                        ):
                            intrinsic = np.array(intrinsic_obj)
                        else:
                            intrinsic = None
                        # Ensure it is a 3x3 matrix.
                        if intrinsic is not None and intrinsic.shape != (3, 3):
                            intrinsic = (
                                intrinsic.reshape(3, 3) if intrinsic.size == 9 else None
                            )
                    except Exception:
                        logger.warning(
                            "Failed to parse intrinsic for camera %s in %s",
                            cam_name,
                            log_filename,
                        )
                        intrinsic = None

                    try:
                        # distortion is a list.
                        distortion_obj = pickle.loads(row["distortion"])
                        if isinstance(distortion_obj, np.ndarray):
                            distortion = distortion_obj
                        elif isinstance(distortion_obj, (list, tuple)):
                            distortion = np.array(distortion_obj)
                        else:
                            distortion = None
                    except Exception:
                        logger.warning(
                            "Failed to parse distortion for camera %s in %s",
                            cam_name,
                            log_filename,
                        )
                        distortion = None

                    sensor_calib["cameras"][cam_name] = {
                        "channel": cam_name,
                        "sensor2ego_rotation": rotation,
                        "sensor2ego_translation": translation,
                        "intrinsic": intrinsic,
                        "distortion": distortion,
                        "height": 1080,
                        "width": 1920,
                    }
            except sqlite3.OperationalError:
                logger.debug(
                    "Camera table not found in %s.db, skipping camera calibration",
                    log_filename,
                )
            except Exception:
                logger.warning(
                    "Unexpected error reading camera calibration from %s.db",
                    log_filename,
                    exc_info=True,
                )

        except Exception:
            logger.warning(
                "Failed to load sensor calibration for %s", log_filename, exc_info=True
            )

        return sensor_calib

    def close_db(self) -> None:
        self.cursor.close()
        self.connection.close()


def nuplan_type_to_unified_type(nuplan_type: str) -> AgentType:
    # TODO map traffic cones, barriers to static; generic_object to pedestrian
    if nuplan_type == "pedestrian":
        return AgentType.PEDESTRIAN
    elif nuplan_type == "bicycle":
        return AgentType.BICYCLE
    elif nuplan_type == "vehicle":
        return AgentType.VEHICLE
    else:
        return AgentType.UNKNOWN


def create_splits_logs() -> Dict[str, List[str]]:
    try:
        import nuplan.planning.script.config.common as common_cfg
    except ModuleNotFoundError:
        logger.warning(
            "NuPlan devkit is not installed; using empty nuPlan log split metadata. "
            "This is valid for fully cached datasets, but rebuilding from raw NuPlan "
            "data requires the NuPlan devkit."
        )
        return {}

    yaml_filepath = Path(common_cfg.__path__[0]) / "splitter" / "nuplan.yaml"
    try:
        with open(yaml_filepath, "r") as stream:
            splits = yaml.safe_load(stream)
    except FileNotFoundError:
        logger.warning(
            "NuPlan splitter config %s was not found; using empty nuPlan log split "
            "metadata. This is valid for fully cached datasets, but rebuilding from "
            "raw NuPlan data requires a complete NuPlan devkit installation.",
            yaml_filepath,
        )
        return {}

    return splits["log_splits"]


def extract_lane_and_edges(
    nuplan_map: NuPlanMap, lane_record, lane_connector_idxs: pd.Series
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, Tuple[str, str]]:
    lane_midline = np.stack(lane_record["geometry"].xy, axis=-1)

    # Getting the bounding polygon vertices.
    boundary_df = nuplan_map._vector_map["boundaries"]
    if np.isfinite(lane_record["lane_fid"]):
        fid = str(int(lane_record["lane_fid"]))
        lane_info = nuplan_map._vector_map["lanes_polygons"].loc[fid]
    elif np.isfinite(lane_record["lane_connector_fid"]):
        fid = int(lane_record["lane_connector_fid"])
        lane_info = nuplan_map._vector_map[
            "gen_lane_connectors_scaled_width_polygons"
        ].iloc[lane_connector_idxs[fid]]
    else:
        raise ValueError("Both lane_fid and lane_connector_fid are NaN!")

    lane_fid = str(fid)
    boundary_info = (
        str(lane_info["left_boundary_fid"]),
        str(lane_info["right_boundary_fid"]),
    )

    left_pts = np.stack(boundary_df.loc[boundary_info[0]]["geometry"].xy, axis=-1)
    right_pts = np.stack(boundary_df.loc[boundary_info[1]]["geometry"].xy, axis=-1)

    # Final ordering check, ensuring that left_pts and right_pts can be combined
    # into a polygon without the endpoints intersecting.
    # Reversing the one lane edge that does not match the ordering of the midline.
    if map_utils.endpoints_intersect(left_pts, right_pts):
        if not map_utils.order_matches(left_pts, lane_midline):
            left_pts = left_pts[::-1]
        else:
            right_pts = right_pts[::-1]

    # Ensuring that left and right have the same number of points.
    # This is necessary, not for data storage but for later rasterization.
    if left_pts.shape[0] < right_pts.shape[0]:
        left_pts = map_utils.interpolate(left_pts, num_pts=right_pts.shape[0])
    elif right_pts.shape[0] < left_pts.shape[0]:
        right_pts = map_utils.interpolate(right_pts, num_pts=left_pts.shape[0])

    return (lane_fid, lane_midline, left_pts, right_pts, boundary_info)


def extract_area(nuplan_map: NuPlanMap, area_record) -> np.ndarray:
    return np.stack(area_record["geometry"].exterior.xy, axis=-1)


def populate_vector_map(
    vector_map: VectorMap,
    nuplan_map: NuPlanMap,
    lane_connector_idxs: pd.Series,
    max_lane_length: Optional[float] = None,
) -> None:
    # Setting the map bounds.
    # NOTE: min_pt is especially important here since the world coordinates of nuPlan
    # are quite large in magnitude. We make them relative to the bottom-left by
    # subtracting all positions by min_pt and registering that offset as part of
    # the map_from_world (and related) transforms later.
    min_pt = np.min(
        [
            layer_df["geometry"].total_bounds[:2]
            for layer_df in nuplan_map._vector_map.values()
        ],
        axis=0,
    )
    max_pt = np.max(
        [
            layer_df["geometry"].total_bounds[2:]
            for layer_df in nuplan_map._vector_map.values()
        ],
        axis=0,
    )

    # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    vector_map.extent = np.array(
        [
            min_pt[0],
            min_pt[1],
            0.0,
            max_pt[0],
            max_pt[1],
            0.0,
        ]
    )

    overall_pbar = tqdm(
        total=len(nuplan_map._vector_map["baseline_paths"])
        + len(nuplan_map._vector_map["drivable_area"])
        + len(nuplan_map._vector_map["crosswalks"])
        + len(nuplan_map._vector_map["walkways"]),
        desc=f"Getting {nuplan_map.map_name} Elements",
        position=1,
        leave=False,
    )

    # This dict stores boundary IDs and which lanes are to the left and right of them.
    boundary_connectivity_dict: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # This dict stores lanes' boundary IDs.
    lane_boundary_dict: Dict[str, Tuple[str, str]] = dict()
    for _, lane_info in nuplan_map._vector_map["baseline_paths"].iterrows():
        (
            lane_id,
            center_pts,
            left_pts,
            right_pts,
            boundary_info,
        ) = extract_lane_and_edges(nuplan_map, lane_info, lane_connector_idxs)

        lane_boundary_dict[lane_id] = boundary_info
        left_boundary_id, right_boundary_id = boundary_info

        # The left boundary of Lane A has Lane A to its right.
        boundary_connectivity_dict[left_boundary_id]["right"].append(lane_id)

        # The right boundary of Lane A has Lane A to its left.
        boundary_connectivity_dict[right_boundary_id]["left"].append(lane_id)

        # Find road areas that this lane intersects for faster lane-based lookup later.
        intersect_filt = nuplan_map._vector_map["drivable_area"].intersects(
            lane_info["geometry"]
        )
        isnear_filt = (
            nuplan_map._vector_map["drivable_area"].distance(lane_info["geometry"])
            < 3.0
        )
        road_area_ids = set(
            nuplan_map._vector_map["drivable_area"][intersect_filt | isnear_filt][
                "fid"
            ].values
        )
        if not road_area_ids:
            print(f"Warning: no road lane associated with lane {lane_id}")

        # "partial" because we aren't adding lane connectivity until later.
        partial_new_lane = RoadLane(
            id=lane_id,
            center=Polyline(center_pts),
            left_edge=Polyline(left_pts),
            right_edge=Polyline(right_pts),
            road_area_ids=road_area_ids,
        )
        if max_lane_length is not None:
            split_lanes = split_lane_segments(partial_new_lane, max_len=max_lane_length)
            for lane in split_lanes:
                vector_map.add_map_element(lane)
                lane_boundary_dict[lane.id] = boundary_info
        else:
            vector_map.add_map_element(partial_new_lane)
        overall_pbar.update()

    for fid, polygon_info in nuplan_map._vector_map["drivable_area"].iterrows():
        polygon_pts = extract_area(nuplan_map, polygon_info)

        new_road_area = RoadArea(id=fid, exterior_polygon=Polyline(polygon_pts))
        for hole in polygon_info["geometry"].interiors:
            hole_pts = extract_area(nuplan_map, hole)
            new_road_area.interior_holes.append(Polyline(hole_pts))

        vector_map.add_map_element(new_road_area)
        overall_pbar.update()

    for fid, ped_area_record in nuplan_map._vector_map["crosswalks"].iterrows():
        polygon_pts = extract_area(nuplan_map, ped_area_record)

        new_ped_crosswalk = PedCrosswalk(id=fid, polygon=Polyline(polygon_pts))
        vector_map.add_map_element(new_ped_crosswalk)
        overall_pbar.update()

    for fid, ped_area_record in nuplan_map._vector_map["walkways"].iterrows():
        polygon_pts = extract_area(nuplan_map, ped_area_record)

        new_ped_walkway = PedWalkway(id=fid, polygon=Polyline(polygon_pts))
        vector_map.add_map_element(new_ped_walkway)
        overall_pbar.update()

    overall_pbar.close()

    # Lane connectivity
    lane_connectivity_exit_dict = defaultdict(list)
    lane_connectivity_entry_dict = defaultdict(list)
    for lane_connector_fid, lane_connector in tqdm(
        nuplan_map._vector_map["lane_connectors"].iterrows(),
        desc="Getting Lane Connectivity",
        total=len(nuplan_map._vector_map["lane_connectors"]),
        position=1,
        leave=False,
    ):
        lane_connectivity_exit_dict[str(lane_connector["exit_lane_fid"])].append(
            lane_connector_fid
        )
        lane_connectivity_entry_dict[lane_connector_fid].append(
            str(lane_connector["exit_lane_fid"])
        )

        lane_connectivity_exit_dict[lane_connector_fid].append(
            str(lane_connector["entry_lane_fid"])
        )
        lane_connectivity_entry_dict[str(lane_connector["entry_lane_fid"])].append(
            lane_connector_fid
        )

    map_elem: RoadLane
    for map_elem in tqdm(
        vector_map.elements[MapElementType.ROAD_LANE].values(),
        desc="Storing Lane Connectivity",
        position=1,
        leave=False,
    ):
        map_elem.prev_lanes.update(lane_connectivity_entry_dict[map_elem.id])
        map_elem.next_lanes.update(lane_connectivity_exit_dict[map_elem.id])

        lane_id: str = map_elem.id
        left_boundary_id, right_boundary_id = lane_boundary_dict[lane_id]

        map_elem.adj_lanes_left.update(
            boundary_connectivity_dict[left_boundary_id]["left"]
        )
        map_elem.adj_lanes_right.update(
            boundary_connectivity_dict[right_boundary_id]["right"]
        )
