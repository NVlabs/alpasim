# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Long Nguyen, Micha Fauth, Bernhard Jaeger, Daniel Dauner,
# Maximilian Igl, Andreas Geiger, Kashyap Chitta

# flake8: noqa F722


import glob
import gzip
import json
import math
import os
import pickle
import typing
from dataclasses import dataclass
from enum import IntEnum, auto

import cv2
import jaxtyping as jt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from beartype import beartype
from omegaconf import OmegaConf
from torch import nn


class CameraPointCloudIndex(IntEnum):
    """Index to access point cloud array of camera."""

    X = 0
    Y = 1
    Z = 2
    UNREAL_SEMANTICS_ID = 3
    UNREAL_INSTANCE_ID = 4


class TargetDataset(IntEnum):
    """Dataset we target to collect data/train/evaluate on."""

    UNKNOWN = 0
    CARLA_LEADERBOARD2_3CAMERAS = auto()
    CARLA_LEADERBOARD2_6CAMERAS = auto()
    NAVSIM_4CAMERAS = auto()
    WAYMO_E2E_2025_3CAMERAS = auto()


class SourceDataset(IntEnum):
    """Dataset intenum a sample originates from."""

    UNKNOWN = 0
    CARLA = auto()
    NAVSIM = auto()
    WAYMO_E2E_2025 = auto()


SOURCE_DATASET_NAME_MAP = {
    SourceDataset.CARLA: "carla",
    SourceDataset.NAVSIM: "navsim",
    SourceDataset.WAYMO_E2E_2025: "waymo_e2e_2025",
}


class RadarLabels(IntEnum):
    """Index to access radar label array."""

    X = 0
    Y = 1
    V = 2
    VALID = 3


class RadarDataIndex(IntEnum):
    """Index to access radar data array."""

    X = 0
    Y = 1
    Z = 2
    V = 3
    SENSOR_ID = 4


class TransfuserBoundingBoxIndex(IntEnum):
    """Index to access bounding array of TransFuser."""

    X = 0
    Y = 1
    W = 2
    H = 3
    YAW = 4
    VELOCITY = 5
    BRAKE = 6
    CLASS = 7
    SCORE = 8  # Only available for prediction
    VISIBLE_PIXELS = 8  # Only available for ground truth
    NUM_POINTS = 9  # Only available for ground truth
    ID = 10  # Only available for ground truth
    NUM_RADAR_POINTS = 11  # Only available for ground truth


class TransfuserBoundingBoxClass(IntEnum):
    """Bounding box classes used in TransFuser."""

    VEHICLE = 0
    WALKER = 1
    TRAFFIC_LIGHT = 2
    STOP_SIGN = 3
    SPECIAL = 4
    OBSTACLE = 5
    PARKING = 6
    BIKER = 7


class CarlaNavigationCommand(IntEnum):
    """Source: https://carla.org/Doxygen/html/d0/db7/namespacecarla_1_1traffic__manager.html#a5734807dba08623eeca046a963ade360"""  # noqa E501

    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class ChaffeurNetBEVSemanticClass(IntEnum):
    """Indicies to access BEV semantic map produced by ChaffeurNet."""

    UNLABELED = 0
    ROAD = 1
    SIDEWALK = 2
    LANE_MARKERS = 3
    LANE_MARKERS_BROKEN = 4
    STOP_SIGNS = 5
    TRAFFIC_GREEN = 6
    TRAFFIC_YELLOW = 7
    TRAFFIC_RED = 8


class TransfuserBEVSemanticClass(IntEnum):
    """Indicies to access BEV semantic map produced by TransFuser."""

    UNLABELED = 0
    ROAD = 1
    LANE_MARKERS = 2
    STOP_SIGNS = 3
    VEHICLE = 4
    WALKER = 5
    OBSTACLE = 6
    PARKING_VEHICLE = 7
    SPECIAL_VEHICLE = 8
    BIKER = 9
    TRAFFIC_GREEN = 10
    TRAFFIC_RED_NORMAL = 11
    TRAFFIC_RED_NOT_NORMAL = 12


class TransfuserBEVOccupancyClass(IntEnum):
    """Indicies to access BEV occupancy map produced by TransFuser."""

    UNLABELED = 0
    VEHICLE = 1
    WALKER = 2
    OBSTACLE = 3
    PARKING_VEHICLE = 4
    SPECIAL_VEHICLE = 5
    BIKER = 6
    TRAFFIC_GREEN = 7
    TRAFFIC_RED_NORMAL = 8
    TRAFFIC_RED_NOT_NORMAL = 9


class WeatherVisibility(IntEnum):
    """Visibility conditions in CARLA simulator, classified by TransFuser."""

    CLEAR = 0
    OK = 1
    LIMITED = 2
    VERY_LIMITED = 3


class CarlaSemanticSegmentationClass(IntEnum):
    """https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera"""

    Unlabeled = 0
    Roads = 1
    SideWalks = 2
    Building = 3
    Wall = 4
    Fence = 5
    Pole = 6
    TrafficLight = 7
    TrafficSign = 8
    Vegetation = 9
    Terrain = 10
    Sky = 11
    Pedestrian = 12
    Rider = 13
    Car = 14
    Truck = 15
    Bus = 16
    Train = 17
    Motorcycle = 18
    Bicycle = 19
    Static = 20
    Dynamic = 21
    Other = 22
    Water = 23
    RoadLine = 24
    Ground = 25
    Bridge = 26
    RailTrack = 27
    GuardRail = 28
    # --- Own classes ---
    ConeAndTrafficWarning = 29
    SpecialVehicles = 30
    StopSign = 31


class TransfuserSemanticSegmentationClass(IntEnum):
    """Semantic segmentation classes used in TransFuser."""

    UNLABELED = 0
    VEHICLE = 1
    ROAD = 2
    TRAFFIC_LIGHT = 3
    PEDESTRIAN = 4
    ROAD_LINE = 5
    OBSTACLE = 6
    SPECIAL_VEHICLE = 7
    STOP_SIGN = 8
    BIKER = 9


def rgb(r, g, b):
    """Dummy function to help with visualizing RGB colors by using a VSCode extension."""
    return (r, g, b)


# Other visualization
LIDAR_COLOR = rgb(90, 107, 249)
EGO_BB_COLOR = rgb(151, 15, 48)
TP_DEFAULT_COLOR = rgb(255, 10, 10)
RADAR_COLOR = rgb(24, 237, 3)
RADAR_DETECTION_COLOR = rgb(255, 24, 0)

# Planning visualization
GROUNDTRUTH_BB_WP_COLOR = rgb(15, 60, 255)
GROUNDTRUTH_FUTURE_WAYPOINT_COLOR = rgb(0, 0, 0)
GROUND_TRUTH_PAST_WAYPOINT_COLOR = rgb(0, 0, 0)
PREDICTION_WAYPOINT_COLOR = rgb(255, 0, 0)
PREDICTION_ROUTE_COLOR = rgb(0, 0, 255)
PREDICTION_WAYPOINT_RADIUS = 10
PREDICTION_ROUTE_RADIUS = 6

# TransFuser BEV Semantic class to color
CARLA_TRANSFUSER_BEV_SEMANTIC_COLOR_CONVERTER = {
    TransfuserBEVSemanticClass.UNLABELED: rgb(0, 0, 0),
    TransfuserBEVSemanticClass.ROAD: rgb(250, 250, 250),
    TransfuserBEVSemanticClass.LANE_MARKERS: rgb(255, 255, 0),
    TransfuserBEVSemanticClass.STOP_SIGNS: rgb(160, 160, 0),
    TransfuserBEVSemanticClass.VEHICLE: rgb(15, 60, 255),
    TransfuserBEVSemanticClass.WALKER: rgb(0, 255, 0),
    TransfuserBEVSemanticClass.OBSTACLE: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.PARKING_VEHICLE: rgb(116, 150, 65),
    TransfuserBEVSemanticClass.SPECIAL_VEHICLE: rgb(255, 0, 255),
    TransfuserBEVSemanticClass.BIKER: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.TRAFFIC_GREEN: rgb(0, 255, 0),
    TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL: rgb(255, 0, 0),
    TransfuserBEVSemanticClass.TRAFFIC_RED_NOT_NORMAL: rgb(0, 0, 255),
}

# TransFuser++ semantic segmentation colors for CARLA data
TRANSFUSER_SEMANTIC_COLORS = {
    TransfuserSemanticSegmentationClass.UNLABELED: rgb(0, 0, 0),
    TransfuserSemanticSegmentationClass.VEHICLE: rgb(31, 119, 180),
    TransfuserSemanticSegmentationClass.ROAD: rgb(128, 64, 128),
    TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT: rgb(250, 170, 30),
    TransfuserSemanticSegmentationClass.PEDESTRIAN: rgb(0, 255, 60),
    TransfuserSemanticSegmentationClass.ROAD_LINE: rgb(157, 234, 50),
    TransfuserSemanticSegmentationClass.OBSTACLE: rgb(255, 0, 0),
    TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE: rgb(255, 255, 0),
    TransfuserSemanticSegmentationClass.STOP_SIGN: rgb(125, 0, 0),
    TransfuserSemanticSegmentationClass.BIKER: rgb(220, 20, 60),
}

# TransFuser++ bounding box colors
TRANSFUSER_BOUNDING_BOX_COLORS = {
    TransfuserBoundingBoxClass.VEHICLE: rgb(0, 0, 255),
    TransfuserBoundingBoxClass.WALKER: rgb(0, 255, 0),
    TransfuserBoundingBoxClass.TRAFFIC_LIGHT: rgb(255, 0, 0),
    TransfuserBoundingBoxClass.STOP_SIGN: rgb(250, 160, 160),
    TransfuserBoundingBoxClass.SPECIAL: rgb(0, 0, 255),
    TransfuserBoundingBoxClass.OBSTACLE: rgb(0, 255, 13),
    TransfuserBoundingBoxClass.PARKING: rgb(116, 150, 65),
    TransfuserBoundingBoxClass.BIKER: rgb(255, 0, 0),
}

# Mapping from CARLA semantic segmentation classes to TransFuser semantic segmentation classes
SEMANTIC_SEGMENTATION_CONVERTER = {
    CarlaSemanticSegmentationClass.Unlabeled: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Roads: TransfuserSemanticSegmentationClass.ROAD,
    CarlaSemanticSegmentationClass.SideWalks: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Building: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Wall: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Fence: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pole: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.TrafficLight: TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT,
    CarlaSemanticSegmentationClass.TrafficSign: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Vegetation: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Terrain: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Sky: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Pedestrian: TransfuserSemanticSegmentationClass.PEDESTRIAN,
    CarlaSemanticSegmentationClass.Rider: TransfuserSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Car: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Truck: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bus: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Train: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Motorcycle: TransfuserSemanticSegmentationClass.VEHICLE,
    CarlaSemanticSegmentationClass.Bicycle: TransfuserSemanticSegmentationClass.BIKER,
    CarlaSemanticSegmentationClass.Static: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Dynamic: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Other: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Water: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RoadLine: TransfuserSemanticSegmentationClass.ROAD_LINE,
    CarlaSemanticSegmentationClass.Ground: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.Bridge: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.RailTrack: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.GuardRail: TransfuserSemanticSegmentationClass.UNLABELED,
    CarlaSemanticSegmentationClass.ConeAndTrafficWarning: TransfuserSemanticSegmentationClass.OBSTACLE,
    CarlaSemanticSegmentationClass.SpecialVehicles: TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE,
    CarlaSemanticSegmentationClass.StopSign: TransfuserSemanticSegmentationClass.STOP_SIGN,
}

# Mapping from ChaffeurNet BEV semantic classes to TransFuser BEV semantic classes
CHAFFEURNET_TO_TRANSFUSER_BEV_SEMANTIC_CONVERTER = {
    ChaffeurNetBEVSemanticClass.UNLABELED: TransfuserBEVSemanticClass.UNLABELED,  # unlabeled
    ChaffeurNetBEVSemanticClass.ROAD: TransfuserBEVSemanticClass.ROAD,  # road
    ChaffeurNetBEVSemanticClass.SIDEWALK: TransfuserBEVSemanticClass.UNLABELED,  # sidewalk
    ChaffeurNetBEVSemanticClass.LANE_MARKERS: TransfuserBEVSemanticClass.LANE_MARKERS,  # lane_markers
    ChaffeurNetBEVSemanticClass.LANE_MARKERS_BROKEN: TransfuserBEVSemanticClass.LANE_MARKERS,  # lane_markers broken
    ChaffeurNetBEVSemanticClass.STOP_SIGNS: TransfuserBEVSemanticClass.STOP_SIGNS,  # stop_signs
    ChaffeurNetBEVSemanticClass.TRAFFIC_GREEN: TransfuserBEVSemanticClass.UNLABELED,  # traffic light green
    ChaffeurNetBEVSemanticClass.TRAFFIC_YELLOW: TransfuserBEVSemanticClass.UNLABELED,  # traffic light yellow
    ChaffeurNetBEVSemanticClass.TRAFFIC_RED: TransfuserBEVSemanticClass.UNLABELED,  # traffic light red
}

SCENARIO_TYPES = [
    "Accident",
    "AccidentTwoWays",
    "BlockedIntersection",
    "ConstructionObstacle",
    "ConstructionObstacleTwoWays",
    "ControlLoss",
    "CrossJunctionDefectTrafficLight",
    "CrossingBicycleFlow",
    "DynamicObjectCrossing",
    "EnterActorFlow",
    "EnterActorFlowV2",
    "HardBreakRoute",
    "HazardAtSideLane",
    "HazardAtSideLaneTwoWays",
    "HighwayCutIn",
    "HighwayExit",
    "InterurbanActorFlow",
    "InterurbanAdvancedActorFlow",
    "InvadingTurn",
    "MergerIntoSlowTraffic",
    "MergerIntoSlowTrafficV2",
    "NonSignalizedJunctionLeftTurn",
    "NonSignalizedJunctionLeftTurnEnterFlow",
    "NonSignalizedJunctionRightTurn",
    "noScenarios",
    "OppositeVehicleRunningRedLight",
    "OppositeVehicleTakingPriority",
    "ParkedObstacle",
    "ParkedObstacleTwoWays",
    "ParkingCrossingPedestrian",
    "ParkingCutIn",
    "ParkingExit",
    "PedestrianCrossing",
    "PriorityAtJunction",
    "RedLightWithoutLeadVehicle",
    "SequentialLaneChange",
    "SignalizedJunctionLeftTurn",
    "SignalizedJunctionLeftTurnEnterFlow",
    "SignalizedJunctionRightTurn",
    "StaticCutIn",
    "T_Junction",
    "VanillaNonSignalizedTurn",
    "VanillaNonSignalizedTurnEncounterStopsign",
    "VanillaSignalizedTurnEncounterGreenLight",
    "VanillaSignalizedTurnEncounterRedLight",
    "VehicleOpensDoorTwoWays",
    "VehicleTurningRoute",
    "VehicleTurningRoutePedestrian",
    "YieldToEmergencyVehicle",
    "NA",
]

EMERGENCY_MESHES = {
    "vehicle.dodge.charger_police_2020",
    "vehicle.dodge.charger_police",
    "vehicle.ford.ambulance",
    "vehicle.carlamotors.firetruck",
}

CONSTRUCTION_MESHES = {"static.prop.constructioncone", "static.prop.trafficwarning"}

BIKER_MESHES = {
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
    "vehicle.bh.crossbike",
    "vehicle.harley-davidson.low_rider",
    "vehicle.kawasaki.ninja",
    "vehicle.vespa.zx125",
    "vehicle.yamaha.yzf",
}

URBAN_MAX_SPEED_LIMIT = 15
SUBURBAN_MAX_SPEED_LIMIT = 25
HIGHWAY_MAX_SPEED_LIMIT = 35

LOOKUP_TABLE = {
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": [
        2.44619083404541,
        1.115301489830017,
        0.7606233954429626,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": [
        2.5039126873016357,
        1.0485419034957886,
        0.7673624753952026,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": [  # noqa E501
        2.2210919857025146,
        0.9388753771781921,
        0.9936029314994812,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": [
        2.6828393936157227,
        0.9732309579849243,
        0.7874829173088074,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": [  # noqa E501
        2.782914400100708,
        1.217571496963501,
        1.022573471069336,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": [
        2.3368194103240967,
        1.0011461973190308,
        0.7259762287139893,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": [
        2.3958897590637207,
        1.081725001335144,
        0.7438300848007202,
    ],
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": [
        2.2763495445251465,
        1.0926425457000732,
        0.8835831880569458,
    ],
    "/Game/Carla/Static/Dynamic/Garden/SM_PlasticTable.SM_PlasticTable": [
        1.241101622581482,
        1.241101622581482,
        1.239898920059204,
    ],
    "/Game/Carla/Static/Dynamic/Garden/SM_PlasticChair.SM_PlasticChair": [
        0.36523768305778503,
        0.37522444128990173,
        0.6356779336929321,
    ],
    "/Game/Carla/Static/Dynamic/Construction/SM_ConstructionCone.SM_ConstructionCone": [
        0.1720348298549652,
        0.1720348298549652,
        0.2928849756717682,
    ],
}

CONSTRUCTION_CONE_BB_SIZE = [0.1720348298549652, 0.1720348298549652]

TRAFFIC_WARNING_BB_SIZE = [1.186714768409729, 1.4352929592132568]

OLD_TOWNS = {
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
}

# List of all CARLA towns for logging purposes
ALL_TOWNS = [
    "Town01",
    "Town02",
    "Town03",
    "Town04",
    "Town05",
    "Town06",
    "Town07",
    "Town10HD",
    "Town11",
    "Town12",
    "Town13",
    "Town15",
]

# Mapping from town name to a zero-padded index for logging (e.g., Town01 -> "01")
TOWN_NAME_TO_INDEX = {
    "Town01": "01",
    "Town02": "02",
    "Town03": "03",
    "Town04": "04",
    "Town05": "05",
    "Town06": "06",
    "Town07": "07",
    "Town10HD": "10HD",
    "Town11": "11",
    "Town12": "12",
    "Town13": "13",
    "Town15": "15",
}

# NavSim/NuPlan camera calibration parameters
NUPLAN_CAMERA_CALIBRATION = {
    "CAM_L0": {
        "pos": [0.2567803445203347, -0.14912709068475835, 1.9611907818710856],
        "rot": [-1.7908743283844297, -0.18030832979657968, -56.0098600797478],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
    "CAM_F0": {
        "pos": [0.3435966588946209, 0.00981503465349912, 1.9520959734648988],
        "rot": [0.004047525205852703, -2.344563417746492, -1.0253844128360994],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
    "CAM_R0": {
        "pos": [0.362775037177945, 0.16380984892156114, 1.9540305698009064],
        "rot": [0.8908895493227222, -0.5177262293066659, 54.26239302855094],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
    "CAM_B0": {
        "pos": [-1.8371898892894447, 0.023124645489646514, 1.9105230244574516],
        "rot": [0.004047525205852703, 1.9819092884563787, 180],
        "fov": 63.71,
        "width": 1920 // 4,
        "height": 1120 // 4,
        "cropped_height": 1080 // 4,
    },
}


class NavSimBEVSemanticClass(IntEnum):
    """Indicies to access BEV semantic map produced by NavSim.

    See: https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_config.py#L83
    """

    UNLABELED = 0
    ROAD = auto()
    WALKWAYS = auto()
    CENTERLINE = auto()
    STATIC_OBJECTS = auto()
    VEHICLES = auto()
    PEDESTRIANS = auto()


class NavSimBoundingBoxIndex(IntEnum):
    """Index to access NavSim bounding box attribute array.

    See: https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/transfuser_features.py#L174
    """

    X = 0
    Y = 1
    HEADING = 2
    LENGTH = 3
    WIDTH = 4


class NavSimBBClass(IntEnum):
    """Bounding box classes used in NavSim."""

    GENERIC_CLASS = 0


class NavSimStatusFeature(IntEnum):
    """Status feature indices used in NavSim."""

    DRIVING_COMMAND_LEFT = 0
    DRIVING_COMMAND_RIGHT = 1
    DRIVING_COMMAND_STRAIGHT = 2
    DRIVING_COMMAND_UNKNOWN = 3
    EGO_VELOCITY_X = 4
    EGO_VELOCITY_Y = 5
    ACCELERATION_X = 6
    ACCELERATION_Y = 7


SIM2REAL_SEMANTIC_SEGMENTATION_CONVERTER = {
    TransfuserSemanticSegmentationClass.UNLABELED: TransfuserSemanticSegmentationClass.UNLABELED,
    TransfuserSemanticSegmentationClass.VEHICLE: TransfuserSemanticSegmentationClass.VEHICLE,
    TransfuserSemanticSegmentationClass.ROAD: TransfuserSemanticSegmentationClass.ROAD,
    TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT: TransfuserSemanticSegmentationClass.TRAFFIC_LIGHT,
    TransfuserSemanticSegmentationClass.PEDESTRIAN: TransfuserSemanticSegmentationClass.PEDESTRIAN,
    TransfuserSemanticSegmentationClass.ROAD_LINE: TransfuserSemanticSegmentationClass.ROAD_LINE,
    TransfuserSemanticSegmentationClass.OBSTACLE: TransfuserSemanticSegmentationClass.OBSTACLE,
    TransfuserSemanticSegmentationClass.SPECIAL_VEHICLE: TransfuserSemanticSegmentationClass.VEHICLE,
    TransfuserSemanticSegmentationClass.STOP_SIGN: TransfuserSemanticSegmentationClass.STOP_SIGN,
    TransfuserSemanticSegmentationClass.BIKER: TransfuserSemanticSegmentationClass.BIKER,
}


SIM2REAL_BEV_SEMANTIC_SEGMENTATION_CONVERTER = {
    TransfuserBEVSemanticClass.UNLABELED: TransfuserBEVSemanticClass.UNLABELED,
    TransfuserBEVSemanticClass.ROAD: TransfuserBEVSemanticClass.ROAD,
    TransfuserBEVSemanticClass.LANE_MARKERS: TransfuserBEVSemanticClass.LANE_MARKERS,
    TransfuserBEVSemanticClass.STOP_SIGNS: TransfuserBEVSemanticClass.UNLABELED,
    TransfuserBEVSemanticClass.VEHICLE: TransfuserBEVSemanticClass.VEHICLE,
    TransfuserBEVSemanticClass.WALKER: TransfuserBEVSemanticClass.WALKER,
    TransfuserBEVSemanticClass.OBSTACLE: TransfuserBEVSemanticClass.OBSTACLE,
    TransfuserBEVSemanticClass.PARKING_VEHICLE: TransfuserBEVSemanticClass.VEHICLE,
    TransfuserBEVSemanticClass.SPECIAL_VEHICLE: TransfuserBEVSemanticClass.SPECIAL_VEHICLE,
    TransfuserBEVSemanticClass.BIKER: TransfuserBEVSemanticClass.BIKER,
    TransfuserBEVSemanticClass.TRAFFIC_GREEN: TransfuserBEVSemanticClass.TRAFFIC_GREEN,
    TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL: TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL,
    TransfuserBEVSemanticClass.TRAFFIC_RED_NOT_NORMAL: TransfuserBEVSemanticClass.TRAFFIC_RED_NORMAL,
}

SIM2REAL_BOUNDING_BOX_CLASS_CONVERTER = {
    TransfuserBoundingBoxClass.VEHICLE: TransfuserBoundingBoxClass.VEHICLE,
    TransfuserBoundingBoxClass.WALKER: TransfuserBoundingBoxClass.WALKER,
    TransfuserBoundingBoxClass.TRAFFIC_LIGHT: TransfuserBoundingBoxClass.TRAFFIC_LIGHT,
    TransfuserBoundingBoxClass.STOP_SIGN: TransfuserBoundingBoxClass.STOP_SIGN,
    TransfuserBoundingBoxClass.SPECIAL: TransfuserBoundingBoxClass.VEHICLE,
    TransfuserBoundingBoxClass.OBSTACLE: TransfuserBoundingBoxClass.OBSTACLE,
    TransfuserBoundingBoxClass.PARKING: TransfuserBoundingBoxClass.VEHICLE,
    TransfuserBoundingBoxClass.BIKER: TransfuserBoundingBoxClass.BIKER,
}

SIM2REAL_BEV_OCCUPANCY_CLASS_CONVERTER = {
    TransfuserBEVOccupancyClass.UNLABELED: TransfuserBEVOccupancyClass.UNLABELED,
    TransfuserBEVOccupancyClass.VEHICLE: TransfuserBEVOccupancyClass.VEHICLE,
    TransfuserBEVOccupancyClass.WALKER: TransfuserBEVOccupancyClass.WALKER,
    TransfuserBEVOccupancyClass.OBSTACLE: TransfuserBEVOccupancyClass.UNLABELED,
    TransfuserBEVOccupancyClass.PARKING_VEHICLE: TransfuserBEVOccupancyClass.VEHICLE,
    TransfuserBEVOccupancyClass.SPECIAL_VEHICLE: TransfuserBEVOccupancyClass.VEHICLE,
    TransfuserBEVOccupancyClass.BIKER: TransfuserBEVOccupancyClass.BIKER,
    TransfuserBEVOccupancyClass.TRAFFIC_GREEN: TransfuserBEVOccupancyClass.TRAFFIC_GREEN,
    TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL: TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL,
    TransfuserBEVOccupancyClass.TRAFFIC_RED_NOT_NORMAL: TransfuserBEVOccupancyClass.TRAFFIC_RED_NORMAL,
}

CARLA_REAR_AXLE = [
    -1.389,
    0.0,
    0.360,
]  # Rear axle position relative to the vehicle center

# Waymo E2E 2025 camera intrinsics
WAYMO_E2E_INTRINSIC = [
    1112.17806333,
    1112.04114501,
    488.128479,
    719.15600586,
    -0.073584,
    -0.036582,
    0.0,
    0.0,
    0.0,
]

# Waymo E2E 2025 camera extrinsics
WAYMO_E2E_2025_CAMERA_SETTING = {
    "FRONT_LEFT": {
        "extrinsic": [
            [0.70549636, -0.70869903, -0.00026687, 1.44509995],
            [0.70868651, 0.70547806, 0.00679447, 0.15270001],
            [-0.00462506, -0.00498361, 0.99996268, 1.80649996],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 972,
        "height": 1440,
        "cropped_height": 1080,
    },
    "FRONT": {
        "extrinsic": [
            [0.99998026, -0.00215741, 0.00378466, 1.51909995],
            [0.00214115, 0.99997577, 0.0045841, 0.0258],
            [-0.00379113, -0.00457491, 0.9999674, 1.80649996],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 972,
        "height": 1440,
        "cropped_height": 1080,
    },
    "FRONT_RIGHT": {
        "extrinsic": [
            [0.70908528, 0.70508644, 0.00506919, 1.48150003],
            [-0.7050955, 0.70909768, -0.00009787, -0.1163],
            [-0.00366141, -0.00350169, 0.99996981, 1.80649996],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 972,
        "height": 1440,
        "cropped_height": 1080,
    },
}


class WaymoE2EIntrinsicIndex(IntEnum):
    # https://github.com/waymo-research/waymo-open-dataset/blob/d16af5cf112a2498d659f87e614ad19f20ca2f56/src/waymo_open_dataset/dataset.proto#L98
    F_U = 0  # Focal length in x direction
    F_V = 1  # Focal length in y direction
    C_U = 2  # Center of the image in x direction
    C_V = 3  # Center of the image in y direction
    K_1 = 4  # Radial distortion coefficient
    K_2 = 5  # Radial distortion coefficient
    P_1 = 6  # Tangential distortion coefficient
    P_2 = 7  # Tangential distortion coefficient
    K_3 = 8  # Radial distortion coefficient


class CarlaImageCroppingType(IntEnum):
    TOP = 0
    BOTTOM = auto()
    BOTH = auto()
    NONE = auto()


WAYMO_DOWN_SAMPLE_FACTOR = 3  # Down-sample factor for Waymo E2E 2025 images
WAYMO_E2E_REAL_DATA_JPEG_LEVEL = (
    50  # JPEG compression level for Waymo E2E 2025 real data
)


@beartype
def normalize_imagenet(
    x: jt.Float[torch.Tensor, "B 3 H W"]
) -> jt.Float[torch.Tensor, "B 3 H W"]:
    """Normalize input images according to ImageNet standards.
    Args:
        x: Input images batch.

    Returns:
        Normalized images batch.
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x


class BaseConfig:
    @property
    def target_dataset(self):
        raise NotImplementedError(
            "Subclasses must implement the target_dataset property."
        )

    # --- Autopilot ---
    # Frame rate used for the bicycle models in the autopilot
    bicycle_frame_rate = 20
    # Number of future route points we save per step
    num_route_points_saved = 50
    # Points sampled per meter when interpolating route
    points_per_meter = 10
    # Pixels per meter used in the semantic segmentation map during data collection
    # On Town 13 2.0 is the highest that opencv can handle
    pixels_per_meter_collection = 2.0
    # Maximum acceleration in meters per tick (1.9 m/tick)
    longitudinal_max_accelerations = 1.89

    # --- Kinematic Bicycle Model ---
    # Time step for the model (20 frames per second)
    time_step = 1.0 / 20.0
    # Distance from the rear axle to the front axle of the vehicle
    front_wheel_base = -0.090769015
    # Distance from the rear axle to the center of the rear wheels
    rear_wheel_base = 1.4178275
    # Gain factor for steering angle to wheel angle conversion
    steering_gain = 0.36848336
    # Deceleration rate when braking (m/s^2) of other vehicles
    brake_acceleration = -4.952399
    # Acceleration rate when throttling (m/s^2) of other vehicles
    throttle_acceleration = 0.5633837
    # Minimum throttle value that has an affect during forecasting the ego vehicle
    throttle_threshold_during_forecasting = 0.3

    # --- Augmentation and Misc ---
    # Frequency (in steps) at which data is saved during data collection
    data_save_freq = 5
    # If true enable camera augmentation during data collection
    augment = False
    # Safety translation augmentation penalty for default scenarios
    default_safety_translation_augmentation_penalty = 0.25
    # Safety translation augmentation penalty for urban scenarios with low speed limits
    urban_safety_translation_augmentation_penalty = 0.4

    # Minimum value by which the augmented camera is shifted left and right
    camera_translation_augmentation_min = 0.1
    # Maximum value by which the augmented camera is shifted left and right
    camera_translation_augmentation_max = 1.0

    # Minimum value by which the augmented camera is rotated around the yaw (degrees)
    camera_rotation_augmentation_min = 5.0
    # Maximum value by which the augmented camera is rotated around the yaw (degrees)
    camera_rotation_augmentation_max = 12.5
    # Epsilon threshold to ignore rotation augmentation around 0.0 degrees
    camera_rotation_epsilon = 0.5
    # If true enable extreme augmentation settings
    extreme_augment = False

    # Probability of applying extreme augmentation during data collection.
    data_collection_extreme_augment_prob = 1.0
    # Minimum extreme rotation augmentation value (degrees)
    camera_extreme_rotation_augmentation_min = 12.5
    # Maximum extreme rotation augmentation value (degrees)
    camera_extreme_rotation_augmentation_max = 40
    # Minimum extreme translation augmentation value
    camera_extreme_translation_augmentation_min = 0.1
    # Maximum extreme translation augmentation value
    camera_extreme_translation_augmentation_max = 1.0
    # Minimum extreme pitch augmentation value (degrees)
    camera_extreme_pitch_augmentation_min = -1.0
    # Maximum extreme pitch augmentation value (degrees)
    camera_extreme_pitch_augmentation_max = 0.5
    # Maximum extreme roll augmentation value (degrees)
    camera_extreme_roll_augmentation_max = 1.0

    # --- LiDAR Compression ---
    # LARS point format used for storing LiDAR data
    point_format = 0
    # Precision up to which LiDAR points are stored (x, y, z coordinates)
    point_precision_x = point_precision_y = point_precision_z = 0.1
    # Maximum height threshold for LiDAR points (meters, points above are discarded)
    max_height_lidar = 10.0
    # Minimum height threshold for LiDAR points (meters, points below are discarded)
    min_height_lidar = -4.0

    # --- Sensor Configuration ---
    # If true use two LiDARs or one
    use_two_lidars = True

    # x, y, z mounting position of the first LiDAR
    @property
    def lidar_pos_1(self):
        return [0.0, 0.0, 2.5]

    # Roll, pitch, yaw rotation of first LiDAR (degrees)
    @property
    def lidar_rot_1(self):
        return [0.0, 0.0, -90.0]

    # x, y, z mounting position of the second LiDAR
    @property
    def lidar_pos_2(self):
        return [0.0, 0.0, 2.5]

    # Roll, pitch, yaw rotation of second LiDAR (degrees)
    @property
    def lidar_rot_2(self):
        return [0.0, 0.0, -270.0]

    # If true accumulate LiDAR data over multiple frames
    @property
    def lidar_accumulation(self):
        return True

    # --- Camera Configuration ---
    @property
    def num_cameras(self):
        """Number of cameras based on the target dataset."""
        return {
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS: 6,
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS: 3,
            TargetDataset.NAVSIM_4CAMERAS: 4,
            TargetDataset.WAYMO_E2E_2025_3CAMERAS: 3,
        }[self.target_dataset]

    @property
    def camera_calibration(self):
        """Camera calibration configuration with positions, rotations, and sensor parameters"""
        if self.target_dataset == TargetDataset.CARLA_LEADERBOARD2_6CAMERAS:
            return {
                1: {
                    "pos": [0.0, -0.3, 2.25],
                    "rot": [0.0, 0.0, -57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                2: {
                    "pos": [0.25, 0.0, 2.25],
                    "rot": [0.0, 0.0, 0.0],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                3: {
                    "pos": [0.0, 0.3, 2.25],
                    "rot": [0.0, 0.0, 57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                4: {
                    "pos": [-0.30, 0.3, 2.25],
                    "rot": [0.0, 0.0, 180 - 57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                5: {
                    "pos": [-0.55, 0.0, 2.25],
                    "rot": [0.0, 0.0, 180.0],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                6: {
                    "pos": [-0.30, -0.3, 2.25],
                    "rot": [0.0, 0.0, -180 + 57.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
            }
        elif self.target_dataset == TargetDataset.CARLA_LEADERBOARD2_3CAMERAS:
            return {
                1: {
                    "pos": [0.1, -0.35, 2.25],
                    "rot": [0.0, 0.0, -54.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                2: {
                    "pos": [0.35, 0.0, 2.25],
                    "rot": [0.0, 0.0, 0.0],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
                3: {
                    "pos": [0.1, 0.35, 2.25],
                    "rot": [0.0, 0.0, 54.5],
                    "width": 1152 // 3,
                    "height": 384,
                    "cropped_height": 384,
                    "fov": 60,
                },
            }
        elif self.target_dataset == TargetDataset.NAVSIM_4CAMERAS:
            return {
                1: NUPLAN_CAMERA_CALIBRATION["CAM_L0"],
                2: NUPLAN_CAMERA_CALIBRATION["CAM_F0"],
                3: NUPLAN_CAMERA_CALIBRATION["CAM_R0"],
                4: NUPLAN_CAMERA_CALIBRATION["CAM_B0"],
            }
        elif self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            from pdm_lite_utils_0916 import waymo_e2e_camera_setting_to_carla

            return {
                1: waymo_e2e_camera_setting_to_carla(
                    WAYMO_E2E_INTRINSIC,
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"]["extrinsic"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"]["width"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"]["height"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_RIGHT"]["cropped_height"],
                ),
                2: waymo_e2e_camera_setting_to_carla(
                    WAYMO_E2E_INTRINSIC,
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["extrinsic"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["width"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["height"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT"]["cropped_height"],
                ),
                3: waymo_e2e_camera_setting_to_carla(
                    WAYMO_E2E_INTRINSIC,
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"]["extrinsic"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"]["width"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"]["height"],
                    WAYMO_E2E_2025_CAMERA_SETTING["FRONT_LEFT"]["cropped_height"],
                ),
            }
        raise ValueError(f"Unsupported target dataset: {self.target_dataset}")

    # --- Radar Configuration ---
    num_radar_sensors = 4

    @property
    def radar_calibration(self):
        """Radar sensor calibration configuration with positions and orientations.

        Currently supports only 4 radar sensors. For other numbers, raises an error.
        """
        return {
            "1": {
                "pos": [2.6, 0, 0.60],  # front-left
                "rot": [0.0, 0.0, -45.0],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
            "2": {
                "pos": [2.6, 0, 0.60],  # front
                "rot": [0.0, 0.0, 45.0],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
            "3": {
                "pos": [-2.6, 0, 0.60],  # front-right
                "rot": [0.0, 0.0, 135],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
            "4": {
                "pos": [-2.6, 0, 0.60],  # rear
                "rot": [0.0, 0.0, 225],
                "horz_fov": 90,
                "vert_fov": 0.1,
            },
        }

    # If true use radar sensors
    use_radars = True
    # If true save radar point cloud as LiDAR format
    save_radar_pc_as_lidar = True
    # If true save LiDAR data only inside bird's eye view area
    save_lidar_only_inside_bev = True
    # If true duplicate radar points near ego vehicle for better detection
    duplicate_radar_near_ego = True
    # Radius around ego vehicle for radar point duplication
    duplicate_radar_radius = 32
    # Multiplication factor for radar point duplication
    duplicate_radar_factor = 5

    # --- Data Storage ---
    # If true save depth images at lower resolution
    save_depth_lower_resolution = True

    @property
    def save_depth_resolution_ratio(self):
        """Resolution reduction ratio for depth image storage."""
        if self.is_on_slurm:
            return 4
        return 4

    # Number of bits used for saving depth images
    save_depth_bits = 8
    # If true save only non-ground LiDAR points
    save_only_non_ground_lidar = True
    # If true save semantic segmentation in grouped format
    save_grouped_semantic = True

    # --- Temporal Data ---
    # Number of temporal data points saved for ego vehicle
    ego_num_temporal_data_points_saved = 200
    # Number of temporal data points saved for other vehicles
    other_vehicles_num_temporal_data_points_saved = 40

    # --- Agent Configuration ---
    # Simulator frames per second
    carla_fps = 20
    # CARLA frame rate in seconds
    carla_frame_rate = 1.0 / carla_fps
    # IoU threshold used for non-maximum suppression on bounding box predictions
    iou_treshold_nms = 0.2
    # Minimum distance to route planner waypoints
    route_planner_min_distance = 7.5
    # Maximum distance to route planner waypoints
    route_planner_max_distance = 50.0
    # Minimum distance to waypoint in dense route that expert follows
    dense_route_planner_min_distance = 2.4
    # Initial frames delay for CARLA initialization
    inital_frames_delay = 1

    # Target point distances for route planning (in meters)
    tp_distances = [
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
        5.25,
        5.5,
        5.75,
        6.0,
        6.25,
        6.5,
        6.75,
        7.0,
        7.25,
        7.5,
        7.75,
        8.0,
        8.25,
        8.5,
        8.75,
        9.0,
        9.25,
        9.5,
        9.75,
        10.0,
    ]

    # Extent of the ego vehicle's bounding box in x direction
    ego_extent_x = 2.4508416652679443
    # Extent of the ego vehicle's bounding box in y direction
    ego_extent_y = 1.0641621351242065
    # Extent of the ego vehicle's bounding box in z direction
    ego_extent_z = 0.7553732395172119

    # Minimum z coordinate of the safety box
    safety_box_z_min = 0.5
    # Maximum z coordinate of the safety box
    safety_box_z_max = 1.5

    # --- Safety Box Properties ---
    @property
    def safety_box_y_min(self):
        """Minimum y coordinate of the safety box relative to ego vehicle."""
        return -self.ego_extent_y * 0.8

    @property
    def safety_box_y_max(self):
        """Maximum y coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_y * 0.8

    @property
    def safety_box_x_min(self):
        """Minimum x coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_x

    @property
    def safety_box_x_max(self):
        """Maximum x coordinate of the safety box relative to ego vehicle."""
        return self.ego_extent_x + 2.5

    @property
    def is_on_slurm(self):
        """Check if running on SLURM cluster environment."""
        return os.getenv("SLURM_JOB_ID") is not None

    @property
    def is_on_tcml(self):
        """Check if running on Training Center for Machine Learning of Tübingen."""
        return os.getenv("TCML") is not None

    # --- Configuration Parsing Methods ---

    def parse(self, loaded_config, env_key: str, raise_error_on_missing_key: bool):
        # --- Parameters coming from environment variables, highest priority
        env_params = os.getenv(env_key, "").strip()
        if not env_params:
            env_params = {}
        else:
            parsed = OmegaConf.create(OmegaConf.from_dotlist(env_params.split()))
            env_params = OmegaConf.to_container(parsed, resolve=True)

        # --- Parameters from loaded file, second priority.
        if loaded_config is None:
            loaded_config = {}

        # ---  We overwrite parameters from loaded file with parameters from environment variables
        for key, value in env_params.items():
            loaded_config[key] = value

        # --- Update config object with loaded config and environment variables
        for key, value in loaded_config.items():
            if hasattr(self, key) and not callable(getattr(self, key)):
                try:
                    setattr(self, key, value)
                except Exception as _:
                    pass
            elif raise_error_on_missing_key:
                raise AttributeError(f"Unknown configuration key: {key}")

        self._loaded_config = (
            loaded_config  # Stored for the `cli_overridable_property` decorator
        )

    def __setattr__(self, name, value):
        # Override __setattr__ to avoid error where we set an attribute that is not defined in the class.
        # This could happen for example when we refactor and rename avariable but the renaming is not done everywhere.
        # Check if the attribute is allowed to be set
        allowed = set()
        for cls in self.__class__.__mro__:
            allowed.update(getattr(cls, "__dict__", {}).keys())
        allowed.update(self.__dict__.keys())

        if name not in allowed and not name.startswith("_"):
            raise AttributeError(
                f"Can't set unknown attribute '{name}'. Please check if this variable might have been renamed."
            )
        super().__setattr__(name, value)

    def base_dict(self):
        out = {}
        for k in dir(self):
            if k.startswith("_"):
                continue
            try:
                v = getattr(self, k)
                if not callable(v):
                    out[k] = v
            except Exception:
                pass
        return out


def overridable_property(fn):
    attr_name = fn.__name__

    @property
    def wrapper(self):
        try:
            if attr_name in self._loaded_config:
                return type(fn(self))(self._loaded_config[attr_name])
        except:
            pass
        return fn(self)

    return wrapper


class TrainingConfig(BaseConfig):
    @property
    def target_dataset(self):
        if self.use_waymo_e2e_data and not self.mixed_data_training:
            return TargetDataset.WAYMO_E2E_2025_3CAMERAS
        elif self.use_navsim_data and not self.use_carla_data:
            return TargetDataset.NAVSIM_4CAMERAS
        elif "carla_leaderboad2_v3" in self.carla_root:
            return TargetDataset.CARLA_LEADERBOARD2_3CAMERAS
        elif "carla_leaderboad2_v8" in self.carla_root:
            return TargetDataset.CARLA_LEADERBOARD2_3CAMERAS
        elif "carla_leaderboad2_v10" in self.carla_root:
            return TargetDataset.CARLA_LEADERBOARD2_6CAMERAS
        elif "carla_leaderboad2_v12" in self.carla_root:
            return TargetDataset.NAVSIM_4CAMERAS
        elif "carla_leaderboad2_v14" in self.carla_root:
            return TargetDataset.WAYMO_E2E_2025_3CAMERAS
        elif "data/carla_today/results/data/garage_v2_2025_06_23" in self.carla_root:
            return TargetDataset.NAVSIM_4CAMERAS
        raise ValueError(f"Unknown CARLA root path: {self.carla_root}")

    @property
    def num_available_cameras(self):
        """Number of available cameras based on the target dataset."""
        return {
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS: 3,
            TargetDataset.CARLA_LEADERBOARD2_6CAMERAS: 6,
            TargetDataset.CARLA_LEADERBOARD2_3CAMERAS: 3,
            TargetDataset.NAVSIM_4CAMERAS: 4,
            TargetDataset.WAYMO_E2E_2025_3CAMERAS: 3,
        }[self.target_dataset]

    @overridable_property
    def used_cameras(self):
        """List indicating which cameras are used based on the target dataset.
        Can be overriden, if a camera is false it will be ignored during training."""
        return [True] * self.num_available_cameras

    @property
    def num_used_cameras(self):
        """Number of cameras used during training."""
        return sum(int(use) for use in self.used_cameras)

    # --- Planning Area ---
    # Maximum planning area coordinate in x direction (meters)
    # How many pixels make up 1 meter in BEV grids.
    pixels_per_meter = 4.0

    @property
    def min_x_meter(self):
        """Back boundary of the planning area in meters."""
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return 0
        return -32

    @property
    def max_x_meter(self):
        """Front boundary of the planning area in meters."""
        if self.carla_leaderboard_mode:
            return 64
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return 64
        return 32

    @property
    def min_y_meter(self):
        """Left boundary of the planning area in meters."""
        if self.carla_leaderboard_mode:
            return -40
        return -32

    @property
    def max_y_meter(self):
        """Right boundary of the planning area in meters."""
        if self.carla_leaderboard_mode:
            return 40
        return 32

    @property
    def lidar_width_pixel(self):
        """Width resolution of LiDAR BEV representation in pixels."""
        return int((self.max_x_meter - self.min_x_meter) * self.pixels_per_meter)

    @property
    def lidar_height_pixel(self):
        """Height resolution of LiDAR BEV representation in pixels."""
        return int((self.max_y_meter - self.min_y_meter) * self.pixels_per_meter)

    @property
    def lidar_width_meter(self):
        """Width of LiDAR coverage area in meters."""
        return int(self.max_x_meter - self.min_x_meter)

    @property
    def lidar_height_meter(self):
        """Height of LiDAR coverage area in meters."""
        return int(self.max_y_meter - self.min_y_meter)

    # Flag to visualize the dataset and deactivate randomization and augmentation.
    visualize_dataset = False
    # Flag to visualize the failed scenarios and deactivate randomization and augmentation.
    visualize_failed_scenarios = False
    # Flag to load the BEV 3rd person images from the dataset for debugging.
    load_bev_3rd_person_images = False

    # --- Training ID, Logging setting ---
    # Training Seed
    seed = 0
    # WandB ID for the experiment. If None, it will be generated automatically.
    wandb_id: str = None
    # must, allow, never
    wandb_resume = "never"
    # Description of the experiment.
    description = "My Experiment."
    # Produce images while training
    visualize_training = True
    # Unique experiment identifier.
    id = "Experiment 1."
    # File to continue training from
    load_file: str = None
    # If true continue the training from the loaded epoch or from 0.
    continue_epoch = False

    @property
    def epoch_checkpoints_keep(self):
        """Number of checkpoints to keep during training."""
        if self.use_carla_data and not self.mixed_data_training:
            return []
        return [sum([1 * 2**i for i in range(n)]) for n in range(3, 10)]

    # --- Training cache ---
    # If true use training session cache. This cache reduces data loading time.
    use_training_session_cache = True
    # If true use persistent cache for training. This cache reduces heavy feature building.
    use_persistent_cache = False
    # If true force rebuild the cache for each training run.
    force_rebuild_data_cache = False

    @property
    def carla_cache_path(self):
        """Tuple of cache characteristics used to identify cached data compatibility."""
        return (
            str(self.image_width_before_camera_subselection),
            str(self.final_image_height),
            str(self.min_x_meter),
            str(self.max_x_meter),
            str(self.min_y_meter),
            str(self.max_y_meter),
            str(self.detect_boxes),
            str(self.use_depth),
            str(self.use_semantic),
            str(self.use_bev_semantic),
            str(self.load_bev_3rd_person_images),
            str(self.training_used_lidar_steps),
            str(self.num_radar_forecast_steps),
            str(self.radar_waypoints_spacing),
        )

    @property
    def ssd_cache_path(self):
        """Path to SSD cache directory."""
        tmp_folder = "/scratch/" + str(os.environ.get("SLURM_JOB_ID"))
        if not self.is_on_tcml:
            tmp_folder = str(os.environ.get("SCRATCH", "/tmp"))
        return tmp_folder

    # Root directory for CARLA sensor data.
    carla_root = "data/carla_leaderboad2_v12/results/data/sensor_data/"

    @property
    def carla_data(self):
        """Path to CARLA data directory."""
        return os.path.join(self.carla_root, "data")

    # --- Training ---
    # Directory to log data to.
    logdir = None
    # PNG compression level for storing images
    training_png_compression_level = 6
    # Minimum number of LiDAR points for a vehicle to be considered valid.
    vehicle_min_num_lidar_points = 1
    # Minimum number of visible pixels for a vehicle to be considered valid.
    vehicle_min_num_visible_pixels = 1
    # Minimum number of LiDAR points for a pedestrian to be considered valid.
    pedestrian_min_num_lidar_points = 5
    # Minimum number of visible pixels for a pedestrian to be considered valid.
    pedestrian_min_num_visible_pixels = 15
    # Minimum number of LiDAR points for a parking vehicle to be considered valid.
    parking_vehicle_min_num_lidar_points = 3
    # Minimum number of visible pixels for a parking vehicle to be considered valid.
    parking_vehicle_min_num_visible_pixels = 5
    # First scale we use for the gradient scaler.
    grad_scaler_init_scale = 1024
    # Factor by which we grow the gradient scaler.
    grad_scaler_growth_factor = 2
    # Factor by which we backoff the gradient scaler if the gradients are too large.
    grad_scaler_backoff_factor = 0.5
    # Number of steps after which we grow the gradient scaler.
    grad_scaler_growth_interval = 256
    # Maximum gradient scale we use for the gradient scaler.
    grad_scaler_max_grad_scale = 2**16

    @property
    def sync_batchnorm(self) -> bool:
        """If true synchronize batch normalization across distributed processes."""
        return False

    # Freeze batch norm layers during training.
    freeze_batchnorm = False

    @overridable_property
    def epochs(self):
        """Total number of training epochs."""
        if self.carla_leaderboard_mode:
            return 31
        if self.target_dataset == TargetDataset.NAVSIM_4CAMERAS:
            return 61
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return 20
        raise ValueError("Unknown target dataset")

    @overridable_property
    def batch_size(self):
        """Batch size for training."""
        if not self.is_on_slurm:  # Local training
            return 4
        return 64

    @property
    def torch_float_type(self):
        """PyTorch float precision type for training."""
        if self.use_mixed_precision_training and self.gpu_name in ["a100", "l40s"]:
            return torch.bfloat16
        return torch.float32

    @property
    def use_mixed_precision_training(self):
        """If true use mixed precision training."""
        return self.gpu_name in ["a100", "l40s"]

    @property
    def need_grad_scaler(self):
        """If true gradient scaling is needed for mixed precision training."""
        return (
            self.use_mixed_precision_training and self.torch_float_type == torch.float16
        )

    # If true use ZeRO redundancy optimizer for distributed training.
    use_zero_redundancy = False

    @property
    def save_model_checkpoint(self):
        """If true save model checkpoints during training."""
        if self.is_on_slurm:
            return True
        return True

    @property
    def is_pretraining(self):
        """If true indicates pretraining phase."""
        return not self.use_planning_decoder

    # --- Training speed and memory optimization ---
    # Number of data loader workers to prefetch batches.
    prefetch_factor = 8

    @property
    def compile(self):
        """If true compile the model for optimization."""
        return True

    @property
    def channel_last(self):
        """If true use channel last memory format for input tensors."""
        return True

    # --- Learning rate and epochs ---
    # Base learning rate for the model.
    lr = 3e-4

    # --- Model input ---
    @property
    def skip_first(self):
        """Number of frames to skip at the beginning of sequences."""
        if self.is_pretraining and not self.mixed_data_training:
            return 1
        return self.num_way_points_prediction

    @property
    def skip_last(self):
        """Number of frames to skip at the end of sequences."""
        if self.is_pretraining:
            return 1
        if self.carla_leaderboard_mode:
            return self.num_way_points_prediction
        if self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return self.num_way_points_prediction * 2
        raise ValueError("Unknown target dataset")

    # --- RaDAR ---
    @property
    def radar_detection(self):
        """If true use radar points as additional input to the model."""
        return self.carla_leaderboard_mode and not self.use_tfpp_planning_decoder

    @overridable_property
    def use_radar_detection(self):
        """If true use radar points as additional input to the model."""
        return self.carla_leaderboard_mode and not self.use_tfpp_planning_decoder

    # Fixed number of radar points per sensor.
    num_radar_points_per_sensor = 75
    # Number of radar queries in the transformer.
    num_radar_queries = 20
    # Hidden dimension for radar tokenizer.
    radar_hidden_dim_tokenizer = 1024
    # Dimension of radar tokens.
    radar_token_dim = 256
    # Feed-forward dimension in radar transformer.
    radar_tf_dim_ff = 1024
    # Dropout rate for radar components.
    radar_dropout = 0.1
    # Number of attention heads in radar transformer.
    radar_num_heads = 8
    # Number of transformer layers for radar processing.
    radar_num_layers = 4
    # Hidden dimension for radar decoder.
    radar_hidden_dim_decoder = 1024
    # Loss weight for radar classification.
    radar_classification_loss_weight = 1.0
    # Loss weight for radar regression.
    radar_forecast_loss_weight = 0.1
    # Loss weight for radar regression.
    radar_regression_loss_weight = 5.0
    # Number of sine features for radar positional encoding.
    radar_sine_features = 128
    # Total number of radar sensors.
    num_radar_sensors = 4
    # If true we forecast motion of radar detections, too.
    forecast_radar_detections = False
    # Number of future steps to forecast for radar detections.
    num_radar_forecast_steps = 8
    # Spacing between waypoints in the prediction. For example: spacing 5 = 4Hz prediction.
    radar_waypoints_spacing = 5
    # Maximum distance for future waypoints.
    max_distance_future_waypoint = 10.0

    # --- Data filtering and bucket system ---
    # If true rebuild buckets collection from scratch.
    force_rebuild_bucket = False
    # If true randomize routes in bucket
    randomize_route_order = False
    # If true use only a subset of the data for training.
    subsample_data = False
    # If true then we skip Town13 routes during training
    hold_out_town13_routes = False

    @property
    def carla_bucket_collection(self):
        """Name of the bucket collection to use for training data."""
        if (
            self.use_carla_data
            and self.use_waymo_e2e_data
            and not self.force_rebuild_bucket
            and not self.force_rebuild_data_cache
        ):
            return "navsim"
        if (
            self.use_carla_data
            and self.use_navsim_data
            and not self.force_rebuild_bucket
            and not self.force_rebuild_data_cache
        ):
            return "navsim"
        if self.visualize_failed_scenarios:
            return "failed"
        if self.subsample_data:
            return "subsampled_posttrain"
        if self.is_pretraining and self.hold_out_town13_routes:
            return "town13_heldout_pretrain"
        if (
            self.is_pretraining
            or self.visualize_dataset
            or self.force_rebuild_data_cache
        ):
            return "full_pretrain"
        if not self.is_pretraining and self.hold_out_town13_routes:
            return "town13_heldout_posttrain"
        return "full_posttrain"

    @property
    def bucket_collection_path(self):
        """Path to bucket collection directory."""
        return os.path.join(self.carla_root, "buckets")

    # --- Training to recover from drift ---
    # If true use rotation and translation perburtation.
    use_sensor_perburtation = True

    # Probability of the perburtated sample being used.
    @overridable_property
    def use_sensor_perburtation_prob(self):
        if not self.carla_leaderboard_mode:
            return 0.8
        return 0.5

    @overridable_property
    def use_extreme_sensor_perburtation_prob(self):
        """Probability of extreme sensor perburtation being applied if we sampled an sensor perburtated sample."""
        if self.is_pretraining and not self.carla_leaderboard_mode:
            return 0.5
        return 0.0

    # --- Regularization ---
    @property
    def use_color_aug(self):
        """If true apply image color based augmentations."""
        # If true apply image color based augmentations
        return not self.visualize_dataset

    @property
    def use_color_aug_prob(self):
        """Probability to apply the different image color augmentations."""
        if self.carla_leaderboard_mode:
            return 0.2
        return 0.1

    # Weight decay for regularization.
    weight_decay = 0.01

    # If true use gradient clipping during training.
    @property
    def use_grad_clip(self):
        """If true use gradient clipping during training."""
        return False

    # Quantile value for gradient norm clipping.
    grad_quantile = 0.99
    # Maximum  gradient norm for gradient clipping.
    grad_history_length = 1000
    # If true use optimizer group
    use_optimizer_groups = False

    # If true use cosine learning rate scheduler with restart, else only one cycle
    @overridable_property
    def use_cosine_annealing_with_restarts(self):
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return False
        return True

    # --- Depth ---
    @overridable_property
    def use_depth(self):
        """If true use depth prediction as auxiliary task."""
        return self.carla_leaderboard_mode

    # --- LiDAR setting ---
    @property
    def training_used_lidar_steps(self):
        """We stack lidar frames for motion cues. Number of past frames we stack for the model input."""
        return 10

    # Minimum Z coordinate for LiDAR points.
    min_z = -4
    # Maximum Z coordinate for LiDAR points.
    max_z = 4
    # Max number of LiDAR points per pixel in voxelized LiDAR.
    hist_max_per_pixel = 5

    @property
    def lidar_vert_anchors(self):
        """Number of vertical anchors for LiDAR feature maps."""
        return self.lidar_height_pixel // 32

    @property
    def lidar_horz_anchors(self):
        """Number of horizontal anchors for LiDAR feature maps."""
        return self.lidar_width_pixel // 32

    # --- Bounding boxes detection ---
    # If true use the bounding box auxiliary task.
    detect_boxes = True
    # If true visualize bounding boxes for debugging.
    debug_boxes_visualization = True
    # If true use global average factor for CenterNet.
    center_net_global_avg_factor = False
    # List of static object types to include in bounding box detection.
    data_bb_static_types_white_list = [
        "static.prop.constructioncone",
        "static.prop.trafficwarning",
    ]
    # Confidence of a bounding box that is needed for the detection to be accepted.
    bb_confidence_threshold = 0.3
    # Maximum number of bounding boxes our system can detect.
    max_num_bbs = 90
    # CenterNet parameters
    # Number of direction bins for object orientation.
    num_dir_bins = 12
    # Top K center keypoints to consider during detection.
    top_k_center_keypoints = 100
    # Kernel size for CenterNet max pooling operation.
    center_net_max_pooling_kernel = 3
    # Number of input channels for bounding box detection head.
    bb_input_channel = 64
    # Extra width to add when car doors are open for safety.
    car_open_door_extra_width = 1.2
    # Total number of bounding box classes to detect.
    num_bb_classes = len(TransfuserBoundingBoxClass)

    # --- Context and statuses ---
    @overridable_property
    def use_discrete_command(self):
        """If true use discrete command input to the network."""
        if self.target_dataset == TargetDataset.WAYMO_E2E_2025_3CAMERAS:
            return False
        return True

    @property
    def discrete_command_dim(self):
        """Dimension of discrete command input."""
        if self.carla_leaderboard_mode:
            return 6
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 4
        elif self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS,
        ]:
            return 4
        raise ValueError("Unknown target dataset")

    # If true add noise to target points for robustness.
    use_noisy_tp = False
    # If true use the velocity as input to the network.
    use_velocity = True

    @property
    def max_speed(self):
        """Maximum speed limit for the vehicle in m/s."""
        if self.carla_leaderboard_mode:
            return 25.0
        if self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 15.0
        if self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS,
        ]:
            return 33.33
        raise ValueError("Unknown target dataset")

    @property
    def use_acceleration(self):
        """If true use the acceleration as input to the network."""
        return not self.carla_leaderboard_mode and self.target_dataset not in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS
        ]

    @property
    def max_acceleration(self):
        """Maximum acceleration for normalization."""
        if self.carla_leaderboard_mode:
            return 10.0
        if self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 4.0

    @property
    def use_previous_tp(self):
        """If true use the previous/visited target point as input to the network."""
        if self.carla_leaderboard_mode and not self.use_tfpp_planning_decoder:
            return True
        return False

    @property
    def use_next_tp(self):
        """If true use the next/subsequent target point as input to the network."""
        if self.carla_leaderboard_mode and not self.use_tfpp_planning_decoder:
            return True
        return False

    @property
    def use_tp(self):
        """If true use the current target point as input to the network."""
        if self.carla_leaderboard_mode:
            return True
        return False

    @property
    def target_points_normalization_constants(self):
        """Normalization constants for target points [x_norm, y_norm]."""
        return [[200.0, 50.0]]

    @property
    def tp_pop_distance(self):
        """Distance threshold for popping target points from route."""
        return 3.25

    @overridable_property
    def use_past_positions(self):
        """If true use past positions as input to the network."""
        return not self.carla_leaderboard_mode and self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS
        ]

    @overridable_property
    def use_past_speeds(self):
        """If true use past speeds as input to the network."""
        return not self.carla_leaderboard_mode and self.target_dataset in [
            TargetDataset.WAYMO_E2E_2025_3CAMERAS
        ]

    @overridable_property
    def num_past_samples_used(self):
        """Number of past samples to use as input to the network."""
        if self.use_past_positions or self.use_past_speeds:
            return 6
        return 0

    # --- Planning decoder configuration ---
    # Number of BEV cross-attention layers in TransFuser.
    transfuser_num_bev_cross_attention_layers = 6
    # Number of attention heads in BEV cross-attention.
    transfuser_num_bev_cross_attention_heads = 8
    # Dimension of tokens in the transformer.
    transfuser_token_dim = 256

    @property
    def predict_target_speed(self):
        """If true predict target speed."""
        return self.carla_leaderboard_mode

    @property
    def predict_spatial_path(self):
        """If true predict spatial path."""
        return self.carla_leaderboard_mode

    # If true predict temporal spatial waypoints.
    predict_temporal_spatial_waypoints = True

    # If true model will use the planning decoder.
    use_planning_decoder = False

    # If true use the carla_garage implementation of planning decoder
    use_tfpp_planning_decoder = False

    # GRU hidden size for planning decoder.
    gru_hidden_size = 64

    @property
    def target_speed_classes(self):
        """Carla target speed prediction classes in m/s."""
        return [
            0.0,
            4.0,
            8.0,
            10.0,
            13.88888888,
            16.0,
            17.77777777,
            20.0,
        ]

    @property
    def target_speeds(self):
        return self.target_speed_classes

    # If true smooth the route points with a spline.
    smooth_route = True
    # Number of route points we use for smoothing.
    num_route_points_smoothing = 20
    # Number of route checkpoints to predict. Needs to be smaller than num_route_points_smoothing!
    num_route_points_prediction = 10

    @property
    def num_way_points_prediction(self):
        """Number of waypoints to predict."""
        if self.carla_leaderboard_mode:
            return 8  # 2 seconds
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 8  # 4 seconds
        elif self.target_dataset in [TargetDataset.WAYMO_E2E_2025_3CAMERAS]:
            return 10  # 5 seconds
        raise ValueError("Unknown target dataset")

    # Spacing between waypoints in the prediction. For example: spacing 5 = 4Hz prediction.
    @property
    def waypoints_spacing(self):
        if self.carla_leaderboard_mode:
            return 5  # 4Hz
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return 10  # 2Hz
        elif self.target_dataset in [TargetDataset.WAYMO_E2E_2025_3CAMERAS]:
            return 10  # 2Hz
        raise ValueError("Unknown target dataset")

    # --- Image config ---
    @property
    def crop_height(self):
        """The amount of pixels cropped from the bottom of the image."""
        return (
            self.camera_calibration[1]["height"]
            - self.camera_calibration[1]["cropped_height"]
        )

    @property
    def carla_crop_height_type(self):
        """Type of cropping applied to CARLA images."""
        if self.carla_leaderboard_mode:
            return CarlaImageCroppingType.NONE
        elif self.target_dataset in [
            TargetDataset.NAVSIM_4CAMERAS,
        ]:
            return CarlaImageCroppingType.BOTTOM
        elif self.target_dataset in [TargetDataset.WAYMO_E2E_2025_3CAMERAS]:
            return CarlaImageCroppingType.NONE
        raise ValueError("Unknown target dataset")

    @property
    def image_width_before_camera_subselection(self):
        """Final width of images after loading from disk but before camera sub-selection."""
        return self.num_available_cameras * self.camera_calibration[1]["width"]

    @property
    def final_image_width(self):
        """Final width of images after cropping and camera sub-selection."""
        return self.num_used_cameras * self.camera_calibration[1]["width"]

    @property
    def final_image_height(self):
        """Final height of images after cropping."""
        return self.camera_calibration[1]["cropped_height"]

    @property
    def img_vert_anchors(self):
        """Number of vertical anchors for image feature maps."""
        return self.final_image_height // 32

    @property
    def img_horz_anchors(self):
        """Number of horizontal anchors for image feature maps."""
        return self.num_used_cameras * self.camera_calibration[1]["width"] // 32

    # --- TransFuser backbone ---
    # If true freeze the backbone weights during training.
    freeze_backbone = False
    # Architecture name for image encoder backbone.
    image_architecture = "resnet34"
    # Architecture name for LiDAR encoder backbone.
    lidar_architecture = "resnet34"
    # Latent TF
    LTF = False
    # Number of stages we do cross-modal fusion at. For example: if num_fusion_stages = 1, then we fuse only the last stage.
    num_fusion_stages = 4
    # If true train a status prediction
    only_ego_status_input = False

    # GPT Encoder
    # Block expansion factor for GPT layers.
    block_exp = 4
    # Number of transformer layers used in the vision backbone.
    n_layer = 2
    # Number of attention heads in transformer.
    n_head = 4
    # Embedding dropout probability.
    embd_pdrop = 0.1
    # Residual connection dropout probability.
    resid_pdrop = 0.1
    # Attention dropout probability.
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT.
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    # --- Semantic segmentation ---
    # If true use semantic segmentation as auxiliary loss.
    use_semantic = True
    # Total number of semantic segmentation classes.
    num_semantic_classes = len(TransfuserSemanticSegmentationClass)
    # Resolution at which the perspective auxiliary tasks are predicted
    perspective_downsample_factor = 1
    # Number of channels at the first deconvolution layer
    deconv_channel_num_0 = 128
    # Number of channels at the second deconvolution layer
    deconv_channel_num_1 = 64
    # Number of channels at the third deconvolution layer
    deconv_channel_num_2 = 32
    # Fraction of the down-sampling factor that will be up-sampled in the first Up-sample
    deconv_scale_factor_0 = 4
    # Fraction of the down-sampling factor that will be up-sampled in the second Up-sample.
    deconv_scale_factor_1 = 8

    # --- BEV Semantic ---
    # If true use bev semantic segmentation as auxiliary loss for training.
    use_bev_semantic = True
    # Total number of BEV semantic segmentation classes.
    num_bev_semantic_classes = len(TransfuserBEVSemanticClass)
    # Scale factor for pedestrian BEV semantic size.
    scale_pedestrian_bev_semantic_size = 2.5
    # Minimum extent for pedestrian BEV representation.
    pedestrian_bev_min_extent = 0.4
    # Number of channels for the BEV feature pyramid.
    bev_features_chanels = 64
    # Resolution at which the BEV auxiliary tasks are predicted.
    bev_down_sample_factor = 4
    # Upsampling factor for BEV features.
    bev_upsample_factor = 2

    # --- Mixed data training settings ---
    # If true use CARLA data for training.
    use_carla_data = True
    # Number of CARLA samples to use in mixed data training. -1 = use all data.
    carla_num_samples = -1
    # If true use NavSim data for training.
    use_navsim_data = False
    # NavSim data root directory.
    navsim_data_root = "data/navsim_training_cache/trainval"
    # Size of NavSim data portion in mixed data training. -1 = use all data.
    navsim_num_samples = -1
    # If true then we also schedule number of samples from CARLA in each batch.
    schedule_carla_num_samples = False
    # If true use Waymo E2E data for training
    use_waymo_e2e_data = False
    # Number of Waymo E2E data from training split
    waymo_e2e_num_training_samples = -1
    # Waymo E2E training data root directory.
    waymo_e2e_training_data_root = (
        "data/waymo_open_dataset_end_to_end_camera_v_1_0_0_training"
    )
    # Waymo E2E validation data root directory.
    waymo_e2e_val_data_root = (
        "data/waymo_open_dataset_end_to_end_camera_v_1_0_0_val_rfm"
    )
    # Waymo E2E test data root directory.
    waymo_e2e_test_data_root = (
        "data/waymo_open_dataset_end_to_end_camera_v_1_0_0_test_submission"
    )
    # Waymo E2E subsample factor for training data.
    waymo_e2e_subsample_factor = 5

    @property
    def navsim_num_bev_semantic_classes(self):
        """Number of BEV semantic classes in NavSim data."""
        return len(NavSimBEVSemanticClass)

    @property
    def navsim_num_bb_classes(self):
        """Number of bb classes in NavSim data."""
        return len(NavSimBBClass)

    @property
    def mixed_data_training(self):
        """If true use mixed data for training."""
        return (
            int(self.use_navsim_data)
            + int(self.use_carla_data)
            + int(self.use_waymo_e2e_data)
            > 1
        )

    @property
    def carla_leaderboard_mode(self):
        """If true use CARLA leaderboard mode settings."""
        return (
            self.target_dataset
            in [
                TargetDataset.CARLA_LEADERBOARD2_3CAMERAS,
                TargetDataset.CARLA_LEADERBOARD2_6CAMERAS,
            ]
            and not self.mixed_data_training
        )

    @beartype
    def detailed_loss_weights(
        self, source_dataset: int, epoch: int
    ) -> dict[str, float]:
        """Computed loss weights for all auxiliary tasks with normalization."""

        weights = {
            "loss_semantic": 1.0,
            "loss_depth": 0.00001,
            "loss_bev_semantic": 1.0,
            "loss_center_net_heatmap": 1.0,
            "loss_center_net_wh": 1.0,
            "loss_center_net_offset": 1.0,
            "loss_center_net_yaw_class": 1.0,
            "loss_center_net_yaw_res": 1.0,
            "loss_center_net_velocity": 1.0,
            "radar_loss": 1.0,
        }

        if source_dataset != SourceDataset.CARLA:
            weights["radar_loss"] = 0.0
            weights["loss_semantic"] = 0.0
            weights["loss_depth"] = 0.0
            weights["loss_center_net_velocity"] = 0.0

        if self.LTF:
            weights["loss_center_net_velocity"] = 0.0

        if not self.use_semantic:
            weights["loss_semantic"] = 0.0

        if not self.use_depth:
            weights["loss_depth"] = 0.0

        if not self.use_bev_semantic:
            weights["loss_bev_semantic"] = 0.0

        if not self.detect_boxes:
            weights["loss_center_net_heatmap"] = 0.0
            weights["loss_center_net_wh"] = 0.0
            weights["loss_center_net_offset"] = 0.0
            weights["loss_center_net_yaw_class"] = 0.0
            weights["loss_center_net_yaw_res"] = 0.0
            weights["loss_center_net_velocity"] = 0.0

        if self.training_used_lidar_steps <= 1:
            weights["loss_center_net_velocity"] = 0.0

        if not self.radar_detection:
            weights["radar_loss"] = 0.0

        # Add prefix to the loss weights based on the source dataset
        prefix = f"{SOURCE_DATASET_NAME_MAP[source_dataset]}_"
        if source_dataset == SourceDataset.CARLA:
            prefix = ""
        weights = {f"{prefix}{k}": v for k, v in weights.items()}

        # Unified planning loss, no source dataset prefix
        weights.update(
            {
                "loss_spatio_temporal_waypoints": 1.0,
                "loss_target_speed": 1.0,
                "loss_spatial_route": 1.0,
            }
        )

        # Disable planning losses during pretraining
        if not self.use_planning_decoder:
            weights["loss_spatio_temporal_waypoints"] = 0.0
            weights["loss_spatial_route"] = 0.0
            weights["loss_target_speed"] = 0.0

        return weights

    @property
    def log_scalars_frequency(self):
        """How often to log scalar values during training."""
        if not self.is_on_slurm:
            return 1
        try:
            with open(
                "scripts/slurm/configs/wandb_log_frequency_training_scalar.txt"
            ) as f:
                return int(f.readline().strip())
        except Exception as e:
            print(f"Error reading log frequency file: {e}.")
            return 1

    @property
    def log_images_frequency(self):
        """How often to log images during training."""
        if not self.is_on_slurm:
            return 1
        try:
            with open(
                "scripts/slurm/configs/wandb_log_frequency_training_images.txt"
            ) as f:
                return int(f.readline().strip())
        except Exception as e:
            print(f"Error reading log frequency file: {e}.")
            return 1000

    @property
    def slurm_job_id(self):
        """Current SLURM job ID if running on SLURM cluster."""
        if self.is_on_slurm:
            return os.environ.get("SLURM_JOB_ID", "0")
        return None

    @property
    def log_wandb(self):
        """If true log metrics to Weights & Biases."""
        if self.is_on_slurm:
            return True
        return False

    # --- Hardware configuration ---
    @property
    def gpu_name(self):
        """Normalized GPU name for hardware-specific configurations."""
        try:
            name = torch.cuda.get_device_name().lower()
            if "rtx 2080 ti" in name:
                return "rtx2080ti"
            elif "gtx 1080 ti" in name:
                return "gtx1080ti"
            elif "a100" in name:
                return "a100"
            elif "l40s" in name:
                return "l40s"
            elif "a4000" in name:
                return "a4000"
            elif "rtx 6000" in name and "ada" in name:
                return "rtx6000ada"
            else:
                raise Exception(f"Unknown GPU name: {name}")
        except RuntimeError:
            return ""

    @property
    def rank(self):
        """Current process rank in distributed training."""
        return int(os.environ.get("RANK", "0"))

    @property
    def world_size(self):
        """Total number of processes in distributed training."""
        return int(os.environ.get("WORLD_SIZE", "1"))

    @property
    def local_rank(self):
        """Local rank of current process on the node."""
        return int(os.environ.get("LOCAL_RANK", "0"))

    @property
    def device(self):
        """PyTorch device to use for training."""
        return torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )

    @property
    def assigned_cpu_cores(self):
        """Number of CPU cores assigned to this job."""
        if "SLURM_JOB_ID" in os.environ:
            cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
            if cpus_per_task:
                return int(cpus_per_task)
        return 8

    @property
    def workers_per_cpu_cores(self):
        """Number of data loader workers per CPU core."""
        if not self.mixed_data_training and not self.use_carla_data:
            return 1  # Use more workers for mixed data training. CARLA loader is slow.
        return 1

    def training_dict(self):
        out = {}
        cls = self.__class__
        for k, v in cls.__dict__.items():
            if isinstance(v, property):
                try:
                    out[k] = getattr(self, k)
                except Exception:
                    pass
            else:
                if not k.startswith("__") and not callable(v):
                    out[k] = v
        for k, v in self.__dict__.items():
            if isinstance(v, property):
                try:
                    out[k] = getattr(self, k)
                except Exception:
                    pass
            else:
                if not k.startswith("__") and not callable(v):
                    out[k] = v
        return out

    def __init__(
        self, loaded_config: dict = None, raise_error_on_missing_key: bool = False
    ):
        """Constructor for training config."""
        super().__init__()
        self.parse(
            loaded_config=loaded_config,
            env_key="TRAINING_CONFIG",
            raise_error_on_missing_key=raise_error_on_missing_key,
        )


class TransfuserBackbone(nn.Module):
    @beartype
    def __init__(self, device: torch.device, config: TrainingConfig):
        super().__init__()
        self.device = device
        self.config = config

        # Image branch
        self.image_encoder = timm.create_model(
            config.image_architecture, pretrained=False, features_only=True
        )
        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        image_start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            image_start_index += 1
        self.num_image_features = self.image_encoder.feature_info.info[
            image_start_index + 3
        ]["num_chs"]

        # LiDAR branch
        if self.config.num_fusion_stages > 1:
            self.lidar_encoder = timm.create_model(
                config.lidar_architecture,
                pretrained=False,
                in_chans=2 if config.LTF else 1,
                features_only=True,
            )
            lidar_start_index = 0
            if len(self.lidar_encoder.return_layers) > 4:
                lidar_start_index += 1
            self.num_lidar_features = self.lidar_encoder.feature_info.info[
                lidar_start_index + 3
            ]["num_chs"]
            self.lidar_channel_to_img = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.lidar_encoder.feature_info.info[lidar_start_index + i][
                            "num_chs"
                        ],
                        self.image_encoder.feature_info.info[image_start_index + i][
                            "num_chs"
                        ],
                        kernel_size=1,
                    )
                    for i in range(4 - config.num_fusion_stages, 4)
                ]
            )
            self.img_channel_to_lidar = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.image_encoder.feature_info.info[image_start_index + i][
                            "num_chs"
                        ],
                        self.lidar_encoder.feature_info.info[lidar_start_index + i][
                            "num_chs"
                        ],
                        kernel_size=1,
                    )
                    for i in range(4 - config.num_fusion_stages, 4)
                ]
            )
        else:
            self.num_lidar_features = self.num_image_features
            self.bev_queries = nn.Parameter(
                torch.randn(
                    1,
                    self.num_lidar_features,
                    self.config.lidar_vert_anchors,
                    self.config.lidar_horz_anchors,
                )
            )
            self.lidar_channel_to_img = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.num_lidar_features,
                        self.image_encoder.feature_info.info[image_start_index + 3][
                            "num_chs"
                        ],
                        kernel_size=1,
                    )
                ]
            )
            self.img_channel_to_lidar = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.image_encoder.feature_info.info[image_start_index + 3][
                            "num_chs"
                        ],
                        self.num_lidar_features,
                        kernel_size=1,
                    )
                ]
            )
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(
            (self.config.lidar_vert_anchors, self.config.lidar_horz_anchors)
        )

        # Fusion transformers
        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[image_start_index + i][
                        "num_chs"
                    ],
                    config=config,
                )
                for i in range(4 - config.num_fusion_stages, 4)
            ]
        )

        # Post-fusion convs
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[image_start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        self.upsample = nn.Upsample(
            scale_factor=self.config.bev_upsample_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.upsample2 = nn.Upsample(
            size=(
                self.config.lidar_height_pixel // self.config.bev_down_sample_factor,
                self.config.lidar_width_pixel // self.config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )
        self.up_conv5 = nn.Conv2d(
            self.config.bev_features_chanels,
            self.config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.up_conv4 = nn.Conv2d(
            self.config.bev_features_chanels,
            self.config.bev_features_chanels,
            (3, 3),
            padding=1,
        )
        self.c5_conv = nn.Conv2d(
            self.num_lidar_features, self.config.bev_features_chanels, (1, 1)
        )

    def top_down(self, x):
        p5 = F.relu(self.c5_conv(x), inplace=True)
        p4 = F.relu(self.up_conv5(self.upsample(p5)), inplace=True)
        p3 = F.relu(self.up_conv4(self.upsample2(p4)), inplace=True)
        return p3

    def forward(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = data["rgb"].to(
            self.device, dtype=self.config.torch_float_type, non_blocking=True
        )
        if self.config.LTF:
            if self.config.num_fusion_stages > 1:
                x = torch.linspace(0, 1, self.config.lidar_width_pixel)
                y = torch.linspace(0, 1, self.config.lidar_height_pixel)
                y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

                lidar = torch.zeros(
                    (
                        rgb.shape[0],
                        2,
                        self.config.lidar_height_pixel,
                        self.config.lidar_width_pixel,
                    ),
                    device=rgb.device,
                )
                lidar[:, 0] = y_grid.unsqueeze(0)  # Top down positional encoding
                lidar[:, 1] = x_grid.unsqueeze(0)  # Left right positional encoding
            else:
                lidar = None
        else:
            lidar = data["lidar"].to(
                self.device, dtype=self.config.torch_float_type, non_blocking=True
            )
        return self._forward(rgb, lidar)

    @jt.jaxtyped(typechecker=beartype)
    def _forward(
        self,
        image: jt.Float[torch.Tensor, "B 3 img_h img_w"],
        lidar: (
            jt.Float[torch.Tensor, "B 1 bev_h bev_w"]
            | jt.Float[torch.Tensor, "B 2 bev_h bev_w"]
            | None
        ),
    ) -> tuple[
        jt.Float[torch.Tensor, "B D1 H1 W1"], jt.Float[torch.Tensor, "B D2 H2 W2"]
    ]:
        """
        Image + LiDAR feature fusion using transformers
        Args:
            image: RGB image.
            lidar: Pseudo-image LiDAR.
        Returns:
            lidar_features: BEV feature map for planning.
            image_features: Image feature map for perception.
        """
        image_features = normalize_imagenet(image)
        lidar_features = lidar

        if self.config.channel_last:
            image = image.to(memory_format=torch.channels_last)
            if lidar is not None:
                lidar = lidar.to(memory_format=torch.channels_last)

        # Generate an iterator for all the layers in the network that one can loop through.
        image_layers = iter(self.image_encoder.items())
        if self.config.num_fusion_stages > 1:
            lidar_layers = iter(self.lidar_encoder.items())

        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
        if (
            self.config.num_fusion_stages > 1
            and len(self.lidar_encoder.return_layers) > 4
        ):
            lidar_features = self.forward_layer_block(
                lidar_layers, self.lidar_encoder.return_layers, lidar_features
            )

        # Loop through the 4 blocks of the network.
        for i in range(4):
            # Branch-specific forward pass
            image_features = self.forward_layer_block(
                image_layers, self.image_encoder.return_layers, image_features
            )
            if self.config.num_fusion_stages > 1:
                lidar_features = self.forward_layer_block(
                    lidar_layers, self.lidar_encoder.return_layers, lidar_features
                )

            # Fusion stages
            if i >= 4 - self.config.num_fusion_stages:
                if self.config.num_fusion_stages == 1:
                    lidar_features = self.bev_queries.expand(
                        image_features.shape[0], -1, -1, -1
                    )  # Expand to batch size
                image_features, lidar_features = self.fuse_features(
                    image_features,
                    lidar_features,
                    i - (4 - self.config.num_fusion_stages),
                )

        return lidar_features, image_features

    @beartype
    def forward_layer_block(
        self, layers, return_layers: dict[str, str], features: torch.Tensor
    ) -> torch.Tensor:
        """Run one forward pass to a block of layers from a TIMM neural network and returns the result.
        Advances the whole network by just one block.

        Args:
            layers: Iterator starting at the current layer block of the target network.
            return_layers: TIMM dictionary describing at which intermediate layers features are returned.
            features: Input features.

        Return:
            torch.Tensor: Processed features
        """
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(
        self, image_features: torch.Tensor, lidar_features: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a TransFuser feature fusion block using a Transformer module.
        Args:
            image_features: Features from the image branch
            lidar_features: Features from the LiDAR branch
            layer_idx: Transformer layer index.
        Returns:
            image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer, lidar_embd_layer
        )

        lidar_features_layer = self.img_channel_to_lidar[layer_idx](
            lidar_features_layer
        )
        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    def __init__(self, n_embd, config):
        super().__init__()
        self.n_embd = n_embd
        self.config = config
        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.config.img_vert_anchors * self.config.img_horz_anchors
                + self.config.lidar_vert_anchors * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for layer in range(config.n_layer)
            ]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor (tensor): B, C, H, W
            lidar_tensor (tensor): B, C, H, W
        """
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        image_tensor = (
            image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        lidar_tensor = (
            lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[:, : self.config.img_vert_anchors * self.config.img_horz_anchors, :]
            .view(bz, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[:, self.config.img_vert_anchors * self.config.img_horz_anchors :, :]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.dropout = attn_pdrop
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()
        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = (
            self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        q = (
            self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        v = (
            self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class PlanningDecoder(nn.Module):
    @beartype
    def __init__(
        self, input_bev_channels: int, config: TrainingConfig, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.planning_context_encoder = PlanningContextEncoder(
            config=self.config,
            input_bev_channels=input_bev_channels,
            device=self.device,
        )

        num_queries = 0
        if self.config.predict_temporal_spatial_waypoints:
            num_queries += self.config.num_way_points_prediction

        self.query = nn.Parameter(
            torch.zeros(
                1,
                num_queries,
                self.config.transfuser_token_dim,
            )
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                self.config.transfuser_token_dim,
                self.config.transfuser_num_bev_cross_attention_heads,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=self.config.transfuser_num_bev_cross_attention_layers,
            norm=nn.LayerNorm(self.config.transfuser_token_dim),
        )

        # Only create decoders if needed
        if self.config.predict_temporal_spatial_waypoints:
            self.wp_decoder = nn.Linear(config.transfuser_token_dim, 2)
            if self.config.use_navsim_data:
                self.heading_decoder = nn.Linear(config.transfuser_token_dim, 1)

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "bs bev_dim height_bev width_bev"],
        data: dict,
        log: dict,
    ) -> tuple[
        jt.Float[torch.Tensor, "B n_waypoints 2"],
        jt.Float[torch.Tensor, "B n_waypoints"] | None,
    ]:
        """
        Args:
            bev_features: BEV features.
            radar_features: Radar features.
            radar_predictions: Radar predictions.
            data: dict
            log: dict
        Returns:
            waypoints: Spatial and temporal path.
            headings: Heading predictions (if using NavSim data).
        """
        self.kv = context_tokens = self.planning_context_encoder(
            bev_features=bev_features, data=data
        )

        bs = context_tokens.shape[0]

        queries = self.transformer_decoder(self.query.repeat(bs, 1, 1), context_tokens)

        # Split the queries flexibly based on what we're predicting
        query_idx = 0
        waypoints = None
        headings = None

        if self.config.predict_temporal_spatial_waypoints:
            waypoints_queries = queries[
                :, query_idx : query_idx + self.config.num_way_points_prediction
            ]
            waypoints = torch.cumsum(self.wp_decoder(waypoints_queries), 1)
            if self.config.use_navsim_data:
                headings = torch.cumsum(self.heading_decoder(waypoints_queries), 1)
            query_idx += self.config.num_way_points_prediction

        return (waypoints, headings.squeeze(-1) if headings is not None else None)


class PlanningContextEncoder(nn.Module):
    @beartype
    def __init__(
        self, config: TrainingConfig, input_bev_channels: int, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.config: TrainingConfig = config

        self.num_status_tokens = 0

        if self.config.use_velocity:
            self.num_status_tokens += 1
            self.velocity_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )

        if self.config.use_acceleration:
            self.num_status_tokens += 1
            self.acceleration_encoder = nn.Sequential(
                nn.Linear(1, self.config.transfuser_token_dim),
            )

        if self.config.use_discrete_command:
            self.num_status_tokens += 1
            self.command_encoder = nn.Sequential(
                nn.Linear(
                    self.config.discrete_command_dim, self.config.transfuser_token_dim
                )
            )

        self.cosine_pos_embeding = PositionEmbeddingSine(
            config, self.config.transfuser_token_dim // 2, normalize=True
        )
        self.status_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_status_tokens, self.config.transfuser_token_dim)
        )

        self.dimension_adapter = nn.Conv2d(
            input_bev_channels, self.config.transfuser_token_dim, kernel_size=1
        )
        self.reset_parameters()

        self.target_points_normalization_constants = torch.tensor(
            self.config.target_points_normalization_constants,
            device=self.device,
            dtype=self.config.torch_float_type,
        )

    def reset_parameters(self):
        nn.init.uniform_(self.status_pos_embedding)

    @beartype
    def forward(
        self,
        bev_features: jt.Float[torch.Tensor, "B C H W"],
        data: dict,
    ) -> jt.Float[torch.Tensor, "B N D"]:
        """
        Args:
            bev_features: Raw BEV features.
            radar_logits: Radar logits.
            radar_predictions: Radar predictions.
            data: dict
            log: dict
        Returns:
            context_tokens: Output tokens for planning transformer decoder.
        """
        # Load data
        if self.config.use_velocity:
            velocity = (
                data["speed"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
        if self.config.use_discrete_command:
            command = data["command"].to(
                self.device, dtype=self.config.torch_float_type
            )

        status_tokens = []

        # Encode speed
        if self.config.use_velocity:
            velocity_token = self.velocity_encoder(
                velocity / self.config.max_speed
            ).reshape(
                -1, 1, self.config.transfuser_token_dim
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(velocity_token)

        # Encode acceleration
        if self.config.use_acceleration:
            acceleration = (
                data["acceleration"]
                .reshape(-1, 1)
                .to(self.device, dtype=self.config.torch_float_type)
            )
            acceleration_token = self.acceleration_encoder(
                acceleration / self.config.max_acceleration
            ).reshape(
                -1, 1, self.config.transfuser_token_dim
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(acceleration_token)

        # Encode command
        if self.config.use_discrete_command:
            command_token = self.command_encoder(command).reshape(
                -1, 1, self.config.transfuser_token_dim
            )  # (bs, 1, transfuser_token_dim)
            status_tokens.append(command_token)

        # Concatenate status tokens if any
        has_statuses = False
        if len(status_tokens) > 0:
            status_tokens = torch.cat(
                status_tokens, dim=1
            )  # (bs, num_status_tokens, transfuser_token_dim)
            has_statuses = True

        # Process BEV features
        context_tokens = self.dimension_adapter(
            bev_features
        )  # (bs, transfuser_token_dim, height, width)

        # Concatenate and add positional embeddings
        if has_statuses:
            context_tokens = context_tokens + self.cosine_pos_embeding(
                context_tokens
            )  # (bs, transfuser_token_dim, height, width)
            context_tokens = torch.flatten(
                context_tokens, start_dim=2
            )  # (bs, transfuser_token_dim, height * width)
            context_tokens = torch.permute(
                context_tokens, (0, 2, 1)
            )  # (bs, height * width, transfuser_token_dim)

            status_tokens = (
                status_tokens + self.status_pos_embedding
            )  # (bs, num_status_tokens, transfuser_token_dim)
            context_tokens = torch.cat(
                [context_tokens, status_tokens], dim=1
            )  # (bs, height * width + num_status_tokens, transfuser_token_dim)

        return context_tokens


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        config: TrainingConfig,
        num_pos_feats=64,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.config = config
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor: torch.Tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos.to(self.config.torch_float_type).contiguous()


@jt.jaxtyped(typechecker=beartype)
@dataclass
class Prediction:
    """Raw output predictions from the model."""

    # Planning prediction
    pred_future_waypoints: jt.Float[torch.Tensor, "bs n_waypoints 2"] | None
    pred_headings: jt.Float[torch.Tensor, "bs n_waypoints"] | None


class Model(nn.Module):
    @beartype
    def __init__(
        self,
        device: torch.device,
        config: TrainingConfig,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.log = {}

        self.backbone = TransfuserBackbone(self.device, self.config)

        self.planning_decoder = PlanningDecoder(
            input_bev_channels=self.backbone.num_lidar_features,
            config=self.config,
            device=self.device,
        ).to(self.device)

    @beartype
    def forward(self, data: dict[str, typing.Any]) -> Prediction:
        self.log = {}
        pred_future_waypoints = pred_headings = None

        bev_features, image_features = self.backbone(data)

        # Planning heads
        if self.config.use_planning_decoder:
            (pred_future_waypoints, pred_headings) = self.planning_decoder(
                bev_features, data, log=self.log
            )

        # Collect predictions
        return Prediction(
            # Planning prediction
            pred_future_waypoints=pred_future_waypoints,
            pred_headings=pred_headings,
        )


def load_tf(model_path: str, device: torch.device):
    base_dir = os.path.dirname(model_path)
    with open(os.path.join(base_dir, "config.json")) as f:
        loaded_config = json.load(f)
    config_training = TrainingConfig(loaded_config)
    # Load model
    model = Model(device=device, config=config_training)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    model.eval()
    return model


class NavsimData(torch.utils.data.Dataset):
    """Data loader for NavSim data"""

    def __init__(self, root, config: TrainingConfig):
        self.root = root
        self.config = config

        self.feature = glob.glob(
            os.path.join(self.root, "**/transfuser_feature.gz"), recursive=True
        )
        self.target = glob.glob(
            os.path.join(self.root, "**/transfuser_target.gz"), recursive=True
        )

        self.size = len(self.feature)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Initialize cache or cache dummy
        feature_path = self.feature[index]
        with gzip.open(feature_path, "rb") as f:
            feature = pickle.load(f)

        rgb = cv2.imdecode(
            np.frombuffer(feature["camera_feature"], np.uint8), cv2.IMREAD_COLOR
        )
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC to CHW

        data = {
            "rgb": rgb,
            "command": feature["status_feature"][:4],
            "speed": np.linalg.norm(feature["status_feature"][4:6]),
            "acceleration": np.linalg.norm(feature["status_feature"][6:8]),
        }
        return data
