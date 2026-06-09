# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from trajdata import MapAPI, VectorMap
from trajdata.maps.vec_map_elements import MapElementType
from trajdata.utils import map_utils


def main():
    cache_path = Path("~/.unified_data_cache").expanduser()
    map_api = MapAPI(cache_path)

    ### Loading random scene and initializing VectorMap.
    env_name: str = np.random.choice(["nusc_mini", "lyft_sample", "nuplan_mini"])
    random_location_dict: Dict[str, str] = {
        "nuplan_mini": np.random.choice(
            ["boston", "singapore", "pittsburgh", "las_vegas"]
        ),
        "nusc_mini": np.random.choice(["boston-seaport", "singapore-onenorth"]),
        "lyft_sample": "palo_alto",
    }

    start = time.perf_counter()
    vec_map: VectorMap = map_api.get_map(
        f"{env_name}:{random_location_dict[env_name]}", incl_road_areas=True
    )
    end = time.perf_counter()
    print(f"Map loading took {(end - start)*1000:.2f} ms")

    start = time.perf_counter()
    vec_map: VectorMap = map_api.get_map(
        f"{env_name}:{random_location_dict[env_name]}", incl_road_areas=True
    )
    end = time.perf_counter()
    print(f"Repeated (cached in memory) map loading took {(end - start)*1000:.2f} ms")

    print(f"Randomly chose {vec_map.env_name}, {vec_map.map_name} map.")

    ### Lane Graph Visualization (with rasterized map in background)
    fig, ax = plt.subplots()

    print(f"Rasterizing Map...")
    start = time.perf_counter()
    map_img, raster_from_world = vec_map.rasterize(
        resolution=2,
        return_tf_mat=True,
        incl_centerlines=False,
        area_color=(255, 255, 255),
        edge_color=(0, 0, 0),
        scene_ts=100,
    )
    end = time.perf_counter()
    print(f"Map rasterization took {(end - start)*1000:.2f} ms")

    ax.imshow(map_img, alpha=0.5, origin="lower")

    lane_idx = np.random.randint(0, len(vec_map.lanes))
    print(f"Visualizing random lane index {lane_idx}...")
    start = time.perf_counter()
    vec_map.visualize_lane_graph(
        origin_lane=lane_idx,
        num_hops=10,
        raster_from_world=raster_from_world,
        ax=ax,
    )
    end = time.perf_counter()
    print(f"Lane visualization took {(end - start)*1000:.2f} ms")

    point = vec_map.lanes[lane_idx].center.xyz[0, :]
    point_raster = map_utils.transform_points(
        point[None, :], transf_mat=raster_from_world
    )
    ax.scatter(point_raster[:, 0], point_raster[:, 1])

    print("Getting nearest road area...")
    start = time.perf_counter()
    area = vec_map.get_closest_area(point, elem_type=MapElementType.ROAD_AREA)
    end = time.perf_counter()
    print(f"Getting nearest area took {(end-start)*1000:.2f} ms")

    raster_pts = map_utils.transform_points(area.exterior_polygon.xy, raster_from_world)
    ax.fill(raster_pts[:, 0], raster_pts[:, 1], alpha=1.0, color="C0")

    print("Getting road areas within 100m...")
    start = time.perf_counter()
    areas = vec_map.get_areas_within(
        point, elem_type=MapElementType.ROAD_AREA, dist=100.0
    )
    end = time.perf_counter()
    print(f"Getting areas within took {(end-start)*1000:.2f} ms")

    for area in areas:
        raster_pts = map_utils.transform_points(
            area.exterior_polygon.xy, raster_from_world
        )
        ax.fill(raster_pts[:, 0], raster_pts[:, 1], alpha=0.2, color="C1")

    ax.axis("equal")
    ax.grid(None)

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
