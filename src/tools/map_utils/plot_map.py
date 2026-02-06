# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""
A tool to plot the map data from an artifact and also visualize the route if requested.
Usage:
    python plot_map.py --artifact <artifact_name> --preview_route
"""

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alpasim_runtime.route_generator import RouteGeneratorMap
from alpasim_utils.artifact import Artifact
from trajdata.dataset_specific.xodr.geo_transform import get_t_rig_enu_from_ecef
from trajdata.dataset_specific.xodr.vector_map_export import (
    populate_vector_map_from_xodr,
)
from trajdata.maps import VectorMap

try:
    matplotlib.use("Qt5Agg")
except Exception:
    print("Could not set matplotlib backend to Qt5Agg, using default backend.")
    pass


def plot_clipgt_map(map_dir):

    df = pd.read_parquet(map_dir + "/crosswalk.parquet")
    for element in df["crosswalk"]:
        x = np.array([loc["x"] for loc in element["location"]])
        y = np.array([loc["y"] for loc in element["location"]])
        plt.plot(x, y, color="green")

    df = pd.read_parquet(map_dir + "/wait_line.parquet")
    for element in df["wait_line"]:
        x_w = np.array([loc["x"] for loc in element["location"]])
        y_w = np.array([loc["y"] for loc in element["location"]])
        if element["category"] == "STOP":
            color = "red"
        else:
            color = "blue"
        plt.plot(x_w, y_w, color=color)

    df = pd.read_parquet(map_dir + "/lane.parquet")
    for element in df["lane"]:
        left_rail = np.array(
            [[point["x"], point["y"]] for point in element["left_rail"]]
        )
        right_rail = np.array(
            [[point["x"], point["y"]] for point in element["right_rail"]]
        )
        center_rail = (left_rail + right_rail) / 2.0
        plt.plot(left_rail[:, 0], left_rail[:, 1], "g--")
        plt.plot(right_rail[:, 0], right_rail[:, 1], "g--")
        plt.plot(center_rail[:, 0], center_rail[:, 1], "k--")

    df = pd.read_parquet(map_dir + "/egomotion_estimate.parquet")
    ests = [est for est in df["egomotion_estimate"]]
    x_e = [est["location"]["x"] for est in ests]
    y_e = [est["location"]["y"] for est in ests]
    plt.plot(x_e, y_e, color="black")


def plot_xodr_map(
    xodr_path: str, rig_trajectories_path: str, show_traffic_signs: bool = False
):
    """Plot XODR map using alpasim's visualization infrastructure."""

    print("Loading XODR map and trajectory data...")

    # Load XODR file
    with open(xodr_path, "r") as f:
        xodr_xml = f.read()

    # Load trajectory data for coordinate transformation
    with open(rig_trajectories_path, "r") as rig_file:
        rig_data = json.load(rig_file)

    # Apply coordinate transformation
    t_world_base = np.asarray(rig_data["T_world_base"])
    t_nurec_map = get_t_rig_enu_from_ecef(t_world_base, xodr_xml)
    transform_mat = np.linalg.inv(t_nurec_map)

    # Parse XML root for road analysis
    root = ET.fromstring(xodr_xml)

    # Create and populate VectorMap
    vmap = VectorMap(map_id="xodr_test:visualization")
    populate_vector_map_from_xodr(vmap, xodr_xml, t_xodr_enu_to_sim=transform_mat)
    vmap.__post_init__()
    vmap.compute_search_indices()

    # Import MapElementType to access elements
    from trajdata.maps.vec_map_elements import MapElementType

    print("Map loaded successfully:")
    print(f"  - Lanes: {len(vmap.lanes)}")
    print(f"  - Road edges: {len(vmap.road_edges)}")
    print(
        f"  - Traffic signs: {len(vmap.elements.get(MapElementType.TRAFFIC_SIGN, {}))}"
    )
    print(f"  - Wait lines: {len(vmap.elements.get(MapElementType.WAIT_LINE, {}))}")
    print(f"  - Sidewalks: {len(vmap.elements.get(MapElementType.PED_WALKWAY, {}))}")
    print(f"  - Crosswalks: {len(vmap.elements.get(MapElementType.PED_CROSSWALK, {}))}")
    print(f"  - Extent: {vmap.extent}")

    # Note: With the refactored code, edge IDs are preserved in their original format
    # (e.g., "roadId_L" or "roadId_R"), so we can still analyze them
    junction_edges = 0
    regular_edges = 0
    for edge in vmap.road_edges:
        try:
            road_id = edge.id.split("_")[0]
            is_junction = any(
                road.attrib.get("junction", "-1") != "-1"
                for road in root.findall("road")
                if road.attrib["id"] == road_id
            )
            if is_junction:
                junction_edges += 1
            else:
                regular_edges += 1
        except Exception:
            # If we can't parse the edge ID, count it as regular
            regular_edges += 1
    print(f"  - Junction road edges: {junction_edges}")
    print(f"  - Regular road edges: {regular_edges}")

    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.set_aspect("equal")
    title = "XODR Map Visualization"
    ax.set_title(f"{title} – XODR Map & Trajectory", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Render map elements manually
    print("Rendering map elements...")

    print(f"  Total lanes: {len(vmap.lanes)}")

    # Plot all lane centers in blue with direction arrows
    for i, lane in enumerate(vmap.lanes):
        points = lane.center.points[:, :2]
        ax.plot(
            points[:, 0],
            points[:, 1],
            "b-",
            linewidth=1.5,
            alpha=0.8,
            label="Lane Centers" if i == 0 else "",
        )

        # Add lane ID annotation at the middle of the lane
        if len(points) > 0:
            mid_idx = len(points) // 2
            mid_x, mid_y = points[mid_idx]
            # Add lane ID text with a background box for better visibility
            ax.annotate(
                f"Lane {lane.id}",
                xy=(mid_x, mid_y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="darkblue",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="darkblue",
                    alpha=0.8,
                ),
                ha="left",
                va="bottom",
            )

        # Add direction arrows every 20 points
        if hasattr(lane.center, "h"):  # Check if heading data exists
            for j in range(0, len(points), 20):
                if j < len(lane.center.h):
                    x, y = points[j]
                    heading = lane.center.h[j]
                    # Arrow points in the direction of travel
                    dx = 3 * np.cos(heading)
                    dy = 3 * np.sin(heading)
                    ax.arrow(
                        x,
                        y,
                        dx,
                        dy,
                        head_width=1.5,
                        head_length=1,
                        fc="red",
                        ec="red",
                        alpha=0.6,
                    )

    # Plot lane boundaries (lighter) - track if we've added to legend
    left_edge_plotted = False
    for lane in vmap.lanes:
        if lane.left_edge is not None:
            points = lane.left_edge.points[:, :2]
            label = "Lane Edges" if not left_edge_plotted else ""
            ax.plot(
                points[:, 0], points[:, 1], "g-", linewidth=0.5, alpha=0.4, label=label
            )
            left_edge_plotted = True
        if lane.right_edge is not None:
            points = lane.right_edge.points[:, :2]
            ax.plot(points[:, 0], points[:, 1], "g-", linewidth=0.5, alpha=0.4)

    # Plot road edges
    road_edge_plotted = False
    for road_edge in vmap.road_edges:
        points = road_edge.polyline.points[:, :2]
        label = "Road Edges" if not road_edge_plotted else ""
        ax.plot(points[:, 0], points[:, 1], "k-", linewidth=1, alpha=0.6, label=label)
        road_edge_plotted = True

    # ------------------------------------------------------------------
    # Plot rig trajectory (ego ground truth)
    # ------------------------------------------------------------------
    """
    try:
        artifact = Artifact(source=usdz_path)
        rig_waypoints = artifact.rig.trajectory.positions  # (N,3) in NuRec space
        ax.plot(
            rig_waypoints[:, 0],
            rig_waypoints[:, 1],
            "r-",
            linewidth=2,
            label="Rig Trajectory",
        )
        # Mark starting point
        ax.scatter(
            rig_waypoints[0, 0],
            rig_waypoints[0, 1],
            c="red",
            s=40,
            marker="o",
            label="Start",
        )
    except Exception as e:
        print(f"Warning: Could not plot rig trajectory: {e}")
    """

    # Plot traffic signs (if enabled)
    traffic_signs = vmap.elements.get(MapElementType.TRAFFIC_SIGN, {})
    if show_traffic_signs and traffic_signs:
        traffic_sign_x = []
        traffic_sign_y = []
        for sign in traffic_signs.values():
            pos = sign.position
            # Handle both scalar and array positions
            if hasattr(pos, "__len__"):
                traffic_sign_x.append(pos[0])
                traffic_sign_y.append(pos[1])
            else:
                # Skip if position is not valid
                continue
        if traffic_sign_x:
            ax.scatter(
                traffic_sign_x,
                traffic_sign_y,
                c="red",
                s=100,
                marker="^",
                edgecolors="darkred",
                linewidth=2,
                label="Traffic Signs",
                zorder=10,
            )
            print(f"  Plotted {len(traffic_sign_x)} traffic signs")

    # Plot wait lines (stop lines and yield lines)
    wait_lines = vmap.elements.get(MapElementType.WAIT_LINE, {})
    stop_line_plotted = False
    yield_line_plotted = False
    stop_count = 0
    yield_count = 0
    if wait_lines:
        for wait_line in wait_lines.values():
            if hasattr(wait_line, "polyline") and wait_line.polyline:
                points = wait_line.polyline.points[:, :2]
                # Determine color and label based on wait line type
                wl_type = getattr(wait_line, "wait_line_type", "Stop")
                if wl_type == "Yield":
                    color = "orange"
                    label = "Yield Lines" if not yield_line_plotted else ""
                    yield_line_plotted = True
                    yield_count += 1
                else:
                    color = "magenta"
                    label = "Stop Lines" if not stop_line_plotted else ""
                    stop_line_plotted = True
                    stop_count += 1
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    color,
                    linewidth=3,
                    alpha=0.8,
                    label=label,
                )
        print(f"  Plotted {stop_count} stop lines and {yield_count} yield lines")

    # Plot sidewalks (PedWalkway)
    sidewalks = vmap.elements.get(MapElementType.PED_WALKWAY, {})
    if sidewalks:
        sidewalk_plotted = False
        for sidewalk in sidewalks.values():
            if hasattr(sidewalk, "polygon") and sidewalk.polygon:
                points = sidewalk.polygon.points[:, :2]
                # Close the polygon for visualization
                points_closed = np.vstack([points, points[0]])
                label = "Sidewalks" if not sidewalk_plotted else ""
                ax.fill(
                    points_closed[:, 0],
                    points_closed[:, 1],
                    color="lightgray",
                    alpha=0.5,
                    edgecolor="gray",
                    linewidth=1,
                    label=label,
                )
                sidewalk_plotted = True
        print(f"  Plotted {len(sidewalks)} sidewalks")

    # Plot crosswalks (PedCrosswalk)
    crosswalks = vmap.elements.get(MapElementType.PED_CROSSWALK, {})
    if crosswalks:
        crosswalk_plotted = False
        for crosswalk in crosswalks.values():
            if hasattr(crosswalk, "polygon") and crosswalk.polygon:
                points = crosswalk.polygon.points[:, :2]
                # Close the polygon for visualization
                points_closed = np.vstack([points, points[0]])
                label = "Crosswalks" if not crosswalk_plotted else ""
                # Use hatched pattern for crosswalks
                ax.fill(
                    points_closed[:, 0],
                    points_closed[:, 1],
                    facecolor="yellow",
                    alpha=0.6,
                    edgecolor="orange",
                    linewidth=2,
                    hatch="//",
                    label=label,
                )
                crosswalk_plotted = True
        print(f"  Plotted {len(crosswalks)} crosswalks")

    # Legend setup with comprehensive color descriptions
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="blue", linewidth=2, label="Lane Centers"),
        Line2D(
            [0],
            [0],
            color="green",
            linewidth=1,
            alpha=0.6,
            label="Lane Edges (Left & Right)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            alpha=0.8,
            label="Road Edges (Outer Boundaries)",
        ),
    ]

    # Add traffic signs to legend if shown and any exist
    if show_traffic_signs and traffic_signs:
        custom_lines.append(
            Line2D(
                [0],
                [0],
                color="red",
                marker="^",
                markersize=8,
                linestyle="None",
                markeredgecolor="darkred",
                markeredgewidth=1,
                label="Traffic Signs",
            )
        )

    # Add stop lines to legend if any exist
    if wait_lines and stop_line_plotted:
        custom_lines.append(
            Line2D(
                [0], [0], color="magenta", linewidth=3, alpha=0.8, label="Stop Lines"
            )
        )

    # Add yield lines to legend if any exist
    if wait_lines and yield_line_plotted:
        custom_lines.append(
            Line2D(
                [0], [0], color="orange", linewidth=3, alpha=0.8, label="Yield Lines"
            )
        )

    # Add sidewalks to legend if any exist
    if sidewalks:
        from matplotlib.patches import Patch

        custom_lines.append(
            Patch(facecolor="lightgray", edgecolor="gray", alpha=0.5, label="Sidewalks")
        )

    # Add crosswalks to legend if any exist
    if crosswalks:
        from matplotlib.patches import Patch

        custom_lines.append(
            Patch(
                facecolor="yellow",
                edgecolor="orange",
                alpha=0.6,
                hatch="//",
                label="Crosswalks",
            )
        )

    ax.legend(handles=custom_lines, loc="best", fontsize=10, framealpha=0.9)

    # Set axis limits based on map extent
    # Recalculate limits from all plotted data
    ax.relim()
    ax.autoscale_view()

    # Optional margin for nicer framing
    margin = 20  # metres
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)

    plt.tight_layout()
    output_path = "xodr_map_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Map visualization saved to: {output_path}")

    plt.show(block=False)
    print("Plot displayed (close window to continue)")

    return vmap


def main(artifact_name, preview_route, no_block):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(artifact_name, "r") as zip_ref:
            if "map.xodr" in zip_ref.namelist():
                zip_ref.extract("map.xodr", temp_dir)
                zip_ref.extract("rig_trajectories.json", temp_dir)
                plot_xodr_map(
                    temp_dir + "/map.xodr", temp_dir + "/rig_trajectories.json"
                )
            else:
                for file_name in zip_ref.namelist():
                    if file_name.startswith("map_data/") or file_name.startswith(
                        "clipgt/"
                    ):
                        zip_ref.extract(file_name, temp_dir)
                        map_dir = temp_dir + "/" + file_name.split("/")[0]
                plot_clipgt_map(map_dir)

        if preview_route:
            artifacts = Artifact.discover_from_glob(artifact_name)
            artifact = artifacts[list(artifacts.keys())[0]]
            route_generator = RouteGeneratorMap(
                artifact.rig.trajectory.positions, artifact.map
            )
            for i in range(
                100
            ):  # extend the waypoints a few times to go past the egomotion history
                route_generator._extend_waypoints()
            route_polyline_in_local = route_generator.route_polyline_in_local
            plt.plot(
                route_polyline_in_local.waypoints[:, 0],
                route_polyline_in_local.waypoints[:, 1],
                "r",
                linewidth=5,
            )

        plt.axis("equal")
        plt.show(block=not no_block)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=str, help="Input artifact name (usdz)")
    parser.add_argument(
        "--preview_route", action="store_true", help="Preview the route"
    )
    parser.add_argument(
        "--no_block", action="store_true", help="Do not block the plot (for testing)"
    )
    args = parser.parse_args()

    main(args.artifact, args.preview_route, args.no_block)
