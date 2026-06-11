# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main(log_file):
    # load csv data
    data = np.loadtxt(log_file, delimiter=",", skiprows=1)

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(data[:, 0], data[:, 1], label="X Position")
    plt.title("Position Over Time")
    plt.ylabel("X Position (m)")
    plt.subplot(3, 1, 2)
    plt.plot(data[:, 0], data[:, 2], label="Y Position")
    plt.ylabel("Y Position (m)")
    plt.subplot(3, 1, 3)
    plt.plot(data[:, 0], data[:, 3], label="Z Position")
    plt.ylabel("Z Position (m)")

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 1], data[:, 2], label="Trajectory")
    plt.title("Trajectory in XY Plane")

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0], data[:, 8], label="X Velocity")
    plt.title("Velocity Over Time")
    plt.ylabel("X Velocity (m/s)")

    plt.subplot(2, 1, 2)
    plt.plot(data[:, 0], data[:, 9], label="Y Velocity")
    plt.ylabel("Y Velocity (m/s)")

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0], data[:, 11], "--", label="steering angle command")
    plt.plot(data[:, 0], data[:, 15], label="steering angle")
    plt.title("Steering Angle Over Time")

    plt.subplot(2, 1, 2)
    plt.plot(data[:, 0], data[:, 12], "--", label="accel command")
    plt.plot(data[:, 0], data[:, 16], label="accel")
    plt.title("Acceleration Over Time")

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(data[:, 0], data[:, 17], label="X Position Error")
    plt.title("Position Error Over Time")
    plt.ylabel("X Position Error (m)")
    plt.subplot(3, 1, 2)
    plt.plot(data[:, 0], data[:, 18], label="Y Position Error")
    plt.ylabel("Y Position Error (m)")
    plt.subplot(3, 1, 3)
    plt.plot(data[:, 0], data[:, 19], label="Yaw Error")
    plt.ylabel("Yaw Error (rad)")

    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Controller")
    parser.add_argument("--log", type=str, required=True, help="Path to data file")
    args = parser.parse_args()

    main(args.log)
