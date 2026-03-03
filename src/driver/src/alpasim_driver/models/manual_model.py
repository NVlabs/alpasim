# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Manual driving model - human-controlled trajectory generation.

This model allows interactive control of the ego vehicle using keyboard input.
It displays camera frames in a pygame window and converts WSAD input into
curved trajectories that the controller module tracks.

Controls:
    W / UP:    Accelerate (increase target speed)
    S / DOWN:  Brake/Decelerate (decrease target speed)
    A / LEFT:  Steer left (positive steering angle)
    D / RIGHT: Steer right (negative steering angle)
    SPACE:     Emergency stop (zero speed)
    ESC / Q:   Quit

The model generates constant-curvature arc trajectories based on the current
steering angle and speed, which are then tracked by the MPC controller.
"""

from __future__ import annotations

import atexit
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

from .base import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput

logger = logging.getLogger(__name__)


@dataclass
class ControlState:
    """Thread-safe container for keyboard control state."""

    target_speed: float = 0.0  # Target speed in m/s
    steering_angle: float = 0.0  # Steering angle in radians (positive = left)
    quit_requested: bool = False

    # Physical limits
    MAX_SPEED: float = 15.0  # Maximum forward speed in m/s (~54 km/h)
    MAX_REVERSE_SPEED: float = 7.5  # Maximum reverse speed in m/s (~27 km/h)
    MAX_STEERING: float = 0.4  # Maximum steering angle in radians (~23 degrees)
    WHEELBASE: float = 2.85  # Wheelbase in meters (Ford Fusion)

    # Control rates
    ACCEL_RATE: float = 3.0  # Speed change rate in m/s per second
    COAST_DECELERATION_RATE: float = 0.9  # Coast deceleration rate in m/s²
    STEER_RATE: float = 0.8  # Steering change rate in rad/s
    STEER_RETURN_RATE: float = 1.5  # Steering return-to-center rate in rad/s


class ManualModelGUI:
    """Pygame-based GUI for displaying camera frames and capturing input.

    Threading model (same on all platforms):
    - Main thread: runs the GUI loop (pygame event handling and rendering)
    - `grpc-server` thread: runs the gRPC async server
    - `ego-driver-worker` thread: runs batched inference, calls `predict()`

    The `predict()` method is called from `ego-driver-worker`, which is where
    simulation time synchronization happens via `wait_until_sim_time()`.

    The main window shows the camera feed with an overlay displaying
    current speed, steering, and control hints.

    Frame display is synchronized with simulation time: frames are displayed
    at wall-clock intervals matching their simulation timestamps, ensuring
    playback runs at "real-time" relative to the simulation.
    """

    WINDOW_WIDTH = 960
    WINDOW_HEIGHT = 540

    # Colors
    BG_COLOR = (20, 20, 30)
    TEXT_COLOR = (220, 220, 230)
    ACCENT_COLOR = (0, 200, 255)
    WARNING_COLOR = (255, 150, 50)
    GAUGE_BG = (40, 40, 50)
    GAUGE_FG = (0, 180, 120)
    GAUGE_STEER = (200, 150, 50)

    # Timing constants
    EVENT_POLL_INTERVAL_S = 0.005  # 5ms between event polls while waiting

    def __init__(self, control_state: ControlState, state_lock: threading.Lock):
        """Initialize the GUI.

        Args:
            control_state: Shared state object for keyboard input.
            state_lock: Lock for thread-safe access to control_state.
        """
        self._control = control_state
        self._lock = state_lock
        self._running = False

        # Key state tracking
        self._keys_pressed: set[int] = set()
        self._last_update_time = time.time()

        # Latest frame (set from ego-driver-worker thread, read by GUI thread)
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_lock = threading.Lock()

        # Simulation time synchronization:
        # We map simulation timestamps to wall-clock time so that playback
        # runs at "real-time" relative to the simulation.
        # Updated by wait_until_sim_time() which blocks ego-driver-worker.
        self._last_sim_time_us: int | None = None
        self._last_wall_clock: float | None = None
        self._sim_time_lock = threading.Lock()

        # Timing statistics (for display)
        self._first_sim_time_us: int | None = None  # First sim timestamp received
        self._first_wall_clock: float | None = None  # Wall clock at first frame

        # Display resources (initialized on GUI thread only)
        self._screen: Any = None  # pygame.Surface when available
        self._fonts: tuple[Any, ...] | None = None
        self._current_frame_surface: Any = None

    def init_display(self) -> None:
        """Initialize pygame display. Must be called on the GUI thread."""
        pygame.init()
        pygame.display.set_caption(
            "AlpaSim Manual Control - WSAD to drive, ESC to quit"
        )
        self._screen = pygame.display.set_mode(
            (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), pygame.RESIZABLE
        )
        self._fonts = (
            pygame.font.Font(None, 48),  # large
            pygame.font.Font(None, 32),  # medium
            pygame.font.Font(None, 24),  # small
        )
        self._running = True
        self._last_update_time = time.time()
        logger.info("ManualModel GUI display initialized")

    def tick(self) -> bool:
        """Process one frame of events and rendering.

        Must be called on the GUI thread (main thread on macOS, background
        thread on Linux). This method handles pygame events, updates keyboard
        controls, and renders the display.

        Timing synchronization happens in wait_until_sim_time(), which is
        called from ego-driver-worker's predict() method.

        Returns:
            True if the GUI should continue running, False if quit was requested.
        """
        if not self._running:
            return False

        if self._screen is None:
            return False

        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        # Process pygame events (main thread only)
        if not self._process_events():
            return False

        # Update control state based on keyboard
        self._update_controls(dt)

        # Update display surface from latest frame (if new frame available)
        self._update_frame_surface()

        # Render
        self._render_frame()

        # Small sleep to target ~60fps for smooth UI
        time.sleep(self.EVENT_POLL_INTERVAL_S)

        return True

    def _process_events(self) -> bool:
        """Process pygame events.

        Must be called on the GUI thread only.

        Returns:
            True to continue running, False if quit was requested.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                with self._lock:
                    self._control.quit_requested = True
                self._running = False
                return False
            elif event.type == pygame.KEYDOWN:
                self._keys_pressed.add(event.key)
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    with self._lock:
                        self._control.quit_requested = True
                    self._running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    with self._lock:
                        self._control.target_speed = 0.0
            elif event.type == pygame.KEYUP:
                self._keys_pressed.discard(event.key)
            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )
        return True

    def _update_frame_surface(self) -> None:
        """Update the display surface from the latest frame.

        Thread-safe: reads from _latest_frame which is set by ego-driver-worker.
        Must be called on the GUI thread (creates pygame surface).
        """
        with self._latest_frame_lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame

        # Create pygame surface (must be on main thread)
        self._current_frame_surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )

    def _render_frame(self) -> None:
        """Render the current frame to the display."""
        if self._screen is None or self._fonts is None:
            return

        font_large, font_medium, font_small = self._fonts
        screen = self._screen

        screen.fill(self.BG_COLOR)

        # Draw camera frame
        if self._current_frame_surface is not None:
            frame_w, frame_h = self._current_frame_surface.get_size()
            scale = min(
                screen.get_width() / frame_w, (screen.get_height() - 120) / frame_h
            )
            scaled_w = int(frame_w * scale)
            scaled_h = int(frame_h * scale)
            scaled_frame = pygame.transform.scale(
                self._current_frame_surface, (scaled_w, scaled_h)
            )
            x = (screen.get_width() - scaled_w) // 2
            screen.blit(scaled_frame, (x, 10))
        else:
            text = font_large.render("Waiting for camera...", True, self.TEXT_COLOR)
            text_rect = text.get_rect(
                center=(screen.get_width() // 2, screen.get_height() // 2 - 60)
            )
            screen.blit(text, text_rect)

        self._draw_overlay(screen, font_large, font_medium, font_small)
        self._draw_timing_overlay(screen, font_medium, font_small)
        pygame.display.flip()

    def run_main_loop(self) -> None:
        """Run the GUI loop (blocking).

        On macOS, this must be called from the main thread (Cocoa requirement).
        The gRPC server runs in a background thread in this case.
        """
        self.init_display()
        logger.info("ManualModel GUI main loop started")

        while self._running:
            if not self.tick():
                break

        pygame.quit()
        logger.info("ManualModel GUI main loop ended")

    def stop(self) -> None:
        """Stop the GUI."""
        self._running = False
        logger.info("ManualModel GUI stopped")

    def set_frame(self, frame: np.ndarray) -> None:
        """Set the latest camera frame for display.

        Thread-safe: called from ego-driver-worker thread.
        The GUI thread's tick() loop will pick up and display this frame.

        Args:
            frame: RGB image as HWC uint8 numpy array.
        """
        with self._latest_frame_lock:
            self._latest_frame = frame.copy()

    def wait_until_sim_time(self, timestamp_us: int) -> None:
        """Block until wall-clock time catches up to the given simulation time.

        This ensures that drive responses are not returned faster than
        real-time relative to the simulation timestamps. Called from
        ego-driver-worker thread to pace responses.

        This method does NOT call any pygame functions, so it's safe to
        call from any thread (required since pygame must run on GUI thread).

        Args:
            timestamp_us: Simulation timestamp to wait for.
        """
        with self._sim_time_lock:
            now = time.time()

            if self._last_sim_time_us is None:
                # First frame - initialize timing
                self._first_sim_time_us = timestamp_us
                self._first_wall_clock = now
                self._last_sim_time_us = timestamp_us
                self._last_wall_clock = now
                return

            sim_time_delta_s = (timestamp_us - self._last_sim_time_us) / 1_000_000.0
            wall_clock_target = self._last_wall_clock + sim_time_delta_s

        # Wait outside the lock - just sleep, no pygame operations
        if now < wall_clock_target:
            time.sleep(wall_clock_target - now)

        # Update timing for next call
        with self._sim_time_lock:
            self._last_sim_time_us = timestamp_us
            self._last_wall_clock = time.time()

    def _update_controls(self, dt: float) -> None:
        """Update control state based on currently pressed keys."""
        with self._lock:
            # Acceleration (W/S or UP/DOWN)
            if pygame.K_w in self._keys_pressed or pygame.K_UP in self._keys_pressed:
                self._control.target_speed += self._control.ACCEL_RATE * dt
            elif (
                pygame.K_s in self._keys_pressed or pygame.K_DOWN in self._keys_pressed
            ):
                self._control.target_speed -= self._control.ACCEL_RATE * dt
            else:
                # Coast to stop when no accel key pressed
                if self._control.target_speed > 0:
                    self._control.target_speed -= (
                        self._control.COAST_DECELERATION_RATE * dt
                    )
                    self._control.target_speed = max(0, self._control.target_speed)
                elif self._control.target_speed < 0:
                    self._control.target_speed += (
                        self._control.COAST_DECELERATION_RATE * dt
                    )
                    self._control.target_speed = min(0, self._control.target_speed)

            # Clamp speed
            self._control.target_speed = np.clip(
                self._control.target_speed,
                -self._control.MAX_REVERSE_SPEED,
                self._control.MAX_SPEED,
            )

            # Steering (A/D or LEFT/RIGHT)
            # A/LEFT = positive steering = turn left
            # D/RIGHT = negative steering = turn right
            if pygame.K_a in self._keys_pressed or pygame.K_LEFT in self._keys_pressed:
                self._control.steering_angle += self._control.STEER_RATE * dt
            elif (
                pygame.K_d in self._keys_pressed or pygame.K_RIGHT in self._keys_pressed
            ):
                self._control.steering_angle -= self._control.STEER_RATE * dt
            else:
                # Return steering to center
                if abs(self._control.steering_angle) > 0.001:
                    return_amount = self._control.STEER_RETURN_RATE * dt
                    if self._control.steering_angle > 0:
                        self._control.steering_angle = max(
                            0, self._control.steering_angle - return_amount
                        )
                    else:
                        self._control.steering_angle = min(
                            0, self._control.steering_angle + return_amount
                        )

            # Clamp steering
            self._control.steering_angle = np.clip(
                self._control.steering_angle,
                -self._control.MAX_STEERING,
                self._control.MAX_STEERING,
            )

    def _draw_overlay(
        self,
        screen: Any,  # pygame.Surface
        font_large: Any,  # pygame.font.Font
        font_medium: Any,  # pygame.font.Font
        font_small: Any,  # pygame.font.Font
    ) -> None:
        """Draw the control overlay at the bottom of the screen."""
        overlay_y = screen.get_height() - 110
        overlay_rect = pygame.Rect(0, overlay_y, screen.get_width(), 110)
        pygame.draw.rect(screen, (30, 30, 40), overlay_rect)
        pygame.draw.line(
            screen,
            self.ACCENT_COLOR,
            (0, overlay_y),
            (screen.get_width(), overlay_y),
            2,
        )

        with self._lock:
            speed = self._control.target_speed
            steering = self._control.steering_angle
            max_speed = self._control.MAX_SPEED
            max_steer = self._control.MAX_STEERING

        # Speed display
        speed_kmh = speed * 3.6
        speed_text = font_large.render(f"{speed_kmh:+5.1f}", True, self.GAUGE_FG)
        screen.blit(speed_text, (20, overlay_y + 15))
        unit_text = font_medium.render("km/h", True, self.TEXT_COLOR)
        screen.blit(unit_text, (20, overlay_y + 65))

        # Speed bar
        bar_x, bar_y = 140, overlay_y + 20
        bar_w, bar_h = 150, 25
        pygame.draw.rect(screen, self.GAUGE_BG, (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * abs(speed) / max_speed)
        fill_color = self.GAUGE_FG if speed >= 0 else self.WARNING_COLOR
        if speed >= 0:
            pygame.draw.rect(screen, fill_color, (bar_x, bar_y, fill_w, bar_h))
        else:
            pygame.draw.rect(
                screen, fill_color, (bar_x + bar_w - fill_w, bar_y, fill_w, bar_h)
            )

        # Steering display
        steer_deg = math.degrees(steering)
        steer_text = font_large.render(f"{steer_deg:+5.1f}°", True, self.GAUGE_STEER)
        screen.blit(steer_text, (320, overlay_y + 15))
        steer_label = font_medium.render("steering", True, self.TEXT_COLOR)
        screen.blit(steer_label, (320, overlay_y + 65))

        # Steering bar (centered, shows left/right)
        sbar_x, sbar_y = 460, overlay_y + 20
        sbar_w, sbar_h = 150, 25
        pygame.draw.rect(screen, self.GAUGE_BG, (sbar_x, sbar_y, sbar_w, sbar_h))
        # Draw center line
        center_x = sbar_x + sbar_w // 2
        pygame.draw.line(
            screen, (80, 80, 90), (center_x, sbar_y), (center_x, sbar_y + sbar_h), 2
        )
        # Draw steering position
        steer_norm = steering / max_steer
        indicator_x = center_x + int(steer_norm * sbar_w // 2)
        pygame.draw.rect(screen, self.GAUGE_STEER, (indicator_x - 3, sbar_y, 6, sbar_h))

        # Controls hint
        hint_text = font_small.render(
            "W/S: Speed   A/D: Steer   SPACE: Stop   ESC: Quit", True, (150, 150, 160)
        )
        screen.blit(hint_text, (640, overlay_y + 45))

        # Steering direction indicator
        if abs(steering) > 0.01:
            direction = "← LEFT" if steering > 0 else "RIGHT →"
            dir_text = font_small.render(direction, True, self.GAUGE_STEER)
            screen.blit(dir_text, (640, overlay_y + 20))

    def _get_timing_stats(self) -> tuple[float, float]:
        """Get elapsed simulation time and real-time ratio.

        Returns:
            Tuple of (elapsed_sim_time_s, realtime_ratio).
            realtime_ratio = sim_time_elapsed / wall_time_elapsed.
            A ratio of 1.0 means running at real-time.
            < 1.0 means running slower than real-time.
        """
        with self._sim_time_lock:
            if self._first_sim_time_us is None or self._last_sim_time_us is None:
                return 0.0, 1.0

            sim_elapsed_s = (
                self._last_sim_time_us - self._first_sim_time_us
            ) / 1_000_000.0
            wall_elapsed_s = time.time() - self._first_wall_clock

        if wall_elapsed_s < 0.001:
            return sim_elapsed_s, 1.0

        realtime_ratio = sim_elapsed_s / wall_elapsed_s
        return sim_elapsed_s, realtime_ratio

    def _draw_timing_overlay(
        self,
        screen: Any,
        font_medium: Any,
        font_small: Any,
    ) -> None:
        """Draw timing information at the top-right of the screen."""
        elapsed_s, realtime_ratio = self._get_timing_stats()

        # Format elapsed time as MM:SS.s
        minutes = int(elapsed_s // 60)
        seconds = elapsed_s % 60
        time_str = f"{minutes:02d}:{seconds:05.2f}"

        # Determine color based on real-time ratio
        # Green if >= 0.95 (close to real-time)
        # Yellow if 0.5-0.95 (somewhat slow)
        # Red if < 0.5 (very slow)
        if realtime_ratio >= 0.95:
            ratio_color = self.GAUGE_FG  # Green
        elif realtime_ratio >= 0.5:
            ratio_color = self.WARNING_COLOR  # Yellow/Orange
        else:
            ratio_color = (255, 80, 80)  # Red

        # Draw semi-transparent background
        padding = 10
        box_width = 180
        box_height = 50
        box_x = screen.get_width() - box_width - padding
        box_y = padding

        bg_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
        bg_surface.fill((30, 30, 40, 200))
        screen.blit(bg_surface, (box_x, box_y))

        # Draw elapsed time
        time_text = font_medium.render(time_str, True, self.TEXT_COLOR)
        screen.blit(time_text, (box_x + 10, box_y + 5))

        # Draw real-time ratio
        ratio_str = f"{realtime_ratio:.2f}x realtime"
        ratio_text = font_small.render(ratio_str, True, ratio_color)
        screen.blit(ratio_text, (box_x + 10, box_y + 28))


class ManualModel(BaseTrajectoryModel):
    """Manual driving model for human control.

    This model generates trajectories based on human keyboard input.
    WSAD controls are converted into constant-curvature arc trajectories
    that the controller module's MPC tracks.

    The model displays a pygame window to show camera frames and capture
    keyboard input. Requires a display (X11 or Wayland).

    Threading model (same on all platforms, see ManualModelGUI for details):
    - Main thread: GUI loop (pygame)
    - `grpc-server` thread: gRPC async server
    - `ego-driver-worker` thread: inference (predict())
    """

    # No GPU/dtype requirements - this is CPU-only
    NUM_CAMERAS = 1

    # Trajectory parameters
    TRAJECTORY_HORIZON_S = 4.0  # 4 seconds of trajectory
    DEFAULT_SPEED_MPS = 5.0  # Default forward speed when no input

    # Singleton GUI instance (shared across all ManualModel instances)
    _gui_instance: ManualModelGUI | None = None
    _gui_lock = threading.Lock()

    def __init__(
        self,
        camera_ids: list[str],
        output_frequency_hz: int,
        context_length: int = 1,
        default_speed_mps: float = 5.0,
    ):
        """Initialize ManualModel.

        Args:
            camera_ids: List of camera IDs to display (currently uses first one).
            output_frequency_hz: Trajectory output frequency in Hz.
            context_length: Number of frames to buffer (default 1, we only need latest).
            default_speed_mps: Default forward speed in m/s for initial trajectory.

        Note:
            The GUI is not started automatically. Call run_gui_loop() from the
            main thread to start the pygame event loop.
        """
        if len(camera_ids) < 1:
            raise ValueError("ManualModel requires at least 1 camera")

        self._camera_ids = camera_ids
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        self._default_speed_mps = default_speed_mps

        # Calculate trajectory length
        self._num_waypoints = int(self.TRAJECTORY_HORIZON_S * output_frequency_hz)

        # Control state (thread-safe)
        self._control = ControlState(target_speed=0.0, steering_angle=0.0)
        self._state_lock = threading.Lock()

        # Create GUI instance but don't start it yet
        self._init_gui_instance()

        logger.info(
            "Initialized ManualModel with %d camera(s), "
            "output_frequency=%dHz, horizon=%.1fs (%d waypoints), "
            "default_speed=%.1f m/s",
            len(camera_ids),
            output_frequency_hz,
            self.TRAJECTORY_HORIZON_S,
            self._num_waypoints,
            default_speed_mps,
        )

    def _init_gui_instance(self) -> None:
        """Create the GUI instance (but don't start it)."""
        with ManualModel._gui_lock:
            if ManualModel._gui_instance is None:
                ManualModel._gui_instance = ManualModelGUI(
                    self._control, self._state_lock
                )
                atexit.register(self._stop_gui)

    def run_gui_loop(self) -> None:
        """Run the GUI loop on the main thread (blocking).

        The gRPC server runs in the `grpc-server` background thread.
        """
        if ManualModel._gui_instance is not None:
            ManualModel._gui_instance.run_main_loop()

    @classmethod
    def _stop_gui(cls) -> None:
        """Stop the GUI (called on exit)."""
        with cls._gui_lock:
            if cls._gui_instance is not None:
                cls._gui_instance.stop()
                cls._gui_instance = None

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz

    def _encode_command(self, command: DriveCommand) -> int:
        """Manual model doesn't use encoded commands - human decides."""
        # Return command as-is for logging purposes
        return int(command)

    def _generate_curved_trajectory(
        self, speed_mps: float, steering_rad: float
    ) -> np.ndarray:
        """Generate a constant-curvature arc trajectory.

        For a bicycle model with steering angle δ and speed v:
        - Turn radius R = wheelbase / tan(δ)
        - Angular velocity ω = v / R
        - At time t: x = R * sin(ω*t), y = R * (1 - cos(ω*t))

        Args:
            speed_mps: Speed in meters per second.
            steering_rad: Steering angle in radians (positive = left turn).

        Returns:
            (N, 2) array of x,y positions in rig frame (x forward, y left).
        """
        # Time between waypoints
        dt = 1.0 / self._output_frequency_hz

        # Time points for waypoints (starting at dt, not 0)
        times = np.arange(1, self._num_waypoints + 1) * dt

        # Minimum steering to avoid numerical issues
        MIN_STEERING = 0.001

        if abs(steering_rad) < MIN_STEERING:
            # Straight line trajectory
            x_positions = times * speed_mps
            y_positions = np.zeros_like(x_positions)
        else:
            # Curved arc trajectory
            wheelbase = self._control.WHEELBASE
            turn_radius = wheelbase / np.tan(steering_rad)

            # Arc angles at each time point
            arc_angles = (speed_mps / turn_radius) * times

            # Position on the arc (bicycle model)
            # Starting at origin facing +x, turning around center at (0, R):
            #   x = R * sin(θ), y = R * (1 - cos(θ))
            # where θ = (v/R) * t is the arc angle.
            #
            # For LEFT turn (positive steering): R > 0, θ > 0
            #   → x = R * sin(+θ) > 0 (forward), y = R * (1 - cos(θ)) > 0 (left)
            # For RIGHT turn (negative steering): R < 0, θ < 0
            #   → x = (-R) * sin(-θ) = (-R) * (-sin(θ)) = R * sin(θ) > 0 (forward)
            #   → y = (-R) * (1 - cos(-θ)) = (-R) * (1 - cos(θ)) < 0 (right)
            #
            # NOTE: Do NOT use abs(turn_radius) for x - the signs cancel correctly!
            x_positions = turn_radius * np.sin(arc_angles)
            y_positions = turn_radius * (1 - np.cos(arc_angles))

        return np.column_stack([x_positions, y_positions])

    def _compute_arc_headings(
        self, speed_mps: float, steering_rad: float
    ) -> np.ndarray:
        """Compute headings for a constant-curvature arc trajectory.

        Args:
            speed_mps: Speed in meters per second.
            steering_rad: Steering angle in radians (positive = left turn).

        Returns:
            (N,) array of heading angles in radians.
        """
        dt = 1.0 / self._output_frequency_hz
        times = np.arange(1, self._num_waypoints + 1) * dt

        MIN_STEERING = 0.001

        if abs(steering_rad) < MIN_STEERING:
            # Straight line - heading is always 0 (forward)
            return np.zeros(self._num_waypoints)
        else:
            wheelbase = self._control.WHEELBASE
            turn_radius = wheelbase / np.tan(steering_rad)
            angular_velocity = speed_mps / turn_radius

            # Heading at each time point (tangent to the arc)
            return angular_velocity * times

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        """Generate trajectory prediction based on keyboard input.

        Called from the `ego-driver-worker` thread. This method synchronizes
        with simulation time: it waits until wall-clock time catches up to the
        simulation timestamp before returning, ensuring responses are paced at
        "real-time" relative to the simulation.

        Manual model uses camera images for display only. Command, speed,
        acceleration, and ego pose history are unused.

        Returns:
            ModelPrediction with curved trajectory in rig frame.
        """
        self._validate_cameras(prediction_input.camera_images)

        # Get the most recent frame and timestamp from the first camera
        first_camera = self._camera_ids[0]
        latest_timestamp_us: int | None = None
        latest_frame: np.ndarray | None = None

        if (
            first_camera in prediction_input.camera_images
            and prediction_input.camera_images[first_camera]
        ):
            latest_timestamp_us, latest_frame = prediction_input.camera_images[
                first_camera
            ][-1]

        if ManualModel._gui_instance is not None and latest_timestamp_us is not None:
            # Wait until wall-clock time catches up to simulation time.
            # This blocks returning the trajectory until the appropriate time,
            # ensuring we don't respond faster than real-time relative to sim.
            ManualModel._gui_instance.wait_until_sim_time(latest_timestamp_us)

            # Set the frame for display (main thread will pick it up)
            if latest_frame is not None:
                ManualModel._gui_instance.set_frame(latest_frame)

        # Get current control state (thread-safe)
        with self._state_lock:
            target_speed = self._control.target_speed
            steering_angle = self._control.steering_angle
            quit_requested = self._control.quit_requested

        # Handle quit request
        if quit_requested:
            logger.info("Quit requested by user")
            # Return zero trajectory to stop the vehicle
            trajectory_xy = np.zeros((self._num_waypoints, 2))
            headings = np.zeros(self._num_waypoints)
            return ModelPrediction(trajectory_xy=trajectory_xy, headings=headings)

        # Use target speed from keyboard, with minimum for trajectory generation
        # If speed is very low, use a small value to generate valid trajectory
        effective_speed = target_speed if abs(target_speed) > 0.1 else 0.0

        # Generate curved trajectory based on steering input
        trajectory_xy = self._generate_curved_trajectory(
            effective_speed, steering_angle
        )
        headings = self._compute_arc_headings(effective_speed, steering_angle)

        logger.debug(
            "ManualModel: speed=%.2f m/s (%.1f km/h), steer=%.1f°, "
            "generating %d waypoints (command=%s ignored)",
            effective_speed,
            effective_speed * 3.6,
            math.degrees(steering_angle),
            len(trajectory_xy),
            prediction_input.command.name,
        )

        return ModelPrediction(trajectory_xy=trajectory_xy, headings=headings)
