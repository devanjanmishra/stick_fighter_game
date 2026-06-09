"""
Walking/movement tracking via shoulder position.

When the player shifts their torso left/right (as seen by the front-facing
camera), the stick figure walks forward/backward in the side-view game world.
Shoulder midpoint lateral movement is the primary signal.

The tracker maintains a baseline shoulder position (calibrated on first few
frames) and maps lateral deviation to game-world horizontal velocity.
"""

from dataclasses import dataclass
from core.pose_estimator import PoseFrame


@dataclass
class MovementConfig:
    """Configuration for shoulder-based movement tracking."""
    # How much normalized shoulder displacement maps to game-world speed (px/frame)
    lateral_sensitivity: float = 800.0
    # Minimum shoulder displacement to start walking (dead zone)
    dead_zone: float = 0.015
    # Maximum game-world speed (px/frame)
    max_speed: float = 6.0
    # Smoothing factor for position updates (0-1, higher = more responsive)
    smoothing: float = 0.3
    # Number of frames to average for baseline calibration
    calibration_frames: int = 15
    # Game world boundaries
    min_x: float = 50.0
    max_x: float = 1230.0


@dataclass
class MovementState:
    """Current movement state of the player character."""
    game_x: float = 300.0       # current game-world x position
    velocity: float = 0.0       # current horizontal velocity (px/frame)
    is_walking: bool = False
    walk_direction: int = 0     # -1 = backward, 0 = still, 1 = forward
    shoulder_offset: float = 0.0  # current shoulder offset from baseline


class MovementTracker:
    """
    Tracks player walking via shoulder lateral movement.

    The front-facing camera sees left/right shoulder shift when the
    player leans or steps. This maps to forward/backward movement
    in the side-view game (facing_right=True means rightward = forward).
    """

    def __init__(self, config: MovementConfig | None = None, initial_x: float = 300.0):
        self.config = config or MovementConfig()
        self._baseline_x: float | None = None
        self._calibration_samples: list[float] = []
        self._state = MovementState(game_x=initial_x)
        self._smoothed_velocity: float = 0.0

    def _get_shoulder_midpoint_x(self, pose: PoseFrame) -> float | None:
        """Get the x-coordinate of the shoulder midpoint."""
        mid = pose.shoulder_midpoint
        if mid is not None:
            return mid[0]
        return None

    def _calibrate(self, shoulder_x: float):
        """Accumulate baseline samples during calibration phase."""
        self._calibration_samples.append(shoulder_x)
        if len(self._calibration_samples) >= self.config.calibration_frames:
            self._baseline_x = sum(self._calibration_samples) / len(self._calibration_samples)

    def update(self, pose: PoseFrame, facing_right: bool = True) -> MovementState:
        """
        Update movement state from a new pose frame.

        Returns the current MovementState with updated position.
        """
        if not pose.valid:
            self._state.velocity = 0.0
            self._state.is_walking = False
            self._state.walk_direction = 0
            return self._state

        shoulder_x = self._get_shoulder_midpoint_x(pose)
        if shoulder_x is None:
            return self._state

        # Calibration phase
        if self._baseline_x is None:
            self._calibrate(shoulder_x)
            return self._state

        # Calculate offset from baseline
        offset = shoulder_x - self._baseline_x
        self._state.shoulder_offset = offset

        # Apply dead zone
        if abs(offset) < self.config.dead_zone:
            target_velocity = 0.0
        else:
            # Remove dead zone from effective offset
            effective = offset - (self.config.dead_zone if offset > 0 else -self.config.dead_zone)
            # Map to velocity
            raw_velocity = effective * self.config.lateral_sensitivity
            # Clamp
            raw_velocity = max(-self.config.max_speed, min(self.config.max_speed, raw_velocity))

            # Direction depends on facing
            # Camera left (negative offset) = walk backward when facing right
            # Camera right (positive offset) = walk forward when facing right
            if not facing_right:
                raw_velocity = -raw_velocity

            target_velocity = raw_velocity

        # Smooth velocity
        self._smoothed_velocity = (
            self.config.smoothing * target_velocity
            + (1.0 - self.config.smoothing) * self._smoothed_velocity
        )

        # Small velocity threshold to stop completely
        if abs(self._smoothed_velocity) < 0.3:
            self._smoothed_velocity = 0.0

        self._state.velocity = self._smoothed_velocity
        self._state.is_walking = abs(self._smoothed_velocity) > 0.0
        self._state.walk_direction = (
            1 if self._smoothed_velocity > 0 else (-1 if self._smoothed_velocity < 0 else 0)
        )

        # Update position
        self._state.game_x += self._smoothed_velocity
        self._state.game_x = max(self.config.min_x, min(self.config.max_x, self._state.game_x))

        return self._state

    @property
    def state(self) -> MovementState:
        return self._state

    @property
    def is_calibrated(self) -> bool:
        return self._baseline_x is not None

    def reset(self, initial_x: float = 300.0):
        """Reset tracker (e.g., new round)."""
        self._baseline_x = None
        self._calibration_samples.clear()
        self._state = MovementState(game_x=initial_x)
        self._smoothed_velocity = 0.0
