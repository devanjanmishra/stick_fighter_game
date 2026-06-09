"""
One Euro Filter for keypoint smoothing.

Reduces jitter from pose estimation while maintaining responsiveness
for fast movements (punches, hooks). The filter adapts its cutoff
frequency based on the speed of change — slow movements get heavy
smoothing, fast movements pass through with minimal lag.

Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems", CHI 2012.
"""

import math
from dataclasses import dataclass
from core.pose_estimator import PoseFrame, Keypoint


class LowPassFilter:
    """Simple exponential low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha
        self._initialized = False
        self._prev: float = 0.0

    def apply(self, value: float, alpha: float | None = None) -> float:
        a = alpha if alpha is not None else self._alpha
        if not self._initialized:
            self._initialized = True
            self._prev = value
            return value
        filtered = a * value + (1.0 - a) * self._prev
        self._prev = filtered
        return filtered

    def reset(self):
        self._initialized = False
        self._prev = 0.0


class OneEuroFilter:
    """
    One Euro Filter for a single scalar value.

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing
                    at low speeds. Default 1.0 is good for body keypoints.
        beta:       Speed coefficient. Higher = less lag for fast movements.
                    Default 5.0 tuned for fighting game punch responsiveness.
        d_cutoff:   Cutoff frequency for the derivative filter. Usually
                    left at 1.0.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 5.0,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._prev_value: float | None = None
        self._prev_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value: float, timestamp: float) -> float:
        if self._prev_time is None:
            self._prev_time = timestamp
            self._prev_value = value
            self._x_filter.apply(value)
            self._dx_filter.apply(0.0)
            return value

        dt = timestamp - self._prev_time
        if dt <= 0:
            dt = 1.0 / 30.0  # assume 30fps if timestamps are equal

        # Estimate derivative
        dx = (value - (self._prev_value or 0.0)) / dt
        # Filter the derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_filtered = self._dx_filter.apply(dx, alpha_d)

        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)

        # Filter the value
        alpha_x = self._alpha(cutoff, dt)
        filtered = self._x_filter.apply(value, alpha_x)

        self._prev_time = timestamp
        self._prev_value = value
        return filtered

    def reset(self):
        self._x_filter.reset()
        self._dx_filter.reset()
        self._prev_value = None
        self._prev_time = None


@dataclass
class SmoothingConfig:
    """Configuration for pose smoothing."""
    min_cutoff: float = 1.0   # Lower = smoother idle, more jitter reduction
    beta: float = 5.0         # Higher = less lag on fast movements (tuned for punches)
    d_cutoff: float = 1.0
    enabled: bool = True


class PoseSmoother:
    """
    Applies One Euro Filter to all keypoints in a PoseFrame.

    Maintains separate filters for each keypoint's x, y, z coordinates.
    This ensures smooth rendering while preserving fast punch movements.
    """

    def __init__(self, config: SmoothingConfig | None = None):
        self.config = config or SmoothingConfig()
        # Dict of keypoint_name -> {coord_name -> OneEuroFilter}
        self._filters: dict[str, dict[str, OneEuroFilter]] = {}

    def _get_filter(self, keypoint_name: str, coord: str) -> OneEuroFilter:
        if keypoint_name not in self._filters:
            self._filters[keypoint_name] = {}
        if coord not in self._filters[keypoint_name]:
            self._filters[keypoint_name][coord] = OneEuroFilter(
                min_cutoff=self.config.min_cutoff,
                beta=self.config.beta,
                d_cutoff=self.config.d_cutoff,
            )
        return self._filters[keypoint_name][coord]

    def smooth(self, pose: PoseFrame) -> PoseFrame:
        """Apply smoothing to all keypoints in the pose."""
        if not self.config.enabled or not pose.valid:
            return pose

        timestamp = pose.timestamp_ms / 1000.0  # Convert to seconds

        smoothed_keypoints: dict[str, Keypoint] = {}
        for name, kp in pose.keypoints.items():
            fx = self._get_filter(name, "x")
            fy = self._get_filter(name, "y")
            fz = self._get_filter(name, "z")

            smoothed_keypoints[name] = Keypoint(
                x=fx.apply(kp.x, timestamp),
                y=fy.apply(kp.y, timestamp),
                z=fz.apply(kp.z, timestamp),
                visibility=kp.visibility,
                name=kp.name,
            )

        return PoseFrame(
            keypoints=smoothed_keypoints,
            timestamp_ms=pose.timestamp_ms,
            frame_index=pose.frame_index,
            valid=pose.valid,
        )

    def reset(self):
        """Reset all filters (e.g., when player re-enters frame)."""
        self._filters.clear()
