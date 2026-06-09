"""
Calibration system for personalized move detection.

The user records 3 examples of each move during a calibration phase.
The system extracts velocity/displacement profiles and stores them as
templates. During gameplay, incoming movements are compared against
these templates using Dynamic Time Warping (DTW) for robust matching.

This adds a layer of personalization — each player's jab, cross, hook,
and uppercut will have slightly different speed/reach/arc profiles.
"""

import math
import json
import os
from dataclasses import dataclass, field
from core.pose_estimator import PoseFrame, Keypoint
from core.move_detector import MoveType


@dataclass
class MoveTemplate:
    """A recorded move template: sequence of (vx, vy, vz) per frame."""
    move_type: MoveType
    hand: str  # "left" or "right"
    velocity_profile: list[tuple[float, float, float]] = field(default_factory=list)
    peak_z_velocity: float = 0.0
    peak_x_velocity: float = 0.0
    peak_y_velocity: float = 0.0
    z_extension: float = 0.0
    x_extension: float = 0.0
    y_extension: float = 0.0
    duration_frames: int = 0


@dataclass
class CalibratedThresholds:
    """Personalized detection thresholds derived from calibration."""
    punch_z_velocity_threshold: float = 0.015
    hook_x_velocity_threshold: float = 0.012
    uppercut_y_velocity_threshold: float = 0.015
    punch_z_extension: float = 0.08
    hook_x_extension: float = 0.04
    uppercut_y_extension: float = 0.06


def dtw_distance(seq_a: list[tuple[float, ...]], seq_b: list[tuple[float, ...]]) -> float:
    """
    Dynamic Time Warping distance between two sequences of tuples.
    Returns the normalized distance (lower = more similar).
    """
    n = len(seq_a)
    m = len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    # Cost matrix
    dtw_matrix: list[list[float]] = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dtw_matrix[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = _euclidean(seq_a[i - 1], seq_b[j - 1])
            dtw_matrix[i][j] = cost + min(
                dtw_matrix[i - 1][j],      # insertion
                dtw_matrix[i][j - 1],      # deletion
                dtw_matrix[i - 1][j - 1],  # match
            )

    return dtw_matrix[n][m] / max(n, m)


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


class CalibrationRecorder:
    """
    Records move examples during calibration phase.

    Usage:
        recorder = CalibrationRecorder()
        recorder.start_recording("jab", "left")
        for each frame during the move:
            recorder.add_frame(pose)
        template = recorder.finish_recording()
    """

    def __init__(self):
        self._recording = False
        self._move_type: str = ""
        self._hand: str = ""
        self._frames: list[PoseFrame] = []

    def start_recording(self, move_type: str, hand: str):
        """Begin recording a move example."""
        self._recording = True
        self._move_type = move_type
        self._hand = hand
        self._frames = []

    def add_frame(self, pose: PoseFrame):
        """Add a frame to the current recording."""
        if self._recording and pose.valid:
            self._frames.append(pose)

    def finish_recording(self) -> MoveTemplate | None:
        """Stop recording and return the extracted template."""
        self._recording = False
        if len(self._frames) < 3:
            return None

        wrist_key = f"{self._hand}_wrist"
        velocity_profile: list[tuple[float, float, float]] = []
        z_values: list[float] = []
        x_values: list[float] = []
        y_values: list[float] = []

        for i in range(1, len(self._frames)):
            prev_kp = self._frames[i - 1].get(wrist_key)
            curr_kp = self._frames[i].get(wrist_key)
            if prev_kp and curr_kp:
                vx = curr_kp.x - prev_kp.x
                vy = curr_kp.y - prev_kp.y
                vz = curr_kp.z - prev_kp.z
                velocity_profile.append((vx, vy, vz))
                z_values.append(curr_kp.z)
                x_values.append(curr_kp.x)
                y_values.append(curr_kp.y)

        if not velocity_profile:
            return None

        peak_z_vel = max(abs(v[2]) for v in velocity_profile)
        peak_x_vel = max(abs(v[0]) for v in velocity_profile)
        peak_y_vel = max(abs(v[1]) for v in velocity_profile)

        z_ext = max(z_values) - min(z_values) if z_values else 0.0
        x_ext = max(x_values) - min(x_values) if x_values else 0.0
        y_ext = max(y_values) - min(y_values) if y_values else 0.0

        return MoveTemplate(
            move_type=MoveType(self._move_type),
            hand=self._hand,
            velocity_profile=velocity_profile,
            peak_z_velocity=peak_z_vel,
            peak_x_velocity=peak_x_vel,
            peak_y_velocity=peak_y_vel,
            z_extension=z_ext,
            x_extension=x_ext,
            y_extension=y_ext,
            duration_frames=len(self._frames),
        )

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frame_count(self) -> int:
        return len(self._frames)


class CalibrationProfile:
    """
    Stores calibrated move templates and derives personalized thresholds.

    After recording 3 examples of each move, call compute_thresholds()
    to generate personalized detection parameters.
    """

    def __init__(self, stance: str = "orthodox"):
        self.stance = stance
        self.templates: dict[str, list[MoveTemplate]] = {
            "jab": [],
            "cross": [],
            "hook": [],
            "uppercut": [],
        }
        self._thresholds: CalibratedThresholds | None = None

    def add_template(self, template: MoveTemplate):
        """Add a recorded template for a move type."""
        key = template.move_type.value
        if key in self.templates:
            self.templates[key].append(template)

    def has_enough_samples(self, move_type: str, min_samples: int = 3) -> bool:
        return len(self.templates.get(move_type, [])) >= min_samples

    def is_fully_calibrated(self, min_samples: int = 3) -> bool:
        return all(
            self.has_enough_samples(m, min_samples)
            for m in ["jab", "cross", "hook", "uppercut"]
        )

    def compute_thresholds(self) -> CalibratedThresholds:
        """
        Derive personalized thresholds from recorded templates.

        Uses 60% of the average peak velocity/extension as thresholds,
        giving a comfortable detection margin.
        """
        threshold_factor = 0.6

        def _avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        # Jab/Cross z-velocity threshold
        jab_z_vels = [t.peak_z_velocity for t in self.templates["jab"]]
        cross_z_vels = [t.peak_z_velocity for t in self.templates["cross"]]
        punch_vels = jab_z_vels + cross_z_vels
        punch_z_thresh = _avg(punch_vels) * threshold_factor if punch_vels else 0.015

        # Hook x-velocity threshold
        hook_x_vels = [t.peak_x_velocity for t in self.templates["hook"]]
        hook_x_thresh = _avg(hook_x_vels) * threshold_factor if hook_x_vels else 0.012

        # Uppercut y-velocity threshold
        upper_y_vels = [t.peak_y_velocity for t in self.templates["uppercut"]]
        upper_y_thresh = _avg(upper_y_vels) * threshold_factor if upper_y_vels else 0.015

        # Extension thresholds
        jab_z_ext = [t.z_extension for t in self.templates["jab"]]
        cross_z_ext = [t.z_extension for t in self.templates["cross"]]
        punch_ext = jab_z_ext + cross_z_ext
        punch_z_ext = _avg(punch_ext) * threshold_factor if punch_ext else 0.08

        hook_x_ext = [t.x_extension for t in self.templates["hook"]]
        hook_x_extension = _avg(hook_x_ext) * threshold_factor if hook_x_ext else 0.04

        upper_y_ext = [t.y_extension for t in self.templates["uppercut"]]
        upper_y_extension = _avg(upper_y_ext) * threshold_factor if upper_y_ext else 0.06

        self._thresholds = CalibratedThresholds(
            punch_z_velocity_threshold=punch_z_thresh,
            hook_x_velocity_threshold=hook_x_thresh,
            uppercut_y_velocity_threshold=upper_y_thresh,
            punch_z_extension=punch_z_ext,
            hook_x_extension=hook_x_extension,
            uppercut_y_extension=upper_y_extension,
        )
        return self._thresholds

    @property
    def thresholds(self) -> CalibratedThresholds | None:
        return self._thresholds

    def match_move(self, velocity_sequence: list[tuple[float, float, float]]) -> tuple[MoveType, float]:
        """
        Match an incoming velocity sequence against stored templates using DTW.
        Returns (best_move_type, confidence).
        """
        if not velocity_sequence:
            return MoveType.IDLE, 0.0

        best_move = MoveType.IDLE
        best_distance = float("inf")

        for move_name, templates in self.templates.items():
            for template in templates:
                dist = dtw_distance(velocity_sequence, template.velocity_profile)
                if dist < best_distance:
                    best_distance = dist
                    best_move = MoveType(move_name)

        # Convert distance to confidence (lower distance = higher confidence)
        if best_distance == float("inf"):
            return MoveType.IDLE, 0.0

        confidence = max(0.0, min(1.0, 1.0 - best_distance / 0.1))
        return best_move, confidence

    def save(self, filepath: str):
        """Save calibration profile to a JSON file."""
        data = {
            "stance": self.stance,
            "templates": {},
        }
        for move_name, templates in self.templates.items():
            data["templates"][move_name] = [
                {
                    "hand": t.hand,
                    "velocity_profile": t.velocity_profile,
                    "peak_z_velocity": t.peak_z_velocity,
                    "peak_x_velocity": t.peak_x_velocity,
                    "peak_y_velocity": t.peak_y_velocity,
                    "z_extension": t.z_extension,
                    "x_extension": t.x_extension,
                    "y_extension": t.y_extension,
                    "duration_frames": t.duration_frames,
                }
                for t in templates
            ]
        if self._thresholds:
            data["thresholds"] = {
                "punch_z_velocity_threshold": self._thresholds.punch_z_velocity_threshold,
                "hook_x_velocity_threshold": self._thresholds.hook_x_velocity_threshold,
                "uppercut_y_velocity_threshold": self._thresholds.uppercut_y_velocity_threshold,
                "punch_z_extension": self._thresholds.punch_z_extension,
                "hook_x_extension": self._thresholds.hook_x_extension,
                "uppercut_y_extension": self._thresholds.uppercut_y_extension,
            }

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CalibrationProfile":
        """Load calibration profile from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        profile = cls(stance=data.get("stance", "orthodox"))
        for move_name, templates_data in data.get("templates", {}).items():
            for td in templates_data:
                template = MoveTemplate(
                    move_type=MoveType(move_name),
                    hand=td["hand"],
                    velocity_profile=[tuple(v) for v in td["velocity_profile"]],
                    peak_z_velocity=td["peak_z_velocity"],
                    peak_x_velocity=td["peak_x_velocity"],
                    peak_y_velocity=td["peak_y_velocity"],
                    z_extension=td["z_extension"],
                    x_extension=td["x_extension"],
                    y_extension=td["y_extension"],
                    duration_frames=td["duration_frames"],
                )
                profile.add_template(template)

        if "thresholds" in data:
            td = data["thresholds"]
            profile._thresholds = CalibratedThresholds(
                punch_z_velocity_threshold=td["punch_z_velocity_threshold"],
                hook_x_velocity_threshold=td["hook_x_velocity_threshold"],
                uppercut_y_velocity_threshold=td["uppercut_y_velocity_threshold"],
                punch_z_extension=td["punch_z_extension"],
                hook_x_extension=td["hook_x_extension"],
                uppercut_y_extension=td["uppercut_y_extension"],
            )

        return profile
