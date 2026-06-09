"""
Rule-based move detection using shoulder-relative wrist coordinates.

ALL wrist tracking is done relative to the shoulder on the same side:
    rel_x = wrist_x - shoulder_x
    rel_y = wrist_y - shoulder_y
    rel_z = wrist_z - shoulder_z

This cancels out:
  - Body leaning forward/backward (shifts absolute z but not relative z)
  - Head tilts and postural sway
  - Hand resting near face (wrist z close to shoulder z -> small relative z)

A real punch is the ONLY thing that makes wrist_z much less than shoulder_z,
because the arm extends past the shoulder plane toward the camera.

Move signatures in relative coords:
  Jab:      Lead wrist rel_z decreases (arm extends toward camera past shoulder)
  Cross:    Rear wrist rel_z decreases
  Hook:     Wrist rel_x changes significantly (lateral sweep away from body)
  Uppercut: Wrist rel_y decreases (wrist rises above shoulder level)
"""

import math
from enum import Enum
from dataclasses import dataclass, field
from core.pose_estimator import PoseFrame, Keypoint


class MoveType(Enum):
    IDLE     = "idle"
    JAB      = "jab"
    CROSS    = "cross"
    HOOK     = "hook"
    UPPERCUT = "uppercut"


class MovePhase(Enum):
    NONE     = "none"
    WINDUP   = "windup"
    ACTIVE   = "active"
    RECOVERY = "recovery"


@dataclass
class DetectedMove:
    move_type:    MoveType
    phase:        MovePhase
    confidence:   float   # 0–1
    hand:         str     # "left" | "right"
    frame_index:  int   = 0
    start_frame:  int   = 0
    peak_velocity: float = 0.0


@dataclass
class MoveDetectorConfig:
    """
    Thresholds in shoulder-relative normalised coords.

    Calibrated from real training video data + live testing:
      Idle rel_z drift:   < 0.020 / frame  (shoulder and wrist not perfectly in sync)
      Punch rel_z peak:   ~ 0.060 / frame
      Threshold at 0.035: clears idle noise, catches 90%+ of real punches.

    Displacement thresholds over the 6-frame tracker window:
      Punch extends ~0.15+ relative to shoulder; we require 0.10 minimum.
    """
    # Velocity thresholds (signed, per-frame average over window)
    # Raised z threshold 0.025->0.035: live testing showed idle breathing
    # creates rel_z noise up to 0.025, causing continuous false cross detections.
    punch_z_velocity_threshold:   float = 0.035
    hook_x_velocity_threshold:    float = 0.015
    uppercut_y_velocity_threshold: float = 0.018

    # Displacement thresholds (must travel this far in 6-frame window)
    # Raised z extension 0.07->0.10: real punches easily clear 0.10 relative
    # displacement; idle drift accumulates slowly and rarely reaches 0.10.
    punch_z_extension:    float = 0.10
    hook_x_extension:     float = 0.05
    uppercut_y_extension: float = 0.05

    # Timing
    min_move_frames:      int   = 3
    max_move_frames:      int   = 20
    cooldown_frames:      int   = 20   # ~0.67s at 30fps
    noise_cooldown_frames: int  = 8
    warmup_frames:        int   = 15   # frames to collect baseline before detecting

    # Confidence gate
    min_confidence: float = 0.35

    # Stance
    stance: str = "orthodox"   # "orthodox" = left lead, "southpaw" = right lead


class RelativeTracker:
    """
    Tracks the shoulder-relative position of one wrist coordinate,
    computing signed velocity and displacement.

    Using relative coords means body movement cancels out:
        rel_z = wrist_z - shoulder_z
    If you lean forward, both wrist_z and shoulder_z decrease equally,
    so rel_z stays constant. Only arm extension changes rel_z.
    """

    def __init__(self, window: int = 6):
        self._history: list[float] = []
        self._window  = window
        self._absent  = 0  # consecutive frames wrist was invisible

    def update(self, rel_val: float):
        self._absent = 0
        self._history.append(rel_val)
        if len(self._history) > self._window + 1:
            self._history = self._history[-(self._window + 1):]

    def mark_absent(self):
        """Wrist not visible this frame. Clear history after 2 frames."""
        self._absent += 1
        if self._absent >= 2:
            self._history.clear()

    # ── Derived signals ────────────────────────────────────────────────────

    @property
    def signed_velocity(self) -> float:
        """Average signed velocity over window. Preserves direction."""
        if len(self._history) < 2:
            return 0.0
        deltas = [self._history[i] - self._history[i-1]
                  for i in range(1, len(self._history))]
        return sum(deltas) / len(deltas)

    @property
    def displacement(self) -> float:
        """Total signed displacement: newest - oldest in window."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-1] - self._history[0]

    @property
    def has_history(self) -> bool:
        return len(self._history) >= 2

    def reset(self):
        self._history.clear()
        self._absent = 0


class MoveDetector:
    """
    Detects fighting moves from a stream of PoseFrames.
    Call detect() once per frame.
    """

    def __init__(self, config: MoveDetectorConfig | None = None):
        self.config = config or MoveDetectorConfig()

        # One tracker per wrist per axis (x, y, z) — all relative to shoulder
        self._trackers: dict[str, dict[str, RelativeTracker]] = {
            f"{hand}_wrist": {"x": RelativeTracker(), "y": RelativeTracker(), "z": RelativeTracker()}
            for hand in ("left", "right")
        }

        self._current_move    = DetectedMove(MoveType.IDLE, MovePhase.NONE, 0.0, "none")
        self._cooldown        = 0
        self._move_frames     = 0
        self._frame_count     = 0

        # Expose velocities for Dojo velocity bars
        self._velocities: dict[str, float] = {"wrist_z": 0.0, "wrist_x": 0.0, "wrist_y": 0.0}

    # ── Public interface ───────────────────────────────────────────────────

    @property
    def lead_hand(self) -> str:
        return "left" if self.config.stance == "orthodox" else "right"

    @property
    def rear_hand(self) -> str:
        return "right" if self.config.stance == "orthodox" else "left"

    def detect(self, pose: PoseFrame) -> DetectedMove:
        """Main entry point — call once per frame."""
        self._frame_count += 1

        if not pose.valid:
            return self._current_move

        self._update_trackers(pose)

        # Cooldown: no new moves allowed
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._cooldown == 0:
                self._reset_all_trackers()
            return DetectedMove(MoveType.IDLE, MovePhase.RECOVERY, 0.0, "none",
                                frame_index=self._frame_count)

        # Warmup: collect baseline before detecting
        if self._frame_count < self.config.warmup_frames:
            return DetectedMove(MoveType.IDLE, MovePhase.NONE, 0.0, "none",
                                frame_index=self._frame_count)

        # Classify both hands, pick the more confident one
        best = MoveType.IDLE
        best_conf = 0.0
        best_hand = "none"
        for hand in (self.lead_hand, self.rear_hand):
            mt, conf = self._classify(hand)
            if conf > best_conf:
                best, best_conf, best_hand = mt, conf, hand

        # Update velocity display
        wrist = f"{self.lead_hand}_wrist"
        self._velocities = {
            "wrist_z": abs(self._trackers[wrist]["z"].signed_velocity),
            "wrist_x": abs(self._trackers[wrist]["x"].signed_velocity),
            "wrist_y": abs(self._trackers[wrist]["y"].signed_velocity),
        }

        # Resolve "none" hand: use move type to infer which hand it must be
        if best_hand == "none" and best != MoveType.IDLE:
            if best == MoveType.JAB:
                best_hand = self.lead_hand
            elif best == MoveType.CROSS:
                best_hand = self.rear_hand
            else:
                best_hand = self.lead_hand  # hook/uppercut default to lead

        if best != MoveType.IDLE and best_conf >= self.config.min_confidence:
            if self._current_move.move_type == MoveType.IDLE:
                # New move started
                if best_hand == "none":
                    best_hand = self.lead_hand
                self._move_frames = 1
                self._current_move = DetectedMove(
                    best, MovePhase.ACTIVE, best_conf, best_hand,
                    frame_index=self._frame_count, start_frame=self._frame_count,
                )
            else:
                # Continue current move (locked — no type switching mid-move)
                self._move_frames += 1
                if self._move_frames > self.config.max_move_frames:
                    self._end_move()
                else:
                    self._current_move = DetectedMove(
                        self._current_move.move_type,
                        MovePhase.ACTIVE,
                        best_conf,
                        self._current_move.hand,
                        frame_index=self._frame_count,
                        start_frame=self._current_move.start_frame,
                    )
        else:
            if self._current_move.move_type != MoveType.IDLE:
                if self._move_frames >= self.config.min_move_frames:
                    self._end_move()
                else:
                    # Too brief — treat as noise
                    self._cooldown = self.config.noise_cooldown_frames
                    self._move_frames = 0
                    self._current_move = DetectedMove(
                        MoveType.IDLE, MovePhase.RECOVERY, 0.0, "none",
                        frame_index=self._frame_count)
            else:
                self._current_move = DetectedMove(
                    MoveType.IDLE, MovePhase.NONE, 0.0, "none",
                    frame_index=self._frame_count)

        return self._current_move

    def reset(self):
        self._reset_all_trackers()
        self._current_move = DetectedMove(MoveType.IDLE, MovePhase.NONE, 0.0, "none")
        self._cooldown = self._move_frames = self._frame_count = 0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _update_trackers(self, pose: PoseFrame):
        """Feed shoulder-relative wrist positions into trackers."""
        for hand in ("left", "right"):
            wrist_name   = f"{hand}_wrist"
            shoulder_name = f"{hand}_shoulder"
            wrist_kp    = pose.get(wrist_name)
            shoulder_kp = pose.get(shoulder_name)
            t = self._trackers[wrist_name]

            if wrist_kp and shoulder_kp:
                # Relative coords — shoulder movement cancels out
                t["x"].update(wrist_kp.x - shoulder_kp.x)
                t["y"].update(wrist_kp.y - shoulder_kp.y)
                t["z"].update(wrist_kp.z - shoulder_kp.z)
            else:
                t["x"].mark_absent()
                t["y"].mark_absent()
                t["z"].mark_absent()

    def _classify(self, hand: str) -> tuple[MoveType, float]:
        """
        Classify what move this hand is doing using signed relative velocities.

        Direction conventions (relative coords):
          rel_z < 0  → wrist is IN FRONT of shoulder (toward camera)
          rel_z → more negative → arm extending = punch
          rel_y < 0  → wrist is ABOVE shoulder level = uppercut
          rel_y → more negative → rising fist
        """
        t = self._trackers[f"{hand}_wrist"]
        if not all(tr.has_history for tr in t.values()):
            return MoveType.IDLE, 0.0

        svz = t["z"].signed_velocity   # neg = extending toward camera
        svx = t["x"].signed_velocity   # lateral
        svy = t["y"].signed_velocity   # neg = rising

        dz = t["z"].displacement       # neg = extended toward camera
        dx = t["x"].displacement       # lateral
        dy = t["y"].displacement       # neg = risen

        cfg = self.config

        # ── UPPERCUT: wrist rises above shoulder (rel_y goes negative) ────
        if (dy < -cfg.uppercut_y_extension
                and svy < -cfg.uppercut_y_velocity_threshold
                and abs(dy) > abs(dx) * 1.2
                and abs(dy) > abs(dz) * 0.5):
            conf = min(1.0, abs(svy) / (cfg.uppercut_y_velocity_threshold * 2))
            return MoveType.UPPERCUT, conf

        # ── HOOK: lateral sweep, rel_x changes significantly ─────────────
        # Either direction (hooks can go both ways).
        # Require lateral dominates over depth (prevents jab-with-x-drift).
        if (abs(dx) > cfg.hook_x_extension
                and abs(svx) > cfg.hook_x_velocity_threshold
                and abs(dx) > abs(dz) * 0.8):
            conf = min(1.0, abs(svx) / (cfg.hook_x_velocity_threshold * 2))
            return MoveType.HOOK, conf

        # ── JAB / CROSS: arm extends toward camera (rel_z goes negative) ─
        # SIGNED check: svz must be NEGATIVE (extending), not positive (retracting).
        # This is the critical fix — retractions have svz > 0, never detected.
        if (dz < -cfg.punch_z_extension
                and svz < -cfg.punch_z_velocity_threshold
                and abs(dz) > abs(dx) * 1.5
                and abs(dz) > abs(dy) * 1.5):
            conf = min(1.0, abs(svz) / (cfg.punch_z_velocity_threshold * 2))
            if hand == self.lead_hand:
                return MoveType.JAB, conf
            else:
                return MoveType.CROSS, conf

        return MoveType.IDLE, 0.0

    def _end_move(self):
        self._cooldown = self.config.cooldown_frames
        self._move_frames = 0
        self._current_move = DetectedMove(
            MoveType.IDLE, MovePhase.RECOVERY, 0.0, "none",
            frame_index=self._frame_count)

    def _reset_all_trackers(self):
        for t in self._trackers.values():
            for tr in t.values():
                tr.reset()

    @property
    def move_history(self) -> list:
        return []
