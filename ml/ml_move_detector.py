"""
ML-based hybrid move detector using velocity peak detection + 1D-CNN classification.

Algorithm:
  1. Extract position + velocity features from each PoseFrame (28 features)
  2. Detect velocity peaks (local maxima in wrist velocity magnitude)
  3. For each peak, extract a 16-frame window and classify with the CNN
  4. Apply velocity heuristics to override ML when confident:
     - HOOK: z-velocity > 0.06 AND z > y*2.0
     - UPPERCUT: y-velocity > 0.035 AND y > z*1.3 AND y > x*1.5
     - JAB/CROSS: Trust ML when confident (>0.5)

Tuned parameters (from tune_v3.py sweep across 7 labeled videos):
  min_peak_distance=10, velocity_threshold=0.040, valley_ratio=0.25
  heavy_gap=35, light_gap=12, hook_z_thresh=0.090
  Result: 52/56 correct (92.9%), 0 idle/walking FP

Maintains the same interface as MoveDetector (detect() returns DetectedMove).
"""
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn

from core.pose_estimator import PoseFrame
from core.move_detector import MoveType, MovePhase, DetectedMove


# Canonical feature names — MUST be alphabetically sorted to match training data
FEATURE_NAMES = sorted([
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
])

CLASS_NAMES = ["idle", "jab", "cross", "hook", "uppercut", "walking"]

# Velocity feature indices for peak detection
_VEL_INDICES = [FEATURE_NAMES.index(f"{h}_{c}")
                for h in ["left", "right"]
                for c in ["vwx", "vwy", "vwz"]]


class MoveClassifierCNN(nn.Module):
    """1D-CNN classifier matching the trained model architecture."""

    def __init__(self, n_features: int = 28, n_classes: int = 6):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch, features, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)  # global average pooling
        return self.classifier(x)


class MLMoveDetector:
    """
    Hybrid move detector: velocity peak detection + ML classification.

    Accumulates a buffer of feature frames. When a velocity peak is detected
    (local maximum exceeding threshold, with minimum distance from previous peak),
    it extracts a 16-frame window around the peak, classifies it with the CNN,
    and applies velocity-based heuristic overrides for robust classification.

    Maintains the same interface as MoveDetector for drop-in replacement.
    """

    DEFAULT_MIN_PEAK_DISTANCE = 10
    DEFAULT_VELOCITY_THRESHOLD = 0.040
    DEFAULT_WINDOW_SIZE = 16
    DEFAULT_VALLEY_RATIO = 0.25
    DEFAULT_HOOK_Z_THRESH = 0.090
    DEFAULT_UPPERCUT_Y_THRESH = 0.050  # raised from 0.035: prevents jab->uppercut misclassification
    DEFAULT_HEAVY_GAP = 35
    DEFAULT_LIGHT_GAP = 12

    def __init__(
        self,
        model_dir: str = "ml/models",
        min_peak_distance: int = DEFAULT_MIN_PEAK_DISTANCE,
        velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stance: str = "orthodox",
        valley_ratio: float = DEFAULT_VALLEY_RATIO,
        hook_z_thresh: float = DEFAULT_HOOK_Z_THRESH,
        uppercut_y_thresh: float = DEFAULT_UPPERCUT_Y_THRESH,
        heavy_gap: int = DEFAULT_HEAVY_GAP,
        light_gap: int = DEFAULT_LIGHT_GAP,
    ):
        self._min_peak_distance = min_peak_distance
        self._velocity_threshold = velocity_threshold
        self._window_size = window_size
        self.stance = stance
        self._valley_ratio = valley_ratio
        self._hook_z_thresh = hook_z_thresh
        self._uppercut_y_thresh = uppercut_y_thresh
        self._heavy_gap = heavy_gap
        self._light_gap = light_gap

        # Load model and normalization stats
        self._device = torch.device("cpu")
        self._model = self._load_model(model_dir)
        self._mean, self._std = self._load_norm_stats(model_dir)

        # Feature buffer: list of 28-element feature vectors (one per frame)
        self._feature_buffer: list[list[float]] = []
        self._frame_count = 0

        # Peak detection state
        self._last_peak_frame = -100
        self._last_peak_vel = 0.0  # velocity of previous accepted peak

        # Post-classification dedup state
        self._last_move_type = ""  # name of last classified move
        self._last_move_frame = -100  # frame of last classified move

        # Current move state
        self._current_move = DetectedMove(
            move_type=MoveType.IDLE, phase=MovePhase.NONE,
            confidence=0.0, hand="none",
        )
        self._move_decay_frames = 0  # frames remaining to show detected move
        self._active_display_duration = 20  # ~0.67s at 30fps

        # Velocity tracking for dojo mode velocity bars
        self._velocities: dict[str, float] = {
            "wrist_z": 0.0, "wrist_x": 0.0, "wrist_y": 0.0,
        }

    def _load_model(self, model_dir: str) -> MoveClassifierCNN:
        config_path = os.path.join(model_dir, "model_config.json")
        model_path = os.path.join(model_dir, "move_classifier.pt")

        with open(config_path) as f:
            config = json.load(f)

        model = MoveClassifierCNN(
            n_features=config["n_features"],
            n_classes=config["n_classes"],
        )
        model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        model.eval()
        return model

    def _load_norm_stats(self, model_dir: str) -> tuple[np.ndarray, np.ndarray]:
        stats = np.load(os.path.join(model_dir, "norm_stats.npz"))
        return stats["mean"], stats["std"]

    def _extract_position_features(self, pose: PoseFrame) -> dict[str, float]:
        """Extract 16 position features from a single PoseFrame."""
        if not pose.valid:
            return {}

        kps = pose.keypoints
        ls = kps.get("left_shoulder")
        rs = kps.get("right_shoulder")
        nose = kps.get("nose")

        if not ls or not rs:
            return {}

        shoulder_mid_x = (ls.x + rs.x) / 2
        shoulder_mid_y = (ls.y + rs.y) / 2
        shoulder_width = math.sqrt((ls.x - rs.x) ** 2 + (ls.y - rs.y) ** 2)
        if shoulder_width < 0.01:
            shoulder_width = 0.1

        features: dict[str, float] = {
            "shoulder_mid_x": shoulder_mid_x,
            "shoulder_mid_y": shoulder_mid_y,
            "shoulder_width": shoulder_width,
            "nose_y": nose.y if nose else shoulder_mid_y - 0.1,
        }

        for hand in ["left", "right"]:
            wrist = kps.get(f"{hand}_wrist")
            shoulder = kps.get(f"{hand}_shoulder")

            if not wrist or not shoulder:
                for suffix in ["wx", "wy", "wz", "dx", "dy", "dz"]:
                    features[f"{hand}_{suffix}"] = 0.0
                continue

            features[f"{hand}_wx"] = wrist.x
            features[f"{hand}_wy"] = wrist.y
            features[f"{hand}_wz"] = wrist.z
            features[f"{hand}_dx"] = (wrist.x - shoulder.x) / shoulder_width
            features[f"{hand}_dy"] = (wrist.y - shoulder.y) / shoulder_width
            features[f"{hand}_dz"] = (wrist.z - shoulder.z) / shoulder_width

        return features

    def _extract_features(self, pose: PoseFrame) -> list[float]:
        """Extract full 28-element feature vector (16 position + 12 velocity)."""
        pos_feats = self._extract_position_features(pose)
        if not pos_feats:
            return [0.0] * 28

        # Compute velocity features from buffer history
        vel_window = 3
        vel_feats: dict[str, float] = {}
        for hand in ["left", "right"]:
            for coord in ["wx", "wy", "wz", "dx", "dy", "dz"]:
                key = f"{hand}_{coord}"
                vel_key = f"{hand}_v{coord}"
                if len(self._feature_buffer) >= vel_window:
                    prev_row = self._feature_buffer[-vel_window]
                    prev_idx = FEATURE_NAMES.index(key)
                    curr_val = pos_feats.get(key, 0.0)
                    if len(prev_row) > prev_idx:
                        vel_feats[vel_key] = (curr_val - prev_row[prev_idx]) / vel_window
                    else:
                        vel_feats[vel_key] = 0.0
                else:
                    vel_feats[vel_key] = 0.0

        # Combine into ordered feature vector matching FEATURE_NAMES
        all_feats: dict[str, float] = {}
        all_feats.update(pos_feats)
        all_feats.update(vel_feats)
        return [all_feats.get(k, 0.0) for k in FEATURE_NAMES]

    def _compute_velocity_magnitude(self, features: list[float]) -> float:
        """Compute total wrist velocity magnitude from feature vector."""
        return math.sqrt(sum(features[idx] ** 2 for idx in _VEL_INDICES))

    def _classify_peak_at(self, buf_idx: int) -> tuple[str, float, str]:
        """
        Classify a velocity peak at a specific buffer index using
        ML model + velocity heuristics.
        Returns (move_name, confidence, reason).
        """
        buf = self._feature_buffer
        n = len(buf)

        half = self._window_size // 2
        start = max(0, buf_idx - half)
        end = min(n, start + self._window_size)
        start = max(0, end - self._window_size)

        if end - start < self._window_size:
            return "idle", 0.0, "skip"

        window = np.array(buf[start:end], dtype=np.float32)
        window_norm = (window - self._mean) / self._std

        with torch.no_grad():
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(self._device)
            probs = torch.softmax(self._model(x), dim=1).squeeze().numpy()

        ml_pred = int(probs.argmax())
        ml_conf = float(probs[ml_pred])

        # Velocity stats around peak (+-5 frames)
        hw = 5
        s = max(0, buf_idx - hw)
        e = min(n, buf_idx + hw + 1)
        win_slice = np.array(buf[s:e], dtype=np.float32)

        def abs_mean(feat_name: str) -> float:
            idx = FEATURE_NAMES.index(feat_name)
            return float(np.mean(np.abs(win_slice[:, idx])))

        y_vel = max(abs_mean("left_vwy"), abs_mean("right_vwy"))
        x_vel = max(abs_mean("left_vwx"), abs_mean("right_vwx"))
        z_vel = max(abs_mean("left_vwz"), abs_mean("right_vwz"))

        # Update velocity tracking for dojo display
        self._velocities["wrist_y"] = y_vel
        self._velocities["wrist_x"] = x_vel
        self._velocities["wrist_z"] = z_vel

        # ── Retraction guard: check wrist z-velocity SIGN at the peak ─────────
        # Punches extend TOWARD camera (z decreasing -> vwz NEGATIVE).
        # Retractions move AWAY from camera (z increasing -> vwz POSITIVE).
        # If the dominant wrist z-velocity at the peak is positive, this is
        # a retraction arc, not a punch — suppress it regardless of ML output.
        def signed_z_mean(feat_name: str) -> float:
            idx = FEATURE_NAMES.index(feat_name)
            return float(np.mean(win_slice[:, idx]))  # keeps sign

        z_signed_l = signed_z_mean("left_vwz")
        z_signed_r = signed_z_mean("right_vwz")
        # Pick the hand with stronger signal
        z_signed = z_signed_l if abs(z_signed_l) > abs(z_signed_r) else z_signed_r

        # If arm is moving away from camera AND z is not tiny, it's retraction
        if z_signed > 0.015:  # positive = away from camera = retraction
            return "idle", 0.0, "RETRACTION_GUARD"

        # Rule 1: HOOK — ML confident or z-velocity very high
        if ml_pred == 3 and ml_conf > 0.7:
            return "hook", ml_conf, "ML"
        if z_vel > self._hook_z_thresh and z_vel > y_vel * 2.0:
            return "hook", max(0.8, float(probs[3])), "VEL:z-dominant"

        # Rule 2: UPPERCUT — y-velocity clearly dominant
        if y_vel > self._uppercut_y_thresh and y_vel > z_vel * 1.3 and y_vel > x_vel * 1.5:
            return "uppercut", max(0.75, float(probs[4])), "VEL:y-dominant"

        # Rule 3: JAB/CROSS — trust ML
        if ml_pred in (1, 2) and ml_conf > 0.5:
            return CLASS_NAMES[ml_pred], ml_conf, "ML"

        # Rule 4: Trust ML for idle/walking (no fallback override)
        if ml_pred in (0, 5):
            return CLASS_NAMES[ml_pred], ml_conf, "ML:trust-idle"

        return CLASS_NAMES[ml_pred], ml_conf, "ML:default"

    def detect(self, pose: PoseFrame) -> DetectedMove:
        """
        Detect the current move from a pose frame.
        Drop-in replacement for MoveDetector.detect().
        """
        self._frame_count += 1

        # Extract features and add to buffer
        features = self._extract_features(pose)
        self._feature_buffer.append(features)

        # Keep buffer bounded (need enough for valley detection between peaks)
        max_buf = self._window_size + max(self._heavy_gap, self._min_peak_distance) + 30
        if len(self._feature_buffer) > max_buf:
            self._feature_buffer = self._feature_buffer[-max_buf:]

        # Update running velocity display values
        vel_mag = self._compute_velocity_magnitude(features)
        if vel_mag > 0.01:
            for hand in ["left", "right"]:
                for axis, key in [("vwz", "wrist_z"), ("vwx", "wrist_x"), ("vwy", "wrist_y")]:
                    idx = FEATURE_NAMES.index(f"{hand}_{axis}")
                    v = abs(features[idx])
                    if v > abs(self._velocities.get(key, 0.0)):
                        self._velocities[key] = v
        # Decay velocities for smooth display
        for k in self._velocities:
            self._velocities[k] *= 0.9

        # If we're still showing the last detected move, keep showing it
        if self._move_decay_frames > 0:
            self._move_decay_frames -= 1
            return self._current_move

        # Need enough frames for peak detection (require +-2 neighborhood)
        n = len(self._feature_buffer)
        if n < 5:
            return DetectedMove(
                move_type=MoveType.IDLE, phase=MovePhase.NONE,
                confidence=0.0, hand="none", frame_index=self._frame_count,
            )

        # Check if frame at buf[n-3] is a velocity peak
        # (we wait 2 frames to confirm it's a local maximum over +-2 neighbors)
        candidate_idx = n - 3
        candidate_frame = self._frame_count - 2

        if candidate_idx >= 2 and candidate_frame - self._last_peak_frame >= self._min_peak_distance:
            cur_vel = self._compute_velocity_magnitude(self._feature_buffer[candidate_idx])
            if cur_vel >= self._velocity_threshold:
                is_peak = True
                for offset in [-2, -1, 1, 2]:
                    neighbor_idx = candidate_idx + offset
                    if 0 <= neighbor_idx < n:
                        neighbor_vel = self._compute_velocity_magnitude(
                            self._feature_buffer[neighbor_idx])
                        if cur_vel < neighbor_vel:
                            is_peak = False
                            break

                if is_peak:
                    # Valley-based peak suppression: require velocity
                    # to drop between consecutive peaks
                    if self._last_peak_frame > 0 and self._last_peak_vel > 0:
                        prev_buf_idx = candidate_idx - (candidate_frame - self._last_peak_frame)
                        if prev_buf_idx >= 0 and prev_buf_idx < candidate_idx - 2:
                            min_vel = min(
                                self._compute_velocity_magnitude(self._feature_buffer[i])
                                for i in range(prev_buf_idx + 1, candidate_idx)
                            )
                            threshold = min(self._last_peak_vel, cur_vel) * self._valley_ratio
                            if min_vel > threshold:
                                # No valley between peaks — keep higher one
                                if cur_vel > self._last_peak_vel:
                                    self._last_peak_frame = candidate_frame
                                    self._last_peak_vel = cur_vel
                                is_peak = False

                if is_peak:
                    pred_name, confidence, _reason = self._classify_peak_at(candidate_idx)

                    if pred_name not in ("idle", "walking"):
                        # Post-classification dedup: enforce type-specific
                        # minimum gaps between consecutive same-type moves
                        if pred_name == self._last_move_type:
                            is_heavy = pred_name in ("hook", "uppercut")
                            required_gap = self._heavy_gap if is_heavy else self._light_gap
                            if candidate_frame - self._last_move_frame < required_gap:
                                # Too close to previous same-type move
                                if cur_vel > self._last_peak_vel:
                                    self._last_peak_frame = candidate_frame
                                    self._last_peak_vel = cur_vel
                                is_peak = False

                if is_peak:
                    pred_name, confidence, _reason = self._classify_peak_at(candidate_idx)

                    if pred_name not in ("idle", "walking"):
                        self._last_peak_frame = candidate_frame
                        self._last_peak_vel = cur_vel
                        self._last_move_type = pred_name
                        self._last_move_frame = candidate_frame

                        move_type_map = {
                            "jab": MoveType.JAB,
                            "cross": MoveType.CROSS,
                            "hook": MoveType.HOOK,
                            "uppercut": MoveType.UPPERCUT,
                        }
                        move_type = move_type_map.get(pred_name, MoveType.IDLE)

                        if move_type in (MoveType.JAB, MoveType.HOOK):
                            hand = "left" if self.stance == "orthodox" else "right"
                        elif move_type in (MoveType.CROSS, MoveType.UPPERCUT):
                            hand = "right" if self.stance == "orthodox" else "left"
                        else:
                            hand = "none"

                        self._current_move = DetectedMove(
                            move_type=move_type,
                            phase=MovePhase.ACTIVE,
                            confidence=confidence,
                            hand=hand,
                            frame_index=self._frame_count,
                            start_frame=candidate_frame,
                            peak_velocity=cur_vel,
                        )
                        self._move_decay_frames = self._active_display_duration
                        return self._current_move

        # No peak detected — return idle
        return DetectedMove(
            move_type=MoveType.IDLE, phase=MovePhase.NONE,
            confidence=0.0, hand="none", frame_index=self._frame_count,
        )

    @property
    def lead_hand(self) -> str:
        return "left" if self.stance == "orthodox" else "right"

    @property
    def rear_hand(self) -> str:
        return "right" if self.stance == "orthodox" else "left"

    def reset(self):
        """Reset all state."""
        self._feature_buffer.clear()
        self._frame_count = 0
        self._last_peak_frame = -100
        self._last_peak_vel = 0.0
        self._last_move_type = ""
        self._last_move_frame = -100
        self._move_decay_frames = 0
        self._current_move = DetectedMove(
            move_type=MoveType.IDLE, phase=MovePhase.NONE,
            confidence=0.0, hand="none",
        )
        self._velocities = {"wrist_z": 0.0, "wrist_x": 0.0, "wrist_y": 0.0}

    @property
    def move_history(self) -> list[DetectedMove]:
        return []

    @property
    def config(self):
        """Compatibility with MoveDetector interface."""
        class _Config:
            def __init__(self, stance: str):
                self.stance = stance
        return _Config(self.stance)
