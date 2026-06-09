"""
Generate synthetic training data for the move classifier.

Creates realistic keypoint time-series for each move type based on
biomechanical models of boxing punches. Variations include:
  - Speed (slow/medium/fast punchers)
  - Amplitude (short/long reach)
  - Body proportions (shoulder width, arm length)
  - Noise (camera jitter, keypoint estimation noise)
  - Stance (orthodox/southpaw)
  - Starting position variation
"""
import math
import random
import numpy as np
from typing import Optional


# Move types
MOVE_TYPES = ["idle", "jab", "cross", "hook", "uppercut"]

# Feature names (must match extract_features.py output)
POSITION_FEATURES = [
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
]

VELOCITY_FEATURES = [
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
]

ALL_FEATURES = POSITION_FEATURES + VELOCITY_FEATURES


def _smooth_trajectory(values: list[float], kernel_size: int = 3) -> list[float]:
    """Simple moving average smoothing."""
    if len(values) < kernel_size:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - kernel_size // 2)
        end = min(len(values), i + kernel_size // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


class BodyModel:
    """Represents a person's body proportions and resting pose."""
    
    def __init__(
        self,
        shoulder_width: float = 0.25,
        shoulder_mid_x: float = 0.5,
        shoulder_mid_y: float = 0.4,
        arm_length: float = 0.15,
        stance: str = "orthodox",
    ):
        self.shoulder_width = shoulder_width
        self.shoulder_mid_x = shoulder_mid_x
        self.shoulder_mid_y = shoulder_mid_y
        self.arm_length = arm_length
        self.stance = stance
        self.nose_y = shoulder_mid_y - 0.12
        
        # Resting wrist positions (guard position)
        hw = shoulder_width / 2
        self.left_shoulder_x = shoulder_mid_x - hw
        self.right_shoulder_x = shoulder_mid_x + hw
        
        # Guard: wrists near chin level, slightly in front
        self.rest = {
            "left_wx": self.left_shoulder_x + 0.02,
            "left_wy": shoulder_mid_y - 0.06,
            "left_wz": -0.05,
            "right_wx": self.right_shoulder_x - 0.02,
            "right_wy": shoulder_mid_y - 0.06,
            "right_wz": -0.05,
        }
    
    @staticmethod
    def random() -> "BodyModel":
        """Generate a random body model with realistic proportions."""
        return BodyModel(
            shoulder_width=random.uniform(0.15, 0.35),
            shoulder_mid_x=random.uniform(0.4, 0.6),
            shoulder_mid_y=random.uniform(0.3, 0.5),
            arm_length=random.uniform(0.10, 0.22),
            stance=random.choice(["orthodox", "southpaw"]),
        )


def _generate_punch_trajectory(
    body: BodyModel,
    hand: str,
    move_type: str,
    duration_frames: int,
    speed_factor: float = 1.0,
    amplitude: float = 1.0,
    noise_std: float = 0.003,
) -> list[dict]:
    """Generate a single punch trajectory as a sequence of feature dicts.
    
    The trajectory follows: rest → windup → extension → retraction → rest
    Using sine-based easing for natural motion.
    """
    frames = []
    rest = body.rest.copy()
    
    # Determine peak displacement based on move type
    # These are in normalized coordinates (0-1 space)
    if move_type == "jab":
        # Forward extension (z decreases = toward camera)
        peak_dz = -0.15 * amplitude
        peak_dx = 0.01 * amplitude  # slight lateral
        peak_dy = 0.01 * amplitude   # slight up
    elif move_type == "cross":
        # Rear hand forward, more z, slight shoulder rotation
        peak_dz = -0.20 * amplitude
        peak_dx = -0.03 * amplitude if hand == "right" else 0.03 * amplitude
        peak_dy = 0.0
    elif move_type == "hook":
        # Large lateral arc then forward
        peak_dz = -0.08 * amplitude
        peak_dx = 0.12 * amplitude if hand == "left" else -0.12 * amplitude
        peak_dy = -0.02 * amplitude  # slight up
    elif move_type == "uppercut":
        # Strong upward movement
        peak_dz = -0.06 * amplitude
        peak_dx = 0.02 * amplitude if hand == "left" else -0.02 * amplitude
        peak_dy = -0.15 * amplitude  # upward (y decreases)
    else:
        peak_dz = 0.0
        peak_dx = 0.0
        peak_dy = 0.0
    
    # Compute shoulder reference
    shoulder_x = body.left_shoulder_x if hand == "left" else body.right_shoulder_x
    
    # Phase timing (as fraction of total duration)
    # windup: 0-20%, strike: 20-50%, hold: 50-60%, retract: 60-100%
    for i in range(duration_frames):
        t = i / max(1, duration_frames - 1)
        
        # Compute displacement using piecewise sine curves
        if t < 0.15:
            # Windup: slight pullback
            phase = t / 0.15
            s = math.sin(phase * math.pi / 2)
            dx = -peak_dx * 0.1 * s
            dy = -peak_dy * 0.1 * s
            dz = -peak_dz * 0.15 * s  # pull back slightly
        elif t < 0.45:
            # Strike: rapid forward
            phase = (t - 0.15) / 0.30
            s = math.sin(phase * math.pi / 2)
            dx = peak_dx * s
            dy = peak_dy * s
            dz = peak_dz * s
        elif t < 0.55:
            # Hold at peak
            dx = peak_dx
            dy = peak_dy
            dz = peak_dz
        else:
            # Retract
            phase = (t - 0.55) / 0.45
            s = 1.0 - math.sin(phase * math.pi / 2)
            dx = peak_dx * s
            dy = peak_dy * s
            dz = peak_dz * s
        
        # Hook has special arc trajectory
        if move_type == "hook":
            if t < 0.3:
                # Lateral phase dominates first
                lat_phase = min(1.0, t / 0.3)
                fwd_phase = min(1.0, max(0, (t - 0.15) / 0.3))
                dx = peak_dx * math.sin(lat_phase * math.pi / 2)
                dz = peak_dz * math.sin(fwd_phase * math.pi / 2)
                dy = peak_dy * math.sin(lat_phase * math.pi / 2)
            elif t < 0.5:
                # Both lateral and forward
                dx = peak_dx
                dz = peak_dz
                dy = peak_dy
            else:
                phase = (t - 0.5) / 0.5
                s = 1.0 - math.sin(phase * math.pi / 2)
                dx = peak_dx * s
                dz = peak_dz * s
                dy = peak_dy * s
        
        # Build feature dict
        feat = {
            "shoulder_mid_x": body.shoulder_mid_x + random.gauss(0, noise_std * 0.3),
            "shoulder_mid_y": body.shoulder_mid_y + random.gauss(0, noise_std * 0.3),
            "shoulder_width": body.shoulder_width + random.gauss(0, noise_std * 0.2),
            "nose_y": body.nose_y + random.gauss(0, noise_std * 0.3),
        }
        
        for h in ["left", "right"]:
            prefix = h
            if h == hand:
                # Active hand
                wx = rest[f"{h}_wx"] + dx + random.gauss(0, noise_std)
                wy = rest[f"{h}_wy"] + dy + random.gauss(0, noise_std)
                wz = rest[f"{h}_wz"] + dz + random.gauss(0, noise_std)
            else:
                # Passive hand: slight sympathetic movement
                sym = 0.05
                wx = rest[f"{h}_wx"] + dx * sym + random.gauss(0, noise_std)
                wy = rest[f"{h}_wy"] + dy * sym + random.gauss(0, noise_std)
                wz = rest[f"{h}_wz"] + dz * sym + random.gauss(0, noise_std)
            
            s_x = body.left_shoulder_x if h == "left" else body.right_shoulder_x
            feat[f"{prefix}_wx"] = wx
            feat[f"{prefix}_wy"] = wy
            feat[f"{prefix}_wz"] = wz
            feat[f"{prefix}_dx"] = (wx - s_x) / body.shoulder_width
            feat[f"{prefix}_dy"] = (wy - body.shoulder_mid_y) / body.shoulder_width
            feat[f"{prefix}_dz"] = (wz - 0.0) / body.shoulder_width
        
        frames.append(feat)
    
    return frames


def _generate_idle_sequence(
    body: BodyModel,
    duration_frames: int,
    noise_std: float = 0.003,
    sway_amplitude: float = 0.01,
) -> list[dict]:
    """Generate idle/guard stance with natural sway."""
    frames = []
    rest = body.rest.copy()
    
    for i in range(duration_frames):
        t = i / max(1, duration_frames - 1)
        
        # Natural body sway
        sway_x = sway_amplitude * math.sin(2 * math.pi * t * 0.5)
        sway_y = sway_amplitude * 0.5 * math.sin(2 * math.pi * t * 0.7)
        
        feat = {
            "shoulder_mid_x": body.shoulder_mid_x + sway_x + random.gauss(0, noise_std * 0.3),
            "shoulder_mid_y": body.shoulder_mid_y + sway_y + random.gauss(0, noise_std * 0.3),
            "shoulder_width": body.shoulder_width + random.gauss(0, noise_std * 0.2),
            "nose_y": body.nose_y + sway_y + random.gauss(0, noise_std * 0.3),
        }
        
        for h in ["left", "right"]:
            wx = rest[f"{h}_wx"] + sway_x + random.gauss(0, noise_std)
            wy = rest[f"{h}_wy"] + sway_y + random.gauss(0, noise_std)
            wz = rest[f"{h}_wz"] + random.gauss(0, noise_std)
            
            s_x = body.left_shoulder_x if h == "left" else body.right_shoulder_x
            feat[f"{h}_wx"] = wx
            feat[f"{h}_wy"] = wy
            feat[f"{h}_wz"] = wz
            feat[f"{h}_dx"] = (wx - s_x) / body.shoulder_width
            feat[f"{h}_dy"] = (wy - body.shoulder_mid_y) / body.shoulder_width
            feat[f"{h}_dz"] = (wz - 0.0) / body.shoulder_width
        
        frames.append(feat)
    
    return frames


def add_velocity_features(frames: list[dict], window: int = 3) -> list[dict]:
    """Compute velocity features for a sequence of feature dicts."""
    enriched = []
    for i, feat in enumerate(frames):
        new_feat = dict(feat)
        for hand in ["left", "right"]:
            for coord in ["wx", "wy", "wz", "dx", "dy", "dz"]:
                key = f"{hand}_{coord}"
                vel_key = f"{hand}_v{coord}"
                if i < window:
                    new_feat[vel_key] = 0.0
                else:
                    prev = frames[i - window]
                    new_feat[vel_key] = (feat[key] - prev[key]) / window
        enriched.append(new_feat)
    return enriched


def generate_dataset(
    n_per_class: int = 500,
    window_size: int = 16,
    stride: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a full training dataset.
    
    Returns:
        X: (N, window_size, n_features) array of feature windows
        y: (N,) array of integer labels
        class_names: list of class name strings
    """
    random.seed(seed)
    np.random.seed(seed)
    
    class_names = MOVE_TYPES  # ["idle", "jab", "cross", "hook", "uppercut"]
    
    all_windows = []  # list of (window_array, label_index)
    
    for class_idx, move_type in enumerate(class_names):
        count = 0
        attempts = 0
        
        while count < n_per_class and attempts < n_per_class * 5:
            attempts += 1
            body = BodyModel.random()
            
            # Randomize parameters
            speed = random.uniform(0.7, 1.5)
            amplitude = random.uniform(0.6, 1.4)
            noise = random.uniform(0.001, 0.008)
            
            if move_type == "idle":
                duration = random.randint(window_size + 5, window_size + 30)
                sway = random.uniform(0.005, 0.025)
                raw_frames = _generate_idle_sequence(body, duration, noise, sway)
            else:
                # Determine active hand based on stance and move
                if body.stance == "orthodox":
                    hand = "left" if move_type in ("jab", "hook") else "right"
                else:
                    hand = "right" if move_type in ("jab", "hook") else "left"
                
                # Sometimes flip hand for variety
                if random.random() < 0.3:
                    hand = "right" if hand == "left" else "left"
                
                duration = random.randint(window_size + 2, window_size + 15)
                raw_frames = _generate_punch_trajectory(
                    body, hand, move_type, duration, speed, amplitude, noise
                )
            
            # Add velocity features
            enriched = add_velocity_features(raw_frames)
            
            # Extract windows
            # For punches: center window around the peak of the punch
            # For idle: random window position
            if move_type == "idle":
                for start in range(0, len(enriched) - window_size, stride):
                    window = enriched[start:start + window_size]
                    feature_matrix = _frames_to_matrix(window)
                    if feature_matrix is not None:
                        all_windows.append((feature_matrix, class_idx))
                        count += 1
                        if count >= n_per_class:
                            break
            else:
                # For punches, take overlapping windows that include the strike phase
                peak_frame = int(len(enriched) * 0.4)  # strike peaks around 40%
                start = max(0, peak_frame - window_size // 2)
                start = min(start, len(enriched) - window_size)
                if start >= 0 and start + window_size <= len(enriched):
                    window = enriched[start:start + window_size]
                    feature_matrix = _frames_to_matrix(window)
                    if feature_matrix is not None:
                        all_windows.append((feature_matrix, class_idx))
                        count += 1
                
                # Also add windows shifted slightly for data augmentation
                for offset in [-3, -1, 1, 3]:
                    s2 = start + offset
                    if 0 <= s2 and s2 + window_size <= len(enriched):
                        window = enriched[s2:s2 + window_size]
                        feature_matrix = _frames_to_matrix(window)
                        if feature_matrix is not None:
                            all_windows.append((feature_matrix, class_idx))
                            count += 1
                            if count >= n_per_class:
                                break
        
        print(f"  {move_type}: generated {count} windows")
    
    # Shuffle and convert to arrays
    random.shuffle(all_windows)
    X = np.array([w[0] for w in all_windows], dtype=np.float32)
    y = np.array([w[1] for w in all_windows], dtype=np.int64)
    
    print(f"\nDataset: {X.shape[0]} samples, window={X.shape[1]}, features={X.shape[2]}")
    print(f"Class distribution: {dict(zip(class_names, [int((y==i).sum()) for i in range(len(class_names))]))}")
    
    return X, y, class_names


def _frames_to_matrix(frames: list[dict]) -> Optional[np.ndarray]:
    """Convert list of feature dicts to a (window_size, n_features) numpy array."""
    if not frames or not frames[0]:
        return None
    
    feature_keys = ALL_FEATURES
    matrix = []
    for feat in frames:
        row = [feat.get(k, 0.0) for k in feature_keys]
        matrix.append(row)
    
    return np.array(matrix, dtype=np.float32)


if __name__ == "__main__":
    X, y, class_names = generate_dataset(n_per_class=1000, window_size=16)
    
    # Save dataset
    import os
    os.makedirs("/home/ubuntu/stick_fighter/ml/data", exist_ok=True)
    np.savez(
        "/home/ubuntu/stick_fighter/ml/data/synthetic_dataset.npz",
        X=X, y=y, class_names=class_names,
    )
    print(f"\nSaved to ml/data/synthetic_dataset.npz")
