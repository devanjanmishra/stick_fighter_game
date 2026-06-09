"""
Step 4: Hybrid detector — velocity peak detection + ML classification.

Approach:
1. Use wrist velocity magnitude to find punch events (peaks)
2. Extract a window around each peak
3. Classify the window with the trained ML model
4. Combine with velocity-based heuristics for final classification

This separates "WHEN does a punch happen" from "WHAT type is it".
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

CLASS_NAMES = ["idle", "jab", "cross", "hook", "uppercut", "walking"]

# Feature name order (alphabetical)
FEATURE_NAMES = sorted([
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
])

# Find indices of velocity features in alphabetical order
def get_feature_idx(name: str) -> int:
    return FEATURE_NAMES.index(name)


class MoveClassifierCNN(nn.Module):
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
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.mean(dim=2)
        return self.classifier(x)


def detect_velocity_peaks(
    features: np.ndarray,
    min_peak_distance: int = 15,
    velocity_threshold: float = 0.02,
) -> list[dict]:
    """Find velocity peaks that indicate punch events."""
    # Get velocity feature indices
    vel_indices = []
    for hand in ["left", "right"]:
        for coord in ["vwx", "vwy", "vwz"]:
            vel_indices.append(get_feature_idx(f"{hand}_{coord}"))

    # Compute velocity magnitude per frame
    vel_mag = np.zeros(len(features))
    for i in range(len(features)):
        v_sum = 0.0
        for idx in vel_indices:
            v_sum += features[i, idx] ** 2
        vel_mag[i] = np.sqrt(v_sum)

    # Find peaks
    peaks = []
    for i in range(2, len(vel_mag) - 2):
        if vel_mag[i] < velocity_threshold:
            continue
        if vel_mag[i] >= vel_mag[i-1] and vel_mag[i] >= vel_mag[i+1]:
            if vel_mag[i] >= vel_mag[i-2] and vel_mag[i] >= vel_mag[i+2]:
                peaks.append({"frame": i, "velocity": float(vel_mag[i])})

    # Filter by minimum distance (keep the strongest peak in each neighborhood)
    filtered = []
    for peak in peaks:
        if not filtered:
            filtered.append(peak)
            continue
        if peak["frame"] - filtered[-1]["frame"] < min_peak_distance:
            if peak["velocity"] > filtered[-1]["velocity"]:
                filtered[-1] = peak
        else:
            filtered.append(peak)

    return filtered, vel_mag


def classify_peaks(
    features: np.ndarray,
    peaks: list[dict],
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    window_size: int = 16,
) -> list[dict]:
    """Classify each peak using the ML model + velocity heuristics."""
    device = torch.device("cpu")
    model.eval()

    # Feature indices for velocity heuristics
    idx_left_vwy = get_feature_idx("left_vwy")
    idx_right_vwy = get_feature_idx("right_vwy")
    idx_left_vwx = get_feature_idx("left_vwx")
    idx_right_vwx = get_feature_idx("right_vwx")
    idx_left_vwz = get_feature_idx("left_vwz")
    idx_right_vwz = get_feature_idx("right_vwz")
    idx_left_vdz = get_feature_idx("left_vdz")
    idx_right_vdz = get_feature_idx("right_vdz")
    idx_left_vdy = get_feature_idx("left_vdy")
    idx_right_vdy = get_feature_idx("right_vdy")
    idx_left_vdx = get_feature_idx("left_vdx")
    idx_right_vdx = get_feature_idx("right_vdx")

    classified = []
    for peak in peaks:
        frame = peak["frame"]

        # Extract window centered on peak
        half = window_size // 2
        start = max(0, frame - half)
        end = min(len(features), start + window_size)
        start = max(0, end - window_size)

        if end - start < window_size:
            continue

        window = features[start:end]

        # ML classification
        window_norm = (window - mean) / std
        with torch.no_grad():
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        ml_pred = int(probs.argmax())
        ml_conf = float(probs[ml_pred])

        # Velocity heuristics around the peak (use a small window)
        hw = 4
        p_start = max(0, frame - hw)
        p_end = min(len(features), frame + hw)
        peak_window = features[p_start:p_end]

        avg_vwy_left = float(np.mean(np.abs(peak_window[:, idx_left_vwy])))
        avg_vwy_right = float(np.mean(np.abs(peak_window[:, idx_right_vwy])))
        avg_vwx_left = float(np.mean(np.abs(peak_window[:, idx_left_vwx])))
        avg_vwx_right = float(np.mean(np.abs(peak_window[:, idx_right_vwx])))
        avg_vwz_left = float(np.mean(np.abs(peak_window[:, idx_left_vwz])))
        avg_vwz_right = float(np.mean(np.abs(peak_window[:, idx_right_vwz])))
        avg_vdy_left = float(np.mean(peak_window[:, idx_left_vdy]))
        avg_vdy_right = float(np.mean(peak_window[:, idx_right_vdy]))
        avg_vdx_left = float(np.mean(np.abs(peak_window[:, idx_left_vdx])))
        avg_vdx_right = float(np.mean(np.abs(peak_window[:, idx_right_vdx])))

        # Y-velocity dominance suggests uppercut
        y_vel = max(avg_vwy_left, avg_vwy_right)
        x_vel = max(avg_vwx_left, avg_vwx_right)
        z_vel = max(avg_vwz_left, avg_vwz_right)

        # Heuristic classification
        heuristic_pred = None
        if y_vel > x_vel and y_vel > z_vel * 0.8:
            # Y-dominant = likely uppercut
            heuristic_pred = 4  # uppercut
        elif x_vel > z_vel and x_vel > y_vel:
            # X-dominant = likely hook
            heuristic_pred = 3  # hook
        # z-dominant = jab or cross (keep ML decision)

        # Final decision: combine ML + heuristics
        final_pred = ml_pred
        final_conf = ml_conf

        # If ML says idle/walking but we have a velocity peak, override
        if ml_pred in (0, 5) and peak["velocity"] > 0.03:
            if heuristic_pred is not None:
                final_pred = heuristic_pred
                final_conf = 0.7
            else:
                final_pred = 2  # default to cross
                final_conf = 0.5

        # If heuristic strongly suggests uppercut but ML disagrees
        if heuristic_pred == 4 and ml_pred != 4:
            # Check if y-velocity is clearly dominant
            if y_vel > 1.5 * z_vel and y_vel > 1.5 * x_vel:
                final_pred = 4
                final_conf = max(0.7, float(probs[4]) * 2)

        # If heuristic suggests hook but ML says jab/cross
        if heuristic_pred == 3 and ml_pred in (1, 2):
            if x_vel > 1.5 * z_vel:
                final_pred = 3
                final_conf = max(0.7, float(probs[3]) * 2)

        classified.append({
            "frame": frame,
            "time": round(frame / 30.0, 2),
            "velocity": round(peak["velocity"], 4),
            "ml_pred": CLASS_NAMES[ml_pred],
            "ml_conf": round(ml_conf, 3),
            "heuristic": CLASS_NAMES[heuristic_pred] if heuristic_pred else "none",
            "final_pred": CLASS_NAMES[final_pred],
            "final_conf": round(final_conf, 3),
            "vel_y": round(y_vel, 4),
            "vel_x": round(x_vel, 4),
            "vel_z": round(z_vel, 4),
        })

    return classified


def main():
    print("=" * 60)
    print("STEP 4: HYBRID DETECTOR (VELOCITY PEAKS + ML)")
    print("=" * 60)

    # Load model
    device = torch.device("cpu")
    model = MoveClassifierCNN(n_features=28, n_classes=6).to(device)
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()

    # Load norm stats
    norm = np.load("ml/models/norm_stats.npz")
    mean = norm["mean"]
    std = norm["std"]

    # Load mixed video features
    mixed = np.load("ml/data/mixed_video_features.npz")
    mixed_X = mixed["X"]
    print(f"  Mixed video: {mixed_X.shape[0]} frames, {mixed_X.shape[1]} features")

    # Step 1: Find velocity peaks
    print("\n--- Velocity Peak Detection ---")
    peaks, vel_mag = detect_velocity_peaks(mixed_X, min_peak_distance=15, velocity_threshold=0.02)
    print(f"  Found {len(peaks)} velocity peaks:")
    for p in peaks:
        ts = p["frame"] / 30.0
        print(f"    Frame {p['frame']} [{ts:.2f}s]: vel={p['velocity']:.4f}")

    # Step 2: Classify each peak
    print("\n--- Peak Classification ---")
    results = classify_peaks(mixed_X, peaks, model, mean, std)

    for r in results:
        marker = " <-- HEURISTIC OVERRIDE" if r["heuristic"] != "none" and r["final_pred"] != r["ml_pred"] else ""
        print(f"    [{r['time']:5.2f}s] ML={r['ml_pred']:>9s}({r['ml_conf']:.2f}) "
              f"H={r['heuristic']:>9s} "
              f"FINAL={r['final_pred']:>9s}({r['final_conf']:.2f}) "
              f"vel(y={r['vel_y']:.3f} x={r['vel_x']:.3f} z={r['vel_z']:.3f})"
              f"{marker}")

    # Summary
    counts = {}
    for r in results:
        t = r["final_pred"]
        counts[t] = counts.get(t, 0) + 1

    print(f"\n    Total: {len(results)} moves detected")
    print(f"    Breakdown: {counts}")
    print(f"    Expected:  jab=2, cross=3, hook=3, uppercut=3 = 11 total")

    # Try different peak distance / threshold combos
    print("\n\n--- PARAMETER SWEEP ---")
    for min_dist in [15, 20, 25, 30]:
        for thresh in [0.02, 0.025, 0.03, 0.035, 0.04]:
            peaks_t, _ = detect_velocity_peaks(mixed_X, min_peak_distance=min_dist,
                                                velocity_threshold=thresh)
            results_t = classify_peaks(mixed_X, peaks_t, model, mean, std)
            cts = {}
            for r in results_t:
                t = r["final_pred"]
                cts[t] = cts.get(t, 0) + 1
            total = len(results_t)
            # Score: how close to expected
            exp = {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3}
            score = 0
            for move, exp_ct in exp.items():
                det = cts.get(move, 0)
                score += min(det, exp_ct)
                score -= max(0, det - exp_ct)  # penalize false positives
            print(f"    dist={min_dist:2d} thresh={thresh:.3f}: "
                  f"{total:2d} moves, score={score:3d}/11, {cts}")


if __name__ == "__main__":
    main()
