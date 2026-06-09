"""
Step 5: Final hybrid detector with tuned parameters.
Analyzes the dist=30 peaks in detail and refines classification.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

CLASS_NAMES = ["idle", "jab", "cross", "hook", "uppercut", "walking"]

FEATURE_NAMES = sorted([
    "shoulder_mid_x", "shoulder_mid_y", "shoulder_width", "nose_y",
    "left_wx", "left_wy", "left_wz", "left_dx", "left_dy", "left_dz",
    "right_wx", "right_wy", "right_wz", "right_dx", "right_dy", "right_dz",
    "left_vwx", "left_vwy", "left_vwz", "left_vdx", "left_vdy", "left_vdz",
    "right_vwx", "right_vwy", "right_vwz", "right_vdx", "right_vdy", "right_vdz",
])


def get_idx(name: str) -> int:
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


def detect_peaks(features, min_peak_distance=30, velocity_threshold=0.03):
    """Find velocity peaks."""
    vel_indices = []
    for hand in ["left", "right"]:
        for coord in ["vwx", "vwy", "vwz"]:
            vel_indices.append(get_idx(f"{hand}_{coord}"))

    vel_mag = np.zeros(len(features))
    for i in range(len(features)):
        v_sum = sum(features[i, idx] ** 2 for idx in vel_indices)
        vel_mag[i] = np.sqrt(v_sum)

    peaks = []
    for i in range(2, len(vel_mag) - 2):
        if vel_mag[i] < velocity_threshold:
            continue
        if (vel_mag[i] >= vel_mag[i-1] and vel_mag[i] >= vel_mag[i+1] and
            vel_mag[i] >= vel_mag[i-2] and vel_mag[i] >= vel_mag[i+2]):
            peaks.append({"frame": i, "velocity": float(vel_mag[i])})

    filtered = []
    for peak in peaks:
        if not filtered:
            filtered.append(peak)
        elif peak["frame"] - filtered[-1]["frame"] < min_peak_distance:
            if peak["velocity"] > filtered[-1]["velocity"]:
                filtered[-1] = peak
        else:
            filtered.append(peak)

    return filtered, vel_mag


def analyze_peak_velocities(features, frame, hw=5):
    """Analyze velocity patterns around a peak frame."""
    start = max(0, frame - hw)
    end = min(len(features), frame + hw + 1)
    win = features[start:end]

    results = {}
    for hand in ["left", "right"]:
        for coord in ["vwx", "vwy", "vwz", "vdx", "vdy", "vdz"]:
            key = f"{hand}_{coord}"
            idx = get_idx(key)
            vals = win[:, idx]
            results[key] = {
                "mean": float(np.mean(vals)),
                "abs_mean": float(np.mean(np.abs(vals))),
                "max": float(np.max(np.abs(vals))),
            }

    # Displacement features (not velocity) - wrist positions relative to shoulder
    for hand in ["left", "right"]:
        for coord in ["dx", "dy", "dz"]:
            key = f"{hand}_{coord}"
            idx = get_idx(key)
            frame_val = features[frame, idx]
            results[f"{key}_pos"] = float(frame_val)

    return results


def classify_move(vel_analysis, ml_probs):
    """
    Classify a move using velocity analysis + ML probabilities.

    Key signatures:
    - JAB: Lead hand z-velocity dominant, smaller amplitude than cross
    - CROSS: Rear hand z-velocity dominant, larger amplitude
    - HOOK: X-velocity dominant (lateral arc), often with z-component
    - UPPERCUT: Y-velocity clearly dominant (upward movement)
    """
    # Extract key velocity magnitudes
    lvy = vel_analysis["left_vwy"]["abs_mean"]
    rvy = vel_analysis["right_vwy"]["abs_mean"]
    lvx = vel_analysis["left_vwx"]["abs_mean"]
    rvx = vel_analysis["right_vwx"]["abs_mean"]
    lvz = vel_analysis["left_vwz"]["abs_mean"]
    rvz = vel_analysis["right_vwz"]["abs_mean"]

    # Displacement velocities
    ldvy = vel_analysis["left_vdy"]["abs_mean"]
    rdvy = vel_analysis["right_vdy"]["abs_mean"]
    ldvx = vel_analysis["left_vdx"]["abs_mean"]
    rdvx = vel_analysis["right_vdx"]["abs_mean"]
    ldvz = vel_analysis["left_vdz"]["abs_mean"]
    rdvz = vel_analysis["right_vdz"]["abs_mean"]

    # Directional velocities (signed)
    lvy_signed = vel_analysis["left_vwy"]["mean"]
    rvy_signed = vel_analysis["right_vwy"]["mean"]

    # Total velocity by axis
    y_total = max(lvy, rvy)
    x_total = max(lvx, rvx)
    z_total = max(lvz, rvz)

    # Displacement velocity totals
    dy_total = max(ldvy, rdvy)
    dx_total = max(ldvx, rdvx)
    dz_total = max(ldvz, rdvz)

    # Overall magnitude
    total_vel = y_total + x_total + z_total

    # ML prediction
    ml_pred = int(np.argmax(ml_probs))
    ml_conf = float(ml_probs[ml_pred])

    # --- Classification rules ---

    # HOOK: High z-velocity (hooks involve fast forward/backward wrist movement
    # in MediaPipe coordinates) AND x-velocity component
    # Hooks have the highest velocity magnitudes typically
    if ml_pred == 3 and ml_conf > 0.7:
        return 3, ml_conf, "ML:hook"

    # HOOK: z-dominant with high magnitude (hooks create the biggest velocity spikes)
    if z_total > 0.06 and z_total > y_total * 1.5 and total_vel > 0.1:
        return 3, max(0.8, float(ml_probs[3])), "VEL:hook(z-dominant-high)"

    # UPPERCUT: y-velocity clearly dominant
    if y_total > z_total * 1.3 and y_total > x_total * 1.3 and y_total > 0.025:
        return 4, max(0.75, float(ml_probs[4])), "VEL:uppercut(y-dominant)"

    # UPPERCUT: y-displacement velocity dominant
    if dy_total > dz_total * 1.2 and dy_total > dx_total and dy_total > 0.1:
        return 4, max(0.7, float(ml_probs[4])), "VEL:uppercut(dy-dominant)"

    # JAB vs CROSS: both z-dominant
    # JAB: typically faster, shorter duration, smaller total displacement
    # CROSS: bigger rotation, more shoulder involvement
    if ml_pred in (1, 2):
        # Check if it's really z-dominant (not misclassified hook/uppercut)
        if z_total > y_total and z_total > x_total:
            return ml_pred, ml_conf, f"ML:{'jab' if ml_pred==1 else 'cross'}"
        # ML says jab/cross but velocity says otherwise
        if y_total > z_total * 1.2:
            return 4, 0.65, "OVERRIDE:uppercut(y>z)"
        return ml_pred, ml_conf, f"ML:{'jab' if ml_pred==1 else 'cross'}(fallback)"

    # Default to ML prediction for anything else
    return ml_pred, ml_conf, f"ML:{CLASS_NAMES[ml_pred]}(default)"


def main():
    print("=" * 60)
    print("STEP 5: FINAL HYBRID DETECTOR")
    print("=" * 60)

    device = torch.device("cpu")
    model = MoveClassifierCNN(n_features=28, n_classes=6).to(device)
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()

    norm = np.load("ml/models/norm_stats.npz")
    mean = norm["mean"]
    std = norm["std"]

    mixed = np.load("ml/data/mixed_video_features.npz")
    mixed_X = mixed["X"]
    print(f"  Mixed video: {mixed_X.shape[0]} frames")

    # Find peaks with dist=30
    peaks, vel_mag = detect_peaks(mixed_X, min_peak_distance=30, velocity_threshold=0.03)
    print(f"  Found {len(peaks)} velocity peaks")

    # Classify each peak
    window_size = 16
    results = []

    for peak in peaks:
        frame = peak["frame"]

        # ML classification
        half = window_size // 2
        start = max(0, frame - half)
        end = min(len(mixed_X), start + window_size)
        start = max(0, end - window_size)

        if end - start < window_size:
            continue

        window = mixed_X[start:end]
        window_norm = (window - mean) / std

        with torch.no_grad():
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        # Velocity analysis
        vel_analysis = analyze_peak_velocities(mixed_X, frame)

        # Combined classification
        pred_idx, conf, reason = classify_move(vel_analysis, probs)

        results.append({
            "frame": frame,
            "time": round(frame / 30.0, 2),
            "velocity": round(peak["velocity"], 4),
            "pred": CLASS_NAMES[pred_idx],
            "conf": round(conf, 3),
            "reason": reason,
            "ml_pred": CLASS_NAMES[int(probs.argmax())],
            "ml_conf": round(float(probs.max()), 3),
            "vel_y": round(max(vel_analysis["left_vwy"]["abs_mean"],
                             vel_analysis["right_vwy"]["abs_mean"]), 4),
            "vel_x": round(max(vel_analysis["left_vwx"]["abs_mean"],
                             vel_analysis["right_vwx"]["abs_mean"]), 4),
            "vel_z": round(max(vel_analysis["left_vwz"]["abs_mean"],
                             vel_analysis["right_vwz"]["abs_mean"]), 4),
        })

    # Filter out idle/walking predictions (those are false peaks)
    move_results = [r for r in results if r["pred"] not in ("idle", "walking")]

    print(f"\n  Detected {len(move_results)} moves:")
    for r in move_results:
        print(f"    [{r['time']:5.2f}s] {r['pred'].upper():>9s} "
              f"(conf={r['conf']:.2f}) "
              f"vel(y={r['vel_y']:.3f} x={r['vel_x']:.3f} z={r['vel_z']:.3f}) "
              f"ML={r['ml_pred']}({r['ml_conf']:.2f}) "
              f"reason={r['reason']}")

    counts = {}
    for r in move_results:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1

    print(f"\n    Total: {len(move_results)} moves")
    print(f"    Breakdown: {counts}")
    print(f"    Expected:  jab=2, cross=3, hook=3, uppercut=3 = 11 total")

    # Score
    expected = {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3}
    correct = 0
    for move, exp_ct in expected.items():
        det = counts.get(move, 0)
        correct += min(det, exp_ct)
    false_pos = len(move_results) - correct
    print(f"    Correct: {correct}/11, False positives: {false_pos}")

    # Save the detection config for the integrated detector
    config = {
        "min_peak_distance": 30,
        "velocity_threshold": 0.03,
        "window_size": 16,
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
    }
    with open("ml/models/detector_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Detector config saved")


if __name__ == "__main__":
    main()
