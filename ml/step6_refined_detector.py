"""
Step 6: Refined hybrid detector with better tuned thresholds.

Key fixes:
1. Trust ML more when confident (jab vs uppercut distinction)
2. Tighter y-velocity threshold for uppercut override (>0.035 not >0.025)
3. Try dist=25 to capture more peaks, then filter by velocity
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


def detect_peaks(features, min_peak_distance=25, velocity_threshold=0.03):
    vel_indices = [get_idx(f"{h}_{c}") for h in ["left", "right"] for c in ["vwx", "vwy", "vwz"]]
    vel_mag = np.array([np.sqrt(sum(features[i, idx] ** 2 for idx in vel_indices))
                        for i in range(len(features))])

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


def get_vel_stats(features, frame, hw=5):
    start = max(0, frame - hw)
    end = min(len(features), frame + hw + 1)
    win = features[start:end]
    stats = {}
    for hand in ["left", "right"]:
        for coord in ["vwx", "vwy", "vwz", "vdx", "vdy", "vdz"]:
            key = f"{hand}_{coord}"
            vals = win[:, get_idx(key)]
            stats[f"{key}_abs"] = float(np.mean(np.abs(vals)))
            stats[f"{key}_mean"] = float(np.mean(vals))
            stats[f"{key}_max"] = float(np.max(np.abs(vals)))
    return stats


def classify_move(stats, ml_probs):
    """Classify with refined rules."""
    # Wrist velocity magnitudes
    lvy = stats["left_vwy_abs"]
    rvy = stats["right_vwy_abs"]
    lvx = stats["left_vwx_abs"]
    rvx = stats["right_vwx_abs"]
    lvz = stats["left_vwz_abs"]
    rvz = stats["right_vwz_abs"]

    # Displacement velocity magnitudes
    ldvy = stats["left_vdy_abs"]
    rdvy = stats["right_vdy_abs"]

    y_total = max(lvy, rvy)
    x_total = max(lvx, rvx)
    z_total = max(lvz, rvz)
    dy_total = max(ldvy, rdvy)

    ml_pred = int(np.argmax(ml_probs))
    ml_conf = float(ml_probs[ml_pred])

    # HOOK: ML confident + high z-velocity (hooks create biggest velocity spikes)
    if ml_pred == 3 and ml_conf > 0.7:
        return 3, ml_conf, "ML:hook"
    if z_total > 0.06 and z_total > y_total * 2.0:
        return 3, max(0.8, float(ml_probs[3])), "VEL:hook"

    # UPPERCUT: Only override if y-velocity is CLEARLY dominant
    # Threshold: y > 0.035 AND y > z * 1.5 AND y > x * 2
    if y_total > 0.035 and y_total > z_total * 1.5 and y_total > x_total * 2:
        return 4, max(0.75, float(ml_probs[4])), "VEL:uppercut"

    # UPPERCUT: displacement y-velocity dominant (more robust)
    if dy_total > 0.15 and dy_total > stats.get("left_vdz_abs", 0) * 1.5:
        return 4, max(0.7, float(ml_probs[4])), "VEL:uppercut(dy)"

    # JAB/CROSS: Trust ML when it's confident
    if ml_pred in (1, 2) and ml_conf > 0.6:
        return ml_pred, ml_conf, f"ML:{'jab' if ml_pred == 1 else 'cross'}"

    # If ML says jab/cross but y is slightly dominant, still trust ML if confident
    if ml_pred in (1, 2) and ml_conf > 0.8:
        return ml_pred, ml_conf, f"ML:{'jab' if ml_pred == 1 else 'cross'}(trusted)"

    # Default
    return ml_pred, ml_conf, f"ML:{CLASS_NAMES[ml_pred]}"


def run_detector(features, model, mean, std, min_peak_dist, vel_thresh, verbose=True):
    """Run full detection pipeline, return results."""
    peaks, vel_mag = detect_peaks(features, min_peak_dist, vel_thresh)

    window_size = 16
    device = torch.device("cpu")
    model.eval()
    results = []

    for peak in peaks:
        frame = peak["frame"]
        half = window_size // 2
        start = max(0, frame - half)
        end = min(len(features), start + window_size)
        start = max(0, end - window_size)
        if end - start < window_size:
            continue

        window = features[start:end]
        window_norm = (window - mean) / std
        with torch.no_grad():
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        stats = get_vel_stats(features, frame)
        pred_idx, conf, reason = classify_move(stats, probs)

        if CLASS_NAMES[pred_idx] in ("idle", "walking"):
            continue

        results.append({
            "frame": frame,
            "time": round(frame / 30.0, 2),
            "velocity": round(peak["velocity"], 4),
            "pred": CLASS_NAMES[pred_idx],
            "conf": round(conf, 3),
            "reason": reason,
            "ml_pred": CLASS_NAMES[int(probs.argmax())],
            "ml_conf": round(float(probs.max()), 3),
            "vel_y": round(max(stats["left_vwy_abs"], stats["right_vwy_abs"]), 4),
            "vel_x": round(max(stats["left_vwx_abs"], stats["right_vwx_abs"]), 4),
            "vel_z": round(max(stats["left_vwz_abs"], stats["right_vwz_abs"]), 4),
        })

    if verbose:
        for r in results:
            print(f"    [{r['time']:5.2f}s] {r['pred'].upper():>9s} "
                  f"(conf={r['conf']:.2f}) "
                  f"vel(y={r['vel_y']:.3f} x={r['vel_x']:.3f} z={r['vel_z']:.3f}) "
                  f"ML={r['ml_pred']}({r['ml_conf']:.2f}) reason={r['reason']}")

    return results


def score_results(results):
    counts = {}
    for r in results:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1

    expected = {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3}
    correct = sum(min(counts.get(m, 0), c) for m, c in expected.items())
    false_pos = len(results) - correct
    return counts, correct, false_pos


def main():
    print("=" * 60)
    print("STEP 6: REFINED HYBRID DETECTOR")
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

    # Parameter sweep
    print("\n--- PARAMETER SWEEP ---")
    best_score = -999
    best_params = None
    for dist in [20, 22, 25, 27, 30, 33, 35]:
        for thresh in [0.025, 0.03, 0.035, 0.04]:
            results = run_detector(mixed_X, model, mean, std, dist, thresh, verbose=False)
            counts, correct, fp = score_results(results)
            score = correct - fp
            if score > best_score or (score == best_score and abs(len(results) - 11) < abs(best_params[2] - 11)):
                best_score = score
                best_params = (dist, thresh, len(results), counts, correct, fp)
            if dist in [25, 30] and thresh == 0.03:
                print(f"    dist={dist:2d} thresh={thresh:.3f}: "
                      f"{len(results):2d} moves, correct={correct}/11, fp={fp}, "
                      f"score={score}, {counts}")

    dist, thresh = best_params[0], best_params[1]
    print(f"\n  Best: dist={dist}, thresh={thresh}, "
          f"moves={best_params[2]}, correct={best_params[4]}/11, "
          f"fp={best_params[5]}, {best_params[3]}")

    # Run with best params (verbose)
    print(f"\n--- RESULTS WITH BEST PARAMS (dist={dist}, thresh={thresh}) ---")
    results = run_detector(mixed_X, model, mean, std, dist, thresh, verbose=True)
    counts, correct, fp = score_results(results)
    print(f"\n    Total: {len(results)} moves")
    print(f"    Breakdown: {counts}")
    print(f"    Expected:  jab=2, cross=3, hook=3, uppercut=3 = 11 total")
    print(f"    Correct: {correct}/11, False positives: {fp}")

    # Also test with dist=30 (known good for hooks)
    if dist != 30:
        print(f"\n--- ALSO TESTING dist=30 thresh=0.03 ---")
        results30 = run_detector(mixed_X, model, mean, std, 30, 0.03, verbose=True)
        counts30, correct30, fp30 = score_results(results30)
        print(f"\n    Total: {len(results30)}, Correct: {correct30}/11, FP: {fp30}, {counts30}")


if __name__ == "__main__":
    main()
