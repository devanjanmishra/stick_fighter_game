"""
Step 7: Final tuned detector. Key insight from analysis:
- Real uppercuts: vel_y > 0.035 (clearly dominant upward movement)
- Real jabs: vel_y < 0.030 (some y-movement but z-dominant)
- Hooks: z_total > 0.06 and z >> y (big velocity spikes)
- Only use vwy threshold for uppercut, remove unreliable dy heuristic
"""
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


def get_idx(name):
    return FEATURE_NAMES.index(name)


class MoveClassifierCNN(nn.Module):
    def __init__(self, n_features=28, n_classes=6):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(n_features, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = conv(x)
        return self.classifier(x.mean(dim=2))


def detect_peaks(features, min_dist, vel_thresh):
    vel_indices = [get_idx(f"{h}_{c}") for h in ["left", "right"] for c in ["vwx", "vwy", "vwz"]]
    vel_mag = np.array([np.sqrt(sum(features[i, idx]**2 for idx in vel_indices)) for i in range(len(features))])

    peaks = []
    for i in range(2, len(vel_mag) - 2):
        if vel_mag[i] < vel_thresh:
            continue
        if all(vel_mag[i] >= vel_mag[i + d] for d in [-2, -1, 1, 2]):
            peaks.append({"frame": i, "vel": float(vel_mag[i])})

    filtered = []
    for p in peaks:
        if not filtered or p["frame"] - filtered[-1]["frame"] >= min_dist:
            filtered.append(p)
        elif p["vel"] > filtered[-1]["vel"]:
            filtered[-1] = p
    return filtered


def classify(features, frame, model, mean, std, window_size=16):
    """Classify a single peak using ML + velocity heuristics."""
    device = torch.device("cpu")

    # ML classification
    half = window_size // 2
    start = max(0, frame - half)
    end = min(len(features), start + window_size)
    start = max(0, end - window_size)
    if end - start < window_size:
        return "idle", 0.0, "skip"

    window = features[start:end]
    window_norm = (window - mean) / std
    with torch.no_grad():
        x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
        probs = torch.softmax(model(x), dim=1).squeeze().numpy()

    ml_pred = int(probs.argmax())
    ml_conf = float(probs[ml_pred])

    # Velocity stats around peak
    hw = 5
    s = max(0, frame - hw)
    e = min(len(features), frame + hw + 1)
    win = features[s:e]

    def abs_mean(feat_name):
        return float(np.mean(np.abs(win[:, get_idx(feat_name)])))

    y = max(abs_mean("left_vwy"), abs_mean("right_vwy"))
    x_v = max(abs_mean("left_vwx"), abs_mean("right_vwx"))
    z = max(abs_mean("left_vwz"), abs_mean("right_vwz"))

    # Rule 1: HOOK — ML confident or z-velocity very high
    if ml_pred == 3 and ml_conf > 0.7:
        return "hook", ml_conf, "ML"
    if z > 0.06 and z > y * 2.0:
        return "hook", max(0.8, float(probs[3])), "VEL:z-dominant"

    # Rule 2: UPPERCUT — y-velocity clearly dominant (threshold from analysis)
    # Only fire when y > 0.035 (all real uppercuts are above this)
    # AND y > z * 1.3 (must be clearly dominant over forward movement)
    if y > 0.035 and y > z * 1.3 and y > x_v * 1.5:
        return "uppercut", max(0.75, float(probs[4])), "VEL:y-dominant"

    # Rule 3: JAB/CROSS — trust ML
    if ml_pred in (1, 2) and ml_conf > 0.5:
        return CLASS_NAMES[ml_pred], ml_conf, "ML"

    # Rule 4: If ML says idle/walking but we have a real velocity peak, classify by velocity
    if ml_pred in (0, 5):
        if z > y and z > x_v:
            return "cross", 0.6, "VEL:fallback-z"
        if y > z:
            return "uppercut", 0.6, "VEL:fallback-y"
        return "jab", 0.5, "VEL:fallback"

    return CLASS_NAMES[ml_pred], ml_conf, "ML:default"


def run_full(features, model, mean, std, min_dist, vel_thresh, verbose=True):
    peaks = detect_peaks(features, min_dist, vel_thresh)
    results = []
    for p in peaks:
        pred, conf, reason = classify(features, p["frame"], model, mean, std)
        if pred in ("idle", "walking"):
            continue
        hw = 5
        s = max(0, p["frame"] - hw)
        e = min(len(features), p["frame"] + hw + 1)
        win = features[s:e]
        y_v = max(float(np.mean(np.abs(win[:, get_idx("left_vwy")]))),
                  float(np.mean(np.abs(win[:, get_idx("right_vwy")]))))
        x_v = max(float(np.mean(np.abs(win[:, get_idx("left_vwx")]))),
                  float(np.mean(np.abs(win[:, get_idx("right_vwx")]))))
        z_v = max(float(np.mean(np.abs(win[:, get_idx("left_vwz")]))),
                  float(np.mean(np.abs(win[:, get_idx("right_vwz")]))))

        results.append({
            "frame": p["frame"], "time": round(p["frame"]/30.0, 2),
            "pred": pred, "conf": round(conf, 3), "reason": reason,
            "vel": round(p["vel"], 4),
            "vy": round(y_v, 4), "vx": round(x_v, 4), "vz": round(z_v, 4),
        })

    if verbose:
        for r in results:
            print(f"  [{r['time']:5.2f}s] {r['pred'].upper():>9s} "
                  f"(conf={r['conf']:.2f}) "
                  f"v(y={r['vy']:.3f} x={r['vx']:.3f} z={r['vz']:.3f}) "
                  f"{r['reason']}")

    counts = {}
    for r in results:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1
    return results, counts


def score(counts):
    expected = {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3}
    correct = sum(min(counts.get(m, 0), c) for m, c in expected.items())
    total = sum(counts.values())
    fp = total - correct
    return correct, fp


def main():
    print("=" * 60)
    print("FINAL TUNED HYBRID DETECTOR")
    print("=" * 60)

    device = torch.device("cpu")
    model = MoveClassifierCNN().to(device)
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()

    norm = np.load("ml/models/norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    mixed = np.load("ml/data/mixed_video_features.npz")
    X = mixed["X"]
    print(f"  {X.shape[0]} frames\n")

    # Comprehensive sweep
    print("--- PARAMETER SWEEP ---")
    best = None
    for dist in range(15, 40):
        for thresh_10k in range(200, 500, 25):
            thresh = thresh_10k / 10000.0
            results, counts = run_full(X, model, mean, std, dist, thresh, verbose=False)
            correct, fp = score(counts)
            s = correct * 2 - fp  # Penalize false positives more
            total = sum(counts.values())
            if best is None or s > best[0] or (s == best[0] and abs(total - 11) < abs(best[4] - 11)):
                best = (s, dist, thresh, counts, total, correct, fp)

    print(f"  Best: dist={best[1]}, thresh={best[2]:.4f}, "
          f"total={best[4]}, correct={best[5]}/11, fp={best[6]}, "
          f"counts={best[3]}")

    print(f"\n--- BEST RESULTS (dist={best[1]}, thresh={best[2]:.4f}) ---")
    results, counts = run_full(X, model, mean, std, best[1], best[2], verbose=True)
    correct, fp = score(counts)
    print(f"\n  Total: {sum(counts.values())} moves")
    print(f"  Breakdown: {counts}")
    print(f"  Expected:  jab=2, cross=3, hook=3, uppercut=3 = 11")
    print(f"  Correct: {correct}/11, FP: {fp}")

    # Also show a few other good configs
    print("\n--- OTHER GOOD CONFIGS ---")
    all_configs = []
    for dist in range(15, 40):
        for thresh_10k in range(200, 500, 25):
            thresh = thresh_10k / 10000.0
            results, counts = run_full(X, model, mean, std, dist, thresh, verbose=False)
            correct, fp = score(counts)
            total = sum(counts.values())
            all_configs.append((correct, fp, total, dist, thresh, dict(counts)))

    all_configs.sort(key=lambda x: (x[0], -x[1], -abs(x[2]-11)), reverse=True)
    seen = set()
    for correct, fp, total, dist, thresh, counts in all_configs[:20]:
        key = (correct, fp, total, tuple(sorted(counts.items())))
        if key in seen:
            continue
        seen.add(key)
        print(f"  correct={correct}/11, fp={fp}, total={total}, "
              f"dist={dist}, thresh={thresh:.4f}, {counts}")


if __name__ == "__main__":
    main()
