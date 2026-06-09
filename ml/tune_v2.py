"""
Tuning v2: Valley-based peak suppression to eliminate retraction false positives.
Between two consecutive peaks, velocity must drop below valley_ratio * min(peak1, peak2).
"""
import os
import sys
import math
import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from ml.step1_extract_all_videos import FEATURE_NAMES, CLASS_NAMES
from ml.step7_final_tuned import MoveClassifierCNN

def get_idx(name):
    return FEATURE_NAMES.index(name)

VEL_INDICES = [get_idx(f"{h}_{c}") for h in ["left", "right"] for c in ["vwx", "vwy", "vwz"]]


def compute_vel_mag(features):
    """Compute per-frame velocity magnitude array."""
    return np.array([
        np.sqrt(sum(features[i, idx]**2 for idx in VEL_INDICES))
        for i in range(len(features))
    ])


def detect_peaks_v2(features, min_dist, vel_thresh, valley_ratio=0.35):
    """Detect peaks with valley requirement between consecutive detections."""
    vel_mag = compute_vel_mag(features)

    # Step 1: Find all local maxima above threshold
    raw_peaks = []
    for i in range(2, len(vel_mag) - 2):
        if vel_mag[i] < vel_thresh:
            continue
        if all(vel_mag[i] >= vel_mag[i + d] for d in [-2, -1, 1, 2]):
            raw_peaks.append({"frame": i, "vel": float(vel_mag[i])})

    # Step 2: Min-distance filter (keep highest in window)
    dist_filtered = []
    for p in raw_peaks:
        if not dist_filtered or p["frame"] - dist_filtered[-1]["frame"] >= min_dist:
            dist_filtered.append(p)
        elif p["vel"] > dist_filtered[-1]["vel"]:
            dist_filtered[-1] = p

    if len(dist_filtered) <= 1:
        return dist_filtered

    # Step 3: Valley filter — require velocity to drop between consecutive peaks
    valley_filtered = [dist_filtered[0]]
    for p in dist_filtered[1:]:
        prev = valley_filtered[-1]
        # Find minimum velocity between prev peak and current peak
        start_f = prev["frame"]
        end_f = p["frame"]
        if end_f - start_f < 3:
            # Too close, just keep higher
            if p["vel"] > prev["vel"]:
                valley_filtered[-1] = p
            continue

        min_vel = float(np.min(vel_mag[start_f + 1:end_f]))
        threshold = min(prev["vel"], p["vel"]) * valley_ratio

        if min_vel <= threshold:
            # Good valley — these are separate movements
            valley_filtered.append(p)
        else:
            # No clear valley — likely same movement (punch + retraction)
            # Keep the higher peak
            if p["vel"] > prev["vel"]:
                valley_filtered[-1] = p

    return valley_filtered


def classify_peak(features, frame, model, mean, std, window_size=16,
                  hook_z_thresh=0.09, hook_z_ratio=2.0,
                  uppercut_y_thresh=0.035, uppercut_yz_ratio=1.3):
    """Classify a peak — NO fallback override (trust ML for idle/walking)."""
    half = window_size // 2
    start = max(0, frame - half)
    end = min(len(features), start + window_size)
    start = max(0, end - window_size)
    if end - start < window_size:
        return "idle", 0.0, "skip"

    window = features[start:end]
    window_norm = (window - mean) / std
    with torch.no_grad():
        x = torch.from_numpy(window_norm).unsqueeze(0).float()
        probs = torch.softmax(model(x), dim=1).squeeze().numpy()

    ml_pred = int(probs.argmax())
    ml_conf = float(probs[ml_pred])

    hw = 5
    s = max(0, frame - hw)
    e = min(len(features), frame + hw + 1)
    win = features[s:e]

    def abs_mean(feat_name):
        return float(np.mean(np.abs(win[:, get_idx(feat_name)])))

    y_vel = max(abs_mean("left_vwy"), abs_mean("right_vwy"))
    x_vel = max(abs_mean("left_vwx"), abs_mean("right_vwx"))
    z_vel = max(abs_mean("left_vwz"), abs_mean("right_vwz"))

    # Rule 1: HOOK — ML confident or z-velocity very high
    if ml_pred == 3 and ml_conf > 0.7:
        return "hook", ml_conf, "ML"
    if z_vel > hook_z_thresh and z_vel > y_vel * hook_z_ratio:
        return "hook", max(0.8, float(probs[3])), "VEL:z-dominant"

    # Rule 2: UPPERCUT — y-velocity clearly dominant
    if y_vel > uppercut_y_thresh and y_vel > z_vel * uppercut_yz_ratio and y_vel > x_vel * 1.5:
        return "uppercut", max(0.75, float(probs[4])), "VEL:y-dominant"

    # Rule 3: JAB/CROSS — trust ML
    if ml_pred in (1, 2) and ml_conf > 0.5:
        return CLASS_NAMES[ml_pred], ml_conf, "ML"

    # Rule 4: If ML says idle/walking, TRUST IT (no override)
    if ml_pred in (0, 5):
        return CLASS_NAMES[ml_pred], ml_conf, "ML:trust-idle"

    return CLASS_NAMES[ml_pred], ml_conf, "ML:default"


def run_detector(features, model, mean, std, min_dist, vel_thresh,
                 valley_ratio=0.35, hook_z_thresh=0.09, uppercut_y_thresh=0.035):
    peaks = detect_peaks_v2(features, min_dist, vel_thresh, valley_ratio=valley_ratio)
    results = []
    for p in peaks:
        pred, conf, reason = classify_peak(
            features, p["frame"], model, mean, std,
            hook_z_thresh=hook_z_thresh, uppercut_y_thresh=uppercut_y_thresh,
        )
        if pred in ("idle", "walking"):
            continue
        results.append({
            "frame": p["frame"],
            "time": round(p["frame"] / 30.0, 2),
            "pred": pred,
            "conf": round(conf, 3),
            "reason": reason,
            "vel": round(p["vel"], 4),
        })

    counts = {}
    for r in results:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1
    return results, counts


def score_video(counts, expected_move, expected_count):
    if expected_move in ("idle", "walking"):
        total = sum(counts.values())
        return 0, total, 0
    elif expected_move == "mixed":
        expected_bd = {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3}
        correct = sum(min(counts.get(m, 0), c) for m, c in expected_bd.items())
        total = sum(counts.values())
        fp = total - correct
        missed = expected_count - correct
        return correct, fp, missed
    else:
        correct_count = counts.get(expected_move, 0)
        correct = min(correct_count, expected_count)
        over = max(0, correct_count - expected_count)
        other = sum(v for k, v in counts.items() if k != expected_move)
        fp = over + other
        missed = max(0, expected_count - correct_count)
        return correct, fp, missed


def main():
    print("=" * 70)
    print("TUNING v2: VALLEY-BASED PEAK SUPPRESSION")
    print("=" * 70)

    model = MoveClassifierCNN()
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()
    norm = np.load("ml/models/norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    # Load cached features
    cache_path = "ml/data/all_video_features_cache.npz"
    cache = np.load(cache_path, allow_pickle=True)
    video_features = {k: cache[k] for k in cache.files}

    expected = {
        "jab": ("jab", 17),
        "cross": ("cross", 9),
        "hook": ("hook", 10),
        "uppercut": ("uppercut", 9),
        "walking": ("walking", 0),
        "idle": ("idle", 0),
        "mixed": ("mixed", 11),
    }

    total_expected = 56  # 17+9+10+9+11

    print("\n--- SWEEPING: min_dist, vel_thresh, valley_ratio, hook_z ---")
    all_configs = []

    for min_dist in [10, 12, 14, 16, 18, 20]:
        for vel_thresh_100 in range(35, 70, 5):
            vel_thresh = vel_thresh_100 / 1000.0
            for valley_100 in [25, 30, 35, 40, 45, 50]:
                valley_ratio = valley_100 / 100.0
                for hook_z_100 in [80, 90, 100, 110]:
                    hook_z = hook_z_100 / 1000.0
                    for uppercut_y_100 in [30, 35, 40]:
                        uppercut_y = uppercut_y_100 / 1000.0

                        total_correct = 0
                        total_fp = 0
                        total_missed = 0
                        per_video = {}

                        for name, feats_arr in video_features.items():
                            if feats_arr.size == 0:
                                continue
                            exp_move, exp_count = expected[name]
                            _, counts = run_detector(
                                feats_arr, model, mean, std,
                                min_dist, vel_thresh,
                                valley_ratio=valley_ratio,
                                hook_z_thresh=hook_z,
                                uppercut_y_thresh=uppercut_y,
                            )
                            c, fp, m = score_video(counts, exp_move, exp_count)
                            total_correct += c
                            total_fp += fp
                            total_missed += m
                            per_video[name] = (c, fp, m, dict(counts))

                        # Score: maximize correct, penalize FP, penalize missed
                        score = total_correct * 3 - total_fp * 2 - total_missed * 1

                        all_configs.append({
                            "score": score,
                            "correct": total_correct,
                            "fp": total_fp,
                            "missed": total_missed,
                            "min_dist": min_dist,
                            "vel_thresh": vel_thresh,
                            "valley_ratio": valley_ratio,
                            "hook_z_thresh": hook_z,
                            "uppercut_y_thresh": uppercut_y,
                            "per_video": per_video,
                        })

    all_configs.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Evaluated {len(all_configs)} configurations")
    print(f"\n--- TOP 15 CONFIGS ---")

    for i, cfg in enumerate(all_configs[:15]):
        print(f"\n  #{i+1}: score={cfg['score']}, correct={cfg['correct']}/{total_expected}, "
              f"fp={cfg['fp']}, missed={cfg['missed']}")
        print(f"       min_dist={cfg['min_dist']}, vel_thresh={cfg['vel_thresh']:.3f}, "
              f"valley={cfg['valley_ratio']:.2f}, hook_z={cfg['hook_z_thresh']:.3f}, "
              f"uppercut_y={cfg['uppercut_y_thresh']:.3f}")
        for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
            if name in cfg["per_video"]:
                c, f, m, counts = cfg["per_video"][name]
                exp_m, exp_c = expected[name]
                print(f"       {name:>10s}: exp={exp_c:>2d}, det={sum(counts.values()):>2d}, "
                      f"correct={c}, fp={f}, missed={m} | {counts}")

    # Find configs where idle+walking have 0 FP
    zero_fp_idle = [c for c in all_configs
                    if c["per_video"].get("idle", (0,0,0,{}))[1] == 0
                    and c["per_video"].get("walking", (0,0,0,{}))[1] == 0]
    zero_fp_idle.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n\n--- TOP 10 WITH ZERO IDLE/WALKING FP ({len(zero_fp_idle)} total) ---")
    for i, cfg in enumerate(zero_fp_idle[:10]):
        print(f"\n  #{i+1}: score={cfg['score']}, correct={cfg['correct']}/{total_expected}, "
              f"fp={cfg['fp']}, missed={cfg['missed']}")
        print(f"       min_dist={cfg['min_dist']}, vel_thresh={cfg['vel_thresh']:.3f}, "
              f"valley={cfg['valley_ratio']:.2f}, hook_z={cfg['hook_z_thresh']:.3f}, "
              f"uppercut_y={cfg['uppercut_y_thresh']:.3f}")
        for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
            if name in cfg["per_video"]:
                c, f, m, counts = cfg["per_video"][name]
                exp_m, exp_c = expected[name]
                print(f"       {name:>10s}: exp={exp_c:>2d}, det={sum(counts.values()):>2d}, "
                      f"correct={c}, fp={f}, missed={m} | {counts}")

    # Detailed output for the best zero-idle-FP config
    if zero_fp_idle:
        best = zero_fp_idle[0]
        print(f"\n\n--- DETAILED BEST (zero idle/walking FP) ---")
        for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
            feats_arr = video_features.get(name)
            if feats_arr is None or feats_arr.size == 0:
                continue
            exp_move, exp_count = expected[name]
            results, counts = run_detector(
                feats_arr, model, mean, std,
                best["min_dist"], best["vel_thresh"],
                valley_ratio=best["valley_ratio"],
                hook_z_thresh=best["hook_z_thresh"],
                uppercut_y_thresh=best["uppercut_y_thresh"],
            )
            c, fp, m = score_video(counts, exp_move, exp_count)
            total = sum(counts.values())
            print(f"\n  {name.upper():>10s}: expected={exp_count:>2d} {exp_move:<10s} "
                  f"detected={total:>2d} {counts}")
            print(f"             correct={c}, fp={fp}, missed={m}")
            for r in results:
                print(f"             [{r['time']:5.2f}s] {r['pred'].upper():>9s} "
                      f"(conf={r['conf']:.2f}, vel={r['vel']:.4f}) {r['reason']}")


if __name__ == "__main__":
    main()
