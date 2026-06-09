"""
Tune detector parameters across ALL individual videos simultaneously.
Optimizes for: correct detections + minimal false positives on idle/walking.
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from ml.step1_extract_all_videos import (
    extract_position_features, add_velocity_features,
    FEATURE_NAMES, CLASS_NAMES,
)
from ml.step7_final_tuned import MoveClassifierCNN

from core.pose_estimator import PoseEstimator, VideoSource
from core.smoothing import PoseSmoother


def get_idx(name):
    return FEATURE_NAMES.index(name)


VEL_INDICES = [get_idx(f"{h}_{c}") for h in ["left", "right"] for c in ["vwx", "vwy", "vwz"]]


def extract_raw_features(video_path: str) -> np.ndarray:
    """Extract per-frame feature matrix from a video via MediaPipe."""
    import cv2

    video_src = VideoSource(video_path)
    if not video_src.is_open:
        return np.array([])

    pose_estimator = PoseEstimator(running_mode="VIDEO")
    smoother = PoseSmoother()

    all_pos_features = []
    while True:
        ret, frame, ts_ms = video_src.read()
        if not ret:
            break
        pose = pose_estimator.process_frame(frame, ts_ms)
        smoothed = smoother.smooth(pose)
        features = extract_position_features(smoothed)
        all_pos_features.append(features)
    video_src.close()

    enriched = add_velocity_features(all_pos_features)
    feat_matrix = []
    for feat in enriched:
        if feat:
            row = [feat.get(k, 0.0) for k in FEATURE_NAMES]
        else:
            row = [0.0] * len(FEATURE_NAMES)
        feat_matrix.append(row)

    return np.array(feat_matrix, dtype=np.float32)


def detect_peaks(features, min_dist, vel_thresh):
    vel_mag = np.array([
        np.sqrt(sum(features[i, idx]**2 for idx in VEL_INDICES))
        for i in range(len(features))
    ])

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


def classify_peak(features, frame, model, mean, std, window_size=16,
                  hook_z_thresh=0.06, hook_z_ratio=2.0,
                  uppercut_y_thresh=0.035, uppercut_yz_ratio=1.3,
                  fallback_mode="none", fallback_vel_thresh=0.08):
    """Classify a peak with configurable heuristic parameters."""
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

    total_vel = np.sqrt(sum(features[frame, idx]**2 for idx in VEL_INDICES))

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

    # Rule 4: If ML says idle/walking — configurable fallback
    if ml_pred in (0, 5):
        if fallback_mode == "none":
            return CLASS_NAMES[ml_pred], ml_conf, "ML:trust"
        elif fallback_mode == "high_vel":
            if total_vel > fallback_vel_thresh:
                if z_vel > y_vel and z_vel > x_vel:
                    return "cross", 0.6, "VEL:fallback-z"
                if y_vel > z_vel:
                    return "uppercut", 0.6, "VEL:fallback-y"
                return "jab", 0.5, "VEL:fallback"
            return CLASS_NAMES[ml_pred], ml_conf, "ML:trust"
        else:  # "always" — original behavior
            if z_vel > y_vel and z_vel > x_vel:
                return "cross", 0.6, "VEL:fallback-z"
            if y_vel > z_vel:
                return "uppercut", 0.6, "VEL:fallback-y"
            return "jab", 0.5, "VEL:fallback"

    return CLASS_NAMES[ml_pred], ml_conf, "ML:default"


def run_detector(features, model, mean, std, min_dist, vel_thresh,
                 hook_z_thresh=0.06, hook_z_ratio=2.0,
                 uppercut_y_thresh=0.035, uppercut_yz_ratio=1.3,
                 fallback_mode="none", fallback_vel_thresh=0.08):
    peaks = detect_peaks(features, min_dist, vel_thresh)
    results = []
    for p in peaks:
        pred, conf, reason = classify_peak(
            features, p["frame"], model, mean, std,
            hook_z_thresh=hook_z_thresh, hook_z_ratio=hook_z_ratio,
            uppercut_y_thresh=uppercut_y_thresh, uppercut_yz_ratio=uppercut_yz_ratio,
            fallback_mode=fallback_mode, fallback_vel_thresh=fallback_vel_thresh,
        )
        if pred in ("idle", "walking"):
            continue
        results.append({"frame": p["frame"], "pred": pred, "conf": conf, "reason": reason})

    counts = {}
    for r in results:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1
    return results, counts


def score_video(counts, expected_move, expected_count):
    """Score a single video. Returns (correct, false_positives, missed)."""
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
    print("TUNING ACROSS ALL VIDEOS")
    print("=" * 70)

    # Load model
    model = MoveClassifierCNN()
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()
    norm = np.load("ml/models/norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    # Extract features from all videos (cached)
    cache_path = "ml/data/all_video_features_cache.npz"
    video_defs = [
        ("jab", os.path.expanduser("~/attachments/7c242094-6815-4982-8183-6ba478e39761/jab.mp4"), 17),
        ("cross", os.path.expanduser("~/attachments/f6777eed-4fc2-4feb-8c3b-9abc9d57dfd0/cross.mp4"), 9),
        ("hook", os.path.expanduser("~/attachments/e707b2dc-e951-4254-9392-d820e336513c/hook.mp4"), 10),
        ("uppercut", os.path.expanduser("~/attachments/624ad648-bd08-49bf-a27c-a490130fcc32/uppercut.mp4"), 9),
        ("walking", os.path.expanduser("~/attachments/da64e209-13fa-4872-b035-300051dc4292/walking_back_forth.mp4"), 0),
        ("idle", os.path.expanduser("~/attachments/d14f6498-0b85-481e-93da-0b2a78819631/idling_notwalking.mp4"), 0),
        ("mixed", os.path.expanduser("~/attachments/772c0aa0-9814-4cb5-b4f1-eace406bb88e/WIN_20260516_18_45_10_Pro.mp4"), 11),
    ]

    if os.path.exists(cache_path):
        print("  Loading cached features...")
        cache = np.load(cache_path, allow_pickle=True)
        video_features = {k: cache[k] for k in cache.files}
    else:
        print("  Extracting features from all videos...")
        video_features = {}
        for name, path, _ in video_defs:
            print(f"    {name}: {path}")
            feats = extract_raw_features(path)
            video_features[name] = feats
            print(f"      -> {feats.shape}")
        np.savez(cache_path, **video_features)
        print(f"  Cached to {cache_path}")

    # Define expected results
    expected = {
        "jab": ("jab", 17),
        "cross": ("cross", 9),
        "hook": ("hook", 10),
        "uppercut": ("uppercut", 9),
        "walking": ("walking", 0),
        "idle": ("idle", 0),
        "mixed": ("mixed", 11),
    }

    total_expected = 17 + 9 + 10 + 9 + 11  # 56 total punches

    # Sweep parameters
    print("\n--- PARAMETER SWEEP ---")
    best_score = -999
    best_config = None

    for min_dist in [12, 14, 16, 18, 20, 22, 25]:
        for vel_thresh_100 in range(30, 65, 5):
            vel_thresh = vel_thresh_100 / 1000.0
            for hook_z_100 in [60, 70, 80, 90, 100]:
                hook_z = hook_z_100 / 1000.0
                for uppercut_y_100 in [30, 35, 40]:
                    uppercut_y = uppercut_y_100 / 1000.0
                    for fallback_mode in ["none", "high_vel"]:
                        total_correct = 0
                        total_fp = 0
                        total_missed = 0

                        for name, feats_arr in video_features.items():
                            if feats_arr.size == 0:
                                continue
                            exp_move, exp_count = expected[name]
                            _, counts = run_detector(
                                feats_arr, model, mean, std,
                                min_dist, vel_thresh,
                                hook_z_thresh=hook_z,
                                uppercut_y_thresh=uppercut_y,
                                fallback_mode=fallback_mode,
                            )
                            c, fp, m = score_video(counts, exp_move, exp_count)
                            total_correct += c
                            total_fp += fp
                            total_missed += m

                        # Score: maximize correct, heavily penalize FP on idle/walking
                        score = total_correct * 3 - total_fp * 2 - total_missed * 1

                        if score > best_score:
                            best_score = score
                            best_config = {
                                "min_dist": min_dist,
                                "vel_thresh": vel_thresh,
                                "hook_z_thresh": hook_z,
                                "uppercut_y_thresh": uppercut_y,
                                "fallback_mode": fallback_mode,
                                "score": score,
                                "correct": total_correct,
                                "fp": total_fp,
                                "missed": total_missed,
                            }

    print(f"\n  BEST CONFIG:")
    for k, v in best_config.items():
        print(f"    {k}: {v}")

    # Run best config with verbose output
    print(f"\n--- DETAILED RESULTS (best config) ---")
    for name, feats_arr in video_features.items():
        if feats_arr.size == 0:
            continue
        exp_move, exp_count = expected[name]
        results, counts = run_detector(
            feats_arr, model, mean, std,
            best_config["min_dist"], best_config["vel_thresh"],
            hook_z_thresh=best_config["hook_z_thresh"],
            uppercut_y_thresh=best_config["uppercut_y_thresh"],
            fallback_mode=best_config["fallback_mode"],
        )
        c, fp, m = score_video(counts, exp_move, exp_count)
        total = sum(counts.values())
        status = "PASS" if fp == 0 and m <= 2 else "NEEDS_WORK"
        print(f"\n  {name.upper():>10s}: expected={exp_count:>2d} {exp_move:<10s} "
              f"detected={total:>2d} {counts}")
        print(f"             correct={c}, fp={fp}, missed={m} -> {status}")
        for r in results:
            t = r['frame'] / 30.0
            print(f"             [{t:5.2f}s] {r['pred'].upper():>9s} "
                  f"(conf={r['conf']:.2f}) {r['reason']}")

    # Also try top-5 configs
    print("\n\n--- TOP CONFIGS ---")
    all_configs = []
    for min_dist in [12, 14, 16, 18, 20, 22, 25]:
        for vel_thresh_100 in range(30, 65, 5):
            vel_thresh = vel_thresh_100 / 1000.0
            for hook_z_100 in [60, 70, 80, 90, 100]:
                hook_z = hook_z_100 / 1000.0
                for uppercut_y_100 in [30, 35, 40]:
                    uppercut_y = uppercut_y_100 / 1000.0
                    for fallback_mode in ["none", "high_vel"]:
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
                                hook_z_thresh=hook_z,
                                uppercut_y_thresh=uppercut_y,
                                fallback_mode=fallback_mode,
                            )
                            c, fp, m = score_video(counts, exp_move, exp_count)
                            total_correct += c
                            total_fp += fp
                            total_missed += m
                            per_video[name] = (c, fp, m, dict(counts))

                        score = total_correct * 3 - total_fp * 2 - total_missed * 1
                        all_configs.append((score, total_correct, total_fp, total_missed,
                                          min_dist, vel_thresh, hook_z, uppercut_y,
                                          fallback_mode, per_video))

    all_configs.sort(key=lambda x: x[0], reverse=True)
    for i, (score, correct, fp, missed, md, vt, hz, uy, fb, pv) in enumerate(all_configs[:10]):
        print(f"\n  #{i+1}: score={score}, correct={correct}/{total_expected}, "
              f"fp={fp}, missed={missed}")
        print(f"       min_dist={md}, vel_thresh={vt:.3f}, "
              f"hook_z={hz:.3f}, uppercut_y={uy:.3f}, fallback={fb}")
        for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
            if name in pv:
                c, f, m, counts = pv[name]
                exp_m, exp_c = expected[name]
                print(f"       {name:>10s}: exp={exp_c:>2d}, det={sum(counts.values()):>2d}, "
                      f"correct={c}, fp={f}, missed={m} | {counts}")


if __name__ == "__main__":
    main()
