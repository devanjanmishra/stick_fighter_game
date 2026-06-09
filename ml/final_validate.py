"""
Final validation: run the tuned v3 detector on all 7 labeled videos
using cached features and report comprehensive results.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from ml.step1_extract_all_videos import FEATURE_NAMES, CLASS_NAMES
from ml.step7_final_tuned import MoveClassifierCNN
from ml.tune_v3 import (
    detect_peaks_v2, score_video, get_idx, VEL_INDICES, compute_vel_mag,
)


def classify_peak(features, frame, model, mean, std, window_size=16,
                  hook_z_thresh=0.09, uppercut_y_thresh=0.035):
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

    if ml_pred == 3 and ml_conf > 0.7:
        return "hook", ml_conf, "ML"
    if z_vel > hook_z_thresh and z_vel > y_vel * 2.0:
        return "hook", max(0.8, float(probs[3])), "VEL:z-dominant"

    if y_vel > uppercut_y_thresh and y_vel > z_vel * 1.3 and y_vel > x_vel * 1.5:
        return "uppercut", max(0.75, float(probs[4])), "VEL:y-dominant"

    if ml_pred in (1, 2) and ml_conf > 0.5:
        return CLASS_NAMES[ml_pred], ml_conf, "ML"

    if ml_pred in (0, 5):
        return CLASS_NAMES[ml_pred], ml_conf, "ML:trust-idle"

    return CLASS_NAMES[ml_pred], ml_conf, "ML:default"


def run_detector(features, model, mean, std,
                 min_dist=10, vel_thresh=0.040, valley_ratio=0.25,
                 hook_z_thresh=0.090, uppercut_y_thresh=0.035,
                 heavy_gap=35, light_gap=12):
    peaks = detect_peaks_v2(features, min_dist, vel_thresh, valley_ratio=valley_ratio)

    classified = []
    for p in peaks:
        pred, conf, reason = classify_peak(
            features, p["frame"], model, mean, std,
            hook_z_thresh=hook_z_thresh, uppercut_y_thresh=uppercut_y_thresh,
        )
        if pred in ("idle", "walking"):
            continue
        classified.append({
            "frame": p["frame"],
            "time": round(p["frame"] / 30.0, 2),
            "pred": pred,
            "conf": round(conf, 3),
            "reason": reason,
            "vel": round(p["vel"], 4),
        })

    if len(classified) <= 1:
        counts = {}
        for r in classified:
            counts[r["pred"]] = counts.get(r["pred"], 0) + 1
        return classified, counts

    deduped = [classified[0]]
    for r in classified[1:]:
        prev = deduped[-1]
        gap = r["frame"] - prev["frame"]

        if r["pred"] == prev["pred"]:
            is_heavy = r["pred"] in ("hook", "uppercut")
            required_gap = heavy_gap if is_heavy else light_gap
            if gap >= required_gap:
                deduped.append(r)
            elif r["vel"] > prev["vel"]:
                deduped[-1] = r
        else:
            deduped.append(r)

    counts = {}
    for r in deduped:
        counts[r["pred"]] = counts.get(r["pred"], 0) + 1
    return deduped, counts


def main():
    print("=" * 70)
    print("FINAL VALIDATION: TUNED ML HYBRID DETECTOR")
    print("Parameters: min_dist=10, vel=0.040, valley=0.25,")
    print("            hook_z=0.090, heavy_gap=35, light_gap=12")
    print("=" * 70)

    model = MoveClassifierCNN()
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()
    norm = np.load("ml/models/norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    cache = np.load("ml/data/all_video_features_cache.npz", allow_pickle=True)
    video_features = {k: cache[k] for k in cache.files}

    expected = {
        "jab": ("jab", 17, "17 jabs"),
        "cross": ("cross", 9, "9 crosses"),
        "hook": ("hook", 10, "10 hooks (4L, 4R, 2L)"),
        "uppercut": ("uppercut", 9, "9 uppercuts (LRLRLRRLR)"),
        "walking": ("walking", 0, "Walking — 0 punches expected"),
        "idle": ("idle", 0, "Idle — 0 punches expected"),
        "mixed": ("mixed", 11, "2 jab + 3 cross + 3 hook + 3 uppercut = 11"),
    }

    total_correct = 0
    total_fp = 0
    total_missed = 0
    total_expected = 56

    for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
        feats = video_features.get(name)
        if feats is None or feats.size == 0:
            print(f"\n  {name}: SKIPPED (no features)")
            continue

        exp_move, exp_count, notes = expected[name]
        results, counts = run_detector(feats, model, mean, std)
        c, fp, m = score_video(counts, exp_move, exp_count)

        total_correct += c
        total_fp += fp
        total_missed += m

        total_det = sum(counts.values())

        if exp_move in ("idle", "walking"):
            status = "PASS" if fp == 0 else f"FAIL ({fp} FP)"
        elif exp_move == "mixed":
            status = f"{'PASS' if c >= exp_count - 1 else 'FAIL'} ({c}/{exp_count} correct, {fp} FP)"
        else:
            correct_type = counts.get(exp_move, 0)
            diff = abs(correct_type - exp_count)
            misclass = {k: v for k, v in counts.items() if k != exp_move}
            status = f"{'PASS' if diff <= 2 else 'FAIL'}"
            if misclass:
                status += f" (misclassified: {misclass})"

        print(f"\n  {name.upper():<12s} Expected: {notes}")
        print(f"               Detected: {total_det} total — {counts}")
        print(f"               Correct={c}, FP={fp}, Missed={m} → {status}")
        for r in results:
            print(f"               [{r['time']:5.2f}s] {r['pred'].upper():>9s} "
                  f"(conf={r['conf']:.2f}, vel={r['vel']:.4f}) {r['reason']}")

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Correct:  {total_correct}/{total_expected} ({100*total_correct/total_expected:.1f}%)")
    print(f"  FP:       {total_fp}")
    print(f"  Missed:   {total_missed}")
    print(f"  Score:    {total_correct * 3 - total_fp * 2 - total_missed}")

    print(f"\n  Per-video summary:")
    print(f"  {'Video':<15s} {'Expected':<10s} {'Detected':<10s} {'Correct':<10s} {'FP':<5s} {'Missed':<8s}")
    print(f"  {'-'*58}")
    for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
        feats = video_features.get(name)
        if feats is None or feats.size == 0:
            continue
        exp_move, exp_count, _ = expected[name]
        _, counts = run_detector(feats, model, mean, std)
        c, fp, m = score_video(counts, exp_move, exp_count)
        total_det = sum(counts.values())
        print(f"  {name:<15s} {exp_count:<10d} {total_det:<10d} {c:<10d} {fp:<5d} {m:<8d}")


if __name__ == "__main__":
    main()
