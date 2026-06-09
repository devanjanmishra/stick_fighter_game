"""
Tuning v4: Refined classification — trust high-confidence ML over velocity heuristics.
Only use velocity heuristics when ML is uncertain.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from ml.step1_extract_all_videos import FEATURE_NAMES, CLASS_NAMES
from ml.step7_final_tuned import MoveClassifierCNN
from ml.tune_v3 import (
    get_idx, VEL_INDICES, compute_vel_mag, detect_peaks_v2, score_video,
)


def classify_peak_v4(features, frame, model, mean, std, window_size=16,
                     hook_z_thresh=0.09, uppercut_y_thresh=0.035,
                     ml_trust_thresh=0.85):
    """Classify with ML-first approach: trust high-confidence ML predictions."""
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

    # Rule 0: If ML says idle/walking, TRUST IT (no override)
    if ml_pred in (0, 5):
        return CLASS_NAMES[ml_pred], ml_conf, "ML:trust-idle"

    # Rule 1: If ML is very confident about any punch type, trust it
    if ml_pred in (1, 2, 3, 4) and ml_conf >= ml_trust_thresh:
        return CLASS_NAMES[ml_pred], ml_conf, "ML:high-conf"

    # Rule 2: ML is uncertain — use velocity heuristics to help
    # HOOK heuristic
    if z_vel > hook_z_thresh and z_vel > y_vel * 2.0:
        return "hook", max(0.8, float(probs[3])), "VEL:z-dominant"

    # UPPERCUT heuristic
    if y_vel > uppercut_y_thresh and y_vel > z_vel * 1.3 and y_vel > x_vel * 1.5:
        return "uppercut", max(0.75, float(probs[4])), "VEL:y-dominant"

    # Trust ML for moderate-confidence predictions
    if ml_pred in (1, 2, 3, 4) and ml_conf > 0.4:
        return CLASS_NAMES[ml_pred], ml_conf, "ML:moderate"

    return CLASS_NAMES[ml_pred], ml_conf, "ML:default"


def run_detector_v4(features, model, mean, std, min_dist, vel_thresh,
                    valley_ratio=0.25, hook_z_thresh=0.09, uppercut_y_thresh=0.035,
                    heavy_gap=35, light_gap=12, ml_trust_thresh=0.85):
    peaks = detect_peaks_v2(features, min_dist, vel_thresh, valley_ratio=valley_ratio)

    classified = []
    for p in peaks:
        pred, conf, reason = classify_peak_v4(
            features, p["frame"], model, mean, std,
            hook_z_thresh=hook_z_thresh, uppercut_y_thresh=uppercut_y_thresh,
            ml_trust_thresh=ml_trust_thresh,
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

    # Post-classification dedup with type-specific gaps
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
    print("TUNING v4: ML-FIRST CLASSIFICATION WITH HEURISTIC FALLBACK")
    print("=" * 70)

    model = MoveClassifierCNN()
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()
    norm = np.load("ml/models/norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    cache = np.load("ml/data/all_video_features_cache.npz", allow_pickle=True)
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

    total_expected = 56

    # Sweep ml_trust_thresh and other key parameters
    print("\n--- SWEEPING ---")
    all_configs = []

    for min_dist in [10, 12, 14]:
        for vel_thresh_100 in [40, 45, 50, 55, 60]:
            vel_thresh = vel_thresh_100 / 1000.0
            for valley_100 in [20, 25, 30]:
                valley_ratio = valley_100 / 100.0
                for heavy_gap in [35, 40, 45, 50]:
                    for light_gap in [12, 15]:
                        for ml_trust_100 in [75, 80, 85, 90]:
                            ml_trust = ml_trust_100 / 100.0
                            for hook_z_100 in [80, 90, 100]:
                                hook_z = hook_z_100 / 1000.0

                                total_correct = 0
                                total_fp = 0
                                total_missed = 0
                                per_video = {}

                                for name, feats_arr in video_features.items():
                                    if feats_arr.size == 0:
                                        continue
                                    exp_move, exp_count = expected[name]
                                    _, counts = run_detector_v4(
                                        feats_arr, model, mean, std,
                                        min_dist, vel_thresh,
                                        valley_ratio=valley_ratio,
                                        hook_z_thresh=hook_z,
                                        heavy_gap=heavy_gap,
                                        light_gap=light_gap,
                                        ml_trust_thresh=ml_trust,
                                    )
                                    c, fp, m = score_video(counts, exp_move, exp_count)
                                    total_correct += c
                                    total_fp += fp
                                    total_missed += m
                                    per_video[name] = (c, fp, m, dict(counts))

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
                                    "heavy_gap": heavy_gap,
                                    "light_gap": light_gap,
                                    "ml_trust_thresh": ml_trust,
                                    "per_video": per_video,
                                })

    all_configs.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n  Evaluated {len(all_configs)} configurations")

    # Filter: zero idle/walking FP
    zero_fp = [c for c in all_configs
               if c["per_video"].get("idle", (0,0,0,{}))[1] == 0
               and c["per_video"].get("walking", (0,0,0,{}))[1] == 0]
    zero_fp.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n--- TOP 10 WITH ZERO IDLE/WALKING FP ({len(zero_fp)} total) ---")
    for i, cfg in enumerate(zero_fp[:10]):
        print(f"\n  #{i+1}: score={cfg['score']}, correct={cfg['correct']}/{total_expected}, "
              f"fp={cfg['fp']}, missed={cfg['missed']}")
        print(f"       min_dist={cfg['min_dist']}, vel={cfg['vel_thresh']:.3f}, "
              f"valley={cfg['valley_ratio']:.2f}, hook_z={cfg['hook_z_thresh']:.3f}, "
              f"heavy_gap={cfg['heavy_gap']}, light_gap={cfg['light_gap']}, "
              f"ml_trust={cfg['ml_trust_thresh']:.2f}")
        for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
            if name in cfg["per_video"]:
                c, f, m, counts = cfg["per_video"][name]
                exp_m, exp_c = expected[name]
                det = sum(counts.values())
                print(f"       {name:>10s}: exp={exp_c:>2d}, det={det:>2d}, "
                      f"correct={c}, fp={f}, missed={m} | {counts}")

    # Detailed output
    if zero_fp:
        best = zero_fp[0]
        print(f"\n\n--- DETAILED BEST ---")
        print(f"  Config: min_dist={best['min_dist']}, vel={best['vel_thresh']:.3f}, "
              f"valley={best['valley_ratio']:.2f}, hook_z={best['hook_z_thresh']:.3f}, "
              f"heavy_gap={best['heavy_gap']}, light_gap={best['light_gap']}, "
              f"ml_trust={best['ml_trust_thresh']:.2f}")
        for name in ["jab", "cross", "hook", "uppercut", "walking", "idle", "mixed"]:
            feats_arr = video_features.get(name)
            if feats_arr is None or feats_arr.size == 0:
                continue
            exp_move, exp_count = expected[name]
            results, counts = run_detector_v4(
                feats_arr, model, mean, std,
                best["min_dist"], best["vel_thresh"],
                valley_ratio=best["valley_ratio"],
                hook_z_thresh=best["hook_z_thresh"],
                heavy_gap=best["heavy_gap"],
                light_gap=best["light_gap"],
                ml_trust_thresh=best["ml_trust_thresh"],
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
