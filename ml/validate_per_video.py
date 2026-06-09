"""
Validate ML hybrid detector performance on each individual labeled video.
Extracts features via MediaPipe, runs the detector, and reports results.
"""
import os
import sys
import math
import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from ml.step1_extract_all_videos import (
    extract_position_features, add_velocity_features, process_video,
    FEATURE_NAMES, CLASS_NAMES,
)
from ml.step7_final_tuned import (
    MoveClassifierCNN, detect_peaks, classify, run_full,
)
from core.pose_estimator import PoseEstimator, VideoSource


def extract_raw_features(video_path: str) -> np.ndarray:
    """Extract per-frame feature matrix from a video via MediaPipe."""
    import cv2
    from core.smoothing import PoseSmoother

    video_src = VideoSource(video_path)
    if not video_src.is_open:
        print(f"  ERROR: Cannot open {video_path}")
        return np.array([])

    fps = video_src.cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = video_src.total_frames
    print(f"  {total} frames @ {fps:.0f} fps")

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


def main():
    print("=" * 70)
    print("ML HYBRID DETECTOR — PER-VIDEO VALIDATION")
    print("=" * 70)

    # Load model
    device = torch.device("cpu")
    model = MoveClassifierCNN().to(device)
    model.load_state_dict(torch.load("ml/models/move_classifier.pt", weights_only=True))
    model.eval()

    norm = np.load("ml/models/norm_stats.npz")
    mean, std = norm["mean"], norm["std"]

    # Detector parameters (tuned)
    min_dist = 18
    vel_thresh = 0.0425

    # Videos with expected results from user
    videos = [
        {
            "path": os.path.expanduser("~/attachments/7c242094-6815-4982-8183-6ba478e39761/jab.mp4"),
            "name": "jab.mp4",
            "expected_move": "jab",
            "expected_count": 17,
            "notes": "17 jabs",
        },
        {
            "path": os.path.expanduser("~/attachments/f6777eed-4fc2-4feb-8c3b-9abc9d57dfd0/cross.mp4"),
            "name": "cross.mp4",
            "expected_move": "cross",
            "expected_count": 9,
            "notes": "9 crosses",
        },
        {
            "path": os.path.expanduser("~/attachments/e707b2dc-e951-4254-9392-d820e336513c/hook.mp4"),
            "name": "hook.mp4",
            "expected_move": "hook",
            "expected_count": 10,
            "notes": "10 hooks (4L, 4R, 2L)",
        },
        {
            "path": os.path.expanduser("~/attachments/624ad648-bd08-49bf-a27c-a490130fcc32/uppercut.mp4"),
            "name": "uppercut.mp4",
            "expected_move": "uppercut",
            "expected_count": 9,
            "notes": "9 uppercuts (LRLRLRRLR)",
        },
        {
            "path": os.path.expanduser("~/attachments/da64e209-13fa-4872-b035-300051dc4292/walking_back_forth.mp4"),
            "name": "walking_back_forth.mp4",
            "expected_move": "walking",
            "expected_count": 0,
            "notes": "Walking forward/backward x2 — should detect 0 punches",
        },
        {
            "path": os.path.expanduser("~/attachments/d14f6498-0b85-481e-93da-0b2a78819631/idling_notwalking.mp4"),
            "name": "idling_notwalking.mp4",
            "expected_move": "idle",
            "expected_count": 0,
            "notes": "Idle/standing — should detect 0 punches",
        },
        {
            "path": os.path.expanduser(
                "~/attachments/772c0aa0-9814-4cb5-b4f1-eace406bb88e/WIN_20260516_18_45_10_Pro.mp4"),
            "name": "mixed_video.mp4",
            "expected_move": "mixed",
            "expected_count": 11,
            "notes": "2 jab, 3 cross, 3 hook, 3 uppercut = 11 total",
            "expected_breakdown": {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3},
        },
    ]

    all_results = []

    for vid in videos:
        print(f"\n{'='*70}")
        print(f"VIDEO: {vid['name']}")
        print(f"Expected: {vid['notes']}")
        print(f"{'='*70}")

        if not os.path.exists(vid["path"]):
            print(f"  SKIPPED — file not found: {vid['path']}")
            all_results.append({"video": vid["name"], "status": "SKIPPED"})
            continue

        features = extract_raw_features(vid["path"])
        if features.size == 0:
            print(f"  SKIPPED — no features extracted")
            all_results.append({"video": vid["name"], "status": "ERROR"})
            continue

        print(f"\n  Running detector (min_dist={min_dist}, vel_thresh={vel_thresh})...")
        results, counts = run_full(features, model, mean, std, min_dist, vel_thresh, verbose=True)

        total_punches = sum(counts.values())
        print(f"\n  RESULTS:")
        print(f"    Total punches detected: {total_punches}")
        print(f"    Breakdown: {counts}")

        if vid["expected_move"] in ("walking", "idle"):
            # Should detect 0 punches
            fp = total_punches
            correct_move_count = 0
            accuracy_str = f"{0 if fp > 0 else 100}% (0 expected, {fp} false positives)"
            status = "PASS" if fp == 0 else f"FAIL ({fp} false positives)"
        elif vid["expected_move"] == "mixed":
            expected_bd = vid.get("expected_breakdown", {})
            correct = 0
            for move, exp_count in expected_bd.items():
                det_count = counts.get(move, 0)
                correct += min(det_count, exp_count)
            fp = total_punches - correct
            accuracy_str = f"{correct}/{vid['expected_count']} correct, {fp} FP"
            status = f"{'PASS' if correct >= vid['expected_count'] - 1 else 'FAIL'} ({accuracy_str})"
        else:
            # Single-move video
            correct_move_count = counts.get(vid["expected_move"], 0)
            other_moves = {k: v for k, v in counts.items() if k != vid["expected_move"]}
            misclassified = sum(other_moves.values())
            accuracy_str = (
                f"{correct_move_count}/{vid['expected_count']} {vid['expected_move']}s detected"
                + (f", {misclassified} misclassified as {other_moves}" if misclassified else "")
            )
            diff = abs(correct_move_count - vid["expected_count"])
            status = f"{'PASS' if diff <= 2 else 'FAIL'} ({accuracy_str})"

        print(f"    Status: {status}")

        all_results.append({
            "video": vid["name"],
            "expected_move": vid["expected_move"],
            "expected_count": vid["expected_count"],
            "detected_total": total_punches,
            "breakdown": dict(counts),
            "detections": results,
            "status": status,
        })

    # Summary
    print(f"\n\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Video':<30s} {'Expected':<25s} {'Detected':<30s} {'Status'}")
    print("-" * 100)
    for r in all_results:
        if r.get("status") == "SKIPPED":
            print(f"{r['video']:<30s} {'—':<25s} {'—':<30s} SKIPPED")
            continue
        expected_str = f"{r['expected_count']} {r['expected_move']}(s)"
        detected_str = f"{r['detected_total']} total: {r.get('breakdown', {})}"
        print(f"{r['video']:<30s} {expected_str:<25s} {detected_str:<30s} {r['status']}")

    pass_count = sum(1 for r in all_results if "PASS" in r.get("status", ""))
    total_count = sum(1 for r in all_results if r.get("status") != "SKIPPED")
    print(f"\n  Result: {pass_count}/{total_count} videos passed")

    return all_results


if __name__ == "__main__":
    main()
