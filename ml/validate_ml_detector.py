"""
Validate the ML move detector against the user's real video.
Compare results with the rule-based detector.
"""
import os
import sys
import time

sys.path.insert(0, '/home/ubuntu/stick_fighter')

from core.pose_estimator import PoseEstimator, VideoSource
from core.smoothing import PoseSmoother
from core.move_detector import MoveDetector, MoveType, MoveDetectorConfig
from ml.ml_move_detector import MLMoveDetector

VIDEO_PATH = os.path.expanduser(
    "~/attachments/772c0aa0-9814-4cb5-b4f1-eace406bb88e/WIN_20260516_18_45_10_Pro.mp4"
)

# Expected moves from user: 2 jab, 3 cross, 3 hook, 3 uppercut = 11 total


def run_detector(detector, all_poses, name):
    """Run a detector on all pose frames and collect detected moves."""
    moves = []
    prev_move = MoveType.IDLE

    for i, pose in enumerate(all_poses):
        detected = detector.detect(pose)

        if (detected.move_type != MoveType.IDLE
                and detected.move_type != prev_move
                and detected.confidence > 0.3):
            ts = pose.timestamp_ms / 1000.0 if pose.valid else i / 30.0
            moves.append({
                "frame": i,
                "time": round(ts, 2),
                "type": detected.move_type.value,
                "hand": detected.hand,
                "confidence": round(detected.confidence, 3),
            })

        prev_move = detected.move_type

    return moves


def main():
    print("=" * 70)
    print("ML MOVE DETECTOR VALIDATION")
    print("=" * 70)

    # Process video
    print("\nProcessing video with MediaPipe...")
    video_src = VideoSource(VIDEO_PATH)
    pose_estimator = PoseEstimator(running_mode="VIDEO")
    smoother = PoseSmoother()

    all_poses = []
    while True:
        ret, frame, ts_ms = video_src.read()
        if not ret:
            break
        pose = pose_estimator.process_frame(frame, ts_ms)
        smoothed = smoother.smooth(pose)
        all_poses.append(smoothed)
    video_src.close()
    print(f"  {len(all_poses)} frames processed")

    # --- Rule-based detector ---
    print("\n" + "-" * 50)
    print("RULE-BASED DETECTOR:")
    print("-" * 50)
    rule_detector = MoveDetector(MoveDetectorConfig(
        cooldown_frames=20,
        min_move_frames=4,
        warmup_frames=12,
    ))
    rule_moves = run_detector(rule_detector, all_poses, "rule-based")

    rule_counts = {}
    for m in rule_moves:
        rule_counts[m["type"]] = rule_counts.get(m["type"], 0) + 1
        print(f"  [{m['time']:.2f}s] {m['type'].upper()} ({m['hand']}, conf={m['confidence']:.2f})")

    print(f"\n  Total: {len(rule_moves)} moves")
    print(f"  Counts: {rule_counts}")

    # --- ML detector ---
    print("\n" + "-" * 50)
    print("ML DETECTOR:")
    print("-" * 50)
    ml_detector = MLMoveDetector(
        model_dir="/home/ubuntu/stick_fighter/ml/models",
        window_size=16,
        classify_every=2,
        confidence_threshold=0.6,
        cooldown_frames=15,
    )
    ml_moves = run_detector(ml_detector, all_poses, "ml")

    ml_counts = {}
    for m in ml_moves:
        ml_counts[m["type"]] = ml_counts.get(m["type"], 0) + 1
        print(f"  [{m['time']:.2f}s] {m['type'].upper()} ({m['hand']}, conf={m['confidence']:.2f})")

    print(f"\n  Total: {len(ml_moves)} moves")
    print(f"  Counts: {ml_counts}")

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("COMPARISON (expected: 2 jab, 3 cross, 3 hook, 3 uppercut = 11 total)")
    print("=" * 70)

    expected = {"jab": 2, "cross": 3, "hook": 3, "uppercut": 3}

    print(f"\n{'Move':<12} {'Expected':>8} {'Rule-Based':>12} {'ML Model':>10}")
    print("-" * 44)
    for move in ["jab", "cross", "hook", "uppercut"]:
        exp = expected[move]
        rb = rule_counts.get(move, 0)
        ml = ml_counts.get(move, 0)
        print(f"{move:<12} {exp:>8} {rb:>12} {ml:>10}")

    total_exp = sum(expected.values())
    total_rb = sum(rule_counts.get(m, 0) for m in expected)
    total_ml = sum(ml_counts.get(m, 0) for m in expected)
    print("-" * 44)
    print(f"{'TOTAL':<12} {total_exp:>8} {total_rb:>12} {total_ml:>10}")

    # Accuracy
    rb_accuracy = sum(min(rule_counts.get(m, 0), expected[m]) for m in expected) / total_exp
    ml_accuracy = sum(min(ml_counts.get(m, 0), expected[m]) for m in expected) / total_exp
    print(f"\n  Rule-based accuracy (capped): {rb_accuracy:.1%}")
    print(f"  ML model accuracy (capped):   {ml_accuracy:.1%}")


if __name__ == "__main__":
    main()
